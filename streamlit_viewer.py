import argparse

import streamlit as st
import streamlit.components.v1 as components
from slist import Slist
from streamlit.delta_generator import DeltaGenerator

from cot_transparency.data_models.models import StageTwoTaskOutput, TaskOutput
from cot_transparency.util import assert_not_none
from cot_transparency.viewer.answer_options import (
    TypeOfAnswerOption,
    select_bias_on_where_option,
    select_left_model_result_option,
)
from cot_transparency.viewer.util import display_task
from cot_transparency.viewer.viewer_cache import (
    DataDropDowns,
    TreeCache,
    TreeCacheKey,
    cached_read_whole_exp_dir,
    cached_read_whole_s2_exp_dir,
    cached_search,
    get_data_dropdowns,
    make_tree,
)

# set to wide
st.set_page_config(layout="wide")

# ruff: noqa: E501

parser = argparse.ArgumentParser()

parser.add_argument(
    "exp_dir",
    default="experiments/finetune",
    help="The experiment directory to load from",
)
parser.add_argument(
    "--stage_two",
    action="store_true",
    default=False,
    help="Whether the experiment dir is a stage two experiment",
)
args = parser.parse_args()
exp_dir: str = args.exp_dir
is_stage_two: bool = args.stage_two


# naughty Slist patch to add __hash__ by id so that lru works
def __hash__(self):  # type: ignore
    return id(self)


Slist.__hash__ = __hash__  # type: ignore


# Ask the user to enter experiment_dir
exp_dir = st.text_input("Enter experiment_dir", exp_dir)
if is_stage_two:
    s2_outputs: Slist[StageTwoTaskOutput] = cached_read_whole_s2_exp_dir(exp_dir=exp_dir)
    everything = s2_outputs.map(lambda x: x.to_s1())
else:
    everything: Slist[TaskOutput] = cached_read_whole_exp_dir(exp_dir=exp_dir)
tree: TreeCache = make_tree(everything)  # type: ignore
st.markdown(f"Loaded {len(everything)} tasks")
# Calculate what mdoels / tasks are available
data_dropdowns: DataDropDowns = get_data_dropdowns(everything)  # type: ignore
task_selection: str = assert_not_none(st.selectbox("Select task", data_dropdowns.tasks))
intervention_drop_down_selection: str | None = st.selectbox("Select intervention", data_dropdowns.interventions)
bias_on_where: TypeOfAnswerOption = select_bias_on_where_option()
answer_result_option: TypeOfAnswerOption = select_left_model_result_option()
# Optional text input
prompt_search: str = st.text_input("Search for text in the prompt for the left model")
completion_search: str = st.text_input("Search for text in final completion for the left model")
put_if_completion_in_user: bool = st.checkbox("Put the assistant if completion in the last user message", value=True)


# Create a button which will increment the counter
increment = st.button("Next")
if "count" not in st.session_state:
    st.session_state.count = 0
if increment:
    st.session_state.count += 1

# A button to decrement the counter
decrement = st.button("Previous")
if decrement:
    st.session_state.count -= 1

# needs to happen after the buttons
components.html(
    r"""
<script>
const doc = window.parent.document;
buttons = Array.from(doc.querySelectorAll('button[kind=secondary]'));
const left_button = buttons.find(el => el.innerText.includes('Previous'));
const right_button = buttons.find(el => el.innerText.includes('Next'));
doc.addEventListener('keydown', function(e) {
    switch (e.keyCode) {
        case 37: // (37 = left arrow)
            left_button.click();
            break;
        case 39: // (39 = right arrow)
            right_button.click();
            break;
    }
});
</script>
""",
    height=0,
    width=0,
)


# split into two columns
left: DeltaGenerator
right: DeltaGenerator
left, right = st.columns(2)
with left:
    i = 0
    formatter_drop_down_selection: str = st.selectbox(
        "Select formatter", data_dropdowns.formatters, key=f"formatter_{i}"
    )  # type: ignore
    model_drop_down_selection: str = st.selectbox("Select model", data_dropdowns.models, key=f"model_{i}")  # type: ignore
    filtered = cached_search(
        completion_search=completion_search,
        prompt_search=prompt_search,
        tree_cache_key=TreeCacheKey(
            task=task_selection,
            model=model_drop_down_selection,
            formatter=formatter_drop_down_selection,
            intervention=intervention_drop_down_selection,
        ),
        tree_cache=tree,
        bias_on_where=bias_on_where,
        task_hash=None,
        answer_result_option=answer_result_option,
    )
    st.markdown(f"Showing {len(filtered)} tasks matching criteria")
    show_item_idx = st.session_state.count % len(filtered) if len(filtered) > 0 else 0
    first = filtered[show_item_idx] if len(filtered) > 0 else None
    first_task_hash: str | None = filtered[show_item_idx].task_spec.task_hash if len(filtered) > 0 else None
    if first:
        display_task(first, put_if_completion_in_user=put_if_completion_in_user)
with right:
    i = 1
    formatter_drop_down_selection: str = assert_not_none(
        st.selectbox("Select formatter", data_dropdowns.formatters, key=f"formatter_{i}")
    )
    model_drop_down_selection: str = assert_not_none(
        st.selectbox("Select model", data_dropdowns.models, key=f"model_{i}")
    )
    filtered = cached_search(
        tree_cache_key=TreeCacheKey(
            task=task_selection,
            model=model_drop_down_selection,
            formatter=formatter_drop_down_selection,
            intervention=intervention_drop_down_selection,
        ),
        task_hash=first_task_hash,
        tree_cache=tree,
        # For the right model, no filters for the below since we just want to join on the task_hash
        completion_search=None,
        prompt_search=None,
        answer_result_option=TypeOfAnswerOption.anything,
        bias_on_where=TypeOfAnswerOption.anything,
    )
    st.markdown(f"Showing {len(filtered)} tasks matching criteria")
    first: TaskOutput | None = filtered.first_option
    if first:
        display_task(first, put_if_completion_in_user=put_if_completion_in_user)
    else:
        st.write("No tasks matching")
