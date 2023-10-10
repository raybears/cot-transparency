import argparse

import streamlit as st
import streamlit.components.v1 as components
from slist import Slist
from streamlit.delta_generator import DeltaGenerator
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.util import assert_not_none
from cot_transparency.data_models.models import (
    TaskOutput,
)
from scripts.streamlit_viewer_components.answer_options import (
    select_bias_on_where_option,
    TypeOfAnswerOption,
    select_left_model_result_option,
)
from scripts.streamlit_viewer_components.viewer_cache import (
    cached_read_whole_exp_dir,
    cached_search,
    get_data_dropdowns,
    DataDropDowns,
    make_tree,
    TreeCache,
    TreeCacheKey,
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
args = parser.parse_args()
exp_dir: str = args.exp_dir


def display_task(task: TaskOutput):
    model_output = task.inference_output.parsed_response
    ground_truth = task.task_spec.ground_truth
    bias_on = task.task_spec.biased_ans
    is_correct = model_output == ground_truth
    emoji = "✔️" if is_correct else "❌"
    st.markdown(f"Ground truth: {ground_truth}")
    st.markdown(f"Model output: {model_output} {emoji}")
    st.markdown(f"Bias on: {bias_on}")

    messages: list[ChatMessage] = task.task_spec.messages
    messages = messages + [ChatMessage(role=MessageRole.assistant, content=task.inference_output.raw_response)]

    for msg in messages:
        # pattern match on msg.role
        match msg.role:
            case MessageRole.none:
                st.markdown(msg.content)
            case MessageRole.system:
                st.markdown("### System")
                st.markdown(msg.content.replace("\n", "  \n"))
            case MessageRole.user:
                with st.chat_message("user"):
                    st.markdown("### User")
                    st.markdown(msg.content.replace("\n", "  \n"))
            case MessageRole.assistant:
                with st.chat_message("assistant"):
                    st.markdown("### Assistant")
                    st.markdown(msg.content.replace("\n", "  \n"))
            case MessageRole.assistant_if_completion:
                with st.chat_message("assistant"):
                    st.markdown("### Assistant if completion")
                    st.markdown(msg.content.replace("\n", "  \n"))


# naughty Slist patch to add __hash__ by id so that lru works
def __hash__(self):  # type: ignore
    return id(self)


Slist.__hash__ = __hash__  # type: ignore

# Ask the user to enter experiment_dir
exp_dir = st.text_input("Enter experiment_dir", exp_dir)
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
    formatter_drop_down_selection: str = st.selectbox("Select formatter", data_dropdowns.formatters, key=f"formatter_{i}")  # type: ignore
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
        display_task(first)
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
        display_task(first)
    else:
        st.write("No tasks matching")
