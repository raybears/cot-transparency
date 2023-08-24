import streamlit as st
from slist import Slist
from streamlit.delta_generator import DeltaGenerator

from cot_transparency.data_models.models import (
    TaskOutput,
    ChatMessage,
    StrictChatMessage,
    MessageRole,
    StrictMessageRole,
)
from cot_transparency.model_apis import Prompt, ModelType
from scripts.better_viewer_cache import (
    cached_read_whole_exp_dir,
    cached_search,
    get_drop_downs,
    DropDowns,
    make_tree,
    TreeCache,
    TreeCacheKey,
)
import streamlit.components.v1 as components




def display_task(task: TaskOutput):
    model_output = task.inference_output.parsed_response
    ground_truth = task.task_spec.ground_truth
    is_correct = model_output == ground_truth
    emoji = "✔️" if is_correct else "❌"
    st.markdown(f"Ground truth: {ground_truth}")
    st.markdown(f"Model output: {model_output} {emoji}")

    messages: list[ChatMessage] = task.task_spec.messages
    model_type: ModelType = ModelType.from_model_name(task.task_spec.inference_config.model)
    strict: list[StrictChatMessage] = Prompt(
        messages=messages + [ChatMessage(role=MessageRole.assistant, content=task.inference_output.raw_response)]
    ).get_strict_messages(model_type=model_type)
    for msg in strict:
        # pattern match on msg.role
        match msg.role:
            case StrictMessageRole.none:
                st.code(msg.content)
            case StrictMessageRole.system:
                st.markdown("### System")
                st.code(msg.content, None)
            case StrictMessageRole.user:
                with st.chat_message("user"):
                    st.markdown("### User")
                    st.code(msg.content, None)
            case StrictMessageRole.assistant:
                with st.chat_message("assistant"):
                    st.markdown("### Assistant")
                    st.code(msg.content, None)

    # # write the final response
    # with st.chat_message("assistant"):
    #     st.markdown("### Assistant")
    #     st.code(task.inference_output.raw_response, None)


# naughty Slist patch to add __hash__ by id
def __hash__(self):
    return id(self)


Slist.__hash__ = __hash__

# Ask the user to enter experiment_dir
exp_dir = st.text_input("Enter experiment_dir", "experiments/math_scale")
everything: Slist[TaskOutput] = cached_read_whole_exp_dir(exp_dir=exp_dir)
tree: TreeCache = make_tree(everything)
st.markdown(f"Loaded {len(everything)} tasks")
# Optional text input
completion_search: str = st.text_input("Search for text in final completion")
drop_downs: DropDowns = get_drop_downs(everything)
task_selection: str = st.selectbox("Select task", drop_downs.tasks)
intervention_drop_down_selection: str | None = st.selectbox("Select intervention", drop_downs.interventions)
bias_on_wrong_answer: bool = st.checkbox("Show only bias on wrong answer", value=True)

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
    formatter_drop_down_selection: str = st.selectbox("Select formatter", drop_downs.formatters, key=f"formatter_{i}")
    model_drop_down_selection: str = st.selectbox("Select model", drop_downs.models, key=f"model_{i}")
    filtered = cached_search(
        completion_search=completion_search,
        tree_cache_key=TreeCacheKey(
            task=task_selection,
            model=model_drop_down_selection,
            formatter=formatter_drop_down_selection,
            intervention=intervention_drop_down_selection,
        ),
        tree_cache=tree,
        only_bias_on_wrong_answer=bias_on_wrong_answer,
        task_hash=None,
    )
    st.markdown(f"Showing {len(filtered)} tasks matching criteria")
    show_item_idx = st.session_state.count % len(filtered)
    first = filtered[show_item_idx]
    first_task_hash: str | None = filtered[show_item_idx].task_spec.task_hash
    if first:
        display_task(first)
with right:
    i = 1
    formatter_drop_down_selection: str = st.selectbox("Select formatter", drop_downs.formatters, key=f"formatter_{i}")
    model_drop_down_selection: str = st.selectbox("Select model", drop_downs.models, key=f"model_{i}")
    filtered = cached_search(
        completion_search=completion_search,
        tree_cache_key=TreeCacheKey(
            task=task_selection,
            model=model_drop_down_selection,
            formatter=formatter_drop_down_selection,
            intervention=intervention_drop_down_selection,
        ),
        tree_cache=tree,
        only_bias_on_wrong_answer=bias_on_wrong_answer,
        task_hash=first_task_hash,
    )
    st.markdown(f"Showing {len(filtered)} tasks matching criteria")
    first: TaskOutput | None = filtered.first_option
    if first:
        display_task(first)
    else:
        st.write("No tasks matching")
