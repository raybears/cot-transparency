import argparse
import time
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components
from slist import Slist

from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.viewer.util import cached_read_finetune_from_url, display_messages
from cot_transparency.viewer.viewer_cache import cached_read_finetune

# set to wide
st.set_page_config(layout="wide")

# ruff: noqa: E501

parser = argparse.ArgumentParser()

parser.add_argument(
    "jsonl_file",
    help="The jsonl file to read",
)
args = parser.parse_args()
jsonl_file: str = args.jsonl_file


# naughty Slist patch to add __hash__ by id so that lru works
def __hash__(self):  # type: ignore
    return id(self)


Slist.__hash__ = __hash__  # type: ignore


# Ask the user to enter experiment_dir
jsonl_file = st.text_input("Enter jsonl file", jsonl_file)
everything: Slist[FinetuneSample] = (
    cached_read_finetune(jsonl_file) if "https://" not in jsonl_file else cached_read_finetune_from_url(jsonl_file)
)

st.markdown(f"Loaded {len(everything)} tasks")
# Calculate what mdoels / tasks are available


def search(
    items: Slist[FinetuneSample],
    prompt_search: Optional[str],
    completion_search: Optional[str],
) -> Slist[FinetuneSample]:
    ts = time.time()
    result = items.filter(
        lambda sample: completion_search in sample.messages[-1].content if completion_search else True
    ).filter(
        lambda sample: prompt_search in " ".join(Slist(sample.messages[:-1]).map(lambda x: x.content))
        if prompt_search
        else True
    )
    print(f"Search took {time.time() - ts} seconds")
    return result


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
left = st.columns(1)

i = 0

filtered: Slist[FinetuneSample] = search(
    items=everything,
    completion_search=completion_search,
    prompt_search=prompt_search,
)
st.markdown(f"Showing {len(filtered)} finetuning samples ")
show_item_idx = st.session_state.count % len(filtered) if len(filtered) > 0 else 0
first = filtered[show_item_idx] if len(filtered) > 0 else None
if first is not None:
    display_messages([i.to_chat_message() for i in first.messages])
