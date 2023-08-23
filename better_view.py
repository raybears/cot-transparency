from functools import lru_cache
from typing import List, Sequence

import streamlit as st
import numpy as np
from pydantic import BaseModel
from slist import Slist, identity

from cot_transparency.data_models.models import TaskOutput, ChatMessage, StrictChatMessage, MessageRole
from cot_transparency.model_apis import Prompt, ModelType
from scripts.intervention_investigation import read_whole_exp_dir


@lru_cache(maxsize=32)
def cached_read_whole_exp_dir(exp_dir: str) -> Slist[TaskOutput]:
    # everything you click a button, streamlit reruns the whole script
    # so we need to cache the results of read_whole_exp_dir
    return read_whole_exp_dir(exp_dir=exp_dir)


def display_task(task: TaskOutput):
    messages: list[ChatMessage] = task.task_spec.messages
    model_type: ModelType = ModelType.from_model_name(task.task_spec.inference_config.model)
    strict: list[StrictChatMessage] = Prompt(messages=messages).get_strict_messages(model_type=model_type)
    for msg in strict:
        # pattern match on msg.role
        match msg.role:
            case MessageRole.none:
                st.write(msg.content)
            case MessageRole.system:
                st.markdown(f"### System")
                st.code(msg.content)
            case MessageRole.user:
                with st.chat_message("user"):
                    st.markdown(f"### User")
                    st.code(msg.content)
            case MessageRole.assistant:
                with st.chat_message("assistant"):
                    st.markdown(f"### Assistant")
                    st.code(msg.content)

    # write the final response
    with st.chat_message("assistant"):
        st.markdown(f"### Assistant")
        st.code(task.inference_output.raw_response)


# naughty Slist patch to add __hash__ by id
def __hash__(self):
    return id(self)


Slist.__hash__ = __hash__


@lru_cache(maxsize=32)
def cached_search(
    completion_search: str,
    everything: Slist[TaskOutput],
    formatter_selection: str,
    intervention_selection: str | None,
    task_selection: str,
    only_bias_on_wrong_answer: bool,
) -> Slist[TaskOutput]:
    return (
        everything.filter(lambda task: completion_search in task.inference_output.raw_response)
        .filter(lambda task: task.task_spec.formatter_name == formatter_selection)
        .filter(lambda task: task.task_spec.intervention_name == intervention_selection)
        .filter(lambda task: task.task_spec.task_name == task_selection)
        .filter(lambda task: task.bias_on_wrong_answer == only_bias_on_wrong_answer if only_bias_on_wrong_answer else True)
    )


class DropDowns(BaseModel):
    formatters: Sequence[str]
    interventions: Sequence[str | None]
    tasks: Sequence[str]


@lru_cache()
def get_drop_downs(items: Slist[TaskOutput]) -> DropDowns:
    formatters = items.map(lambda task: task.task_spec.formatter_name).distinct_unsafe()
    interventions = items.map(lambda task: task.task_spec.intervention_name).distinct_unsafe()
    tasks = items.map(lambda task: task.task_spec.task_name).distinct_unsafe()
    return DropDowns(formatters=formatters, interventions=interventions, tasks=tasks)


# Ask the user to enter experiment_dir
exp_dir = st.text_input("Enter experiment_dir", "experiments/math_scale")
everything: Slist[TaskOutput] = cached_read_whole_exp_dir(exp_dir=exp_dir)
st.markdown(f"Loaded {len(everything)} tasks")
# Optional text input
completion_search: str = st.text_input("Search for text in final completion")
drop_downs: DropDowns = get_drop_downs(everything)
task_selection: str = st.selectbox("Select task", drop_downs.tasks)
formatter_drop_down_selection: str = st.selectbox("Select formatter", drop_downs.formatters)
intervention_drop_down_selection: str | None = st.selectbox("Select intervention", drop_downs.interventions)
bias_on_wrong_answer: bool = st.checkbox("Show only bias on wrong answer", value=True)
filtered = cached_search(
    completion_search=completion_search,
    everything=everything,
    formatter_selection=formatter_drop_down_selection,
    intervention_selection=intervention_drop_down_selection,
    task_selection=task_selection,
    only_bias_on_wrong_answer=bias_on_wrong_answer,
)
st.markdown(f"Showing {len(filtered)} tasks matching criteria")
first: TaskOutput | None = filtered.first_option
if first:
    display_task(first)
