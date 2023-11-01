import pathlib
from typing import cast, Sequence

import streamlit as st
from slist import Slist

from cot_transparency.apis.base import FileCacheRow
from cot_transparency.apis.openai import append_assistant_preferred_to_last_user
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.data_models.models import TaskOutput, ModelOutput, TaskSpec


def display_messages(messages: Sequence[ChatMessage]):
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
                    content = msg.content.replace("\n", "  \n")
                    st.markdown(content)

            case MessageRole.assistant:
                with st.chat_message("assistant"):
                    st.markdown("### Assistant")
                    st.markdown(msg.content.replace("\n", "  \n"))
            case MessageRole.assistant_if_completion:
                with st.chat_message("assistant"):
                    st.markdown("### Assistant if completion")
                    st.markdown(msg.content.replace("\n", "  \n"))


def display_task(task: TaskOutput, put_if_completion_in_user: bool = True):
    model_output = task.inference_output.parsed_response
    ground_truth = task.task_spec.ground_truth
    bias_on = task.task_spec.biased_ans
    is_correct = model_output == ground_truth
    emoji = "✔️" if is_correct else "❌"
    st.markdown(f"Ground truth: {ground_truth}")
    st.markdown(f"Model output: {model_output} {emoji}")
    st.markdown(f"Bias on: {bias_on}")

    messages: Sequence[ChatMessage] = (
        cast(
            Sequence[ChatMessage],
            append_assistant_preferred_to_last_user(task.task_spec.messages),
        )
        if put_if_completion_in_user
        else task.task_spec.messages
    )
    messages = list(messages) + [ChatMessage(role=MessageRole.assistant, content=task.inference_output.raw_response)]
    display_messages(messages)


def file_cache_row_to_task_output(row: FileCacheRow) -> Sequence[TaskOutput]:
    responses: Sequence[str] = row.response.response.raw_responses
    task_hash = Slist(row.response.messages).map(str).mk_string("")
    task_spec = TaskSpec(
        task_name="not_used",
        inference_config=row.response.config,
        messages=row.response.messages,
        out_file_path=pathlib.Path("not_used"),
        ground_truth="A",
        formatter_name="not_used",
        task_hash=task_hash,
    )
    return [
        TaskOutput(
            task_spec=task_spec,
            inference_output=ModelOutput(raw_response=response, parsed_response=None),
        )
        for response in responses
    ]
