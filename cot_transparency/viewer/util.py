from cot_transparency.data_models.messages import ChatMessage, MessageRole


import streamlit as st


from typing import Sequence

from cot_transparency.data_models.models import TaskOutput


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
    display_messages(messages)