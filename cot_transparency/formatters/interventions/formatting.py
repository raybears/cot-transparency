from cot_transparency.data_models.data.bbh import MilesBBHRawData
from cot_transparency.data_models.models import TaskOutput, ChatMessage, MessageRole
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter, ZeroShotSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.formatters.instructions import END_SINGLE_SHOT_SEP
from cot_transparency.model_apis import Prompt


def add_to_final_assistant(messages: list[ChatMessage], new_message: str) -> list[ChatMessage]:
    # If the final message is from the assistant, then we need to add the final assistant message
    # Otherwise, we need to add a new assistant message
    new_list = messages.copy()
    if messages[-1].role == MessageRole.assistant or messages[-1].role == MessageRole.assistant_if_completion:
        new_list[-1] = ChatMessage(
            role=MessageRole.assistant, content=messages[-1].content.rstrip() + " " + new_message
        )
    else:
        new_list.append(ChatMessage(role=MessageRole.assistant, content=new_message))
    return new_list


def prepend_to_front_first_user_message(messages: list[ChatMessage], prepend: str) -> list[ChatMessage]:
    """Prepend a string to the first user message."""
    new_messages = []
    for m in messages:
        if m.role == MessageRole.user:
            new_messages.append(ChatMessage(role=MessageRole.user, content=prepend + m.content))
        else:
            new_messages.append(m)
    return new_messages


def format_pair_cot(task: TaskOutput) -> Prompt:
    read = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    messages: list[ChatMessage] = add_to_final_assistant(
        ZeroShotCOTSycophancyFormatter.format_example(read),
        new_message=task.model_output.raw_response + END_SINGLE_SHOT_SEP,
    ) + add_to_final_assistant(
        ZeroShotCOTUnbiasedFormatter.format_example(read),
        new_message=task.model_output.raw_response + END_SINGLE_SHOT_SEP,
    )
    return Prompt(messages=messages)


def format_pair_non_cot(task: TaskOutput) -> Prompt:
    read = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    messages: list[ChatMessage] = add_to_final_assistant(
        ZeroShotSycophancyFormatter.format_example(read), new_message=read.ground_truth + END_SINGLE_SHOT_SEP
    ) + add_to_final_assistant(
        ZeroShotUnbiasedFormatter.format_example(read), new_message=read.ground_truth + END_SINGLE_SHOT_SEP
    )
    return Prompt(messages=messages)


def format_unbiased_question_cot(task: TaskOutput) -> Prompt:
    read = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    messages: list[ChatMessage] = add_to_final_assistant(
        ZeroShotCOTUnbiasedFormatter.format_example(read),
        new_message=task.model_output.raw_response + END_SINGLE_SHOT_SEP,
    )
    return Prompt(messages=messages)


def format_biased_question_cot(task: TaskOutput) -> Prompt:
    read = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    messages: list[ChatMessage] = add_to_final_assistant(
        ZeroShotCOTSycophancyFormatter.format_example(read),
        new_message=task.model_output.raw_response + END_SINGLE_SHOT_SEP,
    )
    return Prompt(messages=messages)
