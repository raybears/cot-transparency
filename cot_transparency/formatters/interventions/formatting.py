from typing import Type, Sequence

from slist import Slist

from cot_transparency.data_models.data.bbh import MilesBBHRawData
from cot_transparency.data_models.models import TaskOutput, ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter, ZeroShotSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.formatters.instructions import END_SINGLE_SHOT_SEP
from cot_transparency.formatters.more_biases.deceptive_assistant import (
    DeceptiveAssistantBiasedNoCOTFormatter,
)
from cot_transparency.formatters.more_biases.more_reward import (
    MoreRewardBiasedNoCOTFormatter,
    MoreRewardBiasedFormatter,
)
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
)
from cot_transparency.formatters.util import add_to_final_assistant
from cot_transparency.formatters.verbalize.formatters import StanfordNoCOTFormatter, StanfordBiasedFormatter
from cot_transparency.model_apis import Prompt


def prepend_to_front_first_user_message(messages: list[ChatMessage], prepend: str) -> list[ChatMessage]:
    """Prepend a string to the first user message."""
    new_messages = []
    for m in messages:
        if m.role == MessageRole.user:
            new_messages.append(ChatMessage(role=MessageRole.user, content=prepend + m.content))
        else:
            new_messages.append(m)
    return new_messages


def insert_to_after_system_message(messages: list[ChatMessage], to_insert: list[ChatMessage]) -> list[ChatMessage]:
    """
    if there is a system message, insert the to_insert after the system message
    otherwise, just insert at the start
    """
    new_messages = []
    first_message = messages[0]
    if first_message.role == MessageRole.system:
        new_messages.append(first_message)
        new_messages.extend(to_insert)
        new_messages.extend(messages[1:])
    else:
        new_messages.extend(to_insert)
        new_messages.extend(messages)

    return new_messages


def format_pair_cot(task: TaskOutput) -> Prompt:
    read = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    messages: list[ChatMessage] = add_to_final_assistant(
        ZeroShotCOTSycophancyFormatter.format_example(read),
        new_message=" " + task.inference_output.raw_response + END_SINGLE_SHOT_SEP,
    ) + add_to_final_assistant(
        ZeroShotCOTUnbiasedFormatter.format_example(read),
        new_message=" " + task.inference_output.raw_response + END_SINGLE_SHOT_SEP,
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
        new_message=" " + task.inference_output.raw_response + END_SINGLE_SHOT_SEP,
    )
    return Prompt(messages=messages)


def format_few_shot_for_prompt_sen(
    task: TaskOutput, Formatter: Type[StageOneFormatter] = ZeroShotUnbiasedFormatter
) -> Prompt:
    read = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    resp = task.inference_output.parsed_response
    assert resp is not None, "This should be a valid response"
    messages: list[ChatMessage] = add_to_final_assistant(
        Formatter.format_example(read),
        new_message=resp + END_SINGLE_SHOT_SEP,
    )
    return Prompt(messages=messages)


def format_unbiased_question_non_cot_add_(
    task: TaskOutput, Formatter: Type[StageOneFormatter] = ZeroShotUnbiasedFormatter
) -> Prompt:
    read = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    resp = task.inference_output.parsed_response
    assert resp is not None, "This should be a valid response"
    messages: list[ChatMessage] = add_to_final_assistant(
        Formatter.format_example(read),
        new_message=resp + END_SINGLE_SHOT_SEP,
    )
    return Prompt(messages=messages)


def format_biased_question_non_cot_sycophancy(task: TaskOutput) -> Prompt:
    read = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    resp = task.inference_output.parsed_response
    assert resp is not None, "This should be a valid response"
    messages: list[ChatMessage] = add_to_final_assistant(
        ZeroShotSycophancyFormatter.format_example(read),
        new_message=resp + END_SINGLE_SHOT_SEP,
    )
    return Prompt(messages=messages)


def format_biased_question_non_cot_random_formatter(task: TaskOutput, formatter: Type[StageOneFormatter]) -> Prompt:
    read = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    resp = task.inference_output.parsed_response
    assert resp is not None, "This should be a valid response"
    formatter_to_use = get_formatter_for_few_shot_non_cot(answer_formatter=formatter, seed=read.hash())
    messages: list[ChatMessage] = add_to_final_assistant(
        formatter_to_use.format_example(read),
        new_message=resp + END_SINGLE_SHOT_SEP,
    )
    return Prompt(messages=messages)


def format_biased_question_cot(task: TaskOutput, formatter: Type[StageOneFormatter]) -> Prompt:
    read = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    formatter_to_use = get_formatter_for_few_shot_cot(exclude_formattter=formatter, seed=read.hash())
    messages: list[ChatMessage] = add_to_final_assistant(
        formatter_to_use.format_example(read),
        new_message=" " + task.inference_output.raw_response + END_SINGLE_SHOT_SEP,
    )
    return Prompt(messages=messages)


def get_formatter_for_few_shot_cot(
    exclude_formattter: Type[StageOneFormatter] | None,
    seed: str,
    use_formatters: Sequence[Type[StageOneFormatter]] = Slist(),
) -> Type[StageOneFormatter]:
    if exclude_formattter:
        assert exclude_formattter in BIASED_FORMATTERS_FEW_SHOT_COT
    sample_from = Slist(use_formatters) or BIASED_FORMATTERS_FEW_SHOT_COT
    formatter_used: Type[StageOneFormatter] = (
        # We don't want to use the same formatter for few shot
        sample_from.filter(lambda f: f is not exclude_formattter)
        .shuffle(seed=seed)
        .first_or_raise()
    )
    return formatter_used


def get_formatter_for_few_shot_non_cot(answer_formatter: Type[StageOneFormatter], seed: str) -> Type[StageOneFormatter]:
    formatter_used: Type[StageOneFormatter] = (
        # We don't want to use the same formatter for few shot
        BIASED_FORMATTERS_FEW_SHOT_NON_COT.filter(lambda f: f is not answer_formatter)
        .shuffle(seed=seed)
        .first_or_raise()
    )
    return formatter_used


BIASED_FORMATTERS_FEW_SHOT_NON_COT: Slist[Type[StageOneFormatter]] = Slist(
    [
        ZeroShotSycophancyFormatter,
        StanfordNoCOTFormatter,
        DeceptiveAssistantBiasedNoCOTFormatter,
        MoreRewardBiasedNoCOTFormatter,
    ]
)

BIASED_FORMATTERS_FEW_SHOT_COT: Slist[Type[StageOneFormatter]] = Slist(
    [
        ZeroShotCOTSycophancyFormatter,
        StanfordBiasedFormatter,
        # DeceptiveAssistantBiasedFormatter,
        WrongFewShotIgnoreMistakesBiasedFormatter,
        MoreRewardBiasedFormatter,
    ]
)
