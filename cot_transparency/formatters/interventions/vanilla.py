from functools import partial
from typing import Type

from pyparsing import Optional
from cot_transparency.data_models.data.bbh import MilesBBHRawData
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ChatMessage, MessageRole, TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotUnbiasedFormatter
from cot_transparency.formatters.interventions.few_shots_loading import get_correct_cots
from cot_transparency.formatters.interventions.formatting import (
    add_to_final_assistant,
    format_unbiased_question_non_cot,
    prepend_to_front_first_user_message,
)
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.model_apis import Prompt


def format_unbiased_question_non_cot(
    task: TaskOutput, Formatter: Type[StageOneFormatter] = ZeroShotUnbiasedFormatter
) -> Prompt:
    read: MilesBBHRawData = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    resp = task.model_output.parsed_response
    assert resp is not None, "This should be a valid response"

    read = read.to_variant(Formatter.data_format_spec)
    ground_truth = read.ground_truth_indicator
    q = Formatter.format_example(read)
    a = ChatMessage(role=MessageRole.assistant, content=ground_truth + ")")
    messages = q + [a]
    return Prompt(messages=messages)


class VanillaFewShotLabelOnly10(Intervention):
    # Non cot, only the label
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)

        f = partial(format_unbiased_question_non_cot, Formatter=formatter)

        prompt: Prompt = (
            get_correct_cots().sample(10, seed=question.hash()).map(f).sum_or_raise()
        )
        msgs = (prompt + Prompt(messages=messages)).messages
        return msgs


class VanillaFewShotLabelOnly20(Intervention):
    # Non cot, only the label
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots().sample(20, seed=question.hash()).map(format_unbiased_question_non_cot).sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class VanillaFewShotLabelOnly30(Intervention):
    # Non cot, only the label
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots().sample(30, seed=question.hash()).map(format_unbiased_question_non_cot).sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new
