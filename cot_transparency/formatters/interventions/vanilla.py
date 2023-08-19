from functools import partial
from typing import Type

from cot_transparency.data_models.data.bbh import MilesBBHRawData
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ChatMessage, MessageRole, TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotUnbiasedFormatter
from cot_transparency.formatters.interventions.few_shots_loading import get_correct_cots
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.model_apis import Prompt


def format_few_shot_for_prompt_sen(
    task: TaskOutput, Formatter: Type[StageOneFormatter] = ZeroShotUnbiasedFormatter
) -> Prompt:
    read: MilesBBHRawData = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    resp = task.model_output.parsed_response
    assert resp is not None, "This should be a valid response"

    specific_data_format = Formatter.get_data_format_spec()
    if specific_data_format:
        read = read.to_variant(specific_data_format)

    ground_truth = read.ground_truth_indicator
    q = Formatter.format_example(read)
    a = ChatMessage(role=MessageRole.assistant, content=ground_truth + ")")
    messages = q + [a]
    return Prompt(messages=messages)


class VanillaFewShotLabelOnly10(Intervention):
    n_samples: int = 10

    # Non cot, only the label
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        question_hash = question.hash()
        messages = formatter.format_example(question)

        f = partial(format_few_shot_for_prompt_sen, Formatter=formatter)

        prompt: Prompt = get_correct_cots().sample(cls.n_samples, seed=question_hash).map(f).sum_or_raise()
        msgs = (prompt + Prompt(messages=messages)).messages
        return msgs


class VanillaFewShotLabelOnly20(VanillaFewShotLabelOnly10):
    n_samples: int = 20


class VanillaFewShotLabelOnly30(VanillaFewShotLabelOnly10):
    n_samples: int = 30
