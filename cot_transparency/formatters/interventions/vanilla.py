from functools import partial
import random
from typing import Optional, Type

from cot_transparency.data_models.data.bbh import MilesBBHRawData
from cot_transparency.data_models.example_base import DataExampleBase, combine_indicator_with_separator
from cot_transparency.data_models.models import ChatMessage, MessageRole, TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.interventions.few_shots_loading import get_correct_cots
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.model_apis import Prompt


def format_few_shot_for_prompt_sen(
    task: TaskOutput,
    Formatter: Type[StageOneFormatter],
    model: Optional[str] = None,
    randomize_question_format: bool = False,
    seed: str = "42",
) -> Prompt:
    read: MilesBBHRawData = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    resp = task.inference_output.parsed_response
    assert resp is not None, "This should be a valid response"

    if randomize_question_format:
        d = StageOneFormatter.all_formatters()
        rng = random.Random(seed + resp)
        Formatter = rng.choice(list(d.values()))

    specific_data_format = Formatter.get_data_format_spec()
    if specific_data_format:
        read = read.to_variant(specific_data_format)

    ground_truth = read.ground_truth_indicator
    combined = combine_indicator_with_separator(ground_truth, read.data_format.indicator_separator)
    opt_string = read._get_options()[read.ground_truth_idx()]
    ans = f"The best answer is: {combined}{opt_string}"

    q = Formatter.format_example(read, model)
    a = ChatMessage(role=MessageRole.assistant, content=ans)
    messages = q + [a]
    return Prompt(messages=messages)


def format_few_shot_for_prompt_sen_cot(
    task: TaskOutput,
    Formatter: Type[StageOneFormatter],
    model: Optional[str] = None,
    randomize_question_format: bool = False,
    seed: str = "42",
) -> Prompt:
    read: MilesBBHRawData = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    resp = task.inference_output.parsed_response
    assert resp is not None, "This should be a valid response"

    if randomize_question_format:
        d = StageOneFormatter.all_formatters()
        rng = random.Random(seed + resp)
        Formatter = rng.choice(list(d.values()))

    specific_data_format = Formatter.get_data_format_spec()
    if specific_data_format:
        read = read.to_variant(specific_data_format)

    ground_truth = read.ground_truth_indicator
    combined = combine_indicator_with_separator(ground_truth, read.data_format.indicator_separator)
    opt_string = read._get_options()[read.ground_truth_idx()]
    ans = f"Therefore, the best answer is: {combined}{opt_string}"

    cot = task.inference_output.raw_response
    cot_pre = cot.split("Therefore, the best")[0]  # remove the part as we want to make sure the answer is in our format
    cot = cot_pre + ans

    q = Formatter.format_example(read, model)
    a = ChatMessage(role=MessageRole.assistant, content=cot)
    messages = q + [a]
    return Prompt(messages=messages)


class VanillaFewShotLabelOnly10(Intervention):
    n_samples: int = 10

    @classmethod
    def formatted_name(cls) -> str:
        return f"{cls.n_samples} Few Shot No COT"

    # Non cot, only the label
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> list[ChatMessage]:
        question_hash = question.hash()
        messages = formatter.format_example(question, model=model)

        f = partial(format_few_shot_for_prompt_sen, Formatter=formatter, model=model)

        prompt: Prompt = get_correct_cots().sample(cls.n_samples, seed=question_hash).map(f).sum_or_raise()
        msgs = (prompt + Prompt(messages=messages)).messages
        return msgs


class VanillaFewShotLabelOnly20(VanillaFewShotLabelOnly10):
    n_samples: int = 20


class VanillaFewShotLabelOnly30(VanillaFewShotLabelOnly10):
    n_samples: int = 30


class MixedFormatFewShotLabelOnly10(Intervention):
    n_samples: int = 10

    @classmethod
    def formatted_name(cls) -> str:
        return f"{cls.n_samples} Few Shot No COT (Mixed Format)"

    # Non cot, only the label
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> list[ChatMessage]:
        question_hash = question.hash()
        messages = formatter.format_example(question, model=model)

        f = partial(
            format_few_shot_for_prompt_sen,
            Formatter=formatter,
            model=model,
            seed=question_hash,
            randomize_question_format=True,
        )

        prompt: Prompt = get_correct_cots().sample(cls.n_samples, seed=question_hash).map(f).sum_or_raise()
        msgs = (prompt + Prompt(messages=messages)).messages
        return msgs


class VanillaFewShot10(Intervention):
    n_samples: int = 10

    @classmethod
    def formatted_name(cls) -> str:
        return f"{cls.n_samples} Few Shot COT"

    # Non cot, only the label
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> list[ChatMessage]:
        question_hash = question.hash()
        messages = formatter.format_example(question, model=model)

        f = partial(format_few_shot_for_prompt_sen_cot, Formatter=formatter, model=model)

        prompt: Prompt = get_correct_cots().sample(cls.n_samples, seed=question_hash).map(f).sum_or_raise()
        msgs = (prompt + Prompt(messages=messages)).messages
        return msgs


class MixedFormatFewShot10(Intervention):
    n_samples: int = 10

    @classmethod
    def formatted_name(cls) -> str:
        return f"{cls.n_samples} Few Shot COT (Mixed Format)"

    # Non cot, only the label
    @classmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> list[ChatMessage]:
        question_hash = question.hash()
        messages = formatter.format_example(question, model=model)

        f = partial(
            format_few_shot_for_prompt_sen_cot,
            Formatter=formatter,
            model=model,
            seed=question_hash,
            randomize_question_format=True,
        )

        prompt: Prompt = get_correct_cots().sample(cls.n_samples, seed=question_hash).map(f).sum_or_raise()
        msgs = (prompt + Prompt(messages=messages)).messages
        return msgs
