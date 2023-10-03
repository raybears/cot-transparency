import random
from functools import partial
from typing import Optional, Type

from cot_transparency.data_models.data.bbh import MilesBBHRawData
from cot_transparency.data_models.example_base import DataExampleBase, combine_indicator_with_separator
from cot_transparency.data_models.models import ChatMessage, MessageRole, TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.interventions.few_shots_loading import get_correct_cots
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.model_apis import Prompt


def _format_few_shot_example(
    task: TaskOutput, FormatterForFinalQuestion: Type[StageOneFormatter], randomize_question_format: bool, seed: str
):
    few_shot_data_example: MilesBBHRawData = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    resp = task.inference_output.parsed_response
    assert resp is not None, "This should be a valid response"

    # We can either randomize the format (how the question is asked) of the few shot examples
    # or we just use the same format as the final question
    if randomize_question_format:
        d = FormatterForFinalQuestion.all_formatters()
        rng = random.Random(seed + few_shot_data_example.hash())
        FormatterForFewShotExample = rng.choice(list(d.values()))
    else:
        FormatterForFewShotExample = FormatterForFinalQuestion

    specific_data_format = FormatterForFewShotExample.get_data_format_spec()
    if specific_data_format:
        few_shot_data_example = few_shot_data_example.to_variant(specific_data_format)

    ground_truth = few_shot_data_example.ground_truth_indicator
    combined = combine_indicator_with_separator(ground_truth, few_shot_data_example.data_format.indicator_separator)
    opt_string = few_shot_data_example._get_options()[few_shot_data_example.ground_truth_idx()]
    return FormatterForFewShotExample, few_shot_data_example, combined, opt_string


def format_few_shot_for_prompt_sen(
    task: TaskOutput,
    Formatter: Type[StageOneFormatter],
    model: Optional[str] = None,
    randomize_question_format: bool = False,
    seed: str = "42",
) -> Prompt:
    FormatterForFewShotExample, read, combined, opt_string = _format_few_shot_example(
        task, Formatter, randomize_question_format, seed
    )

    ans = f"The best answer is: {combined}{opt_string}"

    q = FormatterForFewShotExample.format_example(read, model)
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
    FormatterForFewShotExample, read, combined, opt_string = _format_few_shot_example(
        task, Formatter, randomize_question_format, seed
    )

    ans = f"Therefore, the best answer is: {combined}{opt_string}"
    cot = task.inference_output.raw_response
    cot_pre = cot.split("Therefore, the best")[0]  # remove the part as we want to make sure the answer is in our format
    cot = cot_pre + ans

    q = FormatterForFewShotExample.format_example(read, model)
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
        assert not formatter.is_cot, "You probably want to use MixedFormatFewShot"

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


class MixedFormatFewShotLabelOnly20(MixedFormatFewShotLabelOnly10):
    n_samples: int = 20


class MixedFormatFewShotLabelOnly30(MixedFormatFewShotLabelOnly10):
    n_samples: int = 30


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

        assert formatter.is_cot, "You probably want to use MixedFormatFewShotLabelOnly*"

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
