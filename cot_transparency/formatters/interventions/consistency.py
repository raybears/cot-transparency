from typing import Type

from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ChatMessage
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.interventions.few_shots_loading import get_correct_cots
from cot_transparency.formatters.interventions.intervention import (
    Intervention,
)
from cot_transparency.formatters.interventions.formatting import (
    format_pair_cot,
    format_unbiased_question_cot,
    format_biased_question_cot,
    prepend_to_front_first_user_message,
    format_unbiased_question_non_cot,
    format_biased_question_non_cot_random_formatter,
    format_pair_non_cot,
    format_biased_question_non_cot_sycophancy,
)
from cot_transparency.model_apis import Prompt


class PairedConsistency6(Intervention):
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = get_correct_cots().sample(3, seed=question.hash()).map(format_pair_cot).sum_or_raise()
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class PairedConsistency10(Intervention):
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = get_correct_cots().sample(5, seed=question.hash()).map(format_pair_cot).sum_or_raise()
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class BiasedConsistency10(Intervention):
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            # Not a pair so, sample 10
            get_correct_cots()
            .sample(10, seed=question.hash())
            .map(lambda task: format_biased_question_cot(task=task, formatter=formatter))
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class NaiveFewShot3(Intervention):
    # Simply use unbiased few shot
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots().sample(3, seed=question.hash()).map(format_unbiased_question_cot).sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class NaiveFewShot6(Intervention):
    # Simply use unbiased few shot
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots().sample(6, seed=question.hash()).map(format_unbiased_question_cot).sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class NaiveFewShot10(Intervention):
    # Simply use unbiased few shot
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots().sample(10, seed=question.hash()).map(format_unbiased_question_cot).sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class NaiveFewShot16(Intervention):
    # Simply use unbiased few shot
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots().sample(16, seed=question.hash()).map(format_unbiased_question_cot).sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new

class NaiveFewShotLabelOnly10(Intervention):
    # Non cot, only the label
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots().sample(10, seed=question.hash()).map(format_unbiased_question_non_cot).sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class NaiveFewShotLabelOnly20(Intervention):
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


class NaiveFewShotLabelOnly30(Intervention):
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


class SycophancyConsistencyLabelOnly10(Intervention):
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots()
            .sample(10, seed=question.hash())
            .map(format_biased_question_non_cot_sycophancy)
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class SycoConsistencyLabelOnly30(Intervention):
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots()
            .sample(30, seed=question.hash())
            .map(format_biased_question_non_cot_sycophancy)
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class BiasedConsistencyLabelOnly10(Intervention):
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots()
            .sample(10, seed=question.hash())
            .map(lambda task: format_biased_question_non_cot_random_formatter(task=task, formatter=formatter))
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new

    @classmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        raise NotImplementedError("This should not be called")


class BiasedConsistencyLabelOnly20(Intervention):
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots()
            .sample(20, seed=question.hash())
            .map(lambda task: format_biased_question_non_cot_random_formatter(task=task, formatter=formatter))
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class BiasedConsistencyLabelOnly30(Intervention):
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        prompt: Prompt = (
            get_correct_cots()
            .sample(10, seed=question.hash())
            .map(lambda task: format_biased_question_non_cot_random_formatter(task=task, formatter=formatter))
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new

    @classmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        raise NotImplementedError("This should not be called")


class PairedFewShotLabelOnly10(Intervention):
    # Non cot, only the label
    @classmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        prompt: Prompt = get_correct_cots().sample(5, seed=question.hash()).map(format_pair_non_cot).sum_or_raise()
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class PairedFewShotLabelOnly30(Intervention):
    # Non cot, only the label
    @classmethod
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages: list[ChatMessage] = formatter.format_example(question)
        prompt: Prompt = get_correct_cots().sample(15, seed=question.hash()).map(format_pair_non_cot).sum_or_raise()
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new
