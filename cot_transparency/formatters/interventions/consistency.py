from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ChatMessage
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
    format_biased_question_non_cot,
)
from cot_transparency.model_apis import Prompt


class PairedConsistency6(Intervention):
    @classmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        prompt: Prompt = get_correct_cots().sample(3, seed=question.hash()).map(format_pair_cot).sum_or_raise()
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class PairedConsistency10(Intervention):
    @classmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        prompt: Prompt = get_correct_cots().sample(5, seed=question.hash()).map(format_pair_cot).sum_or_raise()
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class BiasedConsistency10(Intervention):
    @classmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        prompt: Prompt = (
            # Not a pair so, sample 10
            get_correct_cots()
            .sample(10, seed=question.hash())
            .map(format_biased_question_cot)
            .sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class NaiveFewShot10(Intervention):
    # Simply use unbiased few shot
    @classmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        prompt: Prompt = (
            get_correct_cots().sample(10, seed=question.hash()).map(format_unbiased_question_cot).sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class NaiveFewShotLabelOnly10(Intervention):
    # Non cot, only the label
    @classmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
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
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
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
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        prompt: Prompt = (
            get_correct_cots().sample(30, seed=question.hash()).map(format_unbiased_question_non_cot).sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class BiasedConsistencyLabelOnly20(Intervention):
    @classmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        prompt: Prompt = (
            get_correct_cots().sample(20, seed=question.hash()).map(format_biased_question_non_cot).sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


class BiasedConsistencyLabelOnly30(Intervention):
    @classmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        prompt: Prompt = (
            get_correct_cots().sample(30, seed=question.hash()).map(format_biased_question_non_cot).sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new
