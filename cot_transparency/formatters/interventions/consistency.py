from typing import Type

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


VALID_INTERVENTIONS: dict[str, Type[Intervention]] = {
    PairedConsistency6.name(): PairedConsistency6,
    PairedConsistency10.name(): PairedConsistency10,
    BiasedConsistency10.name(): BiasedConsistency10,
    NaiveFewShot10.name(): NaiveFewShot10,
}


def get_valid_stage1_interventions(interventions: list[str]) -> list[Type[Intervention]]:
    # assert that the formatters are valid
    for intervention in interventions:
        if intervention not in VALID_INTERVENTIONS:
            raise ValueError(
                f"intervention {intervention} is not valid. Valid intervention are {list(VALID_INTERVENTIONS.keys())}"
            )

    validated: list[Type[Intervention]] = [VALID_INTERVENTIONS[i] for i in interventions]
    return validated
