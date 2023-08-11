from typing import Type

from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ChatMessage
from cot_transparency.formatters.consistency_prompting.few_shots import few_shots_to_sample
from cot_transparency.formatters.interventions.intervention import (
    Intervention,
    prepend_to_front_first_user_message,
    NoIntervention,
)
from scripts.biased_wrong_ans import sample_few_shots_cot, sample_few_shots_cot_with_max


class PairedConsistency3(Intervention):
    @classmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=sample_few_shots_cot(few_shots_to_sample, seed=question.hash(), n=5).convert_to_completion_str(),
        )
        return new


class PairedConsistency5(Intervention):
    @classmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=sample_few_shots_cot_with_max(
                few_shots_to_sample, seed=question.hash(), n=5, max_tokens=6000
            ).convert_to_completion_str(),
        )
        return new


class PairedConsistency10(Intervention):
    @classmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=sample_few_shots_cot(few_shots_to_sample, seed=question.hash(), n=10).convert_to_completion_str(),
        )
        return new


five_shot_intervention = PairedConsistency5
VALID_INTERVENTIONS: dict[str, Type[Intervention]] = {
    five_shot_intervention.name(): five_shot_intervention,
    NoIntervention.name(): NoIntervention,
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
