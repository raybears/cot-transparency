from typing import Type


from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ChatMessage
from cot_transparency.formatters.consistency_prompting.few_shots import few_shots_to_sample
from cot_transparency.formatters.interventions.intervention import (
    Intervention,
    prepend_to_front_first_user_message,
)
from cot_transparency.model_apis import Prompt
from scripts.biased_wrong_ans import (
    paired_sample_few_shots_cot,
    sample_few_shots_cot_with_max,
    biased_qn_with_raw_response,
    unbiased_qn_with_raw_response,
)


class PairedConsistency3(Intervention):
    @classmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=paired_sample_few_shots_cot(
                few_shots_to_sample, seed=question.hash(), n=5
            ).convert_to_completion_str(),
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
            prepend=paired_sample_few_shots_cot(
                few_shots_to_sample, seed=question.hash(), n=10
            ).convert_to_completion_str(),
        )
        return new


class BiasedConsistency10(Intervention):
    @classmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        prompt: Prompt = (
            few_shots_to_sample.sample(10, seed=question.hash()).map(biased_qn_with_raw_response).sum_or_raise()
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
            few_shots_to_sample.sample(10, seed=question.hash()).map(unbiased_qn_with_raw_response).sum_or_raise()
        )
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=prompt.convert_to_completion_str(),
        )
        return new


VALID_INTERVENTIONS: dict[str, Type[Intervention]] = {
    PairedConsistency5.name(): PairedConsistency5,
    BiasedConsistency10.name(): BiasedConsistency10,
    PairedConsistency10.name(): PairedConsistency10,
    PairedConsistency3.name(): PairedConsistency3,
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
