from typing import Type

from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ChatMessage
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.consistency_prompting.few_shots import few_shots_to_sample
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.interventions.intervention import Intervention, prepend_to_front_first_user_message
from cot_transparency.formatters.more_biases.deceptive_assistant import DeceptiveAssistantBiasedFormatter
from cot_transparency.formatters.more_biases.more_reward import MoreRewardConsistencyFormatter
from scripts.biased_wrong_ans import sample_few_shots_cot


class PairedConsistency3(Intervention):
    def hook(self, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=sample_few_shots_cot(few_shots_to_sample, seed=question.hash(), n=5).convert_to_completion_str(),
        )
        return new


class PairedConsistency5(Intervention):
    def hook(self, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=sample_few_shots_cot(few_shots_to_sample, seed=question.hash(), n=5).convert_to_completion_str(),
        )
        return new


class PairedConsistency10(Intervention):
    def hook(self, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        new = prepend_to_front_first_user_message(
            messages=messages,
            prepend=sample_few_shots_cot(few_shots_to_sample, seed=question.hash(), n=10).convert_to_completion_str(),
        )
        return new


five_shot_intervention = PairedConsistency5()
PairedConsistency5_ZeroShotCOTSycophancyFormatter: Type[StageOneFormatter] = five_shot_intervention.intervene(
    ZeroShotCOTSycophancyFormatter
)
PairedConsistency5_ZeroShotCOTUnbiasedFormatter: Type[StageOneFormatter] = five_shot_intervention.intervene(
    ZeroShotCOTUnbiasedFormatter
)
PairedConsistency5_DeceptiveAssistantBiasedFormatter: Type[StageOneFormatter] = five_shot_intervention.intervene(
    DeceptiveAssistantBiasedFormatter
)
PairedConsistency5_MoreRewardConsistencyFormatter: Type[StageOneFormatter] = five_shot_intervention.intervene(
    MoreRewardConsistencyFormatter
)
