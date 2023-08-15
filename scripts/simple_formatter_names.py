from typing import Type, Optional

from cot_transparency.formatters import PromptFormatter
from cot_transparency.formatters.interventions.consistency import (
    PairedConsistency10,
    NaiveFewShot10,
    BiasedConsistency10,
    NaiveFewShotLabelOnly10,
    NaiveFewShotLabelOnly30,
)
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.formatters.more_biases.user_wrong_cot import UserBiasedWrongCotFormatter
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotBiasedFormatter,
)
from cot_transparency.formatters.more_biases.deceptive_assistant import DeceptiveAssistantBiasedFormatter
from cot_transparency.formatters.more_biases.more_reward import MoreRewardBiasedFormatter

FORMATTER_TO_SIMPLE_NAME: dict[Type[PromptFormatter], str] = {
    DeceptiveAssistantBiasedFormatter: "Tell model to be deceptive",
    MoreRewardBiasedFormatter: "More reward for an option",
    UserBiasedWrongCotFormatter: "User says wrong reasoning",
    WrongFewShotBiasedFormatter: "Wrong label in the few shot",
}


INTERVENTION_TO_SIMPLE_NAME: dict[Optional[Type[Intervention]], str] = {
    None: "No intervention, biased context",
    PairedConsistency10: "Biased and Unbiased question pairs in few shot",
    BiasedConsistency10: "Biased questions in few shot",
    NaiveFewShot10: "10 Unbiased questions, with COT and ground truth labels",
    NaiveFewShotLabelOnly10: "10 Unbiased questions, with ground truth labels",
    NaiveFewShotLabelOnly30: "30 Unbiased questions, with ground truth labels",
}
