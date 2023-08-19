from typing import Type, Optional

from cot_transparency.formatters import PromptFormatter
from cot_transparency.formatters.interventions.consistency import (
    PairedConsistency10,
    NaiveFewShot10,
    BiasedConsistency10,
    NaiveFewShotLabelOnly10,
    NaiveFewShotLabelOnly30,
    NaiveFewShot16,
    NaiveFewShot6,
    NaiveFewShot3,
)
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.formatters.more_biases.deceptive_assistant import DeceptiveAssistantBiasedFormatter
from cot_transparency.formatters.more_biases.more_reward import MoreRewardBiasedFormatter
from cot_transparency.formatters.more_biases.user_wrong_cot import UserBiasedWrongCotFormatter
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotBiasedFormatter,
)
from cot_transparency.formatters.more_biases.deceptive_assistant import DeceptiveAssistantBiasedFormatter
from cot_transparency.formatters.more_biases.more_reward import MoreRewardBiasedFormatter


INTERVENTION_TO_SIMPLE_NAME: dict[Optional[Type[Intervention]], str] = {
    None: "No intervention, biased context",
    PairedConsistency10: "Biased and Unbiased question pairs in few shot",
    BiasedConsistency10: "Biased questions in few shot",
    NaiveFewShot3: "3 Unbiased questions with COT answer",
    NaiveFewShot6: "6 Unbiased questions with COT answer",
    NaiveFewShot10: "10 Unbiased questions with COT answer",
    NaiveFewShot16: "16 Unbiased questions with COT answer",
    NaiveFewShotLabelOnly10: "10 Unbiased questions, with ground truth labels",
    NaiveFewShotLabelOnly30: "30 Unbiased questions, with ground truth labels",
}
