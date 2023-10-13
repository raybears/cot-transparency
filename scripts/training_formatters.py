from typing import Sequence, Type

from cot_transparency.formatters import StageOneFormatter
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter, ZeroShotSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.formatters.more_biases.anchor_initial_wrong import ZeroShotInitialWrongFormatter
from cot_transparency.formatters.more_biases.deceptive_assistant import (
    DeceptiveAssistantTargetedFormatter,
)
from cot_transparency.formatters.more_biases.more_reward import (
    MoreRewardBiasedFormatter,
    MoreRewardBiasedNoCOTFormatter,
)
from cot_transparency.formatters.more_biases.random_bias_formatter import (
    RandomBiasedFormatter,
    RandomAgainstBiasedFormatter,
    RandomBiasedQuotedFormatter,
    RandomBiasedNoCOTFormatter,
    RandomAgainstBiasedNoCOTFormatter,
    RandomAgainstQuotedBiasedFormatter,
    RandomBiasedQuotedNoCOTFormatter,
    RandomAgainstBiasedQuotedNoCOTFormatter,
)
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
)
from cot_transparency.formatters.verbalize.formatters import (
    StanfordBiasedFormatter,
    CheckmarkBiasedFormatter,
    CrossBiasedFormatter,
    StanfordNoCOTFormatter,
    CrossNoCOTFormatter,
    CheckmarkNoCOTFormatter,
)


TRAINING_COT_FORMATTERS_ZEROSHOT = [
    StanfordBiasedFormatter,
    MoreRewardBiasedFormatter,
    ZeroShotCOTSycophancyFormatter,
    RandomBiasedFormatter,
    RandomBiasedQuotedFormatter,
    RandomAgainstBiasedFormatter,
    RandomAgainstQuotedBiasedFormatter,
    ZeroShotInitialWrongFormatter,  # There is only a COT version of this formatter
]

TRAINING_COT_FORMATTERS_FEW_SHOT = [
    WrongFewShotIgnoreMistakesBiasedFormatter,
    CheckmarkBiasedFormatter,
    CrossBiasedFormatter,
    # TODO: Add "All answer A"
]


TRAINING_COT_FORMATTERS: Sequence[Type[StageOneFormatter]] = (
    TRAINING_COT_FORMATTERS_ZEROSHOT + TRAINING_COT_FORMATTERS_FEW_SHOT
)
TRAINING_COT_FORMATTERS_WITH_UNBIASED = list(TRAINING_COT_FORMATTERS) + [ZeroShotCOTUnbiasedFormatter]

TRAINING_NO_COT_FORMATTERS_ZEROSHOT = [
    StanfordNoCOTFormatter,
    MoreRewardBiasedNoCOTFormatter,
    ZeroShotSycophancyFormatter,
    RandomBiasedNoCOTFormatter,
    RandomBiasedQuotedNoCOTFormatter,
    RandomAgainstBiasedNoCOTFormatter,
    RandomAgainstBiasedQuotedNoCOTFormatter,
]

TRAINING_NO_COT_FORMATTERS_FEWSHOT = [
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
    CheckmarkNoCOTFormatter,
    CrossNoCOTFormatter,
    # TODO: Add "All answer A"
]


TRAINING_NO_COT_FORMATTERS = TRAINING_NO_COT_FORMATTERS_ZEROSHOT + TRAINING_NO_COT_FORMATTERS_FEWSHOT
TRAINING_NO_COT_FORMATTERS_WITH_UNBIASED = TRAINING_NO_COT_FORMATTERS + [ZeroShotUnbiasedFormatter]
TRAINING_DECEPTIVE_COT = DeceptiveAssistantTargetedFormatter
