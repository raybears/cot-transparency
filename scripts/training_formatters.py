from typing import Sequence, Type

from slist import Slist

from cot_transparency.formatters import StageOneFormatter
from cot_transparency.formatters.core.answer_always_a import (
    AnswerAlwaysAFormatter,
    AnswerAlwaysANoCOTFormatter,
)
from cot_transparency.formatters.core.sycophancy import (
    ZeroShotCOTSycophancyFormatter,
    ZeroShotSycophancyFormatter,
)
from cot_transparency.formatters.core.unbiased import (
    ZeroShotCOTUnbiasedFormatter,
    ZeroShotUnbiasedFormatter,
)
from cot_transparency.formatters.more_biases.anchor_initial_wrong import (
    ZeroShotInitialWrongFormatter,
)
from cot_transparency.formatters.more_biases.deceptive_assistant import (
    DeceptiveAssistantTargetedFormatter,
)
from cot_transparency.formatters.more_biases.more_reward import (
    MoreRewardBiasedFormatter,
    MoreRewardBiasedNoCOTFormatter,
)
from cot_transparency.formatters.more_biases.random_bias_formatter import (
    RandomAgainstBiasedFormatter,
    RandomAgainstBiasedNoCOTFormatter,
    RandomAgainstBiasedQuotedNoCOTFormatter,
    RandomAgainstQuotedBiasedFormatter,
    RandomBiasedFormatter,
    RandomBiasedNoCOTFormatter,
    RandomBiasedQuotedFormatter,
    RandomBiasedQuotedNoCOTFormatter,
)
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
)
from cot_transparency.formatters.verbalize.formatters import (
    CheckmarkBiasedFormatter,
    CheckmarkNoCOTFormatter,
    CrossBiasedFormatter,
    CrossNoCOTFormatter,
    StanfordBiasedFormatter,
    StanfordNoCOTFormatter,
)

# COT FORMATTERS

TRAINING_COT_FORMATTERS_ZERO_SHOT = [
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
    AnswerAlwaysAFormatter,
]
HAS_STRONG_EFFECT_FEW_SHOT_FORMATTERS: Sequence[Type[StageOneFormatter]] = [
    WrongFewShotIgnoreMistakesBiasedFormatter,
    CheckmarkBiasedFormatter,
    CrossBiasedFormatter,
    AnswerAlwaysANoCOTFormatter,  # use non cot for this since the COT version doesn't bias so much
]


TRAINING_COT_FORMATTERS: Sequence[Type[StageOneFormatter]] = (
    TRAINING_COT_FORMATTERS_ZERO_SHOT + TRAINING_COT_FORMATTERS_FEW_SHOT
)
TRAINING_COT_FORMATTERS_WITH_UNBIASED = list(TRAINING_COT_FORMATTERS) + [
    ZeroShotCOTUnbiasedFormatter
]


# NON-COT FORMATTERS

TRAINING_NO_COT_FORMATTERS_ZERO_SHOT: Slist[Type[StageOneFormatter]] = Slist(
    [
        StanfordNoCOTFormatter,
        MoreRewardBiasedNoCOTFormatter,
        ZeroShotSycophancyFormatter,
        RandomBiasedNoCOTFormatter,
        RandomBiasedQuotedNoCOTFormatter,
        RandomAgainstBiasedNoCOTFormatter,
        RandomAgainstBiasedQuotedNoCOTFormatter,
    ]
)

TRAINING_NO_COT_FORMATTERS_FEW_SHOT: Slist[Type[StageOneFormatter]] = Slist(
    [
        WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
        CheckmarkNoCOTFormatter,
        CrossNoCOTFormatter,
        AnswerAlwaysANoCOTFormatter,
    ]
)


TRAINING_NO_COT_FORMATTERS = (
    TRAINING_NO_COT_FORMATTERS_ZERO_SHOT + TRAINING_NO_COT_FORMATTERS_FEW_SHOT
)
TRAINING_NO_COT_FORMATTERS_WITH_UNBIASED = TRAINING_NO_COT_FORMATTERS + Slist(
    [ZeroShotUnbiasedFormatter]
)
TRAINING_DECEPTIVE_COT = DeceptiveAssistantTargetedFormatter
