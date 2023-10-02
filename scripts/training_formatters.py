from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter, ZeroShotSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.formatters.more_biases.deceptive_assistant import DeceptiveAssistantBiasedFormatter, \
    DeceptiveAssistantTargetedFormatter
from cot_transparency.formatters.more_biases.more_reward import (
    MoreRewardBiasedFormatter,
    MoreRewardBiasedNoCOTFormatter,
)
from cot_transparency.formatters.more_biases.random_bias_formatter import (
    RandomBiasedFormatter,
    RandomAgainstBiasedFormatter,
    RandomBiasedNoCOTFormatter,
    RandomAgainstBiasedNoCOTFormatter,
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

TRAINING_COT_FORMATTERS = [
    ZeroShotCOTUnbiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedFormatter,
    StanfordBiasedFormatter,
    MoreRewardBiasedFormatter,
    ZeroShotCOTSycophancyFormatter,
    CheckmarkBiasedFormatter,
    CrossBiasedFormatter,
    RandomBiasedFormatter,
    RandomAgainstBiasedFormatter,
]
TRAINING_NO_COT_FORMATTERS = [
    ZeroShotUnbiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
    StanfordNoCOTFormatter,
    MoreRewardBiasedNoCOTFormatter,
    ZeroShotSycophancyFormatter,
    CheckmarkNoCOTFormatter,
    CrossNoCOTFormatter,
    RandomBiasedNoCOTFormatter,
    RandomAgainstBiasedNoCOTFormatter,
]
TRAINING_DECEPTIVE_COT = DeceptiveAssistantTargetedFormatter
