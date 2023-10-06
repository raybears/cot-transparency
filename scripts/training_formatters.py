from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter, ZeroShotSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
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


TRAINING_COT_FORMATTERS = [
    WrongFewShotIgnoreMistakesBiasedFormatter,
    StanfordBiasedFormatter,
    MoreRewardBiasedFormatter,
    ZeroShotCOTSycophancyFormatter,
    CheckmarkBiasedFormatter,
    CrossBiasedFormatter,
    RandomBiasedFormatter,
    RandomBiasedQuotedFormatter,
    RandomAgainstBiasedFormatter,
    RandomAgainstQuotedBiasedFormatter,
]
TRAINING_COT_FORMATTERS_WITH_UNBIASED = TRAINING_COT_FORMATTERS + [ZeroShotCOTUnbiasedFormatter]
TRAINING_NO_COT_FORMATTERS = [
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
    StanfordNoCOTFormatter,
    MoreRewardBiasedNoCOTFormatter,
    ZeroShotSycophancyFormatter,
    CheckmarkNoCOTFormatter,
    CrossNoCOTFormatter,
    RandomBiasedNoCOTFormatter,
    RandomBiasedQuotedNoCOTFormatter,
    RandomAgainstBiasedNoCOTFormatter,
    RandomAgainstBiasedQuotedNoCOTFormatter,
]
TRAINING_NO_COT_FORMATTERS_WITH_UNBIASED = TRAINING_NO_COT_FORMATTERS + [ZeroShotUnbiasedFormatter]
TRAINING_DECEPTIVE_COT = DeceptiveAssistantTargetedFormatter
