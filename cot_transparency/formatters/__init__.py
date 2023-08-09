from typing import Type

from cot_transparency.formatters.base_class import PromptFormatter
from cot_transparency.formatters.biased_wrong_cot.formatters import UserBiasedWrongCotFormatter
from cot_transparency.formatters.core.sycophancy import (
    ZeroShotCOTSycophancyFormatter,
    ZeroShotCOTSycophancyNoRoleFormatter,
    ZeroShotCOTSycophancyToldBiasFormatter,
    ZeroShotSycophancyFormatter,
    ZeroShotSycophancyNoRoleFormatter,
)
from cot_transparency.formatters.core.unbiased import (
    ZeroShotCOTUnbiasedFormatter,
    ZeroShotCOTUnbiasedNoRoleFormatter,
    ZeroShotUnbiasedFormatter,
    ZeroShotUnbiasedNoRoleFormatter,
    FewShotCOTUnbiasedNoRoleFormatter,
    FewShotUnbiasedNoRoleFormatter,
)

from cot_transparency.formatters.verbalize.formatters import (
    StanfordBiasedFormatter,
    StanfordTreatmentFormatter,
    CrossBiasedFormatter,
    CrossTreatmentFormatter,
    CheckmarkBiasedFormatter,
    CheckmarkTreatmentFormatter,
    IThinkAnswerTreatmentFormatter,
    IThinkAnswerBiasedFormatter,
    StanfordCalibratedFormatter,
    CrossNoCOTFormatter,
    CheckmarkNoCOTFormatter,
    StanfordNoCOTFormatter,
)

from cot_transparency.formatters.transparency.mistakes import (
    CompletePartialCOT,
    FewShotGenerateMistakeFormatter,
)

from cot_transparency.formatters.transparency.util import FullCOTFormatter

from cot_transparency.formatters.transparency.s1_baselines import (
    FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter,
    FewShotCOTUnbiasedTameraTFormatter,
    ZeroShotCOTUnbiasedChatTameraTFormatter,
)

from cot_transparency.formatters.transparency.interventions.logical_consequence import (
    LogicalConsequenceChatFormatter,
)
from cot_transparency.formatters.wrong_few_shot.formatters import (
    WrongFewShotBiasedFormatter,
    DeceptiveAssistantBiasedFormatter,
    MoreRewardBiasedFormatter,
)


def bias_to_unbiased_formatter(biased_formatter_name: str) -> str:
    if not name_to_formatter(biased_formatter_name).is_biased:
        return biased_formatter_name

    mapping = {
        ZeroShotCOTSycophancyFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        ZeroShotSycophancyFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        ZeroShotSycophancyNoRoleFormatter.name(): ZeroShotUnbiasedNoRoleFormatter.name(),
        ZeroShotCOTSycophancyNoRoleFormatter.name(): ZeroShotCOTUnbiasedNoRoleFormatter.name(),
        ZeroShotCOTSycophancyToldBiasFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        StanfordBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        StanfordTreatmentFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        CrossBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        CrossTreatmentFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        CheckmarkBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        CheckmarkTreatmentFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        IThinkAnswerBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        IThinkAnswerTreatmentFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        StanfordCalibratedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        CrossNoCOTFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        CheckmarkNoCOTFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        StanfordNoCOTFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        UserBiasedWrongCotFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        WrongFewShotBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        DeceptiveAssistantBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        MoreRewardBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
    }
    return mapping[biased_formatter_name]


def name_to_formatter(name: str) -> Type[PromptFormatter]:
    mapping = PromptFormatter.all_formatters()
    return mapping[name]


__all__ = [
    "bias_to_unbiased_formatter",
    "name_to_formatter",
    "PromptFormatter",
    "ZeroShotCOTSycophancyFormatter",
    "ZeroShotCOTSycophancyNoRoleFormatter",
    "ZeroShotCOTSycophancyToldBiasFormatter",
    "ZeroShotSycophancyFormatter",
    "ZeroShotSycophancyNoRoleFormatter",
    "ZeroShotCOTUnbiasedFormatter",
    "ZeroShotCOTUnbiasedNoRoleFormatter",
    "ZeroShotUnbiasedFormatter",
    "ZeroShotUnbiasedNoRoleFormatter",
    "FewShotCOTUnbiasedNoRoleFormatter",
    "FewShotUnbiasedNoRoleFormatter",
    "StanfordBiasedFormatter",
    "StanfordTreatmentFormatter",
    "CrossBiasedFormatter",
    "CrossTreatmentFormatter",
    "CheckmarkBiasedFormatter",
    "CheckmarkTreatmentFormatter",
    "IThinkAnswerTreatmentFormatter",
    "IThinkAnswerBiasedFormatter",
    "StanfordCalibratedFormatter",
    "CrossNoCOTFormatter",
    "CheckmarkNoCOTFormatter",
    "StanfordNoCOTFormatter",
    "FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter",
    "FewShotCOTUnbiasedTameraTFormatter",
    "CompletePartialCOT",
    "FullCOTFormatter",
    "FewShotGenerateMistakeFormatter",
    "ZeroShotCOTUnbiasedChatTameraTFormatter",
    "ZeroShotCOTUnbiasedChatTameraTFormatter",
    "LogicalConsequenceChatFormatter",
    "UserBiasedWrongCotFormatter",
]
