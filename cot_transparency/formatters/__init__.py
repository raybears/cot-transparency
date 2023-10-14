from cot_transparency.formatters.base_class import PromptFormatter, StageOneFormatter
from cot_transparency.formatters.core.answer_always_a import AnswerAlwaysANoCOTFormatter, AnswerAlwaysAFormatter
from cot_transparency.formatters.name_mapping import name_to_formatter
from cot_transparency.formatters.core.no_latex import (
    ZeroShotCOTUnbiasedNoLatexFormatter,
    ZeroShotUnbiasedNoLatexFormatter,
)
from cot_transparency.formatters.core.prompt_sensitivity_map import (
    no_cot_sensitivity_formatters,
    cot_sensitivity_formatters,
)
from cot_transparency.formatters.more_biases.anchor_initial_wrong import ZeroShotInitialWrongFormatter
from cot_transparency.formatters.more_biases.baseline_be_unbiased import BeUnbiasedCOTSycophancyFormatter
from cot_transparency.formatters.more_biases.model_written_evals import (
    ModelWrittenBiasedFormatter,
    ModelWrittenBiasedCOTFormatter,
    ModelWrittenBiasedCOTWithNoneAssistantFormatter,
    ModelWrittenBiasedCOTWithNoneFormatter,
    ModelWrittenBiasedCOTWithNoneAssistantPerspectiveFormatter,
    ModelWrittenBiasedWithNoneAssistantPerspectiveFormatter,
    ModelWrittenBiasedWithNoneFormatter,
)
from cot_transparency.formatters.more_biases.random_bias_formatter import (
    RandomAgainstBiasedFormatter,
    RandomBiasedFormatter,
    RandomBiasedNoCOTFormatter,
    RandomBiasedQuotedFormatter,
    RandomAgainstBiasedNoCOTFormatter,
    RandomAgainstQuotedBiasedFormatter,
    RandomAgainstBiasedQuotedNoCOTFormatter,
    RandomBiasedQuotedNoCOTFormatter,
)
from cot_transparency.formatters.more_biases.user_wrong_cot import (
    UserBiasedWrongCotFormatter,
    ModelBiasedWrongCotFormatter,
)
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

from cot_transparency.formatters.symbol_tuning.bbq_symbol_few_shot import BBQSymbolTuningCOTFewShot

from cot_transparency.formatters.task_decomposition.decompose_step_by_step import (
    DecomposeUnbiasedFormatter,
    DecomposeBiasedFormatter,
    DecomposeStanfordBiasedFormatter,
    DecomposeMoreRewardBiasedFormatter,
)

from cot_transparency.formatters.prompt_addition_python.pal_few_shot import (
    PALFewShot,
)

from cot_transparency.formatters.verbalize.formatters import (
    StanfordBiasedFormatter,
    StanfordTreatmentFormatter,
    CrossBiasedLabelFormatter,
    CrossTreatmentFormatter,
    CheckmarkBiasedLabelFormatter,
    CheckmarkTreatmentFormatter,
    IThinkAnswerTreatmentFormatter,
    IThinkAnswerBiasedFormatter,
    StanfordCalibratedFormatter,
    CrossNoCOTFormatter,
    CheckmarkNoCOTFormatter,
    StanfordNoCOTFormatter,
    CrossBiasedFormatter,
    CheckmarkBiasedFormatter,
    StanfordBiasedLabelFormatter,
)

from cot_transparency.formatters.transparency.mistakes import (
    CompletePartialCOT,
    FewShotGenerateMistakeFormatter,
)

from cot_transparency.formatters.transparency.util import FullCOTFormatter

from cot_transparency.formatters.transparency.s1_baselines import (
    FewShotCOTUnbiasedCompletionNoRoleTameraTFormatter,
    FewShotCOTUnbiasedTameraTFormatter,
    ZeroShotCOTUnbiasedTameraTFormatter,
)

from cot_transparency.formatters.transparency.interventions.logical_consequence import (
    LogicalConsequenceChatFormatter,
    LogicalConsequence2ChatFormatter,
    LogicalConsequence2ChatFS2Formatter,
    LogicalConsequence2ChatFS3Formatter,
    LogicalConsequence2ChatFS5Formatter,
    LogicalConsequence2ChatFS10Formatter,
    LogicalConsequence2ChatFS15Formatter,
    LogicalConsequence3ChatFormatter,
    LogicalConsequence3ChatFS2Formatter,
    LogicalConsequence3ChatFS3Formatter,
    LogicalConsequence3ChatFS5Formatter,
    LogicalConsequence3ChatFS10Formatter,
    LogicalConsequence3ChatFS15Formatter,
)
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
    WrongFewShotIgnoreMistakesBiasedFormatter,
)
from cot_transparency.formatters.more_biases.deceptive_assistant import (
    DeceptiveAssistantBiasedFormatter,
    DeceptiveAssistantBiasedNoCOTFormatter,
    DeceptiveAssistantTargetedFormatter,
)
from cot_transparency.formatters.more_biases.more_reward import (
    MoreRewardBiasedFormatter,
    MoreRewardBiasedNoCOTFormatter,
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
        CrossBiasedLabelFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        CrossTreatmentFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        CheckmarkBiasedLabelFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        CheckmarkTreatmentFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        IThinkAnswerBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        IThinkAnswerTreatmentFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        StanfordCalibratedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        CrossNoCOTFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        CheckmarkNoCOTFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        StanfordNoCOTFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        UserBiasedWrongCotFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        WrongFewShotBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        WrongFewShotIgnoreMistakesBiasedNoCOTFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        DeceptiveAssistantBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        MoreRewardBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        BeUnbiasedCOTSycophancyFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        CrossBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        ModelBiasedWrongCotFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        CheckmarkBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        MoreRewardBiasedNoCOTFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        DeceptiveAssistantBiasedNoCOTFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        DeceptiveAssistantTargetedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        StanfordBiasedLabelFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        ModelWrittenBiasedFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        ModelWrittenBiasedCOTFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        ModelWrittenBiasedCOTWithNoneFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        ModelWrittenBiasedCOTWithNoneAssistantFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        ModelWrittenBiasedCOTWithNoneAssistantPerspectiveFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        ModelWrittenBiasedWithNoneAssistantPerspectiveFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        ModelWrittenBiasedWithNoneFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        WrongFewShotIgnoreMistakesBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        BBQSymbolTuningCOTFewShot.name(): BBQSymbolTuningCOTFewShot.name(),
        DecomposeUnbiasedFormatter.name(): DecomposeUnbiasedFormatter.name(),
        DecomposeBiasedFormatter.name(): DecomposeBiasedFormatter.name(),
        DecomposeStanfordBiasedFormatter.name(): DecomposeStanfordBiasedFormatter.name(),
        DecomposeMoreRewardBiasedFormatter.name(): DecomposeMoreRewardBiasedFormatter.name(),
        PALFewShot.name(): PALFewShot.name(),
        ZeroShotInitialWrongFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        RandomAgainstBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        RandomBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        RandomBiasedNoCOTFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        RandomBiasedQuotedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        RandomAgainstBiasedNoCOTFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        RandomAgainstQuotedBiasedFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        RandomAgainstBiasedQuotedNoCOTFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        RandomBiasedQuotedNoCOTFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        ZeroShotCOTUnbiasedNoLatexFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        ZeroShotUnbiasedNoLatexFormatter.name(): ZeroShotUnbiasedFormatter.name(),
        AnswerAlwaysAFormatter.name(): ZeroShotCOTUnbiasedFormatter.name(),
        AnswerAlwaysANoCOTFormatter.name(): ZeroShotUnbiasedFormatter.name(),
    }

    for formatter in no_cot_sensitivity_formatters:
        mapping[formatter.name()] = ZeroShotCOTUnbiasedFormatter.name()
    for formatter in cot_sensitivity_formatters:
        mapping[formatter.name()] = ZeroShotCOTUnbiasedFormatter.name()

    return mapping[biased_formatter_name]


__all__ = [
    "bias_to_unbiased_formatter",
    "StageOneFormatter",
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
    "CrossBiasedLabelFormatter",
    "CrossTreatmentFormatter",
    "CheckmarkBiasedLabelFormatter",
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
    "ZeroShotCOTUnbiasedTameraTFormatter",
    "LogicalConsequenceChatFormatter",
    "UserBiasedWrongCotFormatter",
    "LogicalConsequence2ChatFormatter",
    "BBQSymbolTuningCOTFewShot",
    "DecomposeUnbiasedFormatter",
    "DecomposeBiasedFormatter",
    "DecomposeStanfordBiasedFormatter",
    "DecomposeMoreRewardBiasedFormatter",
    "PALFewShot",
    "LogicalConsequence2ChatFS2Formatter",
    "LogicalConsequence2ChatFS3Formatter",
    "LogicalConsequence2ChatFS5Formatter",
    "LogicalConsequence2ChatFS10Formatter",
    "LogicalConsequence2ChatFS15Formatter",
    "LogicalConsequence3ChatFormatter",
    "LogicalConsequence3ChatFS2Formatter",
    "LogicalConsequence3ChatFS3Formatter",
    "LogicalConsequence3ChatFS5Formatter",
    "LogicalConsequence3ChatFS10Formatter",
    "LogicalConsequence3ChatFS15Formatter",
]
