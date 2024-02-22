import dataclasses
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
    InitialWrongNonCOTFormatter,
    PostHocAnchor,
    PostHocDontAnchor,
    InitialWrongMoreClearFormatter,
    PostHocNoPlease,
    ZeroShotInitialWrongFormatter,
)
from cot_transparency.formatters.more_biases.deceptive_assistant import (
    DeceptiveAssistantTargetedFormatter,
)
from cot_transparency.formatters.more_biases.distractor_fact import (
    FirstLetterDistractor,
    FirstAndLastWordDistractor,
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
from cot_transparency.formatters.more_biases.user_wrong_cot import (
    DistractorArgumentNeedHelp,
    DistractorArgumentNotsure,
    ImprovedDistractorArgument,
    DistractorArgumentCorrectOrWrong,
    DistractorArgumentImportant,
    DistractorArgumentNoTruthfullyAnswer,
    DistractorAnswerWithoutInfluence,
    ReadOnInternetNoCotFormatter,
)
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
    WrongFewShotMoreClearlyLabelledAtBottom,
)
from cot_transparency.formatters.prompt_sensitivity.automated_generations import (
    AskWithDistractorFactNoCot,
    DistractorFactMaybeWrong,
    DistractorFactMiles,
)
from cot_transparency.formatters.verbalize.formatters import (
    BlackSquareBiasedFormatter,
    BlackSquareBiasedNoCoTFormatter,
    CheckmarkBiasedFormatter,
    CheckmarkNoCOTFormatter,
    CrossBiasedFormatter,
    CrossNoCOTFormatter,
    StanfordBiasedFormatter,
    StanfordNoCOTFormatter,
)

# COT FORMATTERS

TRAINING_COT_FORMATTERS_ZERO_SHOT = [
    # StanfordBiasedFormatter,
    # MoreRewardBiasedFormatter,  # removed because unclear whether models should follow the reward or not
    # ZeroShotCOTSycophancyFormatter,
    RandomBiasedFormatter,
    RandomBiasedQuotedFormatter,
    RandomAgainstBiasedFormatter,
    RandomAgainstQuotedBiasedFormatter,
    InitialWrongMoreClearFormatter,
    ZeroShotInitialWrongFormatter,  # There is only a COT version of this formatter
    PostHocDontAnchor,
    PostHocAnchor,
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

# INTERESTING FORMATTERS FOR THE GRID
INTERESTING_FORMATTERS = [
    RandomBiasedFormatter,  # Suggested answer
    PostHocNoPlease,
    # WrongFewShotIgnoreMistakesBiasedFormatter,  # Wrong Few Shot
    # WrongFewShotMoreClearlyLabelled,
    WrongFewShotMoreClearlyLabelledAtBottom,
    BlackSquareBiasedFormatter,  # Spurious Few Shot
    # AskWithDistractorFact,
    # DistractorFactMaybeWrong,
    FirstLetterDistractor,
    # DistractorFactMiles,
    # FirstAndLastWordDistractor,
    ImprovedDistractorArgument,  # Distractor Argument V2
    DistractorAnswerWithoutInfluence,
    DistractorArgumentCorrectOrWrong,
    DistractorArgumentImportant,
    DistractorArgumentNotsure,
    DistractorArgumentNoTruthfullyAnswer,
    ZeroShotCOTUnbiasedFormatter,  # unbiased baseline
    # EmptyDistractorFact,
    # Ed's Distractor Argument
    # StanfordBiasedFormatter,
    # MoreRewardBiasedFormatter,
    # ZeroShotCOTSycophancyFormatter,
    # CheckmarkBiasedFormatter,
    # CrossBiasedFormatter,
    # PostHocDontAnchor,
    # PostHocAnchor,
    # PostHocAnchor2,
    # PostHocAnchor3,
]

ARE_YOU_SURE_COT_NAME = "2) Are you sure"


JUDGE_INCONSISTENCY_NAME = "Answer Choice Ordering"
ANSWER_CHOICE_CLAUDES = "Answer Choice Ordering (Claude 2.1 vs Claude Instant 1.2)"
ANSWER_CHOICE_NAME = "zzz10a ) Answer Choice Ordering (GPT 3.5 vs GPT 4)"
ANSWER_CHOICE_ALLOW_COUNT_TIES = "zzz10b ) Answer Choice Ordering allow and count ties(GPT 3.5 vs GPT 4)"
ANSWER_CHOICE_ALLOW_DONT_COUNT_TIES = "zzz10c ) Answer Choice Ordering allow but exclude ties(GPT 3.5 vs GPT 4)"
# put numbering so that when pandas groupby its in order
FORMATTERS_TO_PAPER_NAME = {
    "RandomBiasedFormatter": "1) Suggested answer",
    ARE_YOU_SURE_COT_NAME: "2) Are you sure",
    "InitialWrongMoreClearFormatter2": "3) Post Hoc",
    PostHocNoPlease.name(): "3) Post Hoc",
    "WrongFewShotIgnoreMistakesBiasedFormatter": "4) Wrong Few Shot",
    "WrongFewShotMoreClearlyLabelled": "4b) Wrong Few Shot without human and assistant text",
    WrongFewShotMoreClearlyLabelledAtBottom.name(): "4c) Wrong Few Shot without human and assistant text, instructions at the bottom",
    "BlackSquareBiasedFormatter": "5) Spurious Few Shot: Squares",
    "hindsight_neglect": "6) Spurious Few Shot: Hindsight",
    "hindsight_neglect_baseline": "zzz13) Hindsight Unbiased Baseline",
    "ReadOnInternetCotFormatter": "8) Distractor: Argument",
    ImprovedDistractorArgument.name(): "8a) Distractor: Argument",
    DistractorAnswerWithoutInfluence.name(): "8a) Distractor: Argument",
    DistractorArgumentNoTruthfullyAnswer.name(): "8a) Distractor: Argument",
    DistractorArgumentCorrectOrWrong.name(): "8a) Distractor: Argument",
    DistractorArgumentImportant.name(): "8a) Distractor: Argument",
    DistractorArgumentNotsure.name(): "8a) Distractor: Argument",
    DistractorArgumentNeedHelp.name(): "8a) Distractor: Argument",
    # ImprovedDistractorArgument.name(): "7a) Distractor: Argument",
    # DistractorAnswerWithoutInfluence.name(): "7b) Distractor: Argument, Answer without influence",
    # DistractorArgumentNoTruthfullyAnswer.name(): "7c) Distractor: Argument, no truthfully answer statement",
    # DistractorArgumentCorrectOrWrong.name(): "7d) Distractor: Argument, specified the argument may be correct or wrong",
    # DistractorArgumentImportant.name(): "7e) Distractor: Argument, added important: at the beginning of statement",
    # DistractorArgumentNotsure.name(): "7f) Distractor: Argument, not sure",
    # DistractorArgumentNeedHelp.name(): "7g) Distractor: Argument, need help",
    "AskWithDistractorFact": "7) Distractor: Fact",
    DistractorFactMaybeWrong.name(): "7b) Distractor: Fact, pointed out maybe wrong",
    FirstLetterDistractor.name(): "7c) Distractor: Fact, first letter",
    FirstAndLastWordDistractor.name(): "7d) Distractor: Fact, first letter even more",
    DistractorFactMiles.name(): "7) Distractor: Fact, miles",
    "ZeroShotCOTUnbiasedFormatter": "zzz11) Unbiased Baseline on COT",
    "ZeroShotUnbiasedFormatter": "zzz12) Unbiased Baseline on Non COT",
}

INTERESTING_FORMATTERS_NO_COT = [
    RandomBiasedNoCOTFormatter,
    # InitialWrongMoreClearFormatter2,  # Asks to explain reasoning so cannot be used for non CoT
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
    BlackSquareBiasedNoCoTFormatter,
    ReadOnInternetNoCotFormatter,
    ZeroShotUnbiasedFormatter,
    AskWithDistractorFactNoCot,
]

INTERESTING_FORMATTERS_COT_AND_NO_COT = INTERESTING_FORMATTERS + INTERESTING_FORMATTERS_NO_COT

TRAINING_COT_FORMATTERS_WITH_UNBIASED = list(TRAINING_COT_FORMATTERS) + [ZeroShotCOTUnbiasedFormatter]


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
        InitialWrongNonCOTFormatter,
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


TRAINING_NO_COT_FORMATTERS = TRAINING_NO_COT_FORMATTERS_ZERO_SHOT + TRAINING_NO_COT_FORMATTERS_FEW_SHOT
TRAINING_NO_COT_FORMATTERS_WITH_UNBIASED = TRAINING_NO_COT_FORMATTERS + Slist([ZeroShotUnbiasedFormatter])
TRAINING_DECEPTIVE_COT = DeceptiveAssistantTargetedFormatter


@dataclasses.dataclass(kw_only=True)
class BiasCotNonCot:
    name: str
    cot: Type[StageOneFormatter]
    non_cot: Type[StageOneFormatter] | None

    def as_list(self) -> Sequence[Type[StageOneFormatter] | None]:
        return [self.cot, self.non_cot]


BIAS_PAIRS: Sequence[BiasCotNonCot] = [
    BiasCotNonCot(name="Stanford", cot=StanfordBiasedFormatter, non_cot=StanfordNoCOTFormatter),
    BiasCotNonCot(name="More Reward", cot=MoreRewardBiasedFormatter, non_cot=MoreRewardBiasedNoCOTFormatter),
    BiasCotNonCot(name="Zero Shot Sycophancy", cot=ZeroShotCOTSycophancyFormatter, non_cot=ZeroShotSycophancyFormatter),
    BiasCotNonCot(name="Model generated sycophancy", cot=RandomBiasedFormatter, non_cot=RandomBiasedNoCOTFormatter),
    BiasCotNonCot(
        name="Model generated sycophancy Quoted",
        cot=RandomBiasedQuotedFormatter,
        non_cot=RandomBiasedQuotedNoCOTFormatter,
    ),
    BiasCotNonCot(
        name="Model generated against Bias", cot=RandomAgainstBiasedFormatter, non_cot=RandomAgainstBiasedNoCOTFormatter
    ),
    BiasCotNonCot(
        name="Random generated Against Quoted Bias",
        cot=RandomAgainstQuotedBiasedFormatter,
        non_cot=RandomAgainstBiasedQuotedNoCOTFormatter,
    ),
    # This is a special case, since there is no non-cot version
    BiasCotNonCot(name="Model is Initially Wrong", cot=ZeroShotInitialWrongFormatter, non_cot=None),
    BiasCotNonCot(
        name="Wrong Few Shot Ignore Mistakes",
        cot=WrongFewShotIgnoreMistakesBiasedFormatter,
        non_cot=WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
    ),
    BiasCotNonCot(name="Checkmark", cot=CheckmarkBiasedFormatter, non_cot=CheckmarkNoCOTFormatter),
    BiasCotNonCot(name="Cross", cot=CrossBiasedFormatter, non_cot=CrossNoCOTFormatter),
    BiasCotNonCot(name="Answer Always A", cot=AnswerAlwaysAFormatter, non_cot=AnswerAlwaysANoCOTFormatter),
]
