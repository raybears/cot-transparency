from typing import Optional, Type

from cot_transparency.formatters import PromptFormatter
from cot_transparency.formatters.core.answer_always_a import AnswerAlwaysANoCOTFormatter
from cot_transparency.formatters.interventions.consistency import (
    BiasedConsistency10,
    NaiveFewShot3,
    NaiveFewShot6,
    NaiveFewShot10,
    NaiveFewShot12,
    NaiveFewShot16,
    NaiveFewShotLabelOnly10,
    NaiveFewShotLabelOnly30,
    PairedConsistency10,
    PairedConsistency12,
)
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.formatters.more_biases.deceptive_assistant import (
    DeceptiveAssistantBiasedFormatter,
)
from cot_transparency.formatters.more_biases.model_written_evals import (
    ModelWrittenBiasedCOTFormatter,
    ModelWrittenBiasedCOTWithNoneAssistantFormatter,
    ModelWrittenBiasedCOTWithNoneAssistantPerspectiveFormatter,
    ModelWrittenBiasedCOTWithNoneFormatter,
    ModelWrittenBiasedFormatter,
    ModelWrittenBiasedWithNoneAssistantPerspectiveFormatter,
    ModelWrittenBiasedWithNoneFormatter,
)
from cot_transparency.formatters.more_biases.more_reward import (
    MoreRewardBiasedFormatter,
)
from cot_transparency.formatters.more_biases.user_wrong_cot import (
    UserBiasedWrongCotFormatter,
)
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedFormatter,
)
from cot_transparency.formatters.verbalize.formatters import (
    CheckmarkBiasedFormatter,
    CrossBiasedFormatter,
)

# ruff: noqa: E501

FORMATTER_TO_SIMPLE_NAME: dict[Type[PromptFormatter], str] = {
    DeceptiveAssistantBiasedFormatter: "Tell model to be deceptive",
    MoreRewardBiasedFormatter: "More reward for an option",
    UserBiasedWrongCotFormatter: "User says wrong reasoning",
    WrongFewShotBiasedFormatter: "Wrong label in the few shot",
    WrongFewShotIgnoreMistakesBiasedFormatter: "Wrong label in the few shot",
    CheckmarkBiasedFormatter: "Checkmark indicating the correct answer in few shot",
    CrossBiasedFormatter: "Cross indicating the correct answer in few shot",
    AnswerAlwaysANoCOTFormatter: "Answer is always A in few shot",
    ModelWrittenBiasedCOTFormatter: "",
    ModelWrittenBiasedCOTWithNoneFormatter: "+ None of the above",
    ModelWrittenBiasedCOTWithNoneAssistantFormatter: "+ None of the above + You are a helpful assistant",
    ModelWrittenBiasedCOTWithNoneAssistantPerspectiveFormatter: "+ None of the above + You are a helpful assistant. Answer in your own perspective as an AI",
    ModelWrittenBiasedFormatter: "",
    ModelWrittenBiasedWithNoneFormatter: "+ None of the above",
    ModelWrittenBiasedWithNoneAssistantPerspectiveFormatter: "+ None of the above + You are a helpful assistant. Answer in your own perspective as an AI",
}


INTERVENTION_TO_SIMPLE_NAME: dict[Optional[Type[Intervention]], str] = {
    None: "No intervention, biased context",
    PairedConsistency10: "Biased and Unbiased question pairs in few shot",
    PairedConsistency12: "6 Biased and Unbiased question pairs in few shot, ~5k tokens",
    BiasedConsistency10: "Biased questions in few shot",
    NaiveFewShot3: "3 Unbiased questions with COT answer",
    NaiveFewShot6: "6 Unbiased questions with COT answer ~2.5k tokens",
    NaiveFewShot12: "12 Unbiased questions with COT answer ~5k tokens",
    NaiveFewShot10: "10 Unbiased questions with COT answer",
    NaiveFewShot16: "16 Unbiased questions with COT answer",
    NaiveFewShotLabelOnly10: "10 Unbiased questions, with ground truth labels",
    NaiveFewShotLabelOnly30: "30 Unbiased questions, with ground truth labels",
}
