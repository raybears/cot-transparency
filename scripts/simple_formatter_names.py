from typing import Type

from cot_transparency.formatters import PromptFormatter
from cot_transparency.formatters.biased_wrong_cot.formatters import UserBiasedWrongCotFormatter
from cot_transparency.formatters.wrong_few_shot.formatters import (
    DeceptiveAssistantBiasedFormatter,
    MoreRewardBiasedFormatter,
    WrongFewShotBiasedFormatter,
)

FORMATTER_TO_SIMPLE_NAME: dict[Type[PromptFormatter], str] = {
    DeceptiveAssistantBiasedFormatter: "Tell model to be deceptive",
    MoreRewardBiasedFormatter: "More reward for an option",
    UserBiasedWrongCotFormatter: "User says wrong reasoning",
    WrongFewShotBiasedFormatter: "Wrong label in the few shot",
}
