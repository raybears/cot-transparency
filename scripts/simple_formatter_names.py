from typing import Type

from cot_transparency.formatters import PromptFormatter
from cot_transparency.formatters.more_biases.user_wrong_cot import UserBiasedWrongCotFormatter
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotBiasedFormatter,
)
from cot_transparency.formatters.more_biases.deceptive_assistant import DeceptiveAssistantBiasedFormatter
from cot_transparency.formatters.more_biases.more_reward import MoreRewardBiasedFormatter

FORMATTER_TO_SIMPLE_NAME: dict[Type[PromptFormatter], str] = {
    DeceptiveAssistantBiasedFormatter: "Tell model to be deceptive",
    MoreRewardBiasedFormatter: "More reward for an option",
    UserBiasedWrongCotFormatter: "User says wrong reasoning",
    WrongFewShotBiasedFormatter: "Wrong label in the few shot",
}
