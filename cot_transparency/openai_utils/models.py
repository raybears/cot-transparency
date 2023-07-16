from enum import Enum
from typing import Optional, Union

import numpy as np
from pydantic import BaseModel, conlist
from slist import Slist


class OpenaiInferenceConfig(BaseModel):
    # Config for openai
    model: str
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: int
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Union[None, str, conlist(str, min_items=1, max_items=4)] = None  # type: ignore


class TokenProba(BaseModel):
    token: str
    log_prob: float


class TokenInfo(BaseModel):
    token: str  # this is the token that got sampled
    log_prob: float  # the first token in the prompt will always have a log_prob of 0.0
    text_offset: int  # the offset of the token in the text
    # the top 5 tokens in the probability distribution
    # for first token in the prompt this is empty
    top_5_tokens: list[TokenProba]


class FinishReasons(str, Enum):
    stop = "stop"
    length = "length"


class OpenaiRoles(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"
    # If you are OpenAI chat, you need to add this back into the previous user message
    # Anthropic can handle it as per normal like an actual assistant
    assistant_preferred = "assistant_preferred"
    # none is designed for completion tasks where no role / tag will be added
    none = "none"


class ChatMessages(BaseModel):
    role: OpenaiRoles
    content: str


class GPTFullResponse(BaseModel):
    id: Optional[str]
    # list of ChatMessages if its a chat api
    prompt: str | list[ChatMessages]
    completion: str
    prompt_token_infos: list[TokenInfo]
    completion_token_infos: list[TokenInfo]
    completion_total_log_prob: float
    average_completion_total_log_prob: Optional[float]
    finish_reason: FinishReasons

    @property
    def token_infos(self) -> list[TokenInfo]:
        return self.prompt_token_infos + self.completion_token_infos

    @property
    def completion_tokens_length(self) -> int:
        return len(self.completion_token_infos)

    @property
    def average_completion_prob(self) -> Optional[float]:
        completion_token_infos_log_prob: Slist[float] = Slist(self.completion_token_infos).map(
            lambda token_info: token_info.log_prob
        )
        # convert them into probabilities and then average them
        probas: Slist[float] = completion_token_infos_log_prob.map(lambda log_prob: np.exp(log_prob))
        return probas.average()
