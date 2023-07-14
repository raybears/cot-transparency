from enum import Enum
from typing import Any, Dict, Optional, Union

import numpy as np
import openai
from openai import APIError
from openai.error import APIConnectionError, RateLimitError, Timeout
from pydantic import BaseModel, conlist
from retry import retry
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


def parse_chat_prompt_response_dict(
    response_dict: Dict[Any, Any],
    prompt: list[ChatMessages],
) -> GPTFullResponse:
    response_id = response_dict["id"]
    top_choice = response_dict["choices"][0]
    completion = top_choice["message"]["content"]
    finish_reason = top_choice["finish_reason"]
    return GPTFullResponse(
        id=response_id,
        prompt=prompt,
        completion=completion,
        prompt_token_infos=Slist(),
        completion_token_infos=Slist(),
        completion_total_log_prob=0,
        average_completion_total_log_prob=0,
        finish_reason=finish_reason,
    )


def __get_chat_response_dict(
    config: OpenaiInferenceConfig,
    prompt: list[ChatMessages],
) -> Dict[Any, Any]:
    return openai.ChatCompletion.create(  # type: ignore
        model=config.model,
        messages=[chat.dict() for chat in prompt],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        presence_penalty=config.presence_penalty,
        frequency_penalty=config.frequency_penalty,
        top_p=1,
        n=1,
        stream=False,
        stop=[config.stop] if isinstance(config.stop, str) else config.stop,
    )


@retry(
    exceptions=(RateLimitError, APIConnectionError, Timeout, APIError),
    tries=6,
    delay=20,
    backoff=1.2,
)
def get_chat_response_simple(
    config: OpenaiInferenceConfig,
    prompt: str,
) -> GPTFullResponse:
    assert config.model == "gpt-3.5-turbo" or config.model == "gpt-4"
    messages = [ChatMessages(role=OpenaiRoles.user, content=prompt)]
    response = __get_chat_response_dict(
        config=config,
        prompt=messages,
    )
    return parse_chat_prompt_response_dict(prompt=messages, response_dict=response)


@retry(
    exceptions=(RateLimitError, APIConnectionError, Timeout, APIError),
    tries=6,
    delay=20,
    backoff=1.2,
)
def get_chat_response(
    config: OpenaiInferenceConfig,
    messages: list[ChatMessages],
) -> GPTFullResponse:
    assert config.model == "gpt-3.5-turbo" or config.model == "gpt-4"
    response_dict = __get_chat_response_dict(
        config=config,
        prompt=messages,
    )
    return parse_chat_prompt_response_dict(prompt=messages, response_dict=response_dict)
