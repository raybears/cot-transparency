from enum import Enum
import os
from typing import Any, Dict, List, Optional, Union
import anthropic
from dotenv import load_dotenv
import numpy as np

import openai
from openai import APIError
from openai.error import APIConnectionError, RateLimitError, ServiceUnavailableError, Timeout
from pydantic import BaseModel
from retry import retry
from slist import Slist
from cot_transparency.data_models.models import (
    OpenaiInferenceConfig,
    StrictChatMessage,
    StrictMessageRole,
)

import logging

from cot_transparency.openai_utils.rate_limiting import token_rate_limiter
from cot_transparency.util import setup_logger

logger = setup_logger(__name__, logging.INFO)

load_dotenv()
NUM_ORGS = len(os.getenv("OPENAI_ORG_IDS", "").split(","))
RATE_SCALER = float(os.getenv("RATE_SCALER", "1.0"))
total_rate_sf = NUM_ORGS * RATE_SCALER
print("Total rate scaling", total_rate_sf)

retry_openai_failures = retry(
    exceptions=(APIConnectionError, Timeout, APIError, ServiceUnavailableError), tries=20, delay=1, logger=None
)


should_log_rate_limit = os.getenv("LOG_RATE_LIMITS", "false").lower() == "true"
rate_limit_logger = setup_logger("rate_limit_logger", logging.INFO) if should_log_rate_limit else None
retry_openai_rate_limits = retry(
    exceptions=(RateLimitError), tries=-1, delay=10, logger=rate_limit_logger, jitter=(0, 5), max_delay=30
)


class TokenProba(BaseModel):
    token: str
    log_prob: float


class FinishReasons(str, Enum):
    stop = "stop"
    length = "length"


class TokenInfo(BaseModel):
    token: str  # this is the token that got sampled
    log_prob: float  # the first token in the prompt will always have a log_prob of 0.0
    text_offset: int  # the offset of the token in the text
    # the top 5 tokens in the probability distribution
    # for first token in the prompt this is empty
    top_5_tokens: list[TokenProba]


class GPTFullResponse(BaseModel):
    id: Optional[str]
    # list of ChatMessages if its a chat api
    prompt: str | list[StrictChatMessage]
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


def parse_gpt_response(prompt: str, response_dict: Dict[Any, Any], end_tokens: set[str]) -> GPTFullResponse:
    response_id = response_dict["id"]
    completion = response_dict["choices"][0]["text"][len(prompt) :]
    logprobs: List[Union[int, None]] = response_dict["choices"][0]["logprobs"]["token_logprobs"]
    # the first token has a logprob of "None" so we need to change it to 0
    edited_logprobs: Slist[int] = Slist(logprobs).map(lambda x: x if x is not None else 0)
    tokens: Slist[str] = Slist(response_dict["choices"][0]["logprobs"]["tokens"])
    top_5_probabilities: Slist[Slist[TokenProba]] = Slist(response_dict["choices"][0]["logprobs"]["top_logprobs"]).map(
        lambda maybe_dict: Slist.from_dict(maybe_dict).map(lambda tup: TokenProba(token=tup[0], log_prob=tup[1]))
        # the first token has None instead of a dict
        if maybe_dict is not None
        else Slist()
    )

    finish_reason = response_dict["choices"][0]["finish_reason"]
    offsets: Slist[int] = Slist(response_dict["choices"][0]["logprobs"]["text_offset"])

    token_infos: Slist[TokenInfo] = tokens.zip(edited_logprobs, top_5_probabilities, offsets).map(
        lambda tup: TokenInfo(token=tup[0], log_prob=tup[1], top_5_tokens=tup[2], text_offset=tup[3])
    )

    # now you need to find out where the prompt ends and the completion begins
    # using the text_offset
    prompt_offset = len(prompt)
    prompt_token_infos, completion_token_infos = token_infos.split_by(lambda x: x.text_offset < prompt_offset)
    # this is dumb, but sometimes openai adds tokens beyond the end token
    completion_token_infos = completion_token_infos.take_until_inclusive(lambda x: x.token in end_tokens)

    completion_token_infos_log_prob = completion_token_infos.map(lambda token_info: token_info.log_prob)

    return GPTFullResponse(
        id=response_id,
        prompt=prompt,
        completion=completion,
        prompt_token_infos=prompt_token_infos,
        completion_token_infos=completion_token_infos,
        completion_total_log_prob=completion_token_infos_log_prob.sum(),
        average_completion_total_log_prob=completion_token_infos_log_prob.average(),
        finish_reason=finish_reason,
    )


# yolo no rate limits
@retry_openai_failures
@retry_openai_rate_limits
def get_openai_completion(
    config: OpenaiInferenceConfig,
    prompt: str,
) -> GPTFullResponse:
    try:
        response_dict: Dict[Any, Any] = openai.Completion.create(  # type: ignore
            model=config.model,
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            top_p=1,
            n=1,
            stream=False,
            stop=[config.stop] if isinstance(config.stop, str) else config.stop,
            # needed to get logprobs
            logprobs=5,
            # needed to get logprobs of prompt
            echo=True,
        )
    except APIError as e:
        print(f"APIError with prompt: {prompt}")
        raise e

    end_tokens: set[str] = (
        set(config.stop) if isinstance(config.stop, list) else {config.stop} if isinstance(config.stop, str) else set()
    )
    return parse_gpt_response(prompt=prompt, response_dict=response_dict, end_tokens=end_tokens)


def parse_chat_prompt_response_dict(
    response_dict: Dict[Any, Any],
    prompt: list[StrictChatMessage],
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
    prompt: list[StrictChatMessage],
) -> Dict[Any, Any]:
    return openai.ChatCompletion.create(  # type: ignore
        model=config.model,
        messages=[chat.dict() for chat in prompt],
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        presence_penalty=config.presence_penalty,
        frequency_penalty=config.frequency_penalty,
        top_p=config.top_p,
        n=1,
        stream=False,
        stop=[config.stop] if isinstance(config.stop, str) else config.stop,
    )


@retry(
    exceptions=(RateLimitError, APIConnectionError, Timeout, APIError),
    tries=20,
    delay=2,
    logger=logger,
)
def get_chat_response_simple(
    config: OpenaiInferenceConfig,
    prompt: str,
) -> GPTFullResponse:
    assert config.model == "gpt-3.5-turbo" or config.model == "gpt-4"
    messages = [StrictChatMessage(role=StrictMessageRole.user, content=prompt)]
    response = __get_chat_response_dict(
        config=config,
        prompt=messages,
    )
    return parse_chat_prompt_response_dict(prompt=messages, response_dict=response)


@token_rate_limiter(tokens_per_minute=int(90_000 * total_rate_sf), logger=logger)
@retry_openai_failures
@retry_openai_rate_limits
def gpt3_5_rate_limited(config: OpenaiInferenceConfig, messages: list[StrictChatMessage]) -> GPTFullResponse:
    assert "gpt-3.5-turbo" in config.model
    response_dict = __get_chat_response_dict(
        config=config,
        prompt=messages,
    )
    return parse_chat_prompt_response_dict(prompt=messages, response_dict=response_dict)


@token_rate_limiter(tokens_per_minute=int(40_000 * total_rate_sf), logger=logger)
@retry_openai_failures
@retry_openai_rate_limits
def gpt4_rate_limited(config: OpenaiInferenceConfig, messages: list[StrictChatMessage]) -> GPTFullResponse:
    assert config.model == "gpt-4"
    response_dict = __get_chat_response_dict(
        config=config,
        prompt=messages,
    )
    return parse_chat_prompt_response_dict(prompt=messages, response_dict=response_dict)


load_dotenv()
client = anthropic.Anthropic()


@retry(exceptions=(anthropic.APIError, anthropic.APIConnectionError), tries=-1, delay=0.1, logger=rate_limit_logger)
def anthropic_chat(config: OpenaiInferenceConfig, prompt: str) -> str:
    assert "claude" in config.model
    resp = client.completions.create(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model=config.model,
        max_tokens_to_sample=config.max_tokens,
        temperature=config.temperature,  # type: ignore
    )
    return resp.completion
