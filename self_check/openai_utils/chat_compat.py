from typing import Dict, Any, List

import openai
from openai.error import RateLimitError, APIConnectionError, Timeout, APIError
from retry import retry
from slist import Slist

from self_check.openai_utils.models import GPTFullResponse, OpenaiInferenceConfig


def parse_chat_prompt_response_dict(
    response_dict: Dict[Any, Any],
    prompt: str,
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
    messages: List[Dict[str, str]],
) -> Dict[Any, Any]:
    return openai.ChatCompletion.create(  # type: ignore
        model=config.model,
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        presence_penalty=config.presence_penalty,
        frequency_penalty=config.frequency_penalty,
        top_p=1,
        n=1,
        stream=False,
        stop=[config.stop] if isinstance(config.stop, str) else config.stop,
    )


def get_chat_prompt_response_dict(
    config: OpenaiInferenceConfig,
    prompt: str,
) -> Dict[Any, Any]:
    messages = [{"role": "user", "content": prompt}]
    return __get_chat_response_dict(
        config=config,
        messages=messages,
    )


@retry(
    exceptions=(RateLimitError, APIConnectionError, Timeout, APIError),
    tries=5,
    delay=20,
)
def get_chat_prompt_full_response(
    config: OpenaiInferenceConfig,
    prompt: str,
) -> GPTFullResponse:
    assert config.model == "gpt-3.5-turbo" or config.model == "gpt-4"
    response = get_chat_prompt_response_dict(
        config=config,
        prompt=prompt,
    )
    return parse_chat_prompt_response_dict(prompt=prompt, response_dict=response)
