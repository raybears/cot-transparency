from typing import Dict, Any, List, Union

import openai
from openai import APIError
from openai.error import RateLimitError, APIConnectionError, Timeout
from retry import retry
from slist import Slist

from self_check.openai_utils.chat_compat import ChatMessages
from self_check.openai_utils.models import (
    OpenaiInferenceConfig,
    TokenProba,
    TokenInfo,
    GPTFullResponse,
)


def parse_gpt_response(
    prompt: str, response_dict: Dict[Any, Any], end_tokens: set[str]
) -> GPTFullResponse:
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


@retry(exceptions=(RateLimitError, APIConnectionError, Timeout), tries=5, delay=20)
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
