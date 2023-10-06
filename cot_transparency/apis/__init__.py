from typing import Type

from cot_transparency.apis.anthropic import AnthropicCaller
from cot_transparency.apis.base import InferenceResponse, ModelCaller, Prompt, ModelType
from cot_transparency.apis.openai import OpenAIChatCaller, OpenAICompletionCaller
from cot_transparency.apis.openai.set_key import set_opeani_org_from_env_rand
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import ChatMessage


__all__ = [
    "ModelType",
]


CALLER_MAP: dict[str, Type[ModelCaller]] = {
    "gpt-3.5-turbo": OpenAIChatCaller,
    "gpt-4": OpenAIChatCaller,
    "text-davinci-003": OpenAICompletionCaller,
    "code-davinci-002": OpenAICompletionCaller,
    "text-davinci-002": OpenAICompletionCaller,
    "davinci": OpenAICompletionCaller,
    "claude-v1": AnthropicCaller,
    "claude-2": AnthropicCaller,
    "claude-instant-1": AnthropicCaller,
    "gpt-3.5-turbo-16k": OpenAIChatCaller,
    "gpt-4-32k": OpenAIChatCaller,
}

CALLER_STORE: dict[str, ModelCaller] = {}


def call_model_api(messages: list[ChatMessage], config: OpenaiInferenceConfig) -> InferenceResponse:
    if not config.is_openai_finetuned():
        # we should only switch between orgs if we are not finetuned
        # TODO: just pass the org explicitly to the api?
        set_opeani_org_from_env_rand()

    prompt = Prompt(messages=messages)
    model_name = config.model

    caller: ModelCaller
    if model_name in CALLER_STORE:
        caller = CALLER_STORE[model_name]
    else:
        caller = CALLER_MAP[model_name]()
        CALLER_STORE[model_name] = caller

    return caller(prompt, config)
