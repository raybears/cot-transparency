from typing import Type

from cot_transparency.apis.anthropic import AnthropicCaller
from cot_transparency.apis.base import InferenceResponse, ModelCaller, Prompt, ModelType
from cot_transparency.apis.openai import OpenAIChatCaller, OpenAICompletionCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import ChatMessage


__all__ = [
    "ModelType",
]

CALLER_STORE: dict[str, ModelCaller] = {}


def get_caller(model_name: str) -> Type[ModelCaller]:
    if "davinci" in model_name:
        return OpenAICompletionCaller
    elif "claude" in model_name:
        return AnthropicCaller
    elif "gpt" in model_name:
        return OpenAIChatCaller
    else:
        raise ValueError(f"Unknown model name {model_name}")


def call_model_api(messages: list[ChatMessage], config: OpenaiInferenceConfig) -> InferenceResponse:
    prompt = Prompt(messages=messages)
    model_name = config.model

    caller: ModelCaller
    if model_name in CALLER_STORE:
        caller = get_caller(model_name)()
    else:
        caller = get_caller(model_name)()
        CALLER_STORE[model_name] = caller

    return caller(prompt, config)
