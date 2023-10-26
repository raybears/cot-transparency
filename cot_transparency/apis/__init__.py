from typing import Sequence, Type

from cot_transparency.apis.anthropic import AnthropicCaller
from cot_transparency.apis.base import InferenceResponse, ModelCaller, ModelType
from cot_transparency.apis.openai import OpenAIChatCaller, OpenAICompletionCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import ChatMessage

__all__ = [
    "ModelType",
]

CALLER_STORE: dict[str, ModelCaller] = {}


def get_caller_class(model_name: str) -> Type[ModelCaller]:
    if "davinci" in model_name:
        return OpenAICompletionCaller
    elif "claude" in model_name:
        return AnthropicCaller
    elif "gpt" in model_name:
        return OpenAIChatCaller
    else:
        raise ValueError(f"Unknown model name {model_name}")


class UniversalCaller(ModelCaller):
    # A caller that can call (mostly) any model
    # This exists so that James can easily attach a cache to a single caller with with_file_cache
    # He uses a single caller in his script because sometimes its Claude, sometimes its GPT-3.5
    def call(
        self,
        messages: Sequence[ChatMessage],
        config: OpenaiInferenceConfig,
    ) -> InferenceResponse:
        return call_model_api(messages, config)


def get_caller(model_name: str) -> ModelCaller:
    if model_name in CALLER_STORE:
        return CALLER_STORE[model_name]
    else:
        caller = get_caller_class(model_name)()
        CALLER_STORE[model_name] = caller
        return caller


def call_model_api(
    messages: Sequence[ChatMessage], config: OpenaiInferenceConfig
) -> InferenceResponse:
    model_name = config.model
    caller = get_caller(model_name)
    return caller.call(messages, config)
