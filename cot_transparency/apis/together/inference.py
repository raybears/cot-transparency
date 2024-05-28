import logging
from collections.abc import Sequence
import os

from together import Together
from together.error import TogetherException, APIConnectionError, ServiceUnavailableError, RateLimitError

from dotenv import load_dotenv
from retry import retry

from cot_transparency.apis.base import InferenceResponse, ModelCaller, Prompt
from cot_transparency.apis.util import (
    convert_assistant_if_completion_to_assistant,
    messages_has_none_role,
)
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import (
    ChatMessage,
    StrictChatMessage,
    StrictMessageRole,
)
from cot_transparency.util import setup_logger

logger = setup_logger(__name__, logging.INFO)


together_ai_model_mapper = {
    "llama-3": "meta-llama/Llama-3-8b-chat-hf",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
}


class TogetherAIPrompt(Prompt):
    def __str__(self) -> str:
        return str(self.format())

    def format(self) -> list[dict[str, str]]:
        if messages_has_none_role(self.messages):
            raise ValueError(
                f"Together AI chat messages cannot have a None role. Got {self.messages}"
            )  # ruff: noqa: E501
        messages = convert_assistant_if_completion_to_assistant(self.messages)

        formatted_messages: list[dict[str, str]] = []
        for msg in messages:
            match msg.role:
                case StrictMessageRole.user:
                    formatted_messages.append({"role": "user", "content": msg.content})
                case StrictMessageRole.assistant:
                    formatted_messages.append({"role": "assistant", "content": msg.content})
                case StrictMessageRole.none:
                    formatted_messages.append({"role": "none", "content": msg.content})
                case StrictMessageRole.system:
                    formatted_messages.append({"role": "system", "content": msg.content})
        return formatted_messages


class TogetherAICaller(ModelCaller):
    def __init__(self):
        load_dotenv()
        self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    @retry(
        exceptions=(APIConnectionError, ServiceUnavailableError),
        tries=10,
        delay=5,
        logger=logger,
    )
    @retry(exceptions=(RateLimitError), tries=-1, delay=1, logger=logger)
    def call(
        self,
        messages: Sequence[ChatMessage],
        config: OpenaiInferenceConfig,
        try_number: int = 1,
    ) -> InferenceResponse:
        assert (
            config.model in together_ai_model_mapper.keys()
        ), f"Invalid model name. Available models: {together_ai_model_mapper.keys()}"  # ruff: noqa: E501
        formatted_messages = TogetherAIPrompt(messages=messages).format()

        resp = self.client.chat.completions.create(
            model=together_ai_model_mapper[config.model],
            messages=formatted_messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stop=["<|eot_id|>", "<|im_end|>"], # LLaMA-3 and Mistral-7B stop tokens
        )  # type: ignore

        inf_response = InferenceResponse(raw_responses=[resp.choices[0].message.content])  # type: ignore
        return inf_response
