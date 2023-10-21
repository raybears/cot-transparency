from abc import ABC, abstractmethod
from enum import Enum
from typing import Self, Sequence
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import ChatMessage


from pydantic import BaseModel


class ModelType(str, Enum):
    # moves the "assistant preferred" message to the previous (user) message
    chat = "chat"
    completion = "completion"
    chat_with_append_assistant = "anthropic"

    @staticmethod
    def from_model_name(name: str) -> "ModelType":
        if "claude" in name:
            return ModelType.chat_with_append_assistant
        elif "gpt-3.5-turbo" in name or name == "gpt-4" or name == "gpt-4-32k":
            return ModelType.chat
        else:
            return ModelType.completion


class Prompt(BaseModel):
    """
    You shouldn't really need to use this class directly, unless you are debugging or testing
    the specific format that is being sent to an api call. ModelCallers take a
    list of ChatMessages but use subtypes of this class to format the messages into the
    format that the api expects
    """

    def __str__(self) -> str:
        out = ""
        for msg in self.messages:
            out += f"\n\n{msg.role}\n{msg.content}"
        return out

    messages: list[ChatMessage]

    @classmethod
    def from_prompt(cls, prompt: "Prompt") -> Self:
        return cls(messages=prompt.messages)

    def __add__(self, other: Self) -> Self:
        return Prompt(messages=self.messages + other.messages)


class InferenceResponse(BaseModel):
    raw_responses: Sequence[str]

    @property
    def has_multiple_responses(self) -> bool:
        return len(self.raw_responses) > 1

    @property
    def single_response(self) -> str:
        if self.has_multiple_responses:
            raise ValueError("This response has multiple responses")
        else:
            return self.raw_responses[0]


class ModelCaller(ABC):
    @abstractmethod
    def call(
        self,
        messages: list[ChatMessage],
        config: OpenaiInferenceConfig,
    ) -> InferenceResponse:
        raise NotImplementedError()
