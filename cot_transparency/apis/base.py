from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Self, Sequence
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
    messages: list[ChatMessage]

    def __str__(self) -> str:
        out = ""
        for msg in self.messages:
            out += f"\n\n{msg.role}\n{msg.content}"
        return out

    @classmethod
    def from_prompt(cls, prompt: "Prompt") -> Self:
        return cls(messages=prompt.messages)

    def __add__(self, other: Self) -> Self:
        return Prompt(messages=self.messages + other.messages)

    def format(self) -> Any:
        """
        This should return the prompt in the format that the mode expects
        """
        raise NotImplementedError()

    # def convert_to_completion_str(self, model_type=ModelType.completion) -> str:
    #     messages = self.get_strict_messages(model_type)
    #     # Add the required empty assistant tag for Claude models if the last message does not have the assistant role
    #     if messages[-1].role == StrictMessageRole.user and model_type == ModelType.chat_with_append_assistant:
    #         messages.append(StrictChatMessage(role=StrictMessageRole.assistant, content=""))

    #     message = ""
    #     for msg in messages:
    #         match msg.role:
    #             case StrictMessageRole.user:
    #                 message += f"{anthropic.HUMAN_PROMPT} {msg.content}"
    #             case StrictMessageRole.assistant:
    #                 message += f"{anthropic.AI_PROMPT} {msg.content}"
    #             case StrictMessageRole.none:
    #                 message += f"\n\n{msg.content}"
    #             case StrictMessageRole.system:
    #                 # No need to add something infront for system messages
    #                 message += f"\n\n{msg.content}"
    #     return message

    # def convert_to_openai_chat(self) -> list[StrictChatMessage]:
    #     return self.get_strict_messages(model_type=ModelType.chat)

    # def convert_to_llama_chat(self) -> list[StrictChatMessage]:
    #     # we can do the same things as anthropic chat
    #     return self.get_strict_messages(model_type=ModelType.chat_with_append_assistant)


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
    def __call__(
        self,
        prompt: Prompt,
        config: OpenaiInferenceConfig,
    ) -> InferenceResponse:
        raise NotImplementedError()
