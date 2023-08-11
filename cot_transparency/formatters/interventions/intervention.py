from abc import ABC, abstractmethod
from typing import final, Type

from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter


class Intervention(ABC):
    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    @abstractmethod
    def hook(cls, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Define a hook that can be used to intervene in the formatting process."""
        pass

    @classmethod
    @final  # Please don't override this method unless you know what you are doing
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        messages = formatter.format_example(question)
        new_messages = cls.hook(question, messages)
        return new_messages


def prepend_to_front_first_user_message(messages: list[ChatMessage], prepend: str) -> list[ChatMessage]:
    """Prepend a string to the first user message."""
    new_messages = []
    for m in messages:
        if m.role == MessageRole.user:
            new_messages.append(ChatMessage(role=MessageRole.user, content=prepend + m.content))
        else:
            new_messages.append(m)
    return new_messages
