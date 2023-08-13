from abc import ABC, abstractmethod
from typing import Type

from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ChatMessage
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
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        # Please don't override this method unless you know what you are doing. You should only need to override `hook`.
        messages = formatter.format_example(question)
        new_messages = cls.hook(question, messages)
        return new_messages
