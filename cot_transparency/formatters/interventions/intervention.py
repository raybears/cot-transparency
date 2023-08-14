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
    def intervene(cls, question: DataExampleBase, formatter: Type[StageOneFormatter]) -> list[ChatMessage]:
        raise NotImplementedError
