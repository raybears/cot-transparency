from abc import ABC, abstractmethod
from typing import Optional, Self, Sequence, Set, Type

from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage
from cot_transparency.formatters.base_class import StageOneFormatter


class Intervention(ABC):
    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    def formatted_name(cls) -> str:
        return cls.name()

    @classmethod
    @abstractmethod
    def intervene(
        cls,
        question: DataExampleBase,
        formatter: Type[StageOneFormatter],
        model: Optional[str] = None,
    ) -> Sequence[ChatMessage]:
        cls_name = cls.__name__
        raise NotImplementedError(f"Intervention {cls_name} has not implemented intervene.")

    @classmethod
    def all_interventions(cls) -> dict[str, Type[Self]]:
        return {s.name(): s for s in cls.all_subclasses()}

    @classmethod
    def all_subclasses(cls) -> Set[Type[Self]]:
        # get all subclasses recursively
        subclasses: set[Type[Intervention]] = set(cls.__subclasses__())
        return subclasses.union([s for c in subclasses for s in c.all_subclasses()])
