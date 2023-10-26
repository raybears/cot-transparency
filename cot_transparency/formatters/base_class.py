from abc import ABC, abstractmethod
from typing import Mapping, Optional, Self, Sequence, Set, Type

from cot_transparency.data_models.data.bbh import DataExampleBase
from cot_transparency.data_models.example_base import DataFormatSpec
from cot_transparency.data_models.messages import ChatMessage


class PromptFormatter(ABC):
    is_biased: Optional[bool] = None
    is_cot: bool = True
    has_none_of_the_above: bool = False

    @staticmethod
    @abstractmethod
    def parse_answer(
        response: str, question: DataExampleBase, model: Optional[str] = None
    ) -> Optional[str]:
        raise NotImplementedError

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    def all_subclasses(cls) -> Set[Type[Self]]:
        # get all subclasses recursively
        subclasses: set[Type[PromptFormatter]] = set(cls.__subclasses__())
        return subclasses.union([s for c in subclasses for s in c.all_subclasses()])

    @classmethod
    def all_formatters(cls) -> Mapping[str, Type[Self]]:
        # return all subclasses thare are not abstract
        return {s.name(): s for s in cls.all_subclasses() if not s.__abstractmethods__}


class StageOneFormatter(PromptFormatter, ABC):
    @classmethod
    def get_data_format_spec(cls) -> DataFormatSpec:
        # return the default one
        return DataFormatSpec()

    @staticmethod
    @abstractmethod
    def format_example(
        question: DataExampleBase, model: Optional[str] = None
    ) -> Sequence[ChatMessage]:
        raise NotImplementedError

    @classmethod
    def all_formatters(cls) -> Mapping[str, Type[Self]]:
        return super().all_formatters()  # type: ignore
