from cot_transparency.data_models.bbh import MilesBBHRawData
from cot_transparency.data_models.models import ChatMessages


from typing import Optional, Self, Set, Type


class PromptFormatter:
    is_biased: Optional[bool] = None
    is_cot: bool = True

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
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
    def all_formatters(cls) -> dict[str, Type[Self]]:
        return {s.name(): s for s in cls.all_subclasses()}


class StageOneFormatter(PromptFormatter):
    @staticmethod
    def format_example(question: MilesBBHRawData) -> list[ChatMessages]:
        raise NotImplementedError

    @classmethod
    def all_formatters(cls) -> dict[str, Type[Self]]:
        return super().all_formatters()  # type: ignore
