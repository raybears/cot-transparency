from cot_transparency.miles_models import MilesBBHRawData
from cot_transparency.openai_utils.models import ChatMessages


from typing import Optional, Set, Type


class PromptFormatter:
    is_biased: Optional[bool] = None
    is_cot: bool = True

    @staticmethod
    def format_example(question: MilesBBHRawData) -> list[ChatMessages]:
        raise NotImplementedError

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        raise NotImplementedError

    @classmethod
    def name(cls) -> str:
        return cls.__name__

    @classmethod
    def all_subclasses(cls) -> Set[Type["PromptFormatter"]]:
        # get all subclasses recursively
        subclasses: set[Type[PromptFormatter]] = set(cls.__subclasses__())
        return subclasses.union([s for c in subclasses for s in c.all_subclasses()])

    @classmethod
    def all_formatters(cls) -> dict[str, Type["PromptFormatter"]]:
        return {s.name(): s for s in cls.all_subclasses()}
