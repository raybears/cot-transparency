from cot_transparency.formatters.base_class import PromptFormatter
from typing import Optional, Type, Self

from cot_transparency.formatters.extraction import extract_answer_non_cot


class StageTwoFormatter(PromptFormatter):
    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return extract_answer_non_cot(response)

    @classmethod
    def all_formatters(cls) -> dict[str, Type[Self]]:
        return {s.name(): s for s in cls.all_subclasses()}
