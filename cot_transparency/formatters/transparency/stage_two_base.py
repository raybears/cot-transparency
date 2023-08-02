from string import ascii_uppercase
from cot_transparency.data_models.example_base import MultipleChoiceAnswer
from cot_transparency.formatters.base_class import PromptFormatter
from typing import Optional, Type, Self


def parse_stage_two_output(response: str, allow_failure: bool = True) -> Optional[MultipleChoiceAnswer]:
    out: Optional[MultipleChoiceAnswer] = None
    response = response.strip()
    print(response)
    if len(response) >= 1:
        if response[0].upper() in ascii_uppercase:
            out = response[0]  # type: ignore
            if len(response) > 1:
                if response[1] != ")":
                    out = None
        if out is None and allow_failure:
            print(f"Did not find a valid answer in reponse '{response}', but allow_failure is set to True")
            return "NOT_FOUND"
    return out


class StageTwoFormatter(PromptFormatter):
    @staticmethod
    def parse_answer(response: str) -> Optional[MultipleChoiceAnswer]:
        return parse_stage_two_output(response)

    @classmethod
    def all_formatters(cls) -> dict[str, Type[Self]]:
        return {s.name(): s for s in cls.all_subclasses()}
