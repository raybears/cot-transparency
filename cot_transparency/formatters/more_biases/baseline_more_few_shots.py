from abc import ABC, abstractmethod
from typing import Optional, Type, final

from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.consistency_prompting.few_shots import consistency_few_shot
from cot_transparency.formatters.extraction import extract_answer
from cot_transparency.formatters.instructions import add_verbalize_instruction_to_question, COT_ASSISTANT_PROMPT


class BaselineMoreCOTs(StageOneFormatter):
    is_biased = True
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase) -> list[ChatMessage]:
        with_instruction = (
            consistency_few_shot.convert_to_completion_str()
            + "\n"
            + add_verbalize_instruction_to_question(question.get_parsed_input())
        )
        output = [
            ChatMessage(role=MessageRole.user, content=with_instruction),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return extract_answer(response, dump_failed=False)


class Intervention(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def hook(self, question: DataExampleBase, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Define a hook that can be used to intervene in the formatting process."""
        pass

    @final  # Please don't override this unless you know what you are doing.
    def intervene(self, formatter: Type[StageOneFormatter]) -> Type[StageOneFormatter]:
        intervention_self = self

        class NewFormatter(formatter):
            @staticmethod
            def format_example(question: DataExampleBase) -> list[ChatMessage]:
                messages = formatter.format_example(question)
                new_messages = self.hook(question, messages)
                return new_messages

            @staticmethod
            def parse_answer(response: str) -> Optional[str]:
                return formatter.parse_answer(response)

            @classmethod
            def name(cls) -> str:
                return f"{intervention_self.name}_{formatter.name()}"

        return NewFormatter
