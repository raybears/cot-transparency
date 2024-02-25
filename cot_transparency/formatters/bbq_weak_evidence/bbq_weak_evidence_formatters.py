from typing import Optional

from cot_transparency.formatters.base_class import StageOneFormatter

from cot_transparency.formatters.extraction import (
    extract_answer,
)

from cot_transparency.formatters.instructions import (
    COT_ASSISTANT_PROMPT_TESTING,
)

from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage, MessageRole


class BBQWECOTContext1(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        message = question.get_parsed_input(context_idx=0)  # type: ignore
        output = [
            ChatMessage(role=MessageRole.user, content=message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT_TESTING),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class BBQWECOTContext2(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        message = question.get_parsed_input(context_idx=1)  # type: ignore
        output = [
            ChatMessage(role=MessageRole.user, content=message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT_TESTING),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)
