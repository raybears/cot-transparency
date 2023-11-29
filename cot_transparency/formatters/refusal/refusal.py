
from typing import Optional, Sequence

from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter


class RefusalFormatter(StageOneFormatter):
    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        formatted_question = question.get_parsed_input()
        output = [
            ChatMessage(role=MessageRole.user, content=formatted_question),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return response