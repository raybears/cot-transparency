from typing import Optional

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
