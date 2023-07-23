from cot_transparency.data_models.models import MessageRoles
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.extraction import extract_answer, extract_answer_non_cot
from cot_transparency.formatters.instructions import (
    COT_ASSISTANT_PROMPT,
    NON_COT_ASSISTANT_PROMPT,
    add_verbalize_instruction_to_question,
)
from cot_transparency.formatters.sycophancy import remove_role_from_messages
from cot_transparency.data_models.data.bbh import DataExampleBase
from cot_transparency.data_models.models import ChatMessages


from typing import Optional


def format_unbiased_question(question: str) -> str:
    return question


class ZeroShotCOTUnbiasedFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase) -> list[ChatMessages]:
        user_message = add_verbalize_instruction_to_question(question.get_parsed_input())
        output = [
            ChatMessages(role=MessageRoles.user, content=user_message),
            ChatMessages(role=MessageRoles.assistant_preferred, content=COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return extract_answer(response, dump_failed=False)


class ZeroShotCOTUnbiasedTameraTFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: MilesBBHRawData) -> list[ChatMessages]:
        output = [
            ChatMessages(role=MessageRoles.user, content=question.parsed_inputs),
            ChatMessages(role=MessageRoles.assistant, content=COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return extract_answer(response, dump_failed=False)


class ZeroShotUnbiasedFormatter(StageOneFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase) -> list[ChatMessages]:
        formatted_question = format_unbiased_question(question=question.get_parsed_input())
        output = [
            ChatMessages(role=MessageRoles.user, content=formatted_question),
            ChatMessages(role=MessageRoles.assistant_preferred, content=NON_COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)


class ZeroShotCOTUnbiasedNoRoleFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase) -> list[ChatMessages]:
        output = ZeroShotCOTUnbiasedFormatter.format_example(question=question)
        return remove_role_from_messages(output)

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return extract_answer(response, dump_failed=False)


class ZeroShotUnbiasedNoRoleFormatter(StageOneFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase) -> list[ChatMessages]:
        output = ZeroShotUnbiasedFormatter.format_example(question=question)
        return remove_role_from_messages(output)

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)
