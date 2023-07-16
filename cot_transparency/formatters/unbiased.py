from cot_transparency.formatters.base_class import PromptFormatter
from cot_transparency.formatters.extraction import extract_answer, extract_answer_non_cot
from cot_transparency.formatters.instructions import (
    COT_ASSISTANT_PROMPT,
    NON_COT_ASSISTANT_PROMPT,
    add_verbalize_instruction_to_question,
)
from cot_transparency.formatters.sycophancy import remove_role_from_messages
from cot_transparency.miles_models import MilesBBHRawData
from cot_transparency.openai_utils.models import ChatMessages, OpenaiRoles


from typing import Optional


def format_unbiased_question(question: str) -> str:
    return question


class ZeroShotCOTUnbiasedFormatter(PromptFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: MilesBBHRawData) -> list[ChatMessages]:
        user_message = add_verbalize_instruction_to_question(question.parsed_inputs)
        output = [
            ChatMessages(role=OpenaiRoles.user, content=user_message),
            ChatMessages(role=OpenaiRoles.assistant_preferred, content=COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return extract_answer(response, dump_failed=False)


class ZeroShotUnbiasedFormatter(PromptFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: MilesBBHRawData) -> list[ChatMessages]:
        formatted_question = format_unbiased_question(question=question.parsed_inputs)
        output = [
            ChatMessages(role=OpenaiRoles.user, content=formatted_question),
            ChatMessages(role=OpenaiRoles.assistant_preferred, content=NON_COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)


class ZeroShotCOTUnbiasedNoRoleFormatter(PromptFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: MilesBBHRawData) -> list[ChatMessages]:
        output = ZeroShotCOTUnbiasedFormatter.format_example(question=question)
        return remove_role_from_messages(output)

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return extract_answer(response, dump_failed=False)


class ZeroShotUnbiasedNoRoleFormatter(PromptFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: MilesBBHRawData) -> list[ChatMessages]:
        output = ZeroShotUnbiasedFormatter.format_example(question=question)
        return remove_role_from_messages(output)

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)
