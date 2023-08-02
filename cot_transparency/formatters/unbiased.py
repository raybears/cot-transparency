import random
from string import ascii_uppercase
from cot_transparency.data_models.models import MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.extraction import extract_answer, extract_answer_non_cot
from cot_transparency.formatters.instructions import (
    COT_ASSISTANT_PROMPT,
    NON_COT_ASSISTANT_PROMPT,
    add_verbalize_instruction_to_question,
)
from cot_transparency.formatters.sycophancy import remove_role_from_messages
from cot_transparency.data_models.data.bbh import DataExampleBase
from cot_transparency.data_models.models import ChatMessage


from typing import Optional
from cot_transparency.formatters.transparency.early_answering import GIVEN_ALL_OF_THE_ABOVE

from cot_transparency.formatters.util import load_few_shots


def format_unbiased_question(question: str) -> str:
    return question


class ZeroShotCOTUnbiasedFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase) -> list[ChatMessage]:
        user_message = add_verbalize_instruction_to_question(question.get_parsed_input())
        output = [
            ChatMessage(role=MessageRole.user, content=user_message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return extract_answer(response, dump_failed=False)


class ZeroShotCOTUnbiasedTameraTFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase) -> list[ChatMessage]:
        output = [
            ChatMessage(role=MessageRole.user, content=question.get_parsed_input()),
            ChatMessage(role=MessageRole.assistant, content=COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return "Extraction not implemented for this formatter as expected to run stage_two"


class FewShotCOTUnbiasedTameraTFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase) -> list[ChatMessage]:
        few_shots: list[tuple[ChatMessage, ChatMessage]] = load_few_shots("./data/ethan_few_shot.txt")
        more_few_shots: list[tuple[ChatMessage, ChatMessage]] = load_few_shots("./data/gpt4_generated_few_shot.txt")

        few_shots = few_shots + more_few_shots
        random.Random(question.get_parsed_input()).shuffle(few_shots)

        # we want to add two more messages
        few_shot_examples: list[ChatMessage] = []
        for msg in few_shots:
            few_shot_examples.append(msg[0])
            few_shot_examples.append(msg[1])

            given_all_of_the_above = ChatMessage(role=MessageRole.user, content=GIVEN_ALL_OF_THE_ABOVE)
            few_shot_examples.append(given_all_of_the_above)

            # Few shots all have Therefore, the answer is (X). So answer is always the last 3 characters
            answer_from_few_shot = msg[1].content[-3]
            assert answer_from_few_shot in ascii_uppercase
            single_best_answer = ChatMessage(
                role=MessageRole.assistant, content=f"The single, most likely answer is: ({answer_from_few_shot})."
            )
            few_shot_examples.append(single_best_answer)

        output = [
            ChatMessage(role=MessageRole.user, content=question.get_parsed_input()),
            ChatMessage(role=MessageRole.assistant, content=COT_ASSISTANT_PROMPT),
        ]
        return few_shot_examples + output

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return "Extraction not implemented for this formatter as expected to run stage_two"


class ZeroShotUnbiasedFormatter(StageOneFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase) -> list[ChatMessage]:
        formatted_question = format_unbiased_question(question=question.get_parsed_input())
        output = [
            ChatMessage(role=MessageRole.user, content=formatted_question),
            ChatMessage(role=MessageRole.assistant_if_completion, content=NON_COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)


class ZeroShotCOTUnbiasedNoRoleFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase) -> list[ChatMessage]:
        output = ZeroShotCOTUnbiasedFormatter.format_example(question=question)
        return remove_role_from_messages(output)

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return extract_answer(response, dump_failed=False)


class ZeroShotUnbiasedNoRoleFormatter(StageOneFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase) -> list[ChatMessage]:
        output = ZeroShotUnbiasedFormatter.format_example(question=question)
        return remove_role_from_messages(output)

    @staticmethod
    def parse_answer(response: str) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)
