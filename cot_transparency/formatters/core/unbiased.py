from typing import Optional, Sequence

from cot_transparency.data_models.data.bbh import DataExampleBase
from cot_transparency.data_models.example_base import MultipleChoiceAnswer, DataFormatSpec, RandomizeOption
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.sycophancy import remove_role_from_messages
from cot_transparency.formatters.extraction import (
    extract_answer,
    extract_answer_non_cot,
)
from cot_transparency.formatters.instructions import (
    COT_ASSISTANT_PROMPT_TESTING,
    NON_COT_ASSISTANT_PROMPT,
    add_verbalize_instruction_to_question,
)
from cot_transparency.formatters.util import get_few_shot_prompts


def format_unbiased_question(question: str) -> str:
    return question


class ZeroShotCOTUnbiasedFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        user_message = add_verbalize_instruction_to_question(question.get_parsed_input())
        output = [
            ChatMessage(role=MessageRole.user, content=user_message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT_TESTING),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class ZeroShotCOTUnbiasedShuffleFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        new_qn = question.to_variant(DataFormatSpec(randomize_order=RandomizeOption.YES))
        user_message = add_verbalize_instruction_to_question(new_qn.get_parsed_input())
        output = [
            ChatMessage(role=MessageRole.user, content=user_message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT_TESTING),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)

    @classmethod
    def get_data_format_spec(cls) -> DataFormatSpec:
        return DataFormatSpec(randomize_order=RandomizeOption.YES)


class ZeroShotCOTUnbiasedWithNoneFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        user_message = add_verbalize_instruction_to_question(question.get_parsed_input(include_none_of_the_above=True))
        output = [
            ChatMessage(role=MessageRole.user, content=user_message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT_TESTING),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class FewShotCOTUnbiasedNoRoleFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        few_shots: list[tuple[ChatMessage, ChatMessage, MultipleChoiceAnswer]] = get_few_shot_prompts(
            question.get_parsed_input(),
            format_of_final="Therefore, the best answer is: (",
            n=10,
        )
        msgs = []
        for q, a, _ in few_shots:
            q_str = add_verbalize_instruction_to_question(q.content)
            msgs.append(ChatMessage(role=MessageRole.none, content=q_str).add_question_prefix())
            msgs.append(a.add_answer_prefix())

        msgs.append(ChatMessage(role=MessageRole.none, content=question.get_parsed_input()).add_question_prefix())
        msgs.append(ChatMessage(role=MessageRole.none, content=COT_ASSISTANT_PROMPT_TESTING).add_answer_prefix())
        return msgs

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class FewShotUnbiasedNoRoleFormatter(StageOneFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        few_shots: list[tuple[ChatMessage, ChatMessage, MultipleChoiceAnswer]] = get_few_shot_prompts(
            question.get_parsed_input(),
            n=10,
        )
        msgs = []
        for q, _, letter in few_shots:
            msgs.append(q.remove_role().add_question_prefix())
            # need to remove the cot from the answer, the answer is after Therefore the answer is (X).
            answer = NON_COT_ASSISTANT_PROMPT + letter + ")."
            msgs.append(ChatMessage(role=MessageRole.none, content=answer).add_answer_prefix())

        msgs.append(ChatMessage(role=MessageRole.none, content=question.get_parsed_input()).add_question_prefix())
        msgs.append(ChatMessage(role=MessageRole.none, content=NON_COT_ASSISTANT_PROMPT).add_answer_prefix())
        return msgs

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)


class ZeroShotUnbiasedFormatter(StageOneFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        formatted_question = format_unbiased_question(question=question.get_parsed_input())
        output = [
            ChatMessage(role=MessageRole.user, content=formatted_question),
            ChatMessage(
                role=MessageRole.assistant_if_completion,
                content=NON_COT_ASSISTANT_PROMPT,
            ),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)


class ZeroShotUnbiasedFinalQuestionFormatter(StageOneFormatter):
    # Useful to tell the model to answer the final question only
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        formatted_question = (
            format_unbiased_question(question=question.get_parsed_input()) + "\nPlease answer the final question only"
        )
        output = [
            ChatMessage(role=MessageRole.user, content=formatted_question),
            ChatMessage(
                role=MessageRole.assistant_if_completion,
                content=NON_COT_ASSISTANT_PROMPT,
            ),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)


class ZeroShotUnbiasedOnlyChooseValidOptionsFormatter(StageOneFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        formatted_question = format_unbiased_question(question=question.get_parsed_input())
        output = [
            ChatMessage(
                role=MessageRole.user, content=formatted_question + "\nPlease only choose from the available options"
            ),
            ChatMessage(
                role=MessageRole.assistant_if_completion,
                content=NON_COT_ASSISTANT_PROMPT,
            ),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)


class ZeroShotUnbiasedShuffledFormatter(StageOneFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        new_qn = question.to_variant(DataFormatSpec(randomize_order=RandomizeOption.YES))
        formatted_question = format_unbiased_question(question=new_qn.get_parsed_input())
        output = [
            ChatMessage(role=MessageRole.user, content=formatted_question),
            ChatMessage(
                role=MessageRole.assistant_if_completion,
                content=NON_COT_ASSISTANT_PROMPT,
            ),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)

    @classmethod
    def get_data_format_spec(cls) -> DataFormatSpec:
        return DataFormatSpec(randomize_order=RandomizeOption.YES)


class ZeroShotUnbiasedWithNoneFormatter(StageOneFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        formatted_question = format_unbiased_question(
            question=question.get_parsed_input(include_none_of_the_above=True)
        )
        output = [
            ChatMessage(role=MessageRole.user, content=formatted_question),
            ChatMessage(
                role=MessageRole.assistant_if_completion,
                content=NON_COT_ASSISTANT_PROMPT,
            ),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)


class ZeroShotCOTUnbiasedNoRoleFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        output = ZeroShotCOTUnbiasedFormatter.format_example(question=question)
        return remove_role_from_messages(output)

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class ZeroShotUnbiasedNoRoleFormatter(StageOneFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        output = ZeroShotUnbiasedFormatter.format_example(question=question)
        return remove_role_from_messages(output)

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)
