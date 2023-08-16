import itertools
from cot_transparency.data_models.example_base import (
    ChoiceVariant,
    DataFormatSpec,
    JoinStr,
    MultipleChoiceAnswer,
    QuestionPrefix,
)
from cot_transparency.data_models.models import MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.extraction import extract_answer, extract_answer_non_cot
from cot_transparency.formatters.instructions import (
    COT_ASSISTANT_PROMPT,
    NON_COT_ASSISTANT_PROMPT,
    add_verbalize_instruction_to_question,
)
from cot_transparency.formatters.core.sycophancy import remove_role_from_messages
from cot_transparency.data_models.data.bbh import DataExampleBase
from cot_transparency.data_models.models import ChatMessage


from typing import Optional

from cot_transparency.formatters.util import get_few_shot_prompts


def format_unbiased_question(question: str) -> str:
    return question


class ZeroShotCOTUnbiasedFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        user_message = add_verbalize_instruction_to_question(question.get_parsed_input())
        output = [
            ChatMessage(role=MessageRole.user, content=user_message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, dump_failed=False)


class FewShotCOTUnbiasedNoRoleFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
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
        msgs.append(ChatMessage(role=MessageRole.none, content=COT_ASSISTANT_PROMPT).add_answer_prefix())
        return msgs

    @staticmethod
    def parse_answer(response: str, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, dump_failed=False)


class FewShotUnbiasedNoRoleFormatter(StageOneFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
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
    def parse_answer(response: str, model: Optional[str] = None) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)


class ZeroShotUnbiasedFormatter(StageOneFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        formatted_question = format_unbiased_question(question=question.get_parsed_input())
        output = [
            ChatMessage(role=MessageRole.user, content=formatted_question),
            ChatMessage(role=MessageRole.assistant_if_completion, content=NON_COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, model: Optional[str] = None) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)


def prompt_sensitivity_factory(data_format_spec: DataFormatSpec):
    class ZeroShotPromptSenFormatter(ZeroShotUnbiasedFormatter):
        is_biased = False
        is_cot = False

        @staticmethod
        def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
            question = question.to_variant(data_format_spec)
            return ZeroShotUnbiasedFormatter.format_example(question=question, model=model)

        @staticmethod
        def parse_answer(response: str, model: Optional[str] = None) -> Optional[str]:
            return extract_answer_non_cot(response, dump_failed=False, input_format=data_format_spec.choice_variant)

        @classmethod
        def name(cls) -> str:
            return f"{cls.__name__}_{data_format_spec.choice_variant.name}_{data_format_spec.question_variant.name}_{data_format_spec.join_variant.name}"

    return ZeroShotPromptSenFormatter


def register_prompt_sensitivity_formatters():
    choice_variants = [i for i in ChoiceVariant]
    question_prefix = [i for i in QuestionPrefix]
    join_str = [i for i in JoinStr]

    combinations = itertools.product(choice_variants, question_prefix, join_str)
    formatters = [
        prompt_sensitivity_factory(DataFormatSpec(choice_variant=c, question_variant=q, join_variant=j))
        for c, q, j in combinations
    ]
    return formatters


class ZeroShotCOTUnbiasedNoRoleFormatter(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        output = ZeroShotCOTUnbiasedFormatter.format_example(question=question)
        return remove_role_from_messages(output)

    @staticmethod
    def parse_answer(response: str, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, dump_failed=False)


class ZeroShotUnbiasedNoRoleFormatter(StageOneFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        output = ZeroShotUnbiasedFormatter.format_example(question=question)
        return remove_role_from_messages(output)

    @staticmethod
    def parse_answer(response: str, model: Optional[str] = None) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)
