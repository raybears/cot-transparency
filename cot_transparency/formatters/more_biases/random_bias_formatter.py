from typing import Optional

from slist import Slist

from cot_transparency.data_models.example_base import DataExampleBase, ChoiceVariant
from cot_transparency.data_models.messages import MessageRole
from cot_transparency.data_models.messages import ChatMessage
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.extraction import extract_answer, extract_answer_non_cot
from cot_transparency.formatters.instructions import (
    add_verbalize_instruction_to_question,
    COT_ASSISTANT_PROMPT,
    NON_COT_ASSISTANT_PROMPT,
)
from cot_transparency.formatters.more_biases.random_biases.anchor_against_variants import (
    format_anchor_against_bias_question,
)
from cot_transparency.formatters.more_biases.random_biases.anchor_variants import format_anchor_bias_question


class RandomBiasedFormatter(StageOneFormatter):
    is_biased = True
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        message = format_anchor_bias_question(
            question=question.get_parsed_input(), biased_answer=question.biased_ans_variant
        )
        with_label_instruction = add_verbalize_instruction_to_question(message)
        messages = [
            ChatMessage(role=MessageRole.user, content=with_label_instruction),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT),
        ]
        return messages

    @staticmethod
    def parse_answer(
        response: str, question: Optional[DataExampleBase] = None, model: Optional[str] = None
    ) -> Optional[str]:
        return extract_answer(response, dump_failed=False)


class RandomBiasedNoCOTFormatter(StageOneFormatter):
    is_biased = True
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        message = format_anchor_bias_question(
            question=question.get_parsed_input(), biased_answer=question.biased_ans_variant
        )
        messages = [
            ChatMessage(role=MessageRole.user, content=message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=NON_COT_ASSISTANT_PROMPT),
        ]
        return messages

    @staticmethod
    def parse_answer(
        response: str, question: Optional[DataExampleBase] = None, model: Optional[str] = None
    ) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)


class RandomBiasedQuotedNoCOTFormatter(StageOneFormatter):
    is_biased = True
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        biased_answer = f"'{question.biased_ans_text}'"
        message = format_anchor_bias_question(question=question.get_parsed_input(), biased_answer=biased_answer)
        messages = [
            ChatMessage(role=MessageRole.user, content=message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=NON_COT_ASSISTANT_PROMPT),
        ]
        return messages

    @staticmethod
    def parse_answer(
        response: str, question: Optional[DataExampleBase] = None, model: Optional[str] = None
    ) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)


class RandomBiasedQuotedFormatter(StageOneFormatter):
    is_biased = True
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        biased_answer = f"'{question.biased_ans_text}'"
        message = format_anchor_bias_question(question=question.get_parsed_input(), biased_answer=biased_answer)
        with_label_instruction = add_verbalize_instruction_to_question(message)
        messages = [
            ChatMessage(role=MessageRole.user, content=with_label_instruction),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT),
        ]
        return messages

    @staticmethod
    def parse_answer(
        response: str, question: Optional[DataExampleBase] = None, model: Optional[str] = None
    ) -> Optional[str]:
        return extract_answer(response, dump_failed=False)


class RandomAgainstBiasedFormatter(StageOneFormatter):
    is_biased = True
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        # NOTE: Since we are biasing against rather than for, we need to get a possible index that is not the bias index
        bias_index = question.bias_idx
        possible_index: int = (
            Slist(range(len(question.get_options())))
            .filter(lambda x: x != bias_index)
            .shuffle(seed=question.get_parsed_input())
            .first_or_raise()
        )
        choice_variant: ChoiceVariant = question.data_format.choice_variant
        biased_against_answer = choice_variant.answers_list[possible_index]
        message = format_anchor_against_bias_question(
            question=question.get_parsed_input(), biased_against_answer=biased_against_answer
        )
        with_label_instruction = add_verbalize_instruction_to_question(message)
        messages = [
            ChatMessage(role=MessageRole.user, content=with_label_instruction),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT),
        ]
        return messages

    @staticmethod
    def parse_answer(
        response: str, question: Optional[DataExampleBase] = None, model: Optional[str] = None
    ) -> Optional[str]:
        return extract_answer(response, dump_failed=False)


class RandomAgainstBiasedNoCOTFormatter(StageOneFormatter):
    is_biased = True
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        # NOTE: Since we are biasing against rather than for, we need to get a possible index that is not the bias
        bias_index = question.bias_idx
        possible_index: int = (
            Slist(range(len(question.get_options())))
            .filter(lambda x: x != bias_index)
            .shuffle(seed=question.get_parsed_input())
            .first_or_raise()
        )
        choice_variant: ChoiceVariant = question.data_format.choice_variant
        biased_against_answer = choice_variant.answers_list[possible_index]
        message = format_anchor_against_bias_question(
            question=question.get_parsed_input(), biased_against_answer=biased_against_answer
        )
        messages = [
            ChatMessage(role=MessageRole.user, content=message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=NON_COT_ASSISTANT_PROMPT),
        ]
        return messages

    @staticmethod
    def parse_answer(
        response: str, question: Optional[DataExampleBase] = None, model: Optional[str] = None
    ) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)


class RandomAgainstBiasedQuotedNoCOTFormatter(StageOneFormatter):
    is_biased = True
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        # NOTE: Since we are biasing against rather than for, we need to get a possible index that is not the bias
        bias_ans_text = question.biased_ans_text
        options = question.get_options()
        assert bias_ans_text in options
        possible_option = (
            Slist(options)
            .filter(lambda x: x != bias_ans_text)
            .shuffle(seed=question.get_parsed_input())
            .first_or_raise()
        )

        biased_against_answer = f"'{possible_option}'"
        message = format_anchor_against_bias_question(
            question=question.get_parsed_input(), biased_against_answer=biased_against_answer
        )
        messages = [
            ChatMessage(role=MessageRole.user, content=message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=NON_COT_ASSISTANT_PROMPT),
        ]
        return messages

    @staticmethod
    def parse_answer(
        response: str, question: Optional[DataExampleBase] = None, model: Optional[str] = None
    ) -> Optional[str]:
        return extract_answer_non_cot(response, dump_failed=False)


class RandomAgainstQuotedBiasedFormatter(StageOneFormatter):
    is_biased = True
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> list[ChatMessage]:
        bias_ans_text = question.biased_ans_text
        options = question.get_options()
        assert bias_ans_text in options
        possible_option = (
            Slist(options)
            .filter(lambda x: x != bias_ans_text)
            .shuffle(seed=question.get_parsed_input())
            .first_or_raise()
        )

        biased_against_answer = f"'{possible_option}'"
        message = format_anchor_against_bias_question(
            question=question.get_parsed_input(), biased_against_answer=biased_against_answer
        )
        with_label_instruction = add_verbalize_instruction_to_question(message)
        messages = [
            ChatMessage(role=MessageRole.user, content=with_label_instruction),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT),
        ]
        return messages

    @staticmethod
    def parse_answer(
        response: str, question: Optional[DataExampleBase] = None, model: Optional[str] = None
    ) -> Optional[str]:
        return extract_answer(response, dump_failed=False)
