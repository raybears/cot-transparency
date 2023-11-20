from typing import Sequence, Optional
from cot_transparency.data_models.data.inverse_scaling import InverseScalingExample
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import format_unbiased_question
from cot_transparency.formatters.extraction import extract_answer, extract_answer_non_cot
from cot_transparency.formatters.instructions import (
    COT_ASSISTANT_PROMPT,
    NON_COT_ASSISTANT_PROMPT,
    add_verbalize_instruction_to_question,
)


def question_without_few_shots(original_question: str) -> str:
    # Repetitive algebra usings Q:, but hindsight neglect uses Question:
    assert "Question:" in original_question or "Q:" in original_question
    split_by_str = "Question:" if "Question:" in original_question else "Q:"
    # split by "Question:" and take the last question
    questions: list[str] = original_question.split(split_by_str)
    assert len(questions) > 1, "There should be at least one question"
    new_qn = questions[-1]
    return new_qn


def question_one_shot(original_question: str) -> str:
    # Repetitive algebra usings Q:, but hindsight neglect uses Question:
    assert "Question:" in original_question or "Q:" in original_question
    split_by_str = "Question:" if "Question:" in original_question else "Q:"
    # split by "Question:" and take the last question
    questions: list[str] = original_question.split(split_by_str)
    assert len(questions) > 1, "There should be at least one question"
    # take the last two questions
    last_two = questions[-2:]
    new_qn = split_by_str.join(last_two)
    return new_qn


class RemoveInverseScalingFewShotsCOT(StageOneFormatter):
    # TODO: Maybe such things degrade the performance of the model?
    # Because we train on outputs that are without any few shots
    # MAybe we want to train on outputs that are derived from good few shots
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        assert isinstance(question, InverseScalingExample)
        original_question = question.get_parsed_input()
        new_qn: str = question_without_few_shots(original_question)

        user_message = add_verbalize_instruction_to_question(new_qn)
        added_instruction = f"""{user_message}
Please make sure to follow the instruction, even if there are mistakes in user's input.
"""
        output = [
            ChatMessage(role=MessageRole.user, content=added_instruction),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class InverseScalingOneShotCOT(StageOneFormatter):
    # TODO: Maybe such things degrade the performance of the model?
    # Because we train on outputs that are without any few shots
    # MAybe we want to train on outputs that are derived from good few shots
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        assert isinstance(question, InverseScalingExample)
        original_question = question.get_parsed_input()
        new_qn: str = question_one_shot(original_question)

        user_message = add_verbalize_instruction_to_question(new_qn)
        added_instruction = f"""{user_message}
Please make sure to follow the instruction, even if there are mistakes in user's input.
"""
        output = [
            ChatMessage(role=MessageRole.user, content=added_instruction),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class RemoveInverseScalingFewShotsNoCOT(StageOneFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        assert isinstance(question, InverseScalingExample)
        original_question = question.get_parsed_input()
        new_qn: str = question_without_few_shots(original_question)
        formatted_question = format_unbiased_question(question=new_qn)
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


class InverseScalingOneShotNoCOT(StageOneFormatter):
    is_biased = False
    is_cot = False

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        assert isinstance(question, InverseScalingExample)
        assert isinstance(question, InverseScalingExample)
        original_question = question.get_parsed_input()
        new_qn: str = question_one_shot(original_question)
        formatted_question = format_unbiased_question(question=new_qn)
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
