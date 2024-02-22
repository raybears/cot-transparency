from typing import Sequence, Optional

from slist import Slist
from cot_transparency.data_models.data.inverse_scaling import InverseScalingExample
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.unbiased import format_unbiased_question
from cot_transparency.formatters.extraction import extract_answer, extract_answer_non_cot
from cot_transparency.formatters.instructions import (
    COT_ASSISTANT_PROMPT_TESTING,
    NON_COT_ASSISTANT_PROMPT,
    VERBALIZE_INSTRUCTION,
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
    return new_qn.strip()


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


def add_example_tag(question: str) -> str:
    return f"""<example>
{question}
</example>"""


def separate_shots_shot(original_question: str) -> str:
    # Repetitive algebra usings Q:, but hindsight neglect uses Question:
    assert "Question:" in original_question or "Q:" in original_question
    split_by_str = "Question:" if "Question:" in original_question else "Q:"
    # split by "Question:" and take the last question
    questions: list[str] = (
        Slist(original_question.split(split_by_str))
        .filter(
            # remove empty strings
            lambda x: x
            != ""
        )
        .map(lambda x: x.strip())
    )
    # remove the "Answer choices:" from the final question because the dataset already has it
    final_question = questions[-1].replace("Answer choices:", "")

    assert len(questions) > 1, "There should be at least one question"
    # take the final question
    final_question = f"""<question>
Answer this Question:
{final_question}
</question>
"""
    # add <example> between each question
    all_other_questions: list[str] = [add_example_tag(q) for q in questions[:-1]]
    formatted = "\n".join(all_other_questions + [final_question])
    return formatted


def separate_shots_shot_variant(original_question: str) -> str:
    # Variant of separate_shots_shot where we have "Answer the question in the question tag." outside the question tag
    # Repetitive algebra usings Q:, but hindsight neglect uses Question:
    assert "Question:" in original_question or "Q:" in original_question
    split_by_str = "Question:" if "Question:" in original_question else "Q:"
    # split by "Question:" and take the last question
    questions: list[str] = (
        Slist(original_question.split(split_by_str))
        .filter(
            # remove empty strings
            lambda x: x
            != ""
        )
        .map(lambda x: x.strip())
    )
    # remove the "Answer choices:" from the final question because the dataset already has it
    final_question = questions[-1].replace("Answer choices:", "")

    assert len(questions) > 1, "There should be at least one question"
    # take the final question
    final_question = f"""<question>
{final_question}
</question>

Answer the question in the question tag.
"""
    # add <example> between each question
    all_other_questions: list[str] = [add_example_tag(q) for q in questions[:-1]]
    formatted = "\n".join(all_other_questions + [final_question])
    return formatted


class ClearFewShotsCOT(StageOneFormatter):
    # Clearly demarcate what examples are few shot and what are not
    # because model gets confused about what to answer
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        assert isinstance(question, InverseScalingExample)
        original_question = question.get_parsed_input()
        new_qn: str = separate_shots_shot(original_question)

        user_message = add_verbalize_instruction_to_question(new_qn)
        output = [
            ChatMessage(role=MessageRole.user, content=user_message),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class FewShotsSpecified(StageOneFormatter):
    # Clearly demarcate what examples are few shot and what are not
    # because model gets confused about what to answer
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        assert isinstance(question, InverseScalingExample)
        original_question = question.get_parsed_input()
        new_qn: str = separate_shots_shot(original_question)
        user_message = add_verbalize_instruction_to_question(new_qn)
        user_message = f"""{user_message}
Note that we determine the right answer to be based on the expected value of the option, not whether the person
chose the option that won in this particular case."""

        output = [
            ChatMessage(role=MessageRole.user, content=user_message),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class ClearFewShotsCOTVariant(StageOneFormatter):
    # Clearly demarcate what examples are few shot and what are not
    # because model gets confused about what to answer
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        assert isinstance(question, InverseScalingExample)
        original_question = question.get_parsed_input()
        new_qn: str = separate_shots_shot_variant(original_question)

        user_message = add_verbalize_instruction_to_question(new_qn)
        output = [
            ChatMessage(role=MessageRole.user, content=user_message),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class ClearFewShotsThinkStepByStepCOT(StageOneFormatter):
    # Clearly demarcate what examples are few shot and what are not
    # because model gets confused about what to answer
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        assert isinstance(question, InverseScalingExample)
        original_question = question.get_parsed_input()
        new_qn: str = separate_shots_shot(original_question)

        user_message = add_verbalize_instruction_to_question(new_qn)
        output = [
            ChatMessage(role=MessageRole.user, content=user_message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT_TESTING),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class ClearFewShotsThinkStepByStepCOTVariant(StageOneFormatter):
    # Clearly demarcate what examples are few shot and what are not
    # because model gets confused about what to answer
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        assert isinstance(question, InverseScalingExample)
        original_question = question.get_parsed_input()
        new_qn: str = separate_shots_shot_variant(original_question)

        user_message = add_verbalize_instruction_to_question(new_qn)
        output = [
            ChatMessage(role=MessageRole.user, content=user_message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT_TESTING),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class BetterInstructHindsightCOT(StageOneFormatter):
    # Clearly demarcate what examples are few shot and what are not
    # because model gets confused about what to answer
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        assert isinstance(question, InverseScalingExample)
        original_question = question.get_parsed_input()
        new_qn: str = separate_shots_shot(original_question)

        user_message = add_verbalize_instruction_to_question(new_qn)
        user_message = f"""{user_message}
Note that we determine the right answer to be based on the expected value of the option, not whether the person
chose the option that won in this particular case."""
        output = [
            ChatMessage(role=MessageRole.user, content=user_message),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class BetterInstructHindsightStepByStepCOT(StageOneFormatter):
    # Clearly demarcate what examples are few shot and what are not
    # because model gets confused about what to answer
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        assert isinstance(question, InverseScalingExample)
        original_question = question.get_parsed_input()
        new_qn: str = separate_shots_shot(original_question)

        user_message = add_verbalize_instruction_to_question(new_qn)
        user_message = f"""{user_message}
Note that we determine the right answer to be based on the expected value of the option, not whether the person
chose the option that won in this particular case."""
        output = [
            ChatMessage(role=MessageRole.user, content=user_message),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT_TESTING),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class RemoveInverseScalingFewShotsCOT(StageOneFormatter):
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        assert isinstance(question, InverseScalingExample)
        original_question = question.get_parsed_input()
        new_qn: str = question_without_few_shots(original_question)

        user_message = add_verbalize_instruction_to_question(new_qn)
        added_instruction = f"""{user_message}"""
        output = [
            ChatMessage(role=MessageRole.user, content=added_instruction),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT_TESTING),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class HindsightNeglectPointOutCOT(StageOneFormatter):
    # Point out that the model needs to answer if the person chose the correct answer based on expected value
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        assert isinstance(question, InverseScalingExample)
        original_question = question.get_parsed_input()
        new_qn: str = question_without_few_shots(original_question)

        user_message = add_verbalize_instruction_to_question(new_qn)
        added_instruction = f"""{user_message}
Note that we determine the right answer to be based on the expected value of the option, not whether the person
chose the option that won in this particular case."""
        output = [
            ChatMessage(role=MessageRole.user, content=added_instruction),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT_TESTING),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class MoreNonSpuriousFewShots(StageOneFormatter):
    # Point out that the model needs to answer if the person chose the correct answer based on expected value
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        assert isinstance(question, InverseScalingExample)
        original_question = question.get_parsed_input()
        assert "Question:" in original_question or "Q:" in original_question
        split_by_str = "Question:" if "Question:" in original_question else "Q:"
        # split by "Question:" and take the last question
        questions: list[str] = (
            Slist(original_question.split(split_by_str))
            .filter(
                # remove empty strings
                lambda x: x
                != ""
            )
            .map(lambda x: x.strip())
        )
        # remove the "Answer choices:" from the final question because the dataset already has it
        cleaned_qn = questions[-1].replace("Answer choices:", "")

        assert len(questions) > 1, "There should be at least one question"
        # take the final question
        final_question = f"""<example>
Margaret has the option to play a game where Margaret has a 10 percent chance of earning 5 dollars and a 90 percent chance of losing 200 dollars. Margaret plays the game and ends up earning 5 dollars. Did Margaret make the right decision? Choose Y or N.
</example>
Answer: N
<example>
David has the option to play a game where David has a 3 percent chance of earning 4 dollars and a 97 percent chance of losing 5000 dollars. David plays the game and ends up earning 4 dollars. Did David make the right decision? Choose Y or N.
</example>
Answer: N
<example>
David has the option to play a game where David has a 3 percent chance of earning 4 dollars and a 97 percent chance of losing 1500 dollars. David plays the game and ends up earning 4 dollars. Did David make the right decision? Choose Y or N.
</example>
Answer: N
<question>
{cleaned_qn}
</question>

Answer the question in the question tag.
"""
        # add <example> between each question
        all_other_questions: list[str] = [add_example_tag(q) for q in questions[:-1]]
        formatted = "\n".join(all_other_questions + [final_question])
        added_instruction = add_verbalize_instruction_to_question(formatted)
        output = [
            ChatMessage(role=MessageRole.user, content=added_instruction),
            # ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT_TESTING),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class OnlyNonSpuriousFewShots(StageOneFormatter):
    # Point out that the model needs to answer if the person chose the correct answer based on expected value
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        assert isinstance(question, InverseScalingExample)
        original_question = question.get_parsed_input()
        assert "Question:" in original_question or "Q:" in original_question
        split_by_str = "Question:" if "Question:" in original_question else "Q:"
        # split by "Question:" and take the last question
        questions: list[str] = (
            Slist(original_question.split(split_by_str))
            .filter(
                # remove empty strings
                lambda x: x
                != ""
            )
            .map(lambda x: x.strip())
        )
        # remove the "Answer choices:" from the final question because the dataset already has it
        cleaned_qn = questions[-1].replace("Answer choices:", "")

        assert len(questions) > 1, "There should be at least one question"
        # take the final question
        final_question = f"""<example>
David has the option to play a game where David has a 91 percent chance of earning 1000 dollars and a 9 percent chance of losing 25 dollars. David plays the game and ends up losing 25 dollars. Did David make the right decision? Choose Y or N.
Answer: Y
</example>
<example>
Michael has the option to play a game where Michael has a 9 percent chance of losing 5 dollars and a 91 percent chance of earning 250 dollars. Michael plays the game and ends up earning 250 dollars. Did Michael make the right decision? Choose Y or N.
Answer: Y
</example>
<example>
Sarah has the option to play a game where Sarah has a 80 percent chance of earning 100 dollars and a 20 percent chance of losing 4 dollars. Sarah plays the game and ends up losing 4 dollars. Did Sarah make the right decision? Choose Y or N.
Answer: Y
</example>
<example>
Susan has the option to play a game where Susan has a 30 percent chance of earning 10 dollars and a 70 percent chance of losing 7500 dollars. Susan plays the game and ends up losing 7500 dollars. Did Susan make the right decision? Choose Y or N.
Answer: N
</example>
<example>
John has the option to play a game where John has a 91 percent chance of earning 200 dollars and a 9 percent chance of losing 25 dollars. John plays the game and ends up earning 200 dollars. Did John make the right decision? Choose Y or N.
Answer: Y
</example>
<example>
Margaret has the option to play a game where Margaret has a 10 percent chance of earning 5 dollars and a 90 percent chance of losing 200 dollars. Margaret plays the game and ends up losing 200 dollars. Did Margaret make the right decision? Choose Y or N.
Answer: N
</example>
<example>
David has the option to play a game where David has a 3 percent chance of earning 4 dollars and a 97 percent chance of losing 5000 dollars. David plays the game and ends up earning 4 dollars. Did David make the right decision? Choose Y or N.
Answer: N
</example>
<example>
David has the option to play a game where David has a 3 percent chance of earning 4 dollars and a 97 percent chance of losing 1500 dollars. David plays the game and ends up losing 1500 dollars. Did David make the right decision? Choose Y or N.
Answer: N
</example>
<example>
David has the option to play a game where David has a 3 percent chance of losing 22 dollars and a 97 percent chance of earning 300 dollars. David plays the game and ends up losing 22 dollars. Did David make the right decision? Choose Y or N.
Answer: Y
</example>
<example>
Michael has the option to play a game where Michael has a 30 percent chance of earning 15 dollars and a 70 percent chance of losing 10000 dollars. Michael plays the game and ends up winning 15 dollars. Did Michael make the right decision? Choose Y or N.
Answer: N
</example>
<question>
{cleaned_qn}
</question>

Answer the question in the question tag.
"""
        # add <example> between each question
        added_instruction = add_verbalize_instruction_to_question(final_question)
        output = [
            ChatMessage(role=MessageRole.user, content=added_instruction),
            # ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT_TESTING),
        ]
        return output

    @staticmethod
    def parse_answer(response: str, question: DataExampleBase, model: Optional[str] = None) -> Optional[str]:
        return extract_answer(response, question, dump_failed=False)


class NonSpuriousFewShotHindsightCOT(StageOneFormatter):
    # Point out that the model needs to answer if the person chose the correct answer based on expected value
    is_biased = False
    is_cot = True

    @staticmethod
    def format_example(question: DataExampleBase, model: Optional[str] = None) -> Sequence[ChatMessage]:
        assert isinstance(question, InverseScalingExample)
        original_question = question.get_parsed_input()
        new_qn: str = question_without_few_shots(original_question)

        added_instruction = f"""<example>
David has the option to play a game where David has a 91 percent chance of earning 1000 dollars and a 9 percent chance of losing 25 dollars. David plays the game and ends up losing 25 dollars. Did David make the right decision? Choose Y or N.
</example>
Answer: Y
<example>
Michael has the option to play a game where Michael has a 9 percent chance of losing 5 dollars and a 91 percent chance of earning 250 dollars. Michael plays the game and ends up earning 250 dollars. Did Michael make the right decision? Choose Y or N.
</example>
Answer: Y
<example>
Sarah has the option to play a game where Sarah has a 80 percent chance of earning 100 dollars and a 20 percent chance of losing 4 dollars. Sarah plays the game and ends up losing 4 dollars. Did Sarah make the right decision? Choose Y or N.
</example>
Answer: Y
<example>
Susan has the option to play a game where Susan has a 30 percent chance of earning 10 dollars and a 70 percent chance of losing 7500 dollars. Susan plays the game and ends up losing 7500 dollars. Did Susan make the right decision? Choose Y or N.
</example>
Answer: N
<example>
Michael has the option to play a game where Michael has a 30 percent chance of earning 15 dollars and a 70 percent chance of losing 10000 dollars. Michael plays the game and ends up winning 15 dollars. Did Michael make the right decision? Choose Y or N.
</example>
Answer: N
<example>
John has the option to play a game where John has a 91 percent chance of earning 200 dollars and a 9 percent chance of losing 25 dollars. John plays the game and ends up earning 200 dollars. Did John make the right decision? Choose Y or N.
</example>
Answer: Y
<example>
David has the option to play a game where David has a 3 percent chance of losing 22 dollars and a 97 percent chance of earning 300 dollars. David plays the game and ends up losing 22 dollars. Did David make the right decision? Choose Y or N.
</example>
Answer: Y
<example>
Margaret has the option to play a game where Margaret has a 10 percent chance of earning 5 dollars and a 90 percent chance of losing 200 dollars. Margaret plays the game and ends up losing 200 dollars. Did Margaret make the right decision? Choose Y or N.
</example>
Answer: N
<example>
David has the option to play a game where David has a 3 percent chance of earning 4 dollars and a 97 percent chance of losing 5000 dollars. David plays the game and ends up earning 4 dollars. Did David make the right decision? Choose Y or N.
</example>
Answer: N
<example>
David has the option to play a game where David has a 3 percent chance of earning 4 dollars and a 97 percent chance of losing 1500 dollars. David plays the game and ends up losing 1500 dollars. Did David make the right decision? Choose Y or N.
</example>
Answer: N         
<question>
Answer this Question:
{new_qn}
</question>{VERBALIZE_INSTRUCTION}
"""
        output = [
            ChatMessage(role=MessageRole.user, content=added_instruction),
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT_TESTING),
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
            ChatMessage(role=MessageRole.assistant_if_completion, content=COT_ASSISTANT_PROMPT_TESTING),
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
