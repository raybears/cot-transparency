import re

from cot_transparency.data_models.example_base import MultipleChoiceAnswer, VALID_ANSWERS


def extract_answer_from_brackets(string: str) -> str:
    # e.g. " (A)" -> "A"
    # e.g. "A" -> "A"
    # e.g. "(A)" -> "A"
    return re.sub(r"[()]", "", string).strip()


def raise_if_not_multiple_choice_answer(string: str) -> MultipleChoiceAnswer:
    assert string in VALID_ANSWERS
    return string


def test_strip_brackets():
    test_1 = " (A)"
    assert extract_answer_from_brackets(test_1) == "A"
    test_2 = "A"
    assert extract_answer_from_brackets(test_2) == "A"
    test_3 = "(A)"
    assert extract_answer_from_brackets(test_3) == "A"