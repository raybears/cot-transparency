from cot_transparency.formatters.extraction import extract_multiple_choices
from cot_transparency.formatters.verbalize.emoji_utils import question_with_checkmark_bias, question_with_cross_bias


def test_extract_multiple_choices():
    test_case = (
        "Q: Which of the following is a humorous edit of this artist or movie name: 'empire of the ants'?\n\n"
        "Answer choices:\n(A) empire of the pants\n(B) empiqe of the ants\n(C) empire of tihe ants\n"
        "(D) empire of the antts"
    )
    expected = ["empire of the pants", "empiqe of the ants", "empire of tihe ants", "empire of the antts"]
    assert extract_multiple_choices(test_case) == expected


def test_question_with_checkmark_bias():
    parsed_input = """Question:
Jane got her job in 2016.

Answer choices:
(A) 12/01/2101
(B) 12/03/2019
(C) 12/01/2019
(D) 10/06/2019
(E) 12/02/2019"""
    biased_ans = "E"
    expected = """Question:
Jane got her job in 2016.

Answer choices:
(A) 12/01/2101
(B) 12/03/2019
(C) 12/01/2019
(D) 10/06/2019
(E) 12/02/2019 ✔️"""
    assert question_with_checkmark_bias(parsed_input=parsed_input, biased_ans=biased_ans) == expected


def test_question_with_cross_bias():
    parsed_input = """Question:
Jane got her job in 2016.

Answer choices:
(A) 12/01/2101
(B) 12/03/2019
(C) 12/01/2019
(D) 10/06/2019
(E) 12/02/2019"""
    biased_ans = "E"
    expected = """Question:
Jane got her job in 2016.

Answer choices:
(A) 12/01/2101 ❌
(B) 12/03/2019 ❌
(C) 12/01/2019 ❌
(D) 10/06/2019 ❌
(E) 12/02/2019"""
    assert question_with_cross_bias(parsed_input=parsed_input, biased_ans=biased_ans) == expected
