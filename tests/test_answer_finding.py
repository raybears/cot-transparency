from string import ascii_uppercase
import pytest
import yaml
from cot_transparency.data_models.example_base import ChoiceVariant, DataFormatSpec, IndicatorSeparator

from cot_transparency.formatters.extraction import (
    FindAnswerStringAfterBreakWord,
    FindIndicatorAfterBreakWord,
    FindIndicatorAtStartOfResponse,
    # extract_answer,
    # extract_answer_non_cot,
    # extract_answer_looking_for_option,
)

from scripts.biased_wrong_ans import cot_extraction


# ruff: noqa: E501


def load_test_data():
    with open("tests/data/answer_examples.yaml", "r") as f:
        data = yaml.safe_load(f)
    return data["tests"]


test_data = load_test_data()


@pytest.mark.parametrize("test_case", test_data)
def test_find_letter_after_breakword(test_case: dict[str, str]):
    options = ["random possible answer" for i in range(15)]
    extracted = FindIndicatorAfterBreakWord(options).extract(test_case["text"])

    assert extracted == test_case["answer"]


@pytest.mark.parametrize(
    "test_case",
    [
        "The best answer is: (Animal)",
        "The best answer is: Car",
        "The best answer is: george washington",
    ],
)
def test_find_indicator_after_breakword_returns_none(test_case: str):
    """
    These should not match because they re just wordsd that happen to
    start with a letter that is in the options
    """
    options = [ascii_uppercase[i] for i in range(15)]
    extracted = FindIndicatorAfterBreakWord(options).extract(test_case)
    assert extracted is None


def test_extractor_get_last_answer():
    long = """Assistant: Let's think step by step:
The best answer is: (A)    
Step 1: Determine the current date
Today's date is May 30, 2021.

Step 2: Calculate the date 24 hours later
Adding 24 hours to May 30, 2021, gives us May 31, 2021.

Step 3: Verify the date format
The format requested is MM/DD/YYYY.
Actually I made a mistake
The best answer is: (B) 05/31/2021.
"""
    options = ["some random answer" for i in range(15)]

    assert FindIndicatorAfterBreakWord(options).extract(long) == "B"


def test_extract_latex():
    answer = r"Therefore, the best answer is: $\boxed{\text{(B) } 432}$."
    options = ["some random answer" for i in range(2)]
    assert FindIndicatorAfterBreakWord(options).extract(answer) == "B"


def test_cot_extraction():
    test = """First, we start by facing forward.
The best answer is: (B) False.
"""
    assert cot_extraction(test) == """First, we start by facing forward."""

    none_test = """First, we start by facing forward."""
    assert cot_extraction(none_test) is None


@pytest.mark.parametrize(
    "response, input_variant, expected_output",
    [
        # Test with letters
        ("A", ChoiceVariant.LETTERS, "A"),
        ("b", ChoiceVariant.LETTERS, "B"),
        ("(C)", ChoiceVariant.LETTERS, "C"),
        ("D", ChoiceVariant.LETTERS, "D"),
        # Test with numbers
        ("1", ChoiceVariant.NUMBERS, "A"),
        ("(2)", ChoiceVariant.NUMBERS, "B"),
        ("4", ChoiceVariant.NUMBERS, "D"),
        # Test with Roman numerals
        ("I", ChoiceVariant.ROMAN, "A"),
        ("ii", ChoiceVariant.ROMAN, "B"),
        ("(III)", ChoiceVariant.ROMAN, "C"),
        ("IV", ChoiceVariant.ROMAN, "D"),
        # Test with foo
        ("foo", ChoiceVariant.FOO, "A"),
        ("bar", ChoiceVariant.FOO, "B"),
        ("(baz)", ChoiceVariant.FOO, "C"),
        ("baz)", ChoiceVariant.FOO, "C"),
        ("(baz", ChoiceVariant.FOO, "C"),
        ("(best", ChoiceVariant.FOO, None),
        ("answer", ChoiceVariant.FOO, None),
        ("None of the above", ChoiceVariant.FOO, None),
    ],
)
def test_extract_answer_non_cot(response: str, input_variant: ChoiceVariant, expected_output: str):
    data_format = DataFormatSpec(choice_variant=input_variant)
    options = ["some random answer" for i in range(15)]  # not matching on these
    assert FindIndicatorAtStartOfResponse(options, data_format).extract(response, dump_failed=False) == expected_output


#


@pytest.mark.parametrize(
    "input_str, data_format, expected_output",
    [
        ("The best answer is: (A) 1", DataFormatSpec(choice_variant=ChoiceVariant.LETTERS), "A"),
        (
            "The best answer is: 5. 2",
            DataFormatSpec(choice_variant=ChoiceVariant.NUMBERS, indicator_separator=IndicatorSeparator.DOT),
            "E",
        ),
        (
            "The best answer is: $\\boxed{\\textbf{(5)} 945}$.",
            DataFormatSpec(choice_variant=ChoiceVariant.NUMBERS),
            "E",
        ),
        ("Therefore, the best answer is: $\\boxed{\\text{(B) } 905}$.", DataFormatSpec(), "B"),
    ],
)
def test_extract_answer_format(input_str: str, data_format: DataFormatSpec, expected_output: str):
    options = ["dog", "cat", "zebra", "100", "200"]
    assert FindIndicatorAfterBreakWord(options, data_format).extract(model_answer=input_str) == expected_output


def test_extract_answer_format_with_roman_options():
    # III should be parsed out
    input_str = "The best answer is: (III)"
    data_format = DataFormatSpec(choice_variant=ChoiceVariant.ROMAN)
    expected_output = "C"
    options = ["dog", "cat", "zebra"]
    assert FindIndicatorAfterBreakWord(options, input_format=data_format).extract(input_str) == expected_output


# 'The best answer is: corge) None of the above' not working


def test_extract_answer_format_without_parenthesis2():
    # baz without the parentheses should be parsed out, even though it was formatted with parentheses
    input_str = "The best answer is: corge) None of the above"
    expected_output = "F"
    options = ["dog", "cat", "zebra", "100", "200", "None of the above"]
    assert FindAnswerStringAfterBreakWord(options).extract(input_str) == expected_output


def test_extract_answer_format_without_parenthesis():
    # baz without the parentheses should be parsed out, even though it was formatted with parentheses
    input_str = "The best answer is: baz"
    data_format = DataFormatSpec(choice_variant=ChoiceVariant.FOO)
    expected_output = "C"
    options = ["dog", "cat", "zebra"]
    assert FindIndicatorAfterBreakWord(options, input_format=data_format).extract(input_str) == expected_output


def test_extract_answer_format_missing_last_period():
    # dog should match even though it is missing the last period
    input_str = "The best answer is: dog"
    expected_output = "A"
    # options with ending periods
    options = ["dog.", "cat.", "zebra."]
    assert FindAnswerStringAfterBreakWord(options).extract(input_str) == expected_output


def test_extract_answer_no_separator():
    # qux isn't in brackets, or has a separator, but should still match
    input_str = "The best answer is: qux Avoid examining his or her"
    data_format = DataFormatSpec(choice_variant=ChoiceVariant.FOO)
    expected_output = "D"
    options = ["cat", "dog", "something", "Avoid examining his or her personality"]
    assert FindIndicatorAfterBreakWord(options, data_format).extract(input_str) == expected_output


def test_extract_answer_should_none():
    input_str = "The best answer is: ffff Avoid "
    options = ["cat", "dog", "something", "Avoid examining his or her personality"]
    assert FindAnswerStringAfterBreakWord(options).extract(input_str) is None


if __name__ == "__main__":
    # run the test_extract_answer_non_cot test
    pytest.main(["-s", "-v", f"{__file__}::test_extract_answer_non_cot"])
