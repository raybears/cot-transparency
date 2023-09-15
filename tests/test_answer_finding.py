import pytest
import yaml
from cot_transparency.data_models.example_base import ChoiceVariant, DataFormatSpec, IndicatorSeparator

from cot_transparency.formatters.extraction import (
    extract_answer,
    extract_answer_non_cot,
    extract_answer_looking_for_option,
)

from scripts.biased_wrong_ans import cot_extraction


# ruff: noqa: E501


def load_test_data():
    with open("tests/data/answer_examples.yaml", "r") as f:
        data = yaml.safe_load(f)
    return data["tests"]


test_data = load_test_data()


@pytest.mark.parametrize("test_case", test_data)
def test_extract_answer(test_case: dict[str, str]):
    assert extract_answer(test_case["text"]) == test_case["answer"]


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
    assert extract_answer(long) == "B"


def test_extract_latex():
    answer = r"Therefore, the best answer is: $\boxed{\text{(B) } 432}$."
    assert extract_answer(answer) == "B"


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
    assert extract_answer_non_cot(response, dump_failed=False, input_format=data_format) == expected_output


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
    assert extract_answer_looking_for_option(model_answer=input_str, input_format=data_format) == expected_output


def test_extract_answer_format_with_roman_options():
    # III should be parsed out
    input_str = "The best answer is: (III)"
    data_format = DataFormatSpec(choice_variant=ChoiceVariant.ROMAN)
    expected_output = "C"
    options = ["dog", "cat", "zebra"]
    assert (
        extract_answer_looking_for_option(model_answer=input_str, input_format=data_format, options=options)
        == expected_output
    )


def test_extract_answer_format_without_parenthesis():
    # baz without the parentheses should be parsed out, even though it was formatted with parentheses
    input_str = "The best answer is: baz"
    data_format = DataFormatSpec(choice_variant=ChoiceVariant.FOO)
    expected_output = "C"
    options = ["dog", "cat", "zebra"]
    assert (
        extract_answer_looking_for_option(model_answer=input_str, input_format=data_format, options=options)
        == expected_output
    )
