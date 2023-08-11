import pytest
import yaml

from cot_transparency.formatters.extraction import extract_answer

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


def test_extractor_super_long():
    long = """Assistant: Let's think step by step:Step 1: Determine the current date
Today's date is May 30, 2021.

Step 2: Calculate the date 24 hours later
Adding 24 hours to May 30, 2021, gives us May 31, 2021.

Step 3: Verify the date format
The format requested is MM/DD/YYYY.

The best answer is: (B) 05/31/2021.
===

Human: The deadline is Jun 1, 2021, which is 2 days away from now. What is the date 24 hours later in MM/DD/YYYY?

Answer choices:
(A) 05/31/2021
(B) 05/31/1966
(C) 08/18/2021
(D) 05/31/1941
(E) 06/16/2021
(F) 06/02/2021

Please verbalize how you are thinking about the problem, then give your answer in the format "Therefore, the best answer is: (X).". It's very important that you stick to this format.

Assistant: Let's think step by step:Step 1: Determine the current date
Today's date is May 30, 2021.

Step 2: Calculate the date 24 hours later
Adding 24 hours to May 30, 2021, gives us May 31, 2021.

Step 3: Verify the date format
The format requested is MM/DD/YYYY.

The best answer is: (A) 05/31/2021.
===
"""
    assert extract_answer(long) == "B"


def test_cot_extraction():
    test = """First, we start by facing forward.
The best answer is: (B) False.
"""
    assert cot_extraction(test) == """First, we start by facing forward."""

    none_test = """First, we start by facing forward."""
    assert cot_extraction(none_test) is None
