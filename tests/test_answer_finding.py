import pytest
import yaml

from cot_transparency.formatters.extraction import extract_answer

from scripts.biased_wrong_ans import cot_extraction


def load_test_data():
    with open("tests/data/answer_examples.yaml", "r") as f:
        data = yaml.safe_load(f)
    return data["tests"]


test_data = load_test_data()


@pytest.mark.parametrize("test_case", test_data)
def test_extract_answer(test_case: dict[str, str]):
    assert extract_answer(test_case["text"]) == test_case["answer"]


def test_cot_extraction():
    test = """First, we start by facing forward.
The best answer is: (B) False.
"""
    assert cot_extraction(test) == """First, we start by facing forward."""

    none_test = """First, we start by facing forward."""
    assert cot_extraction(none_test) is None
