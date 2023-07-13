import pytest
import yaml

from cot_transparency.prompt_formatter import extract_answer


def load_test_data():
    with open("tests/data/answer_examples.yaml", "r") as f:
        data = yaml.safe_load(f)
    return data["tests"]


test_data = load_test_data()


@pytest.mark.parametrize("test_case", test_data)
def test_extract_answer(test_case):
    assert extract_answer(test_case["text"]) == test_case["answer"]