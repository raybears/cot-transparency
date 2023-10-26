import re

import pytest

from cot_transparency.data_models.data.arc import ArcExample
from stage_one import TASK_LIST, get_list_of_examples

ANSWER_RE = r"^(\n|.)+\n\nAnswer choices:\n\(A\) .+\n\(B\) .+"
# should have the format
# <question>
# Answer choices:
# (A) <option>
# (B) <option>
# ...


@pytest.mark.parametrize("task", TASK_LIST["transparency"])
def test_transparency_data_format(task: str):
    examples = get_list_of_examples(task, "transparency").take(2)
    assert len(examples) == 2, f"task: {task}, expected 2 examples, got {len(examples)}"

    for example in examples:
        full_question = example.get_parsed_input()
        assert re.match(ANSWER_RE, full_question)


@pytest.mark.parametrize("task", TASK_LIST["bbh"])
def test_bbh_data_format(task: str):
    examples = get_list_of_examples(task, "bbh").take(2)
    assert len(examples) == 2

    for example in examples:
        full_question = example.get_parsed_input()
        assert re.match(ANSWER_RE, full_question)


@pytest.mark.parametrize("task", TASK_LIST["bbh_biased_wrong_cot"])
def test_bbh_biased_wrong_ans_data_format(task: str):
    examples = get_list_of_examples(task, "bbh_biased_wrong_cot").take(2)
    assert len(examples) == 2

    for example in examples:
        full_question = example.get_parsed_input()
        assert re.match(ANSWER_RE, full_question)


def test_arc_examples():
    numeric_example = {
        "id": "NYSEDREGENTS_2014_8_38",
        "question": {
            "stem": "A substance in the solid phase (state) of matter has",
            "choices": [
                {"text": "a definite shape and a definite volume", "label": "1"},
                {"text": "a definite shape, but no definite volume", "label": "2"},
                {"text": "no definite shape, but a definite volume", "label": "3"},
                {"text": "no definite shape and no definite volume", "label": "4"},
            ],
        },
        "answerKey": "1",
    }
    parsed = ArcExample.model_validate(numeric_example)
    assert parsed.ground_truth == "A"
    numeric_example_2 = {
        "id": "NYSEDREGENTS_2011_8_4",
        "question": {
            "stem": "Which type of energy in gasoline is transformed into mechanical energy in a motorcycle engine?",
            "choices": [
                {"text": "chemical", "label": "1"},
                {"text": "nuclear", "label": "2"},
                {"text": "magnetic", "label": "3"},
                {"text": "electrical", "label": "4"},
            ],
        },
        "answerKey": "1",
    }

    parsed_2 = ArcExample.model_validate(numeric_example_2)
    assert parsed_2.ground_truth == "A"
    letter_example = {
        "id": "Mercury_7008435",
        "question": {
            "stem": "What contributes the most to beach erosion?",
            "choices": [
                {"text": "animal activity", "label": "A"},
                {"text": "evaporation", "label": "B"},
                {"text": "precipitation", "label": "C"},
                {"text": "wave action", "label": "D"},
            ],
        },
        "answerKey": "D",
    }
    parsed_3 = ArcExample.model_validate(letter_example)
    assert parsed_3.ground_truth == "D"
