import re

import pytest

from cot_transparency.data_models.data import TASK_LIST
from cot_transparency.data_models.data import get_list_of_examples
from cot_transparency.data_models.data.arc import ArcExample
from scripts.none_few_shot_bias.make_few_shots import DataExampleWithNone
from tests.prompt_formatting.test_prompt_formatter import EMPIRE_OF_PANTS_EXAMPLE

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


def test_replacing_correct_with_none():
    example = EMPIRE_OF_PANTS_EXAMPLE

    original_options = example._get_options()
    original_gt = example.ground_truth

    example_with_none = DataExampleWithNone(wrapped=example, replace="correct")

    none_added = example_with_none._get_options().index("None of these options")
    modified_options = example_with_none._get_options()

    assert original_options != modified_options
    assert modified_options[none_added] == "None of these options"
    assert example_with_none.ground_truth_text == "None of these options"
    assert example_with_none.ground_truth_idx() == example.ground_truth_idx()
    assert original_gt not in example_with_none._get_options()


def test_replacing_incorrect_with_none():
    example = EMPIRE_OF_PANTS_EXAMPLE

    original_options = example._get_options()
    original_gt = example.ground_truth

    example_with_none = DataExampleWithNone(wrapped=example, replace="incorrect")

    none_added = example_with_none._get_options().index("None of these options")
    modified_options = example_with_none._get_options()

    assert original_options != modified_options
    assert modified_options[none_added] == "None of these options"
    assert example_with_none.ground_truth == original_gt
    assert example_with_none.ground_truth_idx() != none_added
    assert example.ground_truth_text in example_with_none._get_options()
