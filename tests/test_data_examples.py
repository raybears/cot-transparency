import re
from stage_one import TASK_LIST, get_list_of_examples
import pytest


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
    assert len(examples) == 2

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
