from functools import lru_cache
from pathlib import Path

from slist import Slist

from cot_transparency.data_models.models import TaskOutput
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel
from cot_transparency.data_models.data.biased_question_unbiased_cot import BiasedQuestionUnbiasedCOT


# Data previously generated with cot-transparency/scripts/dump_correct_cot_data.py
# small brain cache to load only when needed
@lru_cache
def get_correct_cots() -> Slist[TaskOutput]:
    jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
        Path("data/bbh_correct_cots/gpt-4_data.jsonl"), TaskOutput
    )

    only_correct_cots: Slist[TaskOutput] = jsons_tasks
    return only_correct_cots


@lru_cache
def get_correct_cots_claude_2() -> Slist[TaskOutput]:
    jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
        Path("data/bbh_correct_cots/claude-2_data.jsonl"), TaskOutput
    )

    only_correct_cots: Slist[TaskOutput] = jsons_tasks
    return only_correct_cots


# Data previously generated with scripts/dump_big_brain_cot_data.py


@lru_cache
def get_big_brain_cots() -> Slist[BiasedQuestionUnbiasedCOT]:
    jsons_tasks: Slist[BiasedQuestionUnbiasedCOT] = read_jsonl_file_into_basemodel(
        Path("data/bbh_big_brain_cots/data.jsonl"), BiasedQuestionUnbiasedCOT
    )

    return jsons_tasks
