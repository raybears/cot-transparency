from functools import lru_cache
from pathlib import Path

from slist import Slist

from cot_transparency.data_models.data.biased_question_unbiased_cot import BiasedQuestionUnbiasedCOT
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel


@lru_cache
def get_big_brain_cots() -> Slist[BiasedQuestionUnbiasedCOT]:
    # bbh only
    jsons_tasks: Slist[BiasedQuestionUnbiasedCOT] = read_jsonl_file_into_basemodel(
        Path("data/bbh_big_brain_cots/data.jsonl"), BiasedQuestionUnbiasedCOT
    )

    return jsons_tasks


@lru_cache
def get_training_cots_gpt_35_big_brain() -> Slist[BiasedQuestionUnbiasedCOT]:
    # BBH_TASK_LIST + ["arc_easy_train", "arc_challenge_train", "openbook_qa_train"]
    jsons_tasks: Slist[BiasedQuestionUnbiasedCOT] = read_jsonl_file_into_basemodel(
        Path("data/training_cots/gpt-35-turbo-big-brain.jsonl"), BiasedQuestionUnbiasedCOT
    )

    return jsons_tasks
