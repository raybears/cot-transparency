from functools import lru_cache
from pathlib import Path

from slist import Slist

from cot_transparency.data_models.models import TaskOutput
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel


# Data previously generated with cot-transparency/scripts/dump_correct_cot_data.py
# small brain cache to load only when needed
@lru_cache
def get_correct_cots() -> Slist[TaskOutput]:
    jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
        Path("data/bbh_correct_cots/data.jsonl"), TaskOutput
    )

    only_correct_cots: Slist[TaskOutput] = jsons_tasks
    return only_correct_cots
