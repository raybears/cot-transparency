from pathlib import Path

from slist import Slist

from cot_transparency.data_models.io import ExpLoader
from cot_transparency.data_models.models import ExperimentJsonFormat, TaskOutput
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel

if __name__ == "__main__":
    """Produces a dataset containing COT reasoning that
    - should be mostly correct. You can filter out later
    Steps
    1. Run stage one with a biased formatter
     `python stage_one.py --exp_dir experiments/biased_aqua --models '["gpt-4"]' --formatters '["ZeroShotCOTUnbiasedFormatter"]'`
    2. Run this script
    3. This will produce a data.jsonl file in data/bbh_cots
    """
    jsons = ExpLoader.stage_one("experiments/bad_cot")
    for v in jsons.values():
        assert isinstance(v, ExperimentJsonFormat)

    jsons_tasks: Slist[TaskOutput] = Slist(jsons.values()).map(lambda x: x.outputs).flatten_list()  # type: ignore
    selected_formatter = "ZeroShotCOTUnbiasedFormatter"
    print(f"Number of jsons: {len(jsons_tasks)}")
    write_jsonl_file_from_basemodel(path=Path("data/bbh_cots/data.jsonl"), basemodels=jsons_tasks)
