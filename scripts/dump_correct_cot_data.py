from pathlib import Path

from slist import Slist

from cot_transparency.data_models.io import ExpLoader
from cot_transparency.data_models.models import ExperimentJsonFormat, TaskOutput
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from stage_one import BBH_TASK_LIST

# ruff: noqa: E501

if __name__ == "__main__":
    """Produces a dataset containing COT reasoning that
    - should be mostly correct. You can filter out later
    Steps
    1. Run stage one with a biased formatter
     `python stage_one.py --exp_dir experiments/biased_aqua --models '["gpt-4"]' --formatters '["ZeroShotCOTUnbiasedFormatter"]'`
    2. Run this script
    3. This will produce a data.jsonl file in data/bbh_cots
    """
    jsons = ExpLoader.stage_one("experiments/biased_aqua")
    for v in jsons.values():
        assert isinstance(v, ExperimentJsonFormat)
    selected_formatter = "ZeroShotCOTUnbiasedFormatter"

    # intervention_name should be None
    # dataset should be bbh
    # model should be gpt-4
    bbh_tasks = BBH_TASK_LIST
    jsons_tasks: Slist[TaskOutput] = (
        Slist(jsons.values())
        .map(lambda x: x.outputs)
        .flatten_list()
        .filter(lambda x: x.task_spec.intervention_name is None)
        .filter(lambda x: x.task_spec.task_name in bbh_tasks)
        .filter(lambda x: x.task_spec.model_config.model == "gpt-4")
        .filter(lambda x: x.task_spec.formatter_name == selected_formatter)
        # only get the ones that are correct
        .filter(lambda x: x.model_output.parsed_response == x.task_spec.ground_truth)
        .distinct_by(lambda x: x.task_spec.task_hash)
    )
    score = jsons_tasks.map(lambda x: x.is_correct).average()
    print(f"Average score: {score}")
    assert score == 1.0, f"This should be 1.0, got {score}"

    print(f"Number of jsons: {len(jsons_tasks)}")
    write_jsonl_file_from_basemodel(path=Path("data/bbh_correct_cots/data.jsonl"), basemodels=jsons_tasks)
