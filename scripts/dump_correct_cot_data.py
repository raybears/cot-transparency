from pathlib import Path

from slist import Slist

from cot_transparency.data_models.io import ExpLoader
from cot_transparency.data_models.models import ExperimentJsonFormat, TaskOutput
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.model_apis import Prompt
from stage_one import COT_TRAINING_TASKS

# ruff: noqa: E501

if __name__ == "__main__":
    """Produces a dataset containing COT reasoning that
    - should be mostly correct. You can filter out later
    Steps
    1. Run stage one with an unbiased formatter e.g.
     `python stage_one.py --exp_dir experiments/biased_aqua --models '["gpt-4"]' --formatters '["ZeroShotCOTUnbiasedFormatter"]'`
    2. Run this script
    3. This will produce a data.jsonl file in data/bbh_cots
    """
    jsons = ExpLoader.stage_one("experiments/gpt_35_cot")
    model: str = "gpt-3.5-turbo"
    for v in jsons.values():
        assert isinstance(v, ExperimentJsonFormat)
    selected_formatter = "ZeroShotCOTUnbiasedFormatter"

    # intervention_name should be None
    # dataset should be bbh
    # model should be gpt-4
    tasks = COT_TRAINING_TASKS

    jsons_tasks: Slist[TaskOutput] = (
        Slist(jsons.values())
        .map(lambda x: x.outputs)
        .flatten_list()
        .filter(lambda x: x.task_spec.intervention_name is None)
        .filter(lambda x: x.task_spec.task_name in tasks)
        .filter(lambda x: x.task_spec.inference_config.model == model)
        .filter(lambda x: x.task_spec.formatter_name == selected_formatter)
        # only get the ones that are correct
        .filter(lambda x: x.inference_output.parsed_response == x.task_spec.ground_truth)
        # make sure the COTs are distinct
        .distinct_by(
            lambda x: Prompt(messages=x.task_spec.messages).convert_to_completion_str()
            + x.inference_output.raw_response
        )
    )
    score = jsons_tasks.map(lambda x: x.is_correct).average()
    print(f"Average score: {score}")
    assert score == 1.0, f"This should be 1.0, got {score}"

    print(f"Number of jsons: {len(jsons_tasks)}")
    write_jsonl_file_from_basemodel(path=Path("data/training_cots/gpt-35-turbo.jsonl"), basemodels=jsons_tasks)
