from pathlib import Path

from slist import Slist

from cot_transparency.data_models.io import ExpLoader
from cot_transparency.data_models.models import ExperimentJsonFormat, TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotUnbiasedFormatter, ZeroShotCOTUnbiasedFormatter
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.model_apis import Prompt
from stage_one import COT_TRAINING_TASKS


# ruff: noqa: E501


def dump_correct_data(cot_data: bool, exp_dir: str, model: str) -> None:
    """Produces a dataset containing COT reasoning that
    - should be mostly correct. You can filter out later
    Steps
    1. Run stage one with an unbiased formatter e.g.
     `python stage_one.py --exp_dir experiments/biased_aqua --models '["gpt-35-turbo"]' --formatters '["ZeroShotCOTUnbiasedFormatter"]'`
    2. Run this script
    3. This will produce a data.jsonl file in data/bbh_cots
    """
    jsons = ExpLoader.stage_one(exp_dir=exp_dir)
    for v in jsons.values():
        assert isinstance(v, ExperimentJsonFormat)
    selected_formatter = ZeroShotCOTUnbiasedFormatter.name() if cot_data else ZeroShotUnbiasedFormatter.name()

    # intervention_name should be None
    # dataset should be bbh
    # model should be gpt-3.5-turbo
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
    print(f"Number of jsons: {len(jsons_tasks)} for model {model}")
    score = jsons_tasks.map(lambda x: x.is_correct).average()
    print(f"Average score: {score}")
    assert score == 1.0, f"This should be 1.0, got {score}"

    print(f"Number of jsons: {len(jsons_tasks)}")
    # remove the . from the model name so we are compatible with more file systems
    model_file_name = model.replace(".", "")
    if cot_data:
        write_jsonl_file_from_basemodel(
            path=Path(f"data/training_cots/{model_file_name}.jsonl"), basemodels=jsons_tasks
        )
    else:
        write_jsonl_file_from_basemodel(
            path=Path(f"data/training_non_cots/{model_file_name}.jsonl"), basemodels=jsons_tasks
        )


if __name__ == "__main__":
    dump_correct_data(cot_data=False, exp_dir="experiments/training_data_temp_1", model="gpt-3.5-turbo")
    dump_correct_data(cot_data=True, exp_dir="experiments/training_data_temp_1", model="gpt-3.5-turbo")
    dump_correct_data(cot_data=False, exp_dir="experiments/training_data_temp_1_claude_2_unbiased", model="claude-2")
    dump_correct_data(cot_data=True, exp_dir="experiments/training_data_temp_1_claude_2_unbiased", model="claude-2")
