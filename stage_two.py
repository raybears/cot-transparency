from pathlib import Path
from typing import List, Optional, Type
import fire
from cot_transparency.formatters.transparency import EarlyAnsweringFormatter, StageTwoFormatter
from cot_transparency.formatters.transparency.trace_manipulation import get_cot_steps
from cot_transparency.formatters.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.openai_utils.set_key import set_openai_key_from_env

from cot_transparency.tasks import (
    ExperimentJsonFormat,
    StageTwoTaskSpec,
    TaskOutput,
    TaskSpec,
    load_jsons,
    run_tasks_multi_threaded,
)
from cot_transparency.util import get_exp_dir_name
from stage_one import get_valid_stage1_formatters, read_done_experiment

"""
We take traces generated from stage_one.py and run analysis on them
"""


def get_early_answering_tasks(stage_one_output: TaskOutput, exp_dir: str) -> List[TaskSpec]:
    outputs = []

    cot_steps = get_cot_steps(stage_one_output.model_output[0].raw_response)

    partial_cot = ""
    for cot_step in cot_steps:
        partial_cot += cot_step

        out_file_path: Path = Path(
            f"{exp_dir}/{stage_one_output.task_name}/{stage_one_output.config.model}/{EarlyAnsweringFormatter.name()}.json"
        )

        task = StageTwoTaskSpec(
            task_name=stage_one_output.task_name,
            model_config=stage_one_output.config,
            messages=EarlyAnsweringFormatter.format_example(stage_one_output.prompt, partial_cot),
            out_file_path=out_file_path,
            ground_truth=stage_one_output.ground_truth,
            formatter=EarlyAnsweringFormatter,
            task_hash=stage_one_output.task_hash,
            biased_ans=stage_one_output.biased_ans,
            stage_one_hash=stage_one_output.output_hash(),
        )
        outputs.append(task)
    return outputs


def create_stage_2_tasks(
    stage_1_task_outputs: List[TaskOutput],
    exp_dir: str,
) -> List[TaskSpec]:
    tasks_to_run: List[TaskSpec] = []

    task_output: TaskOutput
    for task_output in stage_1_task_outputs:
        assert len(task_output.model_output) == 1  # should only be one model output per task

        early_answering_tasks = get_early_answering_tasks(task_output, exp_dir)
        tasks_to_run.extend(early_answering_tasks)

    return tasks_to_run


def get_valid_stage2_formatters(formatters: list[str]):
    VALID_FORMATTERS = StageTwoFormatter.all_formatters()

    for formatter in formatters:
        if formatter not in VALID_FORMATTERS:
            raise ValueError(
                f"stage_two_formatter {formatter} is not valid. Valid formatters are {list(VALID_FORMATTERS.keys())}"
            )

    validated_formatters: list[Type[StageTwoFormatter]] = [VALID_FORMATTERS[formatter] for formatter in formatters]
    return validated_formatters


def main(
    input_exp_dir: str,
    models: list[str] = ["gpt-3.5-turbo", "gpt-4"],
    stage_one_formatters: list[str] = [ZeroShotCOTUnbiasedFormatter.name()],
    exp_dir: Optional[str] = None,
    experiment_suffix: str = "",
    save_file_every: int = 50,
    batch: int = 10,
):
    experiment_jsons = load_jsons(input_exp_dir)

    valid_stage_one_formatters = get_valid_stage1_formatters(stage_one_formatters)
    for formatter in valid_stage_one_formatters:
        if not formatter.is_cot:
            raise ValueError("stage two metrics only for cot")

    # create flat list of task outputs
    stage_1_task_outputs: List[TaskOutput] = []
    for experiment_json in experiment_jsons.values():
        stage_1_task_outputs.extend(experiment_json.outputs)

    exp_dir = get_exp_dir_name(exp_dir, experiment_suffix, sub_dir="stage_two")

    stage_2_tasks: List[TaskSpec] = create_stage_2_tasks(stage_1_task_outputs, exp_dir)

    # work out which tasks wwe have already done
    loaded_dict: dict[Path, ExperimentJsonFormat] = {}

    # get the counts of done experiments
    paths = {i.out_file_path for i in stage_2_tasks}
    for path in paths:
        if path.exists():
            loaded_dict[path] = read_done_experiment(path)
        else:
            loaded_dict[path] = ExperimentJsonFormat(outputs=[])

    # which ones have we already done
    for path, loaded_experiment_json in loaded_dict.items():
        task_output_hashes = [output.input_hash() for output in loaded_experiment_json.outputs]
        to_run = []
        for task_spec in stage_2_tasks:
            if task_spec not in task_output_hashes:
                to_run.append(task_spec)

    # run the tasks
    run_tasks_multi_threaded(save_file_every, batch=batch, loaded_dict=loaded_dict, tasks_to_run=stage_2_tasks)


if __name__ == "__main__":
    set_openai_key_from_env()
    fire.Fire(main)
