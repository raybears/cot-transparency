import json
from pathlib import Path
from typing import List, Optional, Type

import fire
from cot_transparency.data_models.io import ExpLoader, save_loaded_dict
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.transparency import EarlyAnsweringFormatter, StageTwoFormatter
from cot_transparency.formatters.transparency.trace_manipulation import get_cot_steps
from cot_transparency.data_models.models import (
    ExperimentJsonFormat,
    StageTwoExperimentJsonFormat,
    StageTwoTaskOutput,
    StageTwoTaskSpec,
    TaskOutput,
)
from cot_transparency.openai_utils.set_key import set_openai_key_from_env

from cot_transparency.util import get_exp_dir_name
from stage_one import get_valid_stage1_formatters

"""
We take traces generated from stage_one.py and run analysis on them
"""


def get_early_answering_tasks(
    stage_one_output: TaskOutput, exp_dir: str, temperature: Optional[float] = None
) -> List[StageTwoTaskSpec]:
    outputs = []

    cot_steps = get_cot_steps(stage_one_output.first_raw_response)

    partial_cot = ""
    for i, cot_step in enumerate(cot_steps):
        partial_cot += cot_step

        config = stage_one_output.task_spec.model_config.copy()
        out_file_path: Path = Path(
            f"{exp_dir}/{stage_one_output.task_spec.task_name}/{config.model}/{EarlyAnsweringFormatter.name()}.json"
        )

        if temperature is not None:
            config.temperature = temperature
        config.max_tokens = 1

        # messages / prompt for stage_two
        messages = EarlyAnsweringFormatter.format_example(stage_one_output.task_spec.messages, partial_cot)

        task = StageTwoTaskSpec(
            stage_one_output=stage_one_output,
            model_config=config,
            formatter_name=EarlyAnsweringFormatter.name(),
            messages=messages,
            out_file_path=out_file_path,
            step_in_cot_trace=i,
        )

        outputs.append(task)
    return outputs


def create_stage_2_tasks(
    stage_1_task_outputs: List[TaskOutput],
    exp_dir: str,
    temperature: Optional[float] = None,
) -> List[StageTwoTaskSpec]:
    tasks_to_run: List[StageTwoTaskSpec] = []

    task_output: TaskOutput
    for task_output in stage_1_task_outputs:
        early_answering_tasks = get_early_answering_tasks(task_output, exp_dir, temperature=temperature)
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


def filter_stage1_outputs(
    stage1_outputs: dict[Path, ExperimentJsonFormat],
    stage_one_formatters: list[Type[StageOneFormatter]],
    models: list[str],
) -> dict[Path, ExperimentJsonFormat]:
    outputs: dict[Path, ExperimentJsonFormat] = {}
    for k, exp_json in stage1_outputs.items():
        # only need to check the first example as all examples in the same experiment have the same formatter and model
        if len(exp_json.outputs) == 0:
            continue
        task_spec = exp_json.outputs[0].task_spec
        if task_spec.formatter_name in [i.name() for i in stage_one_formatters]:
            if task_spec.model_config.model in models:
                outputs[k] = exp_json
    return outputs


def main(
    input_exp_dir: str,
    models: list[str] = ["text-davinci-003"],
    stage_one_formatters: Optional[list[str]] = None,
    exp_dir: Optional[str] = None,
    experiment_suffix: str = "",
    save_file_every: int = 50,
    batch: int = 1,
    temperature: float = 0.0,
    example_cap: int = 999999999,
):
    if stage_one_formatters is None:
        # don't do any filtering, just use all stage one outputs
        all_formatters = StageOneFormatter.all_formatters().values()
        stage_one_formatters = [i.name() for i in all_formatters if i.is_cot]  # type: ignore
    valid_stage_one_formatters = get_valid_stage1_formatters(stage_one_formatters)
    for formatter in valid_stage_one_formatters:
        if not formatter.is_cot:
            raise ValueError("stage two metrics only make sense for COT stage_one_formatters")

    experiment_jsons: dict[Path, ExperimentJsonFormat] = ExpLoader.stage_one(input_exp_dir)
    experiment_jsons = filter_stage1_outputs(experiment_jsons, valid_stage_one_formatters, models)
    if len(experiment_jsons) == 0:
        print("No matching data from stage one found, nothing to run")
        exit(1)
    else:
        print(f"Found {len(experiment_jsons)} matching experiments from stage one")

    exp_dir = get_exp_dir_name(exp_dir, experiment_suffix, sub_dir="stage_two")

    # symlink the stage one experiments (input_exp_dir) into stage_two exp_dir
    # as stage_one_exp_dir
    # so we can easily see what stage one experiments were used to generate stage two
    stage_one_exp_dir = Path(f"{exp_dir}/stage_one_exp_dir")
    if not stage_one_exp_dir.exists():
        Path(exp_dir).mkdir(parents=True, exist_ok=True)
        stage_one_exp_dir.symlink_to(Path(input_exp_dir).absolute())
    else:
        assert stage_one_exp_dir.resolve() == Path(input_exp_dir).absolute()

    # create flat list of task outputs
    stage_2_tasks: List[StageTwoTaskSpec] = []
    for experiment_json in experiment_jsons.values():
        stage_2_tasks_for_this_json = create_stage_2_tasks(experiment_json.outputs, exp_dir, temperature=temperature)
        # we example cap here if required
        stage_2_tasks_for_this_json = stage_2_tasks_for_this_json[:example_cap]
        stage_2_tasks.extend(stage_2_tasks_for_this_json)

    # work out which tasks wwe have already done
    loaded_dict: dict[Path, StageTwoExperimentJsonFormat] = {}

    # get the counts of done experiments
    paths = {i.out_file_path for i in stage_2_tasks}
    completed_outputs: dict[str, StageTwoTaskOutput] = {}
    for path in paths:
        if path.exists():
            done_exp = StageTwoExperimentJsonFormat(**json.load(open(path)))
            done_exp.stage = 2
            loaded_dict[path] = done_exp
            for output in loaded_dict[path].outputs:
                completed_outputs[output.task_spec.uid()] = output
        else:
            loaded_dict[path] = StageTwoExperimentJsonFormat(outputs=[])

    to_run = []
    for task_spec in stage_2_tasks:
        if task_spec.uid() not in completed_outputs:
            to_run.append(task_spec)
        else:
            # you can modify already done TaskOutputs here if you need to change
            # already done experiments, e.g.
            # completed_outputs[task_spec.input_hash()].stage_one_hash = task_spec.stage_one_hash
            # will need to run save_loaded_dict(loaded_dict) after this
            pass

    run_tasks_multi_threaded(save_file_every, batch=batch, loaded_dict=loaded_dict, tasks_to_run=to_run)
    # save_loaded_dict(loaded_dict)


if __name__ == "__main__":
    set_openai_key_from_env()
    fire.Fire(main)
