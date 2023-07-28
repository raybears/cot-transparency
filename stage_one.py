from collections import Counter
import json
import random
from pathlib import Path
from typing import Optional, Type

import fire
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import OpenaiInferenceConfig, TaskSpec

from cot_transparency.formatters.base_class import StageOneFormatter

from cot_transparency.data_models.data import aqua, arc, bbh, truthful_qa
from cot_transparency.data_models.models import ChatMessage
from cot_transparency.openai_utils.set_key import set_keys_from_env
from cot_transparency.formatters import ZeroShotCOTSycophancyFormatter, ZeroShotCOTUnbiasedFormatter
from cot_transparency.data_models.models import ExperimentJsonFormat
from cot_transparency.tasks import TaskSetting
from cot_transparency.util import get_exp_dir_name, deterministic_hash_int
from cot_transparency.tasks import run_tasks_multi_threaded

TASK_LIST = {
    "bbh": [
        "sports_understanding",
        "snarks",
        "disambiguation_qa",
        "movie_recommendation",
        "causal_judgment",
        "date_understanding",
        "tracking_shuffled_objects_three_objects",
        "temporal_sequences",
        "ruin_names",
        "web_of_lies",
        "navigate",
        "logical_deduction_five_objects",
        "hyperbaton",
    ],
    "transparency": [
        "aqua",
        "arc_easy",
        "arc_challenge",
        "truthful_qa",
    ],
}
CONFIG_MAP = {
    "gpt-4": OpenaiInferenceConfig(model="gpt-4", temperature=1, max_tokens=1000, top_p=1.0),
    "gpt-3.5-turbo": OpenaiInferenceConfig(model="gpt-3.5", temperature=1, max_tokens=1000, top_p=1.0),
    "text-davinci-003": OpenaiInferenceConfig(model="text-davinci-003", temperature=0.7, max_tokens=1000, top_p=1.0),
    "claude-v1": OpenaiInferenceConfig(model="claude-v1", temperature=1, max_tokens=1000, top_p=1.0),
}


def read_done_experiment(out_file_path: Path) -> ExperimentJsonFormat:
    # read in the json file
    if out_file_path.exists():
        with open(out_file_path, "r") as f:
            _dict = json.load(f)
            return ExperimentJsonFormat(**_dict)
    else:
        return ExperimentJsonFormat(outputs=[])


def create_task_settings(
    tasks: list[str], models: list[str], formatters: list[Type[StageOneFormatter]]
) -> list[TaskSetting]:
    """Create a list of task settings to run"""
    task_settings = []
    for task in tasks:
        for model in models:
            for formatter in formatters:
                task_settings.append(TaskSetting(task=task, formatter=formatter, model=model))
    return task_settings


def validate_dataset_and_task(dataset: str, tasks: Optional[list[str]] = None) -> list[str]:
    # get the tasks we are doing
    if dataset not in TASK_LIST:
        raise ValueError(f"dataset {dataset} is not valid. Valid datasets are {list(TASK_LIST.keys())}")

    if tasks is None:
        tasks = TASK_LIST[dataset]
    else:
        for task in tasks:
            if task not in TASK_LIST[dataset]:
                raise ValueError(
                    f"task {task} is not valid for dataset {dataset}. Valid tasks are {TASK_LIST[dataset]}"
                )
    return tasks


def get_list_of_examples(dataset: str, task: str) -> list[DataExampleBase]:
    data = None
    if dataset == "bbh":
        data = bbh.load_bbh(task)
    elif dataset == "transparency":
        if task == "aqua":
            data = aqua.dev()
        elif task == "arc_easy":
            data = arc.arc_easy_dev()
        elif task == "arc_challenge":
            data = arc.arc_challenge_dev()
        elif task == "truthful_qa":
            data = truthful_qa.eval()

    if data is None:
        raise ValueError(f"dataset and or task is not valid. Valid datasets are {list(TASK_LIST.keys())}")
    return data  # type: ignore


def main(
    dataset: str = "bbh",
    tasks: Optional[list[str]] = None,
    models: list[str] = ["gpt-3.5-turbo", "gpt-4"],
    formatters: list[str] = [ZeroShotCOTSycophancyFormatter.name(), ZeroShotCOTUnbiasedFormatter.name()],
    exp_dir: Optional[str] = None,
    experiment_suffix: str = "",
    # for each model, task, and formatter, how many examples to generate
    example_cap: Optional[int] = 1000000,
    save_file_every: int = 50,
    batch: int = 10,
    repeats_per_question: int = 1,
    # 1 to 10
    subset: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
):
    tasks = validate_dataset_and_task(dataset, tasks)

    validated_formatters = get_valid_stage1_formatters(formatters)

    if dataset == "transparency":
        for formatter in validated_formatters:
            if formatter.is_biased:
                raise ValueError(
                    f"Formatter {formatter.name()} is biased. Transparency tasks should only use unbiased formatters."
                )

    loaded_dict: dict[Path, ExperimentJsonFormat] = {}
    exp_dir = get_exp_dir_name(exp_dir, experiment_suffix, sub_dir="stage_one")
    task_settings: list[TaskSetting] = create_task_settings(tasks=tasks, models=models, formatters=validated_formatters)
    # parse it into MilesBBHRawDataFolder
    # Create tasks
    tasks_to_run: list[TaskSpec] = []
    for setting in task_settings:
        task = setting.task
        model = setting.model
        formatter = setting.formatter
        data: list[DataExampleBase] = get_list_of_examples(dataset, task)

        # Shuffle the data BEFORE we cap it
        random.Random(42).shuffle(data)

        filtered_data = []
        for item in data:
            hash_bucket = deterministic_hash_int(item.hash()) % 10
            if hash_bucket in subset:
                filtered_data.append(item)

        if example_cap:
            filtered_data = filtered_data[:example_cap]

        out_file_path: Path = Path(f"{exp_dir}/{task}/{model}/{formatter.name()}.json")
        already_done = read_done_experiment(out_file_path)
        loaded_dict.update({out_file_path: already_done})
        already_done_hashes_counts = Counter([o.task_spec.task_hash for o in already_done.outputs])
        for item in filtered_data:
            task_hash: str = item.hash()
            if task_hash not in already_done_hashes_counts:
                runs_to_do = repeats_per_question
            else:
                # may be negative
                runs_to_do = max(repeats_per_question - already_done_hashes_counts[task_hash], 0)

            formatted: list[ChatMessage] = formatter.format_example(question=item)
            config = CONFIG_MAP[model].copy()
            config.model = model
            if not formatter.is_cot:
                config.max_tokens = 1
            task_spec = TaskSpec(
                task_name=task,
                model_config=config,
                messages=formatted,
                out_file_path=out_file_path,
                ground_truth=item.ground_truth,
                formatter_name=formatter.name(),
                task_hash=task_hash,
                biased_ans=item.biased_ans,
                data_example=item.dict(),
            )
            for _ in range(runs_to_do):
                tasks_to_run.append(task_spec)

    # Shuffle so we distribute API calls evenly among models
    random.Random(42).shuffle(tasks_to_run)

    if len(tasks_to_run) == 0:
        print("No tasks to run, experiment is already done.")
        return

    run_tasks_multi_threaded(save_file_every, batch, loaded_dict, tasks_to_run)


def get_valid_stage1_formatters(formatters: list[str]) -> list[Type[StageOneFormatter]]:
    VALID_FORMATTERS = StageOneFormatter.all_formatters()

    # assert that the formatters are valid
    for formatter in formatters:
        if formatter not in VALID_FORMATTERS:
            raise ValueError(
                f"formatter {formatter} is not valid. Valid formatters are {list(VALID_FORMATTERS.keys())}"
            )

    validated_formatters: list[Type[StageOneFormatter]] = [VALID_FORMATTERS[formatter] for formatter in formatters]
    return validated_formatters


if __name__ == "__main__":
    set_keys_from_env()
    fire.Fire(main)
