from collections import Counter
import json
import random
from pathlib import Path
from typing import Optional, Type

import fire
from pydantic import ValidationError, BaseModel
from cot_transparency.formatters.base_class import StageOneFormatter

from cot_transparency.miles_models import MilesBBHRawData, MilesBBHRawDataFolder
from cot_transparency.openai_utils.models import ChatMessages, OpenaiInferenceConfig
from cot_transparency.openai_utils.set_key import set_openai_key_from_env
from cot_transparency.formatters import ZeroShotCOTSycophancyFormatter, ZeroShotCOTUnbiasedFormatter
from cot_transparency.tasks import ExperimentJsonFormat, TaskSpec
from cot_transparency.util import get_exp_dir_name
from cot_transparency.tasks import run_tasks_multi_threaded

BBH_TASK_LIST = [
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
]
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


class TaskSetting(BaseModel):
    task: str
    formatter: Type[StageOneFormatter]
    model: str


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


def main(
    tasks: list[str] = BBH_TASK_LIST,
    models: list[str] = ["gpt-3.5-turbo", "gpt-4"],
    formatters: list[str] = [ZeroShotCOTSycophancyFormatter.name(), ZeroShotCOTUnbiasedFormatter.name()],
    exp_dir: Optional[str] = None,
    experiment_suffix: str = "",
    example_cap: Optional[int] = 1000000,
    save_file_every: int = 50,
    batch: int = 10,
    repeats_per_question: int = 1,
):
    # bbh is in data/bbh/task_name
    # read in the json file
    # data/bbh/{task_name}/val_data.json
    validated_formatters = get_valid_stage1_formatters(formatters)

    loaded_dict: dict[Path, ExperimentJsonFormat] = {}
    exp_dir = get_exp_dir_name(exp_dir, experiment_suffix, sub_dir="stage_one")
    task_settings: list[TaskSetting] = create_task_settings(tasks=tasks, models=models, formatters=validated_formatters)
    # parse it into MilesBBHRawDataFolder
    # Create tasks
    tasks_to_run: list[TaskSpec] = []
    for setting in task_settings:
        bbh_task = setting.task
        model = setting.model
        formatter = setting.formatter
        json_path: Path = Path(f"data/bbh/{setting.task}/val_data.json")
        with open(json_path, "r") as f:
            raw_data = json.load(f)
        try:
            data: list[MilesBBHRawData] = MilesBBHRawDataFolder(**raw_data).data
        except ValidationError as e:
            print(f"Error parsing {json_path}")
            raise e

        # Shuffle the data BEFORE we cap it
        random.Random(42).shuffle(data)
        if example_cap:
            data = data[:example_cap]

        out_file_path: Path = Path(f"{exp_dir}/{bbh_task}/{model}/{formatter.name()}.json")
        already_done = read_done_experiment(out_file_path)
        loaded_dict.update({out_file_path: already_done})
        already_done_hashes_counts = Counter([o.task_hash for o in already_done.outputs])
        item: MilesBBHRawData
        for item in data:
            task_hash = item.hash()
            if task_hash not in already_done_hashes_counts:
                runs_to_do = repeats_per_question
            else:
                # may be negative
                runs_to_do = max(repeats_per_question - already_done_hashes_counts[task_hash], 0)

            formatted: list[ChatMessages] = formatter.format_example(question=item)
            config = CONFIG_MAP[model].copy()
            config.model = model
            if not formatter.is_cot:
                config.max_tokens = 1
            task_spec = TaskSpec(
                task_name=bbh_task,
                model_config=config,
                messages=formatted,
                out_file_path=out_file_path,
                ground_truth=item.ground_truth,
                formatter=formatter,
                task_hash=task_hash,
                biased_ans=item.biased_ans,
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
    set_openai_key_from_env()
    fire.Fire(main)
