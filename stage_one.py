from collections import Counter
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Type

import fire
from pydantic import ValidationError, BaseModel
from tqdm import tqdm

from cot_transparency.miles_models import MilesBBHRawData, MilesBBHRawDataFolder
from cot_transparency.openai_utils.models import ChatMessages, OpenaiInferenceConfig
from cot_transparency.openai_utils.set_key import set_openai_key_from_env
from cot_transparency.formatters import ZeroShotCOTSycophancyFormatter, PromptFormatter, ZeroShotCOTUnbiasedFormatter
from cot_transparency.stage_one_tasks import ExperimentJsonFormat, TaskOutput, TaskSpec, save_loaded_dict, task_function
from cot_transparency.util import get_exp_dir_name

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

STANDARD_GPT4_CONFIG: OpenaiInferenceConfig = OpenaiInferenceConfig(
    model="gpt-4", temperature=0.7, max_tokens=1000, top_p=1.0
)


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
    formatter: Type[PromptFormatter]
    model: str


def create_task_settings(
    tasks: list[str], models: list[str], formatters: list[Type[PromptFormatter]]
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
    VALID_FORMATTERS = PromptFormatter.all_formatters()

    # assert that the formatters are valid
    for formatter in formatters:
        if formatter not in VALID_FORMATTERS:
            raise ValueError(
                f"formatter {formatter} is not valid. Valid formatters are {list(VALID_FORMATTERS.keys())}"
            )

    validated_formatters: list[Type[PromptFormatter]] = [VALID_FORMATTERS[formatter] for formatter in formatters]

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
        if example_cap:
            data = data[:example_cap]

        random.Random(42).shuffle(data)
        out_file_path: Path = Path(f"{exp_dir}/{bbh_task}/{model}/{formatter.name()}.json")
        already_done = read_done_experiment(out_file_path)
        loaded_dict.update({out_file_path: already_done})
        already_done_hashes_counts = Counter(already_done.already_done_hashes())
        item: MilesBBHRawData
        for item in data:
            task_hash = item.hash()
            if task_hash not in already_done_hashes_counts:
                runs_to_do = repeats_per_question
            else:
                runs_to_do = repeats_per_question - already_done_hashes_counts[task_hash]

            formatted: list[ChatMessages] = formatter.format_example(question=item)
            config = STANDARD_GPT4_CONFIG.copy()
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

    future_instance_outputs = []

    executor = ThreadPoolExecutor(max_workers=batch)

    def kill_and_save(loaded_dict):
        for future in future_instance_outputs:
            future.cancel()
        executor.shutdown(wait=False)
        save_loaded_dict(loaded_dict)

    for task in tasks_to_run:
        future_instance_outputs.append(executor.submit(task_function, task))

    try:
        for cnt, instance_output in tqdm(
            enumerate(as_completed(future_instance_outputs)), total=len(future_instance_outputs)
        ):
            output: TaskOutput = instance_output.result()
            # extend the existing json file
            loaded_dict[output.out_file_path].outputs.append(output)
            if cnt % save_file_every == 0:
                save_loaded_dict(loaded_dict)

    except Exception as e:
        kill_and_save(loaded_dict)
        raise e

    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, please wait while running tasks finish...")
        kill_and_save(loaded_dict)
        exit(1)

    save_loaded_dict(loaded_dict)


if __name__ == "__main__":
    set_openai_key_from_env()
    fire.Fire(main)
