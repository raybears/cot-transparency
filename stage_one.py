import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Type
import random

import fire
from pydantic import ValidationError
from tqdm import tqdm

from cot_transparency.miles_models import MilesBBHRawData, MilesBBHRawDataFolder
from cot_transparency.openai_utils.models import ChatMessages, OpenaiInferenceConfig
from cot_transparency.openai_utils.set_key import set_openai_key_from_env
from cot_transparency.prompt_formatter import (
    PromptFormatter,
    ZeroShotCOTSycophancyFormatter,
    ZeroShotCOTUnbiasedFormatter,
)
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


def main(
    tasks: list[str] = BBH_TASK_LIST,
    models: list[str] = ["gpt-3.5-turbo", "gpt-4"],
    formatters: list[str] = [ZeroShotCOTSycophancyFormatter.name(), ZeroShotCOTUnbiasedFormatter.name()],
    exp_dir: Optional[str] = None,
    experiment_suffix: str = "",
    example_cap: Optional[int] = 1000000,
    run_few_shot: bool = False,
    save_file_every: int = 10,
    batch: int = 10,
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

    # parse it into MilesBBHRawDataFolder
    # Create tasks
    tasks_to_run: list[TaskSpec] = []
    for bbh_task in tasks:
        json_path: Path = Path(f"data/bbh/{bbh_task}/val_data.json")
        with open(json_path, "r") as f:
            raw_data = json.load(f)
        try:
            data: list[MilesBBHRawData] = MilesBBHRawDataFolder(**raw_data).data
        except ValidationError as e:
            print(f"Error parsing {json_path}")
            raise e
        if example_cap:
            data = data[:example_cap]
        # shuffle the data
        random.seed(42)
        random.shuffle(data)
        for formatter in validated_formatters:
            for model in models:
                out_file_path: Path = Path(f"{exp_dir}/{bbh_task}/{model}/{formatter.name()}.json")
                # read in the json file
                if out_file_path.exists():
                    with open(out_file_path, "r") as f:
                        _dict = json.load(f)
                        already_done: ExperimentJsonFormat = ExperimentJsonFormat(**_dict)
                else:
                    already_done = ExperimentJsonFormat(outputs=[])
                loaded_dict.update({out_file_path: already_done})
                alreay_done_hashes = already_done.already_done_hashes()
                item: MilesBBHRawData
                for item in data:
                    task_hash = item.hash()
                    if task_hash not in alreay_done_hashes:
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
                            times_to_repeat=1,
                            task_hash=task_hash,
                            biased_ans=item.biased_ans,
                        )
                        tasks_to_run.append(task_spec)

    if len(tasks_to_run) == 0:
        print("No tasks to run, experiment is already done.")
        return

    future_instance_outputs = []
    # Actually run the tasks
    with ThreadPoolExecutor(max_workers=batch) as executor:
        for task in tasks_to_run:
            future_instance_outputs.append(executor.submit(task_function, task))

        for cnt, instance_output in tqdm(
            enumerate(as_completed(future_instance_outputs)), total=len(future_instance_outputs)
        ):
            try:
                output: TaskOutput = instance_output.result()
            except Exception as e:
                # kill all future tasks
                for future in future_instance_outputs:
                    future.cancel()
                # save the loaded dict
                save_loaded_dict(loaded_dict)
                raise e
            # extend the existing json file
            loaded_dict[output.out_file_path].outputs.append(output)
            if cnt % save_file_every == 0:
                save_loaded_dict(loaded_dict)
    save_loaded_dict(loaded_dict)


if __name__ == "__main__":
    set_openai_key_from_env()
    fire.Fire(main)
