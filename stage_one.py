import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import fire
from tqdm import tqdm

from cot_transparency.miles_models import MilesBBHRawData, MilesBBHRawDataFolder
from cot_transparency.openai_utils.models import ChatMessages, OpenaiInferenceConfig
from cot_transparency.prompt_formatter import ZeroShotCOTSycophancyFormatter
from cot_transparency.stage_one_tasks import ExperimentJsonFormat, TaskOutput, TaskSpec, save_loaded_dict, task_function

BBH_TASK_LIST = [
    # "sports_understanding",
    # "snarks",
    # 'disambiguation_qa',
    # 'movie_recommendation',
    # 'causal_judgment',
    # 'date_understanding',
    # 'tracking_shuffled_objects_three_objects',
    # 'temporal_sequences',
    "ruin_names",
    # 'web_of_lies',
    # 'navigate',
    # 'logical_deduction_five_objects',
    # 'hyperbaton',
]

STANDARD_GPT4_CONFIG: OpenaiInferenceConfig = OpenaiInferenceConfig(
    model="gpt-4", temperature=0.7, max_tokens=1000, top_p=1.0
)


def main(
    tasks: list[str] = ["ruin_names"],
    models: list[str] = ["gpt-3.5-turbo", "gpt-4"],
    bias_type: str = "suggested_answer",
    exp_dir: Optional[str] = None,
    experiment_suffix: str = "",
    example_cap: Optional[int] = 5,
    log_metrics_every: int = 1,
    run_few_shot: bool = False,
    save_file_every: int = 10,
    batch: int = 10,
):
    # bbh is in data/bbh/task_name
    # read in the json file
    # data/bbh/{task_name}/val_data.json

    formatters: list[ZeroShotCOTSycophancyFormatter] = [ZeroShotCOTSycophancyFormatter()]

    loaded_dict: dict[Path, ExperimentJsonFormat] = {}

    # parse it into MilesBBHRawDataFolder
    # Create tasks
    tasks_to_run: list[TaskSpec] = []
    for bbh_task in tasks:
        json_path: Path = Path(f"data/bbh/{bbh_task}/val_data.json")
        with open(json_path, "r") as f:
            raw_data = json.load(f)
        data: list[MilesBBHRawData] = MilesBBHRawDataFolder(**raw_data).data
        if example_cap:
            data = data[:example_cap]
        for formatter in formatters:
            for model in models:
                out_file_path: Path = Path(f"experiments/{bbh_task}/{model}/{formatter.name}.json")
                # read in the json file
                if out_file_path.exists():
                    with open(out_file_path, "r") as f:
                        _dict = json.load(f)
                        already_done: ExperimentJsonFormat = ExperimentJsonFormat(**_dict)
                else:
                    already_done = ExperimentJsonFormat(outputs=[], task=bbh_task, model=model)
                loaded_dict.update({out_file_path: already_done})
                alreay_done_hashes = already_done.already_done_hashes()
                item: MilesBBHRawData
                for item in data:
                    task_hash = item.hash()
                    if task_hash not in alreay_done_hashes:
                        formatted: list[ChatMessages] = formatter.format_example(question=item)
                        config = STANDARD_GPT4_CONFIG.copy()
                        config.model = model
                        task_spec = TaskSpec(
                            model_config=STANDARD_GPT4_CONFIG,
                            messages=formatted,
                            out_file_path=out_file_path,
                            ground_truth=item.ground_truth,
                            formatter=formatter,
                            times_to_repeat=1,
                            task_hash=task_hash,
                        )
                        tasks_to_run.append(task_spec)

    future_instance_outputs = []
    # Actually run the tasks
    for task in tasks_to_run:
        executor = ThreadPoolExecutor(max_workers=batch)
        future_instance_outputs.append(executor.submit(task_function, task))
    for cnt, instance_output in tqdm(enumerate(as_completed(future_instance_outputs))):
        output: TaskOutput = instance_output.result()
        # extend the existing json file
        loaded_dict[output.out_file_path].outputs.append(output)
        if cnt % save_file_every == 0:
            save_loaded_dict(loaded_dict)
    save_loaded_dict(loaded_dict)


if __name__ == "__main__":
    fire.Fire(main)
