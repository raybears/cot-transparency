import random
from pathlib import Path
from typing import Optional, Type

import fire

from cot_transparency.data_models.data.bbh_biased_wrong_cot import BiasedWrongCOTBBH
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import OpenaiInferenceConfig, TaskSpec

from cot_transparency.formatters.base_class import StageOneFormatter

from cot_transparency.data_models.data import aqua, arc, bbh, truthful_qa, logiqa, mmlu, openbook, hellaswag
from cot_transparency.formatters.transparency.s1_baselines import FormattersForTransparency
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel
from cot_transparency.openai_utils.set_key import set_keys_from_env
from cot_transparency.formatters import (
    ZeroShotCOTSycophancyFormatter,
    ZeroShotCOTUnbiasedFormatter,
)
from cot_transparency.tasks import TaskSetting, run_with_caching
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

TASK_LIST = {
    "bbh": BBH_TASK_LIST,
    "bbh_biased_wrong_cot": BBH_TASK_LIST,
    "transparency": [
        "aqua",
        "arc_easy",
        "arc_challenge",
        "truthful_qa",
        "logiqa",
        "mmlu",
        "openbook_qa",
        "hellaswag",
    ],
}
CONFIG_MAP = {
    "gpt-4": OpenaiInferenceConfig(model="gpt-4", temperature=1, max_tokens=1000, top_p=1.0),
    "gpt-3.5-turbo": OpenaiInferenceConfig(model="gpt-3.5-turbo", temperature=1, max_tokens=1000, top_p=1.0),
    "text-davinci-003": OpenaiInferenceConfig(model="text-davinci-003", temperature=1, max_tokens=1000, top_p=1.0),
    "code-davinci-002": OpenaiInferenceConfig(model="code-davinci-002", temperature=1, max_tokens=1000, top_p=1.0),
    "text-davinci-002": OpenaiInferenceConfig(model="text-davinci-002", temperature=1, max_tokens=1000, top_p=1.0),
    "davinci": OpenaiInferenceConfig(model="davinci", temperature=1, max_tokens=1000, top_p=1.0),
    "claude-v1": OpenaiInferenceConfig(model="claude-v1", temperature=1, max_tokens=1000, top_p=1.0),
    "claude-2": OpenaiInferenceConfig(model="claude-1", temperature=1, max_tokens=1000, top_p=1.0),
    "gpt-3.5-turbo-16k": OpenaiInferenceConfig(model="gpt-3.5-turbo-16k", temperature=1, max_tokens=1000, top_p=1.0),
    "gpt-4-32k": OpenaiInferenceConfig(model="gpt-4-32k", temperature=1, max_tokens=1000, top_p=1.0),
    "llama-2-7b-chat-hf": OpenaiInferenceConfig(model="llama-2-7b-chat-hf", temperature=1, max_tokens=1000, top_p=1.0),
}


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


def validate_tasks(tasks: list[str]) -> list[str]:
    # get the tasks we are doing
    # flatten the TASK_LIST to get all tasks
    all_tasks = []
    for dataset in TASK_LIST:
        all_tasks += TASK_LIST[dataset]
    for task in tasks:
        if task not in all_tasks:
            raise ValueError(f"task {task} is not valid. Valid tasks are {all_tasks}")
    return tasks


def get_list_of_examples(task: str, dataset: Optional[str] = None) -> list[DataExampleBase]:
    data = None
    if dataset == "bbh_biased_wrong_cot":
        data = read_jsonl_file_into_basemodel(Path("data/bbh_biased_wrong_cot/data.jsonl"), BiasedWrongCOTBBH).filter(
            lambda x: x.task == task
        )
    elif task in TASK_LIST["bbh"]:
        data = bbh.load_bbh(task)
    else:
        if task == "aqua":
            data = aqua.dev()
        elif task == "arc_easy":
            data = arc.arc_easy_dev()
        elif task == "arc_challenge":
            data = arc.arc_challenge_dev()
        elif task == "truthful_qa":
            data = truthful_qa.eval()
        elif task == "logiqa":
            data = logiqa.eval()
        elif task == "mmlu":
            data = mmlu.test(questions_per_task=20)
        elif task == "openbook_qa":
            data = openbook.test()
        elif task == "hellaswag":
            data = hellaswag.val()

    if data is None:
        raise ValueError(f"dataset and or task is not valid. Valid datasets are {list(TASK_LIST.keys())}")
    return data  # type: ignore




def get_model_caller(models: list[str]) -> dict[str, ModelCaller]:
    pass


def main(
    tasks: Optional[list[str]] = None,
    dataset: Optional[str] = None,
    models: list[str] = ["gpt-3.5-turbo", "gpt-4"],
    formatters: list[str] = [ZeroShotCOTSycophancyFormatter.name(), ZeroShotCOTUnbiasedFormatter.name()],
    exp_dir: Optional[str] = None,
    experiment_suffix: str = "",
    example_cap: Optional[int] = 1000000,
    save_file_every: int = 50,
    batch: int = 10,
    repeats_per_question: int = 1,
    temperature: Optional[float] = None,
):
    if dataset is not None:
        assert tasks is None, "dataset and tasks are mutually exclusive"
        tasks = TASK_LIST[dataset]
    else:
        assert tasks is not None, "dataset and tasks are mutually exclusive"

    for model in models:
        if "llama" in model.lower():
            assert batch == 1, "Llama only supports batch size of 1"

    tasks = validate_tasks(tasks)
    validated_formatters = get_valid_stage1_formatters(formatters)

    exp_dir = get_exp_dir_name(exp_dir, experiment_suffix, sub_dir="stage_one")
    task_settings: list[TaskSetting] = create_task_settings(tasks=tasks, models=models, formatters=validated_formatters)

    tasks_to_run: list[TaskSpec] = []
    for setting in task_settings:
        task = setting.task
        model = setting.model
        formatter = setting.formatter
        data: list[DataExampleBase] = get_list_of_examples(task, dataset=dataset)
        out_file_path: Path = Path(f"{exp_dir}/{task}/{model}/{formatter.name()}.json")

        # Shuffle the data BEFORE we cap it
        random.Random(42).shuffle(data)
        if example_cap:
            data = data[:example_cap]

        # Possible config overrides
        config = CONFIG_MAP[model].copy()
        if issubclass(formatter, FormattersForTransparency):
            few_shot_stops = ["\n\nHuman:", "\n\nAssistant:", "\n\nQuestion:"]
            if isinstance(config.stop, list):
                config.stop += few_shot_stops
            else:
                config.stop = few_shot_stops
            config.max_tokens = 300
            config.temperature = 0.8
            config.top_p = 0.95
        if temperature is not None:
            print("Overriding temperature")
            config.temperature = temperature
        assert config.model == model
        if not formatter.is_cot:
            config.max_tokens = 3

        for item in data:
            for i in range(repeats_per_question):
                task_spec = TaskSpec(
                    task_name=task,
                    model_config=config,
                    messages=formatter.format_example(question=item),
                    out_file_path=out_file_path,
                    ground_truth=item.ground_truth,
                    formatter_name=formatter.name(),
                    task_hash=item.hash(),
                    biased_ans=item.biased_ans,
                    data_example=item.dict(),
                    repeat_idx=i,
                )
                tasks_to_run.append(task_spec)

    run_with_caching(save_every=save_file_every, batch=batch, task_to_run=tasks_to_run)


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
