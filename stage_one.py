import typing
from pathlib import Path
from typing import Literal, Optional, Sequence, Type

import fire
from slist import Slist

from cot_transparency.apis.openai.set_key import set_keys_from_env
from cot_transparency.data_models.config import config_from_default
from cot_transparency.data_models.data import TASK_LIST
from cot_transparency.data_models.data import get_list_of_examples
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.models import TaskSpec
from cot_transparency.formatters import (
    ZeroShotCOTSycophancyFormatter,
    ZeroShotCOTUnbiasedFormatter,
)
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.instructions import FEW_SHOT_STOP_TOKEN
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.formatters.interventions.valid_interventions import (
    get_valid_stage1_interventions,
)
from cot_transparency.formatters.transparency.s1_baselines import (
    FormattersForTransparency,
)
from cot_transparency.formatters.wildcard import match_wildcard_formatters
from cot_transparency.tasks import TaskSetting, run_with_caching
from cot_transparency.util import get_exp_dir_name


def create_task_settings(
    tasks: Sequence[str],
    models: Sequence[str],
    formatters: list[Type[StageOneFormatter]],
    # see cot_transparency/formatters/interventions/valid_interventions.py for valid interventions
    interventions: Sequence[Type[Intervention] | None],
) -> list[TaskSetting]:
    """Create a list of task settings to run"""
    task_settings = []
    for task in tasks:
        for model in models:
            for formatter in formatters:
                task_settings.append(TaskSetting(task=task, formatter=formatter, model=model))
    with_interventions = []
    for setting in task_settings:
        for intervention in interventions:
            with_interventions.append(
                TaskSetting(
                    task=setting.task,
                    formatter=setting.formatter,
                    model=setting.model,
                    intervention=intervention,
                )
            )
    if len(with_interventions) > 0:
        return with_interventions
    else:
        return task_settings


def validate_tasks(tasks: Sequence[str]) -> Sequence[str]:
    # get the tasks we are doing
    # flatten the TASK_LIST to get all tasks
    all_tasks = []
    for dataset in TASK_LIST:
        all_tasks += TASK_LIST[dataset]
    for task in tasks:
        if task not in all_tasks:
            raise ValueError(f"task {task} is not valid. Valid tasks are {all_tasks}")
    return tasks


def create_stage_one_task_specs(
    tasks: Sequence[str] = [],
    dataset: Optional[str] = None,
    models: Sequence[str] = ["gpt-3.5-turbo", "gpt-4"],
    formatters: Sequence[str] = [ZeroShotCOTSycophancyFormatter.name(), ZeroShotCOTUnbiasedFormatter.name()],
    # Pass in a list of interventions to run, indicate None to run no intervention as well
    interventions: Sequence[str | None] = [],
    exp_dir: Optional[str] = None,
    experiment_suffix: str = "",
    example_cap: Optional[int | int] = 1000000,
    formatter_example_cap_override: Optional[dict[type[StageOneFormatter], int]] = None,
    batch: int = 20,
    repeats_per_question: int = 1,
    temperature: Optional[float] = None,
    raise_after_retries: bool = True,
    max_tokens: Optional[int] = None,
    n_responses_per_request: Optional[int] = None,
) -> Sequence[TaskSpec]:
    if dataset is not None:
        # we are using a dataset
        assert len(tasks) == 0, "You have defined a dataset and a task, you can only define one"
        tasks = TASK_LIST[dataset]
    else:
        assert tasks, "You must define a task or a dataset"

    for model in models:
        if "llama" in model.lower():
            assert batch == 1, "Llama only supports batch size of 1"
    print("Number of models to run:", len(models))

    # match formatter name wildcard
    formatters = match_wildcard_formatters(formatters)

    assert len(formatters) > 0, "You must define at least one formatter"

    tasks = validate_tasks(tasks)
    print("Number of tasks to run:", len(tasks))
    validated_formatters = get_valid_stage1_formatters(formatters)
    print("Number of formatters to run:", len(validated_formatters))
    validated_interventions = get_valid_stage1_interventions(interventions)
    print("Number of interventions to run:", len(validated_interventions))
    exp_dir = get_exp_dir_name(exp_dir, experiment_suffix, sub_dir="stage_one")

    task_settings: list[TaskSetting] = create_task_settings(
        tasks=tasks, models=models, formatters=validated_formatters, interventions=validated_interventions
    )

    tasks_to_run: list[TaskSpec] = []
    print("Number of settings to run:", len(task_settings))
    for setting in task_settings:
        task = setting.task
        model = setting.model
        formatter = setting.formatter
        # Shuffle the data BEFORE we cap it
        # Pass 42 to maintain the same shuffle that we had in the past, though slist wants a string instead
        data: Slist[DataExampleBase] = get_list_of_examples(task, dataset=dataset).shuffle(typing.cast(str, 42))
        out_file_path: Path = (
            Path(f"{exp_dir}/{task}/{model}/{formatter.name()}.json")
            if setting.intervention is None
            else Path(f"{exp_dir}/{task}/{model}/{formatter.name()}_and_{setting.intervention.name()}.json")
        )

        if example_cap:
            if formatter_example_cap_override is not None and formatter in formatter_example_cap_override:
                pass  # we will cap later
            else:
                data = data.take(example_cap)

        if formatter_example_cap_override is not None and formatter in formatter_example_cap_override:
            data = data.take(formatter_example_cap_override[formatter])

        # Config Overrides Start ----------------------
        config = config_from_default(model, temperature=temperature, max_tokens=max_tokens)
        if issubclass(formatter, FormattersForTransparency):
            few_shot_stops = ["\n\nHuman:", "\n\nAssistant:", "\n\nQuestion:"]
            if isinstance(config.stop, list):
                config = config.copy_update(stop=config.stop + few_shot_stops)
            else:
                config = config.copy_update(stop=few_shot_stops)
            config = config.copy_update(max_tokens=300, temperature=0.8, top_p=0.95)

        # if you are using an intervention, we need to add SINGLE_SHOT_SEP to the stop list
        if setting.intervention:
            if isinstance(config.stop, list):
                config.stop += [FEW_SHOT_STOP_TOKEN]
                config = config.copy_update(stop=config.stop + [FEW_SHOT_STOP_TOKEN])
            else:
                config = config.copy_update(stop=[FEW_SHOT_STOP_TOKEN])

        assert config.model == model

        if not formatter.is_cot:
            config = config.copy_update(max_tokens=1)

        if n_responses_per_request is not None:
            config = config.copy_update(n=n_responses_per_request)

        # Config Overrides End ----------------------

        if raise_after_retries and temperature == 0:
            raise ValueError("Must set --raise_after_retires=False when temperature is 0 as it will always fail")

        for item in data:
            for i in range(repeats_per_question):
                messages = (
                    setting.intervention.intervene(question=item, formatter=formatter, model=model)
                    if setting.intervention
                    else formatter.format_example(question=item, model=model)
                )
                # Save the format spec defined by the formatter
                new_item: DataExampleBase = item.model_copy()
                format_spec = formatter.get_data_format_spec()
                new_item.data_format = format_spec
                task_spec = TaskSpec(
                    task_name=task,
                    inference_config=config,
                    messages=messages,
                    out_file_path=out_file_path,
                    ground_truth=new_item.ground_truth,  # type: ignore
                    formatter_name=formatter.name(),
                    task_hash=new_item.hash(),
                    biased_ans=new_item.biased_ans,
                    data_example=new_item.model_dump(),
                    repeat_idx=i,
                    intervention_name=setting.intervention.name() if setting.intervention else None,
                )
                tasks_to_run.append(task_spec)

    return tasks_to_run


def main(
    tasks: Sequence[str] = [],
    dataset: Optional[str] = None,
    models: Sequence[str] = ["gpt-3.5-turbo", "gpt-4"],
    formatters: Sequence[str] = [
        ZeroShotCOTSycophancyFormatter.name(),
        ZeroShotCOTUnbiasedFormatter.name(),
    ],
    # Pass in a list of interventions to run, indicate None to run no intervention as well
    interventions: Sequence[str | None] = [],
    exp_dir: Optional[str] = None,
    experiment_suffix: str = "",
    example_cap: Optional[int] = 1000000,
    save_file_every: int = 50,
    batch: int = 20,
    repeats_per_question: int = 1,
    temperature: Optional[float] = None,
    raise_after_retries: bool = True,
    raise_on: Literal["all", "any"] = "all",
    num_tries: int = 10,
    max_tokens: Optional[int] = None,
    n_responses_per_request: Optional[int] = None,
    retry_answers_with_none: bool = False,
):
    tasks_to_run = create_stage_one_task_specs(
        tasks=tasks,
        dataset=dataset,
        models=models,
        formatters=formatters,
        interventions=interventions,
        exp_dir=exp_dir,
        experiment_suffix=experiment_suffix,
        example_cap=example_cap,
        batch=batch,
        repeats_per_question=repeats_per_question,
        temperature=temperature,
        raise_after_retries=raise_after_retries,
        max_tokens=max_tokens,
        n_responses_per_request=n_responses_per_request,
    )

    run_with_caching(
        save_every=save_file_every,
        batch=batch,
        task_to_run=tasks_to_run,
        raise_after_retries=raise_after_retries,
        raise_on=raise_on,
        num_tries=num_tries,
        retry_answers_with_none=retry_answers_with_none,
    )


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
