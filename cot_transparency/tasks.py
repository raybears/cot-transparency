from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
from typing import Literal, Type, Union, Optional

from pydantic import BaseModel
from retry import retry
from tqdm import tqdm
from cot_transparency.apis.base import InferenceResponse
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.io import LoadedJsonType, save_loaded_dict
from cot_transparency.data_models.models import (
    ExperimentJsonFormat,
    StageTwoExperimentJsonFormat,
    StageTwoTaskOutput,
    ModelOutput,
)
from cot_transparency.formatters.interventions.intervention import Intervention

from cot_transparency.apis import call_model_api
from cot_transparency.formatters import PromptFormatter, name_to_formatter, StageOneFormatter
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.data_models.models import StageTwoTaskSpec
from cot_transparency.data_models.models import TaskSpec
from cot_transparency.util import setup_logger
from cot_transparency.apis.rate_limiting import exit_event

logger = setup_logger(__name__)


class AtLeastOneFailed(Exception):
    def __init__(self, e: str, model_outputs: list[ModelOutput]):
        self.e = e
        self.model_outputs = model_outputs


class AnswerNotFound(Exception):
    def __init__(self, e: str, model_output: ModelOutput):
        self.e = e
        self.model_output = model_output


def __call_or_raise(
    task: Union[TaskSpec, StageTwoTaskSpec],
    config: OpenaiInferenceConfig,
    formatter: Type[PromptFormatter],
    raise_on: Union[Literal["all"], Literal["any"]] = "any",
) -> list[ModelOutput]:
    if isinstance(task, StageTwoTaskSpec):
        stage_one_task_spec = task.stage_one_output.task_spec
    else:
        stage_one_task_spec = task

    raw_responses: InferenceResponse = call_model_api(task.messages, config)

    def get_model_output_for_response(raw_response: str) -> Union[ModelOutput, AnswerNotFound]:
        parsed_response: str | None = formatter.parse_answer(
            raw_response,
            model=config.model,
            question=stage_one_task_spec.get_data_example_obj(),
        )
        if parsed_response is not None:
            return ModelOutput(raw_response=raw_response, parsed_response=parsed_response)
        else:
            messages = task.messages
            maybe_second_last = messages[-2] if len(messages) >= 2 else None
            msg = (
                f"Formatter: {formatter.name()}, Model: {config.model}, didnt find answer in model answer:"
                f"\n\n'{raw_response}'\n\n'last two messages were:\n{maybe_second_last}\n\n{messages[-1]}"
            )
            logger.warning(msg)
            model_output = ModelOutput(raw_response=raw_response, parsed_response=None)
            return AnswerNotFound(msg, model_output)

    outputs = [get_model_output_for_response(response) for response in raw_responses.raw_responses]
    failed_examples = [o for o in outputs if isinstance(o, AnswerNotFound)]

    match raise_on:
        case "any":
            should_raise = len(failed_examples) > 0
        case "all":
            should_raise = len(failed_examples) == len(outputs)

    model_outputs: list[ModelOutput] = []

    for o in outputs:
        if isinstance(o, AnswerNotFound):
            model_outputs.append(o.model_output)
        elif isinstance(o, ModelOutput):
            model_outputs.append(o)

    if should_raise:
        raise AtLeastOneFailed(
            f"{len(failed_examples)} of the model outputs were None",
            model_outputs,
        )

    return model_outputs


def call_model_and_raise_if_not_suitable(
    task: Union[TaskSpec, StageTwoTaskSpec],
    config: OpenaiInferenceConfig,
    formatter: Type[PromptFormatter],
    retries: int = 20,
    raise_on: Union[Literal["all"], Literal["any"]] = "any",
) -> list[ModelOutput]:
    responses = retry(exceptions=AtLeastOneFailed, tries=retries)(__call_or_raise)(
        task=task, config=config, formatter=formatter, raise_on=raise_on
    )
    return responses


def call_model_and_catch(
    task: Union[TaskSpec, StageTwoTaskSpec],
    config: OpenaiInferenceConfig,
    formatter: Type[PromptFormatter],
    retries: int = 20,
    raise_on: Union[Literal["all"], Literal["any"]] = "any",
) -> list[ModelOutput]:
    try:
        return call_model_and_raise_if_not_suitable(
            task=task, config=config, formatter=formatter, retries=retries, raise_on=raise_on
        )
    except AtLeastOneFailed as e:
        return e.model_outputs


def task_function(
    task: Union[TaskSpec, StageTwoTaskSpec],
    raise_after_retries: bool,
    raise_on: Union[Literal["all"], Literal["any"]] = "any",
    num_retries: int = 10,
) -> Union[list[TaskOutput], list[StageTwoTaskOutput]]:
    formatter = name_to_formatter(task.formatter_name)

    responses = (
        call_model_and_raise_if_not_suitable(
            task=task,
            config=task.inference_config,
            formatter=formatter,
            retries=num_retries,
            raise_on=raise_on,
        )
        if raise_after_retries
        else call_model_and_catch(
            task=task,
            config=task.inference_config,
            formatter=formatter,
            retries=num_retries,
            raise_on=raise_on,
        )
    )

    if isinstance(task, StageTwoTaskSpec):
        output_class = StageTwoTaskOutput
        outputs = [
            output_class(
                task_spec=task,
                inference_output=responses[i],
                response_idx=i,
            )
            for i in range(len(responses))
        ]
    elif isinstance(task, TaskSpec):
        output_class = TaskOutput
        outputs = [
            output_class(
                task_spec=task,
                inference_output=responses[i],
                response_idx=i,
            )
            for i in range(len(responses))
        ]
    else:
        raise ValueError(f"Unknown task type {type(task)}")

    return outputs


def get_loaded_dict_stage2(paths: set[Path]) -> dict[Path, StageTwoExperimentJsonFormat]:
    # work out which tasks we have already done
    loaded_dict: dict[Path, StageTwoExperimentJsonFormat] = {}
    for path in paths:
        if path.exists():
            with open(path) as f:
                done_exp = StageTwoExperimentJsonFormat(**json.load(f))
            # Override to ensure bwds compat with some old exps that had the wrong stage
            done_exp.stage = 2
            loaded_dict[path] = done_exp
        else:
            loaded_dict[path] = StageTwoExperimentJsonFormat(outputs=[])
    return loaded_dict


def read_done_experiment(out_file_path: Path) -> ExperimentJsonFormat:
    # read in the json file
    if out_file_path.exists():
        with open(out_file_path, "r") as f:
            _dict = json.load(f)
            return ExperimentJsonFormat(**_dict)
    else:
        return ExperimentJsonFormat(outputs=[])


def run_with_caching(
    save_every: int,
    batch: int,
    task_to_run: list[TaskSpec] | list[StageTwoTaskSpec],
    raise_after_retries: bool = False,
    raise_on: Union[Literal["all"], Literal["any"]] = "any",
    num_retries: int = 10,
):
    """
    Take a list of TaskSpecs or StageTwoTaskSpecs and run, skipping completed tasks
    """
    paths = {task.out_file_path for task in task_to_run}

    loaded_dict: Union[dict[Path, ExperimentJsonFormat], dict[Path, StageTwoExperimentJsonFormat]] = {}
    completed_outputs: set[str] = set()
    if isinstance(task_to_run[0], TaskSpec):
        for path in paths:
            already_done = read_done_experiment(path)
            loaded_dict.update({path: already_done})

    elif isinstance(task_to_run[0], StageTwoTaskSpec):
        loaded_dict = get_loaded_dict_stage2(paths)

    for task_output in loaded_dict.values():
        for output in task_output.outputs:
            completed_outputs.add(output.task_spec.uid())

    to_do = []
    for item in task_to_run:
        task_hash = item.uid()
        if task_hash not in completed_outputs:
            to_do.append(item)

    random.Random(42).shuffle(to_do)
    run_tasks_multi_threaded(
        save_file_every=save_every,
        batch=batch,
        loaded_dict=loaded_dict,
        tasks_to_run=to_do,
        raise_after_retries=raise_after_retries,
        raise_on=raise_on,
        num_retries=num_retries,
    )

    outputs: list[TaskOutput | StageTwoTaskOutput] = []
    for exp in loaded_dict.values():
        outputs.extend(exp.outputs)
    return outputs


def run_tasks_multi_threaded(
    save_file_every: int,
    batch: int,
    loaded_dict: LoadedJsonType,
    tasks_to_run: Union[list[TaskSpec], list[StageTwoTaskSpec]],
    raise_after_retries: bool,
    raise_on: Union[Literal["all"], Literal["any"]] = "any",
    num_retries: int = 10,
) -> None:
    if len(tasks_to_run) == 0:
        print("No tasks to run, experiment is already done.")
        return

    future_instance_outputs = []

    print(f"Running {len(tasks_to_run)} tasks with {batch} threads")
    executor = ThreadPoolExecutor(max_workers=batch)

    def kill_and_save(loaded_dict: LoadedJsonType):
        save_loaded_dict(loaded_dict)
        executor.shutdown(wait=False, cancel_futures=True)
        exit_event.set()  # notify rate limiter to exit

    # import ipdb; ipdb.set_trace()

    for task in tasks_to_run:
        future_instance_outputs.append(
            executor.submit(
                task_function, task, raise_after_retries=raise_after_retries, raise_on=raise_on, num_retries=num_retries
            )
        )

    try:
        for cnt, instance_output in tqdm(
            enumerate(as_completed(future_instance_outputs)), total=len(future_instance_outputs)
        ):
            outputs = instance_output.result()
            # extend the existing json file
            loaded_dict[outputs[0].task_spec.out_file_path].outputs.extend(outputs)
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


class TaskSetting(BaseModel):
    task: str
    formatter: Type[StageOneFormatter]
    intervention: Optional[Type[Intervention]] = None
    model: str
