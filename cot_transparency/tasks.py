from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
from typing import Optional, Type, Union

from pydantic import BaseModel
from tqdm import tqdm
from cot_transparency.data_models.io import LoadedJsonType, save_loaded_dict
from cot_transparency.data_models.models import (
    ExperimentJsonFormat,
    OpenaiInferenceConfig,
    StageTwoExperimentJsonFormat,
    StageTwoTaskOutput,
    StrictChatMessage,
)
from cot_transparency.formatters.base_class import StageOneFormatter

from cot_transparency.model_apis import call_model_api
from cot_transparency.data_models.models import ChatMessage
from cot_transparency.formatters import PromptFormatter, name_to_formatter
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.data_models.models import ModelOutput
from cot_transparency.data_models.models import StageTwoTaskSpec
from cot_transparency.data_models.models import TaskSpec
from cot_transparency.util import setup_logger
from cot_transparency.openai_utils.rate_limiting import exit_event

logger = setup_logger(__name__)


class AnswerNotFound(Exception):
    def __init__(self, e: str):
        self.e = e


def call_model_until_suitable_response(
    messages: list[StrictChatMessage] | list[ChatMessage],
    config: OpenaiInferenceConfig,
    formatter: Type[PromptFormatter],
    allow_failure_after_n: Optional[int] = None,
    retries: int = 20,
):
    n_failures = 0

    for _ in range(max(retries, 1)):
        response = call_model_api(messages, config)
        parsed_response = formatter.parse_answer(response)
        if parsed_response:
            break

        msg = (
            f"Formatter: {formatter}, Model: {config.model}, didnt find answer in model answer '{response}'"
            f"last two messages were:\n{messages[-2]}\n\n{messages[-1]}"
        )
        logger.warning(msg)
        if allow_failure_after_n and n_failures > allow_failure_after_n:
            logger.error(f"Allowing failure after {allow_failure_after_n} failures, raising exception")
            parsed_response = "NOT_FOUND"
        n_failures += 1

    if not parsed_response:  # type: ignore
        raise AnswerNotFound(msg)  # type: ignore
    else:
        return ModelOutput(raw_response=response, parsed_response=parsed_response)  # type: ignore


def task_function(
    task: Union[TaskSpec, StageTwoTaskSpec], allow_failure_after_n: Optional[int] = None
) -> Union[TaskOutput, StageTwoTaskOutput]:
    formatter = name_to_formatter(task.formatter_name)
    response = call_model_until_suitable_response(
        messages=task.messages,
        config=task.model_config,
        formatter=formatter,
        allow_failure_after_n=allow_failure_after_n,
    )

    if isinstance(task, StageTwoTaskSpec):
        return StageTwoTaskOutput(
            task_spec=task,
            model_output=response,
        )
    elif isinstance(task, TaskSpec):
        return TaskOutput(
            task_spec=task,
            model_output=response,
        )
    else:
        raise ValueError(f"Unknown task type {type(task)}")


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
    allow_failure_after_n: Optional[int] = None,
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
        allow_failure_after_n=allow_failure_after_n,
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
    allow_failure_after_n: Optional[int] = None,
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

    for task in tasks_to_run:
        future_instance_outputs.append(
            executor.submit(task_function, task, allow_failure_after_n=allow_failure_after_n)
        )

    try:
        for cnt, instance_output in tqdm(
            enumerate(as_completed(future_instance_outputs)), total=len(future_instance_outputs)
        ):
            output = instance_output.result()
            # extend the existing json file
            loaded_dict[output.task_spec.out_file_path].outputs.append(output)
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
    model: str
