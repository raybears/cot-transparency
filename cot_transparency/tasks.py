from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Type, Union

from pydantic import BaseModel
from retry import retry
from tqdm import tqdm
from cot_transparency.data_models.io import LoadedJsonType, save_loaded_dict
from cot_transparency.data_models.models import OpenaiInferenceConfig, StageTwoTaskOutput, StrictChatMessage
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


@retry(exceptions=AnswerNotFound, tries=20, delay=1, logger=logger)
def call_model_until_suitable_response(
    messages: list[StrictChatMessage] | list[ChatMessage],
    config: OpenaiInferenceConfig,
    formatter: Type[PromptFormatter],
) -> ModelOutput:
    # call api
    response = call_model_api(messages, config)
    # extract the answer
    parsed_response = formatter.parse_answer(response)

    if not parsed_response:
        raise AnswerNotFound(
            f"Formatter: {formatter}, Model: {config.model}, didnt find answer in model answer '{response}'"
            f"last two messages were:\n{messages[-2]}\n\n{messages[-1]}"
        )
    return ModelOutput(raw_response=response, parsed_response=parsed_response)


def task_function(task: Union[TaskSpec, StageTwoTaskSpec]) -> Union[TaskOutput, StageTwoTaskOutput]:
    formatter = name_to_formatter(task.formatter_name)
    response = call_model_until_suitable_response(
        messages=task.messages,
        config=task.model_config,
        formatter=formatter,
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


def run_tasks_multi_threaded(
    save_file_every: int,
    batch: int,
    loaded_dict: LoadedJsonType,
    tasks_to_run: Union[list[TaskSpec], list[StageTwoTaskSpec]],
):
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
        future_instance_outputs.append(executor.submit(task_function, task))

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
