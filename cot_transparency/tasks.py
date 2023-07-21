from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Type, Union

from pydantic import BaseModel
from retry import retry
from tqdm import tqdm
from cot_transparency.data_models.io import LoadedJsonType, save_loaded_dict
from cot_transparency.data_models.models_v2 import OpenaiInferenceConfig, StageTwoTaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter

from cot_transparency.model_apis import call_model_api
from cot_transparency.data_models.models_v2 import ChatMessages
from cot_transparency.formatters import PromptFormatter, name_to_formatter
from cot_transparency.data_models.models_v2 import TaskOutput
from cot_transparency.data_models.models_v2 import ModelOutput
from cot_transparency.data_models.models_v2 import StageTwoTaskSpec
from cot_transparency.data_models.models_v2 import TaskSpec
from cot_transparency.util import setup_logger

logger = setup_logger(__name__)


class AnswerNotFound(Exception):
    def __init__(self, e: str):
        self.e = e


@retry(exceptions=AnswerNotFound, tries=10, delay=1, logger=logger)
def call_model_until_suitable_response(
    messages: list[ChatMessages], config: OpenaiInferenceConfig, formatter: Type[PromptFormatter]
) -> ModelOutput:
    # call api
    response = call_model_api(messages, config)
    # extract the answer
    parsed_response = formatter.parse_answer(response)
    if not parsed_response:
        raise AnswerNotFound(f"didnt find answer in model answer '{response}'")
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

    executor = ThreadPoolExecutor(max_workers=batch)

    def kill_and_save(loaded_dict: LoadedJsonType):
        save_loaded_dict(loaded_dict)
        for future in future_instance_outputs:
            future.cancel()
        executor.shutdown(wait=False)

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
