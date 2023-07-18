from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from glob import glob
from pathlib import Path
from typing import Type, Optional, Union

from pydantic import BaseModel
from retry import retry
from tqdm import tqdm
from cot_transparency.hashing import deterministic_hash

from cot_transparency.miles_models import MultipleChoiceAnswer
from cot_transparency.model_apis import call_model_api
from cot_transparency.openai_utils.models import ChatMessages, OpenaiInferenceConfig
from cot_transparency.formatters import PromptFormatter
from cot_transparency.util import setup_logger, safe_file_write

logger = setup_logger(__name__)


class AnswerNotFound(Exception):
    def __init__(self, e: str):
        self.e = e


def deterministic_task_hash(task_name: str, messages: list[ChatMessages], model_config: OpenaiInferenceConfig) -> str:
    hashes: str = ""
    hashes += task_name
    hashes += model_config.d_hash()
    for message in messages:
        hashes += message.d_hash()

    return deterministic_hash(hashes)


class TaskSpec(BaseModel):
    # This is a dataclass because a PromptFormatter isn't serializable
    task_name: str
    model_config: OpenaiInferenceConfig
    messages: list[ChatMessages]
    out_file_path: Path
    ground_truth: MultipleChoiceAnswer
    formatter: Type[PromptFormatter]
    task_hash: str  # linked to the orignal question
    biased_ans: Optional[MultipleChoiceAnswer] = None

    def input_hash(self) -> str:
        return deterministic_task_hash(self.task_name, self.messages, self.model_config)


class StageTwoTaskSpec(TaskSpec):
    stage_one_hash: str


class ModelOutput(BaseModel):
    raw_response: str
    # We always have a suitable response because we keep retrying
    parsed_response: str


class TaskOutput(BaseModel):
    # This is one single experiment
    task_name: str
    prompt: list[ChatMessages]
    model_output: list[ModelOutput]
    ground_truth: MultipleChoiceAnswer
    task_hash: str
    config: OpenaiInferenceConfig
    formatter_name: str
    out_file_path: Path
    biased_ans: Optional[MultipleChoiceAnswer] = None

    def input_hash(self) -> str:
        return deterministic_task_hash(self.task_name, self.prompt, self.config)

    def output_hash(self) -> str:
        inp = self.input_hash()
        output = self.model_output[0].raw_response
        return deterministic_hash(inp + output)


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


def task_function(task: TaskSpec) -> TaskOutput:
    response = call_model_until_suitable_response(
        messages=task.messages, config=task.model_config, formatter=task.formatter
    )
    outputs = [response]  # maintained as list for backwards compatibility
    return TaskOutput(
        task_name=task.task_name,
        prompt=task.messages,
        model_output=outputs,
        ground_truth=task.ground_truth,
        task_hash=task.input_hash(),
        config=task.model_config,
        out_file_path=task.out_file_path,
        formatter_name=task.formatter.name(),
        biased_ans=task.biased_ans,
    )


class ExperimentJsonFormat(BaseModel):
    # e.g. 1000 examples will have 1000 entries
    outputs: list[TaskOutput]

    def already_done_hashes(self) -> list[str]:
        return [o.task_hash for o in self.outputs]


def save_loaded_dict(loaded_dict: dict[Path, ExperimentJsonFormat]):
    for file_out, loaded in loaded_dict.items():
        # create the directory if it doesn't exist
        file_out.parent.mkdir(parents=True, exist_ok=True)
        _json = loaded.json(indent=2)
        safe_file_write(str(file_out), _json)


def load_jsons(exp_dir: str) -> dict[Path, ExperimentJsonFormat]:
    loaded_dict: dict[Path, ExperimentJsonFormat] = {}

    paths = glob(f"{exp_dir}/**/*.json", recursive=True)
    print(f"Found {len(paths)} json files")
    for path in paths:
        _dict = json.load(open(path))
        loaded_dict[Path(path)] = ExperimentJsonFormat(**_dict)
    return loaded_dict


def run_tasks_multi_threaded(
    save_file_every: int,
    batch: int,
    loaded_dict: dict[Path, ExperimentJsonFormat],
    tasks_to_run: Union[list[TaskSpec], list[StageTwoTaskSpec]],
):
    if len(tasks_to_run) == 0:
        print("No tasks to run, experiment is already done.")
        return

    future_instance_outputs = []

    executor = ThreadPoolExecutor(max_workers=batch)

    def kill_and_save(loaded_dict: dict[Path, ExperimentJsonFormat]):
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
