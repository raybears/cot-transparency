from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from cot_transparency.model_apis import call_model_api
from cot_transparency.openai_utils.models import ChatMessages, OpenaiInferenceConfig
from cot_transparency.prompt_formatter import PromptFormatter


@dataclass
class TaskSpec:
    # This is a dataclass because a PromptFormatter isn't serializable
    model_config: OpenaiInferenceConfig
    messages: list[ChatMessages]
    out_file_path: Path
    ground_truth: str
    formatter: PromptFormatter
    times_to_repeat: int
    task_hash: str


class ModelOutput(BaseModel):
    raw_response: str
    parsed_response: Optional[str]


class TaskOutput(BaseModel):
    # This is one single experiment
    prompt: list[ChatMessages]
    # E.g. 10 samples of COT will have a length of 10
    model_output: list[ModelOutput]
    ground_truth: str
    task_hash: str
    config: OpenaiInferenceConfig
    out_file_path: Path


def task_function(task: TaskSpec) -> TaskOutput:
    # TODO: possibly parallelize this
    outputs = []
    for i in range(task.times_to_repeat):
        # call api
        response = call_model_api(task.messages, task.model_config)
        # extract the answer
        parsed_response = task.formatter.parse_answer(response)
        outputs.append(ModelOutput(raw_response=response, parsed_response=parsed_response))
    return TaskOutput(
        prompt=task.messages,
        model_output=outputs,
        ground_truth=task.ground_truth,
        task_hash=task.task_hash,
        config=task.model_config,
        out_file_path=task.out_file_path,
    )


class ExperimentJsonFormat(BaseModel):
    # e.g. 1000 examples will have 1000 entries
    outputs: list[TaskOutput]
    task: str
    model: str

    def already_done_hashes(self) -> set[str]:
        return {o.task_hash for o in self.outputs}


def save_loaded_dict(loaded_dict: dict[Path, ExperimentJsonFormat]):
    for file_out, loaded in loaded_dict.items():
        # create the directory if it doesn't exist
        file_out.parent.mkdir(parents=True, exist_ok=True)
        with open(file_out, "w") as f:
            _json = loaded.json(indent=2)
            f.write(_json)
