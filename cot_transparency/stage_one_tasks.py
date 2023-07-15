from dataclasses import dataclass
from pathlib import Path
from typing import Type, Optional

from pydantic import BaseModel
from retry import retry

from cot_transparency.miles_models import MultipleChoiceAnswer
from cot_transparency.model_apis import call_model_api
from cot_transparency.openai_utils.models import ChatMessages, OpenaiInferenceConfig
from cot_transparency.prompt_formatter import AnswerNotFound, PromptFormatter


@dataclass
class TaskSpec:
    # This is a dataclass because a PromptFormatter isn't serializable
    task_name: str
    model_config: OpenaiInferenceConfig
    messages: list[ChatMessages]
    out_file_path: Path
    ground_truth: MultipleChoiceAnswer
    formatter: Type[PromptFormatter]
    times_to_repeat: int
    task_hash: str
    biased_ans: Optional[MultipleChoiceAnswer] = None


class ModelOutput(BaseModel):
    raw_response: str
    # We always have a suitable response because we keep retrying
    parsed_response: str


class TaskOutput(BaseModel):
    # This is one single experiment
    task_name: str
    prompt: list[ChatMessages]
    # E.g. 10 samples of COT will have a length of 10
    model_output: list[ModelOutput]
    ground_truth: str
    task_hash: str
    config: OpenaiInferenceConfig
    formatter_name: str
    out_file_path: Path
    biased_ans: Optional[MultipleChoiceAnswer] = None


@retry(exceptions=AnswerNotFound, tries=10, delay=1)
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
    # TODO: possibly parallelize this
    outputs = []
    for i in range(task.times_to_repeat):
        response = call_model_until_suitable_response(
            messages=task.messages, config=task.model_config, formatter=task.formatter
        )
        outputs.append(response)
    return TaskOutput(
        task_name=task.task_name,
        prompt=task.messages,
        model_output=outputs,
        ground_truth=task.ground_truth,
        task_hash=task.task_hash,
        config=task.model_config,
        out_file_path=task.out_file_path,
        formatter_name=task.formatter.name(),
        biased_ans=task.biased_ans,
    )


class ExperimentJsonFormat(BaseModel):
    # e.g. 1000 examples will have 1000 entries
    outputs: list[TaskOutput]

    def already_done_hashes(self) -> set[str]:
        return {o.task_hash for o in self.outputs}


def save_loaded_dict(loaded_dict: dict[Path, ExperimentJsonFormat]):
    for file_out, loaded in loaded_dict.items():
        # create the directory if it doesn't exist
        file_out.parent.mkdir(parents=True, exist_ok=True)
        with open(file_out, "w") as f:
            _json = loaded.json(indent=2)
            f.write(_json)
