"""Contains models for all objects that are loaded / saved to experiment jsons"""

from enum import Enum
from pathlib import Path


from pydantic import BaseModel, conlist


from typing import Optional, Union

from cot_transparency.util import deterministic_hash
from cot_transparency.data_models.example_base import MultipleChoiceAnswer


class HashableBaseModel(BaseModel):
    def d_hash(self) -> str:
        as_json = self.json()
        return deterministic_hash(as_json)


class OpenaiInferenceConfig(HashableBaseModel):
    # Config for openai
    model: str
    temperature: Optional[float]
    top_p: Optional[float]
    max_tokens: int
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Union[None, str, conlist(str, min_items=1, max_items=4)] = None  # type: ignore


class MessageRoles(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"
    # If you are OpenAI chat, you need to add this back into the previous user message
    # Anthropic can handle it as per normal like an actual assistant
    assistant_preferred = "assistant_preferred"
    # none is designed for completion tasks where no role / tag will be added
    none = "none"


class ChatMessages(HashableBaseModel):
    role: MessageRoles
    content: str

    class Config:
        frozen = True


class ModelOutput(BaseModel):
    raw_response: str
    # We always have a suitable response because we keep retrying
    parsed_response: str


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
    formatter_name: str
    task_hash: str  # linked to the orignal question
    biased_ans: Optional[MultipleChoiceAnswer] = None

    def uid(self) -> str:
        return deterministic_task_hash(self.task_name, self.messages, self.model_config)


class TaskOutput(BaseModel):
    # This is one single experiment
    task_spec: TaskSpec
    model_output: ModelOutput

    @property
    def first_parsed_response(self) -> str:
        return self.model_output.parsed_response

    @property
    def first_raw_response(self) -> str:
        return self.model_output.raw_response

    def task_spec_uid(self) -> str:
        return self.task_spec.uid()

    def uid(self) -> str:
        inp = self.task_spec_uid()
        response = self.model_output
        return deterministic_hash(inp + response.raw_response)


class StageTwoTaskSpec(BaseModel):
    stage_one_output: TaskOutput
    model_config: OpenaiInferenceConfig
    messages: list[ChatMessages]
    out_file_path: Path
    formatter_name: str
    step_in_cot_trace: Optional[int] = None

    def uid(self) -> str:
        task_name = self.stage_one_output.task_spec.task_name
        return deterministic_task_hash(task_name, self.messages, self.model_config)


class StageTwoTaskOutput(BaseModel):
    task_spec: StageTwoTaskSpec
    model_output: ModelOutput

    def uid(self) -> str:
        inp = self.task_spec.uid()
        return deterministic_hash(inp + self.model_output.raw_response)

    @property
    def first_raw_response(self) -> str:
        return self.model_output.raw_response

    @property
    def first_parsed_response(self) -> str:
        return self.model_output.parsed_response


class ExperimentJsonFormat(BaseModel):
    # e.g. 1000 examples will have 1000 entries
    stage: int = 1
    outputs: list[TaskOutput]

    def already_done_hashes(self) -> list[str]:
        return [o.task_spec.task_hash for o in self.outputs]


class StageTwoExperimentJsonFormat(BaseModel):
    stage: int = 2
    outputs: list[StageTwoTaskOutput]
