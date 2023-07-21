"""Contains models for all objects that are loaded / saved to experiment jsons"""

from pathlib import Path


from pydantic import BaseModel


from typing import Any, Optional, Type

from cot_transparency.util import deterministic_hash
from cot_transparency.data_models.bbh import MultipleChoiceAnswer

from cot_transparency.data_models.models import (
    ChatMessages,
    ModelOutput,
    OpenaiInferenceConfig,
    deterministic_task_hash,
)


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
    # step 2 specific
    step_in_cot_trace: Optional[int] = None
    stage_one_hash: Optional[str] = None

    @property
    def first_parsed_response(self) -> str:
        return self.model_output[0].parsed_response

    def input_hash(self) -> str:
        return deterministic_task_hash(self.task_name, self.prompt, self.config)

    def output_hash(self) -> str:
        inp = self.input_hash()
        output = self.model_output[0].raw_response
        return deterministic_hash(inp + output)


class TaskSpec(BaseModel):
    # This is a dataclass because a PromptFormatter isn't serializable
    task_name: str
    model_config: OpenaiInferenceConfig
    messages: list[ChatMessages]
    out_file_path: Path
    ground_truth: MultipleChoiceAnswer
    formatter: Type[Any]
    task_hash: str  # linked to the orignal question
    biased_ans: Optional[MultipleChoiceAnswer] = None

    def input_hash(self) -> str:
        return deterministic_task_hash(self.task_name, self.messages, self.model_config)


class StageTwoTaskSpec(TaskSpec):
    stage_one_hash: str
    step_in_cot_trace: Optional[int] = None


class ExperimentJsonFormat(BaseModel):
    # e.g. 1000 examples will have 1000 entries
    outputs: list[TaskOutput]

    def already_done_hashes(self) -> list[str]:
        return [o.task_hash for o in self.outputs]
