"""Contains models for all objects that are loaded / saved to experiment jsons"""

from enum import Enum
from pathlib import Path


from pydantic import BaseModel, conlist, Field, AliasChoices, ConfigDict

from typing import Optional, Union, Any, Type


from cot_transparency.util import deterministic_hash
from cot_transparency.data_models.example_base import MultipleChoiceAnswer, GenericDataExample


class HashableBaseModel(BaseModel):
    def d_hash(self) -> str:
        as_json = self.model_dump_json()
        return deterministic_hash(as_json)


class OpenaiInferenceConfig(HashableBaseModel):
    # Config for openai
    model: str
    temperature: float
    top_p: Optional[float]
    max_tokens: int
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Union[None, str, conlist(str, min_length=1, max_length=4)] = None  # type: ignore


class MessageRole(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"
    # If you are OpenAI chat, you need to add this back into the previous user message
    # Anthropic can handle it as per normal like an actual assistant
    assistant_if_completion = "assistant_preferred"
    # none is designed for completion tasks where no role / tag will be added
    none = "none"


class StrictMessageRole(str, Enum):
    # Stricter set of roles that doesn't allow assistant_preferred
    user = "user"
    system = "system"
    assistant = "assistant"
    # none is designed for completion tasks where no role / tag will be added
    none = "none"


class ChatMessage(HashableBaseModel):
    role: MessageRole
    content: str
    model_config = ConfigDict(frozen=True)

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def remove_role(self) -> "ChatMessage":
        return ChatMessage(role=MessageRole.none, content=self.content)

    def add_question_prefix(self) -> "ChatMessage":
        if self.content.startswith("Question: "):
            return self
        return ChatMessage(role=self.role, content=f"Question: {self.content}")

    def add_answer_prefix(self) -> "ChatMessage":
        if self.content.startswith("Answer: "):
            return self
        return ChatMessage(role=self.role, content=f"Answer: {self.content}")


class StrictChatMessage(HashableBaseModel):
    role: StrictMessageRole
    content: str
    model_config = ConfigDict(frozen=True)

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"


class ModelOutput(BaseModel):
    raw_response: str
    # We don't have a suitable response
    parsed_response: Optional[str]


def deterministic_task_hash(
    task_name: str,
    messages: list[ChatMessage] | list[StrictChatMessage],
    model_config: OpenaiInferenceConfig,
    repeat_idx: int = 0,
) -> str:
    hashes: str = ""
    if repeat_idx > 0:
        hashes += str(repeat_idx)
    hashes += task_name
    hashes += model_config.d_hash()
    for message in messages:
        hashes += message.d_hash()

    return deterministic_hash(hashes)


class BaseTaskSpec(BaseModel):
    # We've called this model_config but that clashes with model_config of pydantic v2
    inference_config: OpenaiInferenceConfig = Field(validation_alias=AliasChoices("inference_config", "model_config"))
    messages: list[ChatMessage]
    out_file_path: Path
    formatter_name: str


class TaskSpec(BaseTaskSpec):
    # This is a dataclass because a PromptFormatter isn't serializable
    task_name: str
    # We've called this model_config but that clashes with model_config of pydantic v2
    inference_config: OpenaiInferenceConfig = Field(validation_alias=AliasChoices("inference_config", "model_config"))
    messages: list[ChatMessage]
    out_file_path: Path
    ground_truth: MultipleChoiceAnswer
    formatter_name: str
    intervention_name: Optional[str] = None
    repeat_idx: int = 0
    task_hash: str  # linked to the orignal question
    biased_ans: Optional[MultipleChoiceAnswer] = None
    # Note that this is empty for older experiments
    # This can't be the abstract class DataExampleBase because you can't instantiate it
    data_example: dict[str, Any] = {}

    def read_data_example_or_raise(self, data_type: Type[GenericDataExample]) -> GenericDataExample:
        return data_type(**self.data_example)

    def uid(self) -> str:
        return deterministic_task_hash(self.task_name, self.messages, self.inference_config, self.repeat_idx)

    def task_hash_with_repeat(self) -> str:
        return deterministic_hash(self.task_hash + str(self.repeat_idx))


class BaseTaskOuput(BaseModel):
    task_spec: BaseTaskSpec
    inference_output: ModelOutput = Field(validation_alias=AliasChoices("inference_output", "model_output"))


class TaskOutput(BaseTaskOuput):
    # This is one single experiment
    task_spec: TaskSpec
    inference_output: ModelOutput = Field(validation_alias=AliasChoices("inference_output", "model_output"))

    @property
    def is_correct(self) -> bool:
        return self.inference_output.parsed_response == self.task_spec.ground_truth

    @property
    def first_parsed_response(self) -> Optional[str]:
        return self.inference_output.parsed_response

    @property
    def first_raw_response(self) -> str:
        return self.inference_output.raw_response

    def task_spec_uid(self) -> str:
        return self.task_spec.uid()

    def uid(self) -> str:
        inp = self.task_spec_uid()
        response = self.inference_output
        return deterministic_hash(inp + response.raw_response)


class TraceInfo(BaseModel):
    original_cot: list[str]
    complete_modified_cot: Optional[str] = None
    mistake_inserted_idx: Optional[int] = None
    sentence_with_mistake: Optional[str] = None
    regenerated_cot_post_mistake: Optional[str] = None

    def get_mistake_inserted_idx(self) -> int:
        if self.mistake_inserted_idx is None:
            raise ValueError("Mistake inserted idx is None")
        return self.mistake_inserted_idx

    def get_sentence_with_mistake(self) -> str:
        if self.sentence_with_mistake is None:
            raise ValueError("Sentence with mistake is None")
        return self.sentence_with_mistake.lstrip()

    def get_trace_upto_mistake(self) -> str:
        original_cot = self.original_cot
        mistake_inserted_idx = self.get_mistake_inserted_idx()
        reasoning_step_with_mistake = self.get_sentence_with_mistake()
        partial_cot = original_cot[:mistake_inserted_idx]
        original_sentence = original_cot[mistake_inserted_idx]

        # ensure that the original sentence has the same leading new lines as the original cot
        # as these are striped when we prompt the model to generate mistakes
        if original_sentence.startswith("\n") and not reasoning_step_with_mistake.startswith("\n"):
            leading_chars = original_sentence[: len(original_sentence) - len(original_sentence.lstrip("\n"))]
        elif original_sentence.startswith(" ") and not reasoning_step_with_mistake.startswith(" "):
            leading_chars = original_sentence[: len(original_sentence) - len(original_sentence.lstrip(" "))]
        else:
            leading_chars = ""

        partial_cot_trace = "".join(partial_cot) + leading_chars + reasoning_step_with_mistake
        return partial_cot_trace

    def get_complete_modified_cot(self) -> str:
        if self.complete_modified_cot is not None:
            return self.complete_modified_cot
        else:
            if self.regenerated_cot_post_mistake is None:
                raise ValueError("Regenerated cot post mistake is None")

            if self.get_mistake_inserted_idx() == len(self.original_cot) - 1:
                # mistake was inserted at the end of the cot so just return that
                return self.get_trace_upto_mistake()

            # if self.regenerated_cot_post_mistake already has leading chars, then we don't need to add them
            if self.regenerated_cot_post_mistake.startswith("\n") or self.regenerated_cot_post_mistake.startswith(" "):
                return self.get_trace_upto_mistake() + self.regenerated_cot_post_mistake

            original_sentence = self.original_cot[self.get_mistake_inserted_idx() + 1]
            if original_sentence.startswith("\n"):
                leading_chars = original_sentence[: len(original_sentence) - len(original_sentence.lstrip("\n"))]
            elif original_sentence.startswith(" "):
                leading_chars = original_sentence[: len(original_sentence) - len(original_sentence.lstrip(" "))]
            else:
                leading_chars = ""

            return self.get_trace_upto_mistake() + leading_chars + self.regenerated_cot_post_mistake

    @property
    def has_mistake(self) -> bool:
        return self.mistake_inserted_idx is not None

    @property
    def was_truncated(self) -> bool:
        if not self.has_mistake and self.complete_modified_cot != "".join(self.original_cot):
            return True
        return False


class StageTwoTaskSpec(BaseTaskSpec):
    stage_one_output: TaskOutput
    # We've called this model_config but that clashes with model_config of pydantic v2
    inference_config: OpenaiInferenceConfig = Field(validation_alias=AliasChoices("inference_config", "model_config"))
    messages: list[ChatMessage]
    out_file_path: Path
    formatter_name: str
    trace_info: TraceInfo
    n_steps_in_cot_trace: Optional[int] = None

    def uid(self) -> str:
        task_name = self.stage_one_output.task_spec.task_name
        original_cot = self.trace_info.original_cot
        stage_one_repeat = self.stage_one_output.task_spec.repeat_idx
        h = deterministic_task_hash(task_name, self.messages, self.inference_config, repeat_idx=stage_one_repeat)
        return deterministic_hash(h + "".join(original_cot))


class StageTwoTaskOutput(BaseTaskOuput):
    task_spec: StageTwoTaskSpec
    inference_output: ModelOutput = Field(validation_alias=AliasChoices("inference_output", "model_output"))

    def uid(self) -> str:
        inp = self.task_spec.uid()
        return deterministic_hash(inp + self.inference_output.raw_response)

    @property
    def first_raw_response(self) -> str:
        return self.inference_output.raw_response

    @property
    def first_parsed_response(self) -> Optional[str]:
        return self.inference_output.parsed_response


class ExperimentJsonFormat(BaseModel):
    # e.g. 1000 examples will have 1000 entries
    stage: int = 1
    outputs: list[TaskOutput]

    def already_done_hashes(self) -> list[str]:
        return [o.task_spec.task_hash for o in self.outputs]


class StageTwoExperimentJsonFormat(BaseModel):
    stage: int = 2
    outputs: list[StageTwoTaskOutput]
