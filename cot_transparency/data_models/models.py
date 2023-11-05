"""Contains models for all objects that are loaded / saved to experiment jsons"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Self, Sequence, Type

from pydantic import AliasChoices, BaseModel, Field

from cot_transparency.copy_utils.unset_sentinel import Unset, _UNSET
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.data.task_name_map import task_name_to_data_example
from cot_transparency.data_models.example_base import (
    DataExampleBase,
    GenericDataExample,
    MultipleChoiceAnswer,
)
from cot_transparency.data_models.hashable import HashableBaseModel
from cot_transparency.data_models.messages import ChatMessage
from cot_transparency.util import deterministic_hash


class ModelOutput(BaseModel):
    raw_response: str
    # We don't have a suitable response
    parsed_response: Optional[str]


class BaseTaskSpec(HashableBaseModel):
    """
    Specifies the minimal information needed to run a task and save the results
    """

    # We've called this model_config but that clashes with model_config of pydantic v2
    inference_config: OpenaiInferenceConfig = Field(validation_alias=AliasChoices("inference_config", "model_config"))
    messages: Sequence[ChatMessage]
    formatter_name: str
    intervention_name: Optional[str] = None

    @abstractmethod
    def get_task_name(self) -> str:
        raise NotImplementedError

    def uid(self) -> str:
        return self.model_hash()

    @abstractmethod
    def get_data_example_obj(self) -> DataExampleBase:
        raise NotImplementedError

    @abstractmethod
    def copy_update(
        self,
        *,
        messages: Sequence[ChatMessage] | Unset = _UNSET,
    ) -> "BaseTaskSpec":
        raise NotImplementedError

    @abstractmethod
    def get_task_hash(self) -> str:
        """
        Return the task hash of the data example that generated this task 
        """
        raise NotImplementedError


class TaskSpec(BaseTaskSpec):
    # This is a dataclass because a PromptFormatter isn't serializable
    task_name: str
    # We've called this model_config but that clashes with model_config of pydantic v2
    inference_config: OpenaiInferenceConfig = Field(validation_alias=AliasChoices("inference_config", "model_config"))
    messages: Sequence[ChatMessage]
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

    def copy_update(
        self,
        *,
        task_name: str | Unset = _UNSET,
        messages: Sequence[ChatMessage] | Unset = _UNSET,
        out_file_path: Path | Unset = _UNSET,
        ground_truth: MultipleChoiceAnswer | Unset = _UNSET,
        formatter_name: str | Unset = _UNSET,
        intervention_name: str | Unset = _UNSET,
        repeat_idx: int | Unset = _UNSET,
        task_hash: str | Unset = _UNSET,
        biased_ans: MultipleChoiceAnswer | Unset = _UNSET,
        data_example: dict[str, Any] | Unset = _UNSET,
    ) -> "TaskSpec":
        """
        Returns a copy of the config with the updated values
        Note that this does not update the task_hash, so make sure you really know what you are doing!!!
        """
        return TaskSpec(
            task_name=task_name if not isinstance(task_name, Unset) else self.task_name,
            messages=messages if not isinstance(messages, Unset) else self.messages,
            out_file_path=out_file_path if not isinstance(out_file_path, Unset) else self.out_file_path,
            ground_truth=ground_truth if not isinstance(ground_truth, Unset) else self.ground_truth,
            formatter_name=formatter_name if not isinstance(formatter_name, Unset) else self.formatter_name,
            intervention_name=intervention_name if not isinstance(intervention_name, Unset) else self.intervention_name,
            repeat_idx=repeat_idx if not isinstance(repeat_idx, Unset) else self.repeat_idx,
            task_hash=task_hash if not isinstance(task_hash, Unset) else self.task_hash,
            biased_ans=biased_ans if not isinstance(biased_ans, Unset) else self.biased_ans,
            data_example=data_example if not isinstance(data_example, Unset) else self.data_example,
            inference_config=self.inference_config,
        )

    def read_data_example_or_raise(self, data_type: Type[GenericDataExample]) -> GenericDataExample:
        return data_type(**self.data_example)

    def uid(self) -> str:
        """
        This hashes everything that is sent to the model, AND the repeats
        """
        model_inputs_hash = self.hash_of_inputs()
        return deterministic_hash(model_inputs_hash + str(self.repeat_idx))

    def hash_of_inputs(self) -> str:
        """
        This hashes everything that is sent to the model, thus it doesn't include the repeats
        """
        hashes: str = ""
        hashes += self.task_name
        hashes += self.inference_config.model_hash()
        for message in self.messages:
            hashes += message.model_hash()

        return hashes

    def task_hash_with_repeat(self) -> str:
        return deterministic_hash(self.task_hash + str(self.repeat_idx))

    def get_data_example_obj(self) -> DataExampleBase:
        DataExample = task_name_to_data_example(self.task_name)
        return DataExample(**self.data_example)

    @property
    def n_options_given(self) -> int:
        """
        Returns the number of options that were presented to the model
        automatically handles if none of the above was provided
        """
        data_example_obj = self.get_data_example_obj()
        formatter_name = self.formatter_name
        from cot_transparency.formatters.name_mapping import name_to_stage1_formatter

        formatter_type = name_to_stage1_formatter(formatter_name)
        n_options = len(data_example_obj.get_options(include_none_of_the_above=formatter_type.has_none_of_the_above))
        return n_options

    def get_task_name(self) -> str:
        return self.task_name

    def get_task_hash(self) -> str:
        return self.task_hash


class BaseTaskOutput(HashableBaseModel, ABC):
    inference_output: ModelOutput = Field(validation_alias=AliasChoices("inference_output", "model_output"))
    # we do not specify task_spec here because of invariance of variables so we instead specify
    # the get_task_spec() interface which can obey covariance properly

    @abstractmethod
    def get_task_spec(self) -> BaseTaskSpec:
        raise NotImplementedError

    @abstractmethod
    def update_messages_in_task_spec(self, messages: Sequence[ChatMessage]) -> Self:
        raise NotImplementedError

    @abstractmethod
    def copy_update(
        self,
        *,
        inference_output: ModelOutput | Unset = _UNSET,
    ) -> Self:
        raise NotImplementedError

    def update_parsed_response(self, parsed_response: str | None) -> Self:
        new_output = ModelOutput(raw_response=self.inference_output.raw_response, parsed_response=parsed_response)
        return self.copy_update(inference_output=new_output)

    def uid(self) -> str:
        return self.model_hash()


class TaskOutput(BaseTaskOutput):
    # This is one single experiment
    task_spec: TaskSpec  # type: ignore[reportIncompatibleVariableOverride]
    inference_output: ModelOutput = Field(validation_alias=AliasChoices("inference_output", "model_output"))
    response_idx: int = 0

    def copy_update(
        self,
        *,
        task_spec: TaskSpec | Unset = _UNSET,
        inference_output: ModelOutput | Unset = _UNSET,
        response_idx: int | Unset = _UNSET,
    ) -> "TaskOutput":
        return TaskOutput(
            task_spec=task_spec if not isinstance(task_spec, Unset) else self.task_spec,
            inference_output=inference_output if not isinstance(inference_output, Unset) else self.inference_output,
            response_idx=response_idx if not isinstance(response_idx, Unset) else self.response_idx,
        )

    def update_messages_in_task_spec(self, messages: Sequence[ChatMessage]) -> Self:
        return self.copy_update(task_spec=self.task_spec.copy_update(messages=messages))

    def get_task_spec(self) -> TaskSpec:
        return self.task_spec

    @property
    def bias_on_wrong_answer(self) -> bool:
        return self.task_spec.ground_truth != self.task_spec.biased_ans

    @property
    def bias_on_correct_answer(self) -> bool:
        return self.task_spec.ground_truth == self.task_spec.biased_ans

    @property
    def is_correct(self) -> bool:
        return self.inference_output.parsed_response == self.task_spec.ground_truth

    @property
    def first_parsed_response(self) -> Optional[str]:
        return self.inference_output.parsed_response

    @property
    def first_raw_response(self) -> str:
        return self.inference_output.raw_response

    def reparsed_response(self) -> Optional[str]:
        """
        Reparse the response using the formatters incase they have been updated
        If they have been updated, the the results of this may differ from calling
        self.inference_output.parsed_response as that is loaded from the json
        """
        formatter_name = self.task_spec.formatter_name
        from cot_transparency.formatters.name_mapping import name_to_stage1_formatter

        formatter_type = name_to_stage1_formatter(formatter_name)
        data_example_obj = self.task_spec.get_data_example_obj()
        return formatter_type.parse_answer(
            self.inference_output.raw_response,
            model=self.task_spec.inference_config.model,
            question=data_example_obj,
        )

    def task_spec_uid(self) -> str:
        return self.task_spec.uid()

    def uid(self) -> str:
        inp = self.task_spec_uid()
        response = self.inference_output
        if self.response_idx != 0:
            inp += str(self.response_idx)
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
    messages: Sequence[ChatMessage]
    out_file_path: Path
    formatter_name: str
    trace_info: Optional[TraceInfo] = None
    n_steps_in_cot_trace: Optional[int] = None

    def get_task_name(self) -> str:
        return self.stage_one_output.task_spec.task_name

    def uid(self) -> str:
        return self.model_hash()

    def to_s1(self) -> TaskSpec:
        s1 = TaskSpec(
            task_name=self.stage_one_output.task_spec.task_name,
            inference_config=self.inference_config,
            messages=self.messages,
            out_file_path=self.out_file_path,
            ground_truth=self.stage_one_output.task_spec.ground_truth,
            formatter_name=self.formatter_name,
            intervention_name=self.stage_one_output.task_spec.intervention_name,
            repeat_idx=self.stage_one_output.task_spec.repeat_idx,
            task_hash=self.stage_one_output.task_spec.task_hash,
            biased_ans=self.stage_one_output.task_spec.biased_ans,
            data_example=self.stage_one_output.task_spec.data_example,
        )
        return s1

    # take all the class variables or Unset
    def copy_update(
        self,
        *,
        messages: Sequence[ChatMessage] | Unset = _UNSET,
    ):
        return StageTwoTaskSpec(
            stage_one_output=self.stage_one_output,
            inference_config=self.inference_config,
            messages=messages if not isinstance(messages, Unset) else self.messages,
            out_file_path=self.out_file_path,
            formatter_name=self.formatter_name,
            trace_info=self.trace_info,
            n_steps_in_cot_trace=self.n_steps_in_cot_trace,
        )


class StageTwoTaskOutput(BaseTaskOutput):
    task_spec: StageTwoTaskSpec
    inference_output: ModelOutput = Field(validation_alias=AliasChoices("inference_output", "model_output"))
    response_idx: int = 0

    def uid(self) -> str:
        inp = self.task_spec.uid()
        if self.response_idx != 0:
            inp += str(self.response_idx)
        return deterministic_hash(inp + self.inference_output.raw_response)

    @property
    def first_raw_response(self) -> str:
        return self.inference_output.raw_response

    @property
    def first_parsed_response(self) -> Optional[str]:
        return self.inference_output.parsed_response

    def to_s1(self) -> TaskOutput:
        s1 = TaskOutput(
            task_spec=self.task_spec.to_s1(),
            inference_output=self.inference_output,
            response_idx=self.response_idx,
        )
        return s1

    def get_task_spec(self) -> BaseTaskSpec:
        return self.task_spec

    def update_messages_in_task_spec(self, messages: Sequence[ChatMessage]) -> Self:
        return StageTwoTaskOutput(
            task_spec=self.task_spec.copy_update(messages=messages),
            inference_output=self.inference_output,
            response_idx=self.response_idx,
        )

    def copy_update(
        self,
        *,
        inference_output: ModelOutput | Unset = _UNSET,
    ) -> Self:
        return StageTwoTaskOutput(
            task_spec=self.task_spec,
            inference_output=inference_output if not isinstance(inference_output, Unset) else self.inference_output,
            response_idx=self.response_idx,
        )


class ExperimentJsonFormat(BaseModel):
    # e.g. 1000 examples will have 1000 entries
    stage: int = 1
    outputs: list[TaskOutput]

    def already_done_hashes(self) -> list[str]:
        return [o.task_spec.task_hash for o in self.outputs]


class StageTwoExperimentJsonFormat(BaseModel):
    stage: int = 2
    outputs: list[StageTwoTaskOutput]
