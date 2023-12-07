from pydantic import BaseModel
from cot_transparency.copy_utils.unset_sentinel import _UNSET, Unset
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.data.task_name_map import task_name_to_data_example
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage
from cot_transparency.data_models.models import BaseTaskOutput, BaseTaskSpec, ModelOutput
from typing import Any, Sequence


class ParaphrasedQuestion(BaseModel):
    paraphrased: str
    tags: Sequence[str]


class StreamingTaskSpec(BaseTaskSpec):
    messages: Sequence[ChatMessage]
    formatter_name: str
    task_name: str
    data_example: dict[str, Any] = {}
    inference_config: OpenaiInferenceConfig
    paraphrasing_formatter_name: str | None = None

    def get_task_name(self) -> str:
        return self.task_name

    def get_data_example_obj(self) -> DataExampleBase:
        DataExample = task_name_to_data_example(self.task_name)
        return DataExample(**self.data_example)

    def get_task_hash(self) -> str:
        return self.get_data_example_obj().hash()

    @property
    def n_options_given(self) -> int:
        """
        Returns the number of options that were presented to the model
        automatically handles if none of the above was provided
        """
        data_example_obj = self.get_data_example_obj()
        formatter_name = self.formatter_name
        from cot_transparency.formatters.name_mapping import name_to_formatter

        formatter_type = name_to_formatter(formatter_name)
        n_options = len(data_example_obj.get_options(include_none_of_the_above=formatter_type.has_none_of_the_above))
        return n_options

    def copy_update(
        self,
        *,
        messages: Sequence[ChatMessage] | Unset = _UNSET,
        inference_config: OpenaiInferenceConfig | Unset = _UNSET,
        formatter_name: str | Unset = _UNSET,
    ) -> "StreamingTaskSpec":
        return StreamingTaskSpec(
            messages=messages if not isinstance(messages, Unset) else self.messages,
            formatter_name=formatter_name if not isinstance(formatter_name, Unset) else self.formatter_name,
            task_name=self.task_name,
            data_example=self.data_example,
            inference_config=inference_config if not isinstance(inference_config, Unset) else self.inference_config,
        )


class ParaphrasedTaskSpec(StreamingTaskSpec):
    paraphrased_question: ParaphrasedQuestion


class StreamingTaskOutput(BaseTaskOutput):
    task_spec: StreamingTaskSpec
    inference_output: ModelOutput

    def get_task_spec(self) -> StreamingTaskSpec:
        return self.task_spec

    @property
    def is_correct(self) -> bool:
        return self.inference_output.parsed_response == self.task_spec.get_data_example_obj().ground_truth

    def update_messages_in_task_spec(self, messages: Sequence[ChatMessage]) -> "StreamingTaskOutput":
        return StreamingTaskOutput(
            task_spec=self.task_spec.copy_update(messages=messages),
            inference_output=self.inference_output,
        )

    def copy_update(
        self,
        *,
        inference_output: ModelOutput | Unset = _UNSET,
    ) -> "StreamingTaskOutput":
        return StreamingTaskOutput(
            task_spec=self.task_spec,
            inference_output=inference_output if not isinstance(inference_output, Unset) else self.inference_output,
        )

    @property
    def is_correct(self) -> bool:
        return self.inference_output.parsed_response == self.get_task_spec().get_data_example_obj().ground_truth


class ParaphrasingOutput(StreamingTaskOutput):
    task_spec: StreamingTaskSpec
    inference_output: ModelOutput
    paraphrased_questions: Sequence[ParaphrasedQuestion]
