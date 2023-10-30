"""
This file basically an evolution of tasks.py but simplified and intended to be used with grugstream.
"""
from cot_transparency.apis.base import ModelCaller
from slist import Slist
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.data.task_name_map import task_name_to_data_example
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.hashable import HashableBaseModel
from cot_transparency.data_models.messages import ChatMessage
from cot_transparency.data_models.models import BaseTaskSpec, ModelOutput


from typing import Any, Sequence

from cot_transparency.formatters.base_class import StageOneFormatter, Type
from cot_transparency.formatters.name_mapping import name_to_formatter


class StreamingTaskSpec(BaseTaskSpec):
    messages: Sequence[ChatMessage]
    formatter_name: str
    task_name: str
    data_example: dict[str, Any] = {}
    inference_config: OpenaiInferenceConfig

    def get_task_name(self) -> str:
        return self.task_name

    def get_data_example_obj(self) -> DataExampleBase:
        DataExample = task_name_to_data_example(self.task_name)
        return DataExample(**self.data_example)

    @property
    def task_hash(self) -> str:
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


class StreamingTaskOutput(HashableBaseModel):
    task_spec: StreamingTaskSpec
    inference_outputs: Sequence[ModelOutput]


def data_to_task_spec(
    task_name: str,
    x: DataExampleBase,
    formatters: Sequence[Type[StageOneFormatter]],
    models: Sequence[OpenaiInferenceConfig],
) -> list[StreamingTaskSpec]:
    specs = []
    for formatter in formatters:
        for model in models:
            messages = formatter.format_example(x)
            ts = StreamingTaskSpec(
                messages=messages,
                formatter_name=formatter.name(),
                data_example=x.model_dump(),
                inference_config=model,
                task_name=task_name,
            )
            specs.append(ts)
    return specs


def call_model_with_task_spec(task_spec: StreamingTaskSpec, caller: ModelCaller) -> StreamingTaskOutput:
    responses = Slist(caller.call(messages=task_spec.messages, config=task_spec.inference_config).raw_responses)
    formatter_class = name_to_formatter(task_spec.formatter_name)
    data_example = task_spec.get_data_example_obj()
    outputs = responses.map(
        lambda i: ModelOutput(
            raw_response=i,
            parsed_response=formatter_class.parse_answer(i, data_example, model=task_spec.inference_config.model),
        )
    )
    return StreamingTaskOutput(task_spec=task_spec, inference_outputs=outputs)
