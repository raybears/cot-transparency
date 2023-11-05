"""
This file basically an evolution of tasks.py but simplified and intended to be used with grugstream.
"""
import typing
from cot_transparency.apis.base import ModelCaller
from slist import Slist
from cot_transparency.copy_utils.unset_sentinel import _UNSET, Unset
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.data.task_name_map import task_name_to_data_example
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.messages import ChatMessage
from cot_transparency.data_models.models import BaseTaskOutput, BaseTaskSpec, ModelOutput


from typing import Any, Self, Sequence

from cot_transparency.formatters.base_class import StageOneFormatter, Type
from cot_transparency.formatters.name_mapping import name_to_formatter
from cot_transparency.data_models.data import get_list_of_examples


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
    ) -> Self:
        return StreamingTaskSpec(
            messages=messages if not isinstance(messages, Unset) else self.messages,
            formatter_name=self.formatter_name,
            task_name=self.task_name,
            data_example=self.data_example,
            inference_config=self.inference_config,
        )


class StreamingTaskOutput(BaseTaskOutput):
    task_spec: StreamingTaskSpec
    inference_output: ModelOutput

    def get_task_spec(self) -> StreamingTaskSpec:
        return self.task_spec

    def update_messages_in_task_spec(self, messages: Sequence[ChatMessage]) -> Self:
        return StreamingTaskOutput(
            task_spec=self.task_spec.copy_update(messages=messages),
            inference_output=self.inference_output,
        )

    def copy_update(
        self,
        *,
        inference_output: ModelOutput | Unset = _UNSET,
    ) -> Self:
        return StreamingTaskOutput(
            task_spec=self.task_spec,
            inference_output=inference_output if not isinstance(inference_output, Unset) else self.inference_output,
        )


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


def call_model_with_task_spec(task_spec: StreamingTaskSpec, caller: ModelCaller) -> Sequence[StreamingTaskOutput]:
    """
    Basic building block to call a model with a task spec. Note this returns a sequence because the config in task_spec
    may specify n > 1 in which case we will get multiple responses for a single request. Call .flatten_iterable() if
    using grugstream after this step to flatten the sequence into a single iterable.
    """

    responses = Slist(caller.call(messages=task_spec.messages, config=task_spec.inference_config).raw_responses)
    formatter_class = name_to_formatter(task_spec.formatter_name)
    data_example = task_spec.get_data_example_obj()
    outputs = responses.map(
        lambda i: ModelOutput(
            raw_response=i,
            parsed_response=formatter_class.parse_answer(i, data_example, model=task_spec.inference_config.model),
        )
    )
    return outputs.map(lambda x: StreamingTaskOutput(task_spec=task_spec, inference_output=x))


def get_examples_for_tasks(tasks: Sequence[str], example_cap: int) -> Slist[tuple[str, DataExampleBase]]:
    """
    Returns a list of tuples of (task_name, example_obj)
    """
    ret = Slist()
    for t in tasks:
        examples = get_list_of_examples(t)
        # print(f"Found {len(examples)} examples for task: {t}")

        task_with_name = examples.shuffle(typing.cast(str, 42)).map(lambda x: (t, x)).take(example_cap)
        ret.extend(task_with_name)
    return ret
