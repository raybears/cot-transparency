from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar

import pandas as pd

from cot_transparency.data_models.models import BaseTaskOuput, TaskOutput
from cot_transparency.formatters import name_to_formatter

T_co = TypeVar("T_co")
T_contra = TypeVar("T_contra", contravariant=True)


class BaseExtractor(ABC, Generic[T_contra]):
    """
    column_names: list[str], the names of the columns that will be extracted
    extract: (output: TaskOutput) -> Sequence[str | float | None | bool]
        the function that will be used to extract the data must return a list of the same length as column_names
    """

    column_names: list[str]

    @abstractmethod
    def extract(self, output: T_contra) -> Sequence[str | float | None | bool]:
        pass


def convert_slist_to_df(
    slist: Sequence[T_co],
    extractors: Sequence[BaseExtractor[T_co]],
) -> pd.DataFrame:  # type: ignore
    """
    Converts an slist of TaskOutputs to a pandas dataframe using the extract method of the extractors passed in.
    """

    columns_names = []
    for extractor in extractors:
        columns_names.extend(extractor.column_names)

    rows = []
    for output in slist:
        row = []
        for extractor in extractors:
            row.extend(extractor.extract(output))
        rows.append(row)

    return pd.DataFrame.from_records(rows, columns=columns_names)


class IsCoTExtractor(BaseExtractor[BaseTaskOuput]):
    column_names = ["is_cot"]

    def extract(self, output: BaseTaskOuput) -> Sequence[bool]:
        formatter = name_to_formatter(output.task_spec.formatter_name)
        return [formatter.is_cot]


class BasicExtractor(BaseExtractor[TaskOutput]):
    column_names = [
        "task_name",
        "task_hash",
        "model",
        "formatter_name",
        "intervention_name",
        "parsed_response",
        "ground_truth",
        "input_hash",
    ]

    def extract(self, output: TaskOutput) -> Sequence[str | float | None]:
        return [
            output.task_spec.task_name,
            output.task_spec.task_hash,
            output.task_spec.inference_config.model,
            output.task_spec.formatter_name,
            output.task_spec.intervention_name,
            output.inference_output.parsed_response,
            output.task_spec.ground_truth,
            output.task_spec.uid(),
        ]


class BiasExtractor(BaseExtractor[TaskOutput]):
    column_names = [
        "bias_ans",
    ]

    def extract(self, output: TaskOutput) -> Sequence[str | float | None]:
        return [
            output.task_spec.biased_ans,
        ]
