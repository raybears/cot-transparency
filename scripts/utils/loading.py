from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

from slist import Slist

from cot_transparency.data_models.models import TaskOutput, ExperimentJsonFormat
from cot_transparency.formatters import name_to_formatter
from cot_transparency.data_models.io import read_done_experiment
import pandas as pd


def read_all_for_selections(
    exp_dirs: Sequence[Path],
    formatters: Sequence[str],
    models: Sequence[str],
    tasks: Sequence[str],
    interventions: Sequence[str | None] = [],
) -> Slist[TaskOutput]:
    # More efficient than to load all the experiments in a directory
    task_outputs: Slist[TaskOutput] = Slist()
    # Add None to interventions if empty
    interventions_none = [None] if not interventions else interventions
    for exp_dir in exp_dirs:
        for formatter in formatters:
            for task in tasks:
                for model in models:
                    for intervention in interventions_none:
                        if intervention is None:
                            path = exp_dir / f"{task}/{model}/{formatter}.json"
                        else:
                            path = exp_dir / f"{task}/{model}/{formatter}_and_{intervention}.json"
                        experiment: ExperimentJsonFormat = read_done_experiment(path)
                        task_outputs.extend(experiment.outputs)
    return task_outputs


class BaseExtractor(ABC):
    """
    column_names: list[str], the names of the columns that will be extracted
    extract: (output: TaskOutput) -> Sequence[str | float | None | bool]
        the function that will be used to extract the data must return a list of the same length as column_names
    """

    column_names: list[str]

    @abstractmethod
    def extract(self, output: TaskOutput) -> Sequence[str | float | None | bool]:
        pass


class BasicExtractor(BaseExtractor):
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


class IsCoTExtractor(BaseExtractor):
    column_names = ["is_cot"]

    def extract(self, output: TaskOutput) -> Sequence[bool]:
        formatter = name_to_formatter(output.task_spec.formatter_name)
        return [formatter.is_cot]


def convert_slist_to_df(slist: Slist[TaskOutput], extractors: list[BaseExtractor]) -> pd.DataFrame:  # type: ignore
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
