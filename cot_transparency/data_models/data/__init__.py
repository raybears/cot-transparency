from typing import Type
from cot_transparency.data_models.data.aqua import AquaExample
from cot_transparency.data_models.data.logiqa import LogicQaExample
from cot_transparency.data_models.example_base import DataExampleBase


def task_name_to_data_example(task_name: str) -> Type[DataExampleBase]:
    if task_name == "aqua":
        return AquaExample
    elif task_name == "logiqa":
        return LogicQaExample
    else:
        raise ValueError(f"Unknown task name {task_name}")
