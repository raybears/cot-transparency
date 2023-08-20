from typing import Type
from cot_transparency.data_models.data.aqua import AquaExample
from cot_transparency.data_models.data.arc import ArcExample
from cot_transparency.data_models.data.hellaswag import HellaSwagExample
from cot_transparency.data_models.data.logiqa import LogicQaExample
from cot_transparency.data_models.data.mmlu import MMLUExample
from cot_transparency.data_models.data.truthful_qa import TruthfulQAExample
from cot_transparency.data_models.example_base import DataExampleBase


def task_name_to_data_example(task_name: str) -> Type[DataExampleBase]:
    if task_name == "aqua":
        return AquaExample
    elif task_name == "logiqa":
        return LogicQaExample
    elif task_name == "hellaswag":
        return HellaSwagExample
    elif task_name == "mmlu":
        return MMLUExample
    elif task_name == "arc_easy":
        return ArcExample
    elif task_name == "arc_challenge":
        return ArcExample
    elif task_name == "truthful_qa":
        return TruthfulQAExample
    elif task_name == "openbook_qa":
        return ArcExample
    else:
        raise ValueError(f"Unknown task name {task_name}")
