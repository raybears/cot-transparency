from typing import Type
from cot_transparency.data_models.data.aqua import AquaExample
from cot_transparency.data_models.data.arc import ArcExample
from cot_transparency.data_models.data.bbh import BBH_TASK_LIST, MilesBBHRawData
from cot_transparency.data_models.data.hellaswag import HellaSwagExample
from cot_transparency.data_models.data.john_math import JohnMath
from cot_transparency.data_models.data.logiqa import LogicQaExample
from cot_transparency.data_models.data.mmlu import MMLUExample
from cot_transparency.data_models.data.truthful_qa import TruthfulQAExample
from cot_transparency.data_models.data.bbq import BBQExample
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
    elif task_name == "arc_easy_train":
        return ArcExample
    elif task_name == "arc_challenge":
        return ArcExample
    elif task_name == "arc_challenge_train":
        return ArcExample
    elif task_name == "truthful_qa":
        return TruthfulQAExample
    elif task_name == "openbook_qa":
        return ArcExample
    elif task_name == "openbook_qa_train":
        return ArcExample
    elif task_name == "john_level_3":
        return JohnMath
    elif task_name == "john_level_4":
        return JohnMath
    elif task_name == "john_level_5":
        return JohnMath
    elif task_name in BBH_TASK_LIST:
        return MilesBBHRawData
    elif task_name == "bbq":
        return BBQExample
    else:
        raise ValueError(f"Unknown task name {task_name}")
