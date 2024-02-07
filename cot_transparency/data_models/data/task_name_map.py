from typing import Type

from cot_transparency.data_models.data.aqua import AquaExample
from cot_transparency.data_models.data.arc import ArcExample
from cot_transparency.data_models.data.bbh import BBH_TASK_LIST, MilesBBHRawData
from cot_transparency.data_models.data.bbq import BBQ_TASK_LIST, BBQExample
from cot_transparency.data_models.data.gsm import GSMBiased, GSMUnbiased
from cot_transparency.data_models.data.hellaswag import HellaSwagExample
from cot_transparency.data_models.data.inverse_scaling import InverseScalingExample, InverseScalingTask
from cot_transparency.data_models.data.john_math import JohnMath
from cot_transparency.data_models.data.karina_hallucination import KarinaHallucination
from cot_transparency.data_models.data.logiqa import LogicQaExample
from cot_transparency.data_models.data.mmlu import (
    MMLU_SUPERCATEGORIES,
    MMLU_TASKS,
    MMLUExample,
)
from cot_transparency.data_models.data.model_written_evals import ModelWrittenEvalsExample
from cot_transparency.data_models.data.truthful_qa import TruthfulQAExample
from cot_transparency.data_models.data.bbq_weak_evidence import BBQWeakEvidenceExample
from cot_transparency.data_models.data.stereoset import StereoSetExample
from cot_transparency.data_models.data.bias_in_bios import (
    BiasInBiosGenderExample,
    BiasInBiosProfessionExample,
)
from cot_transparency.data_models.data.discrim_eval import DiscrimEval, DISCRIM_EVAL_TASKS_LIST
from cot_transparency.data_models.data.winomt_bias import WinoMTExample
from cot_transparency.data_models.example_base import DataExampleBase


def task_name_to_data_example(task_name: str) -> Type[DataExampleBase]:
    if task_name == "aqua":
        return AquaExample
    elif task_name == "aqua_train":
        return AquaExample
    elif task_name == "logiqa":
        return LogicQaExample
    elif task_name == "logiqa_train":
        return LogicQaExample
    elif task_name == "hellaswag":
        return HellaSwagExample
    elif task_name == "mmlu":
        return MMLUExample
    elif task_name == "mmlu_easy_train":
        return MMLUExample
    elif task_name == "mmlu_easy_test":
        return MMLUExample
    elif task_name in InverseScalingTask.all_tasks():
        return InverseScalingExample
    elif task_name == "arc_easy":
        return ArcExample
    elif task_name == "arc_easy_train":
        return ArcExample
    elif task_name == "arc_easy_test":
        return ArcExample
    elif task_name == "arc_challenge":
        return ArcExample
    elif task_name == "arc_challenge_train":
        return ArcExample
    elif task_name == "arc_challenge_test":
        return ArcExample
    elif task_name == "truthful_qa":
        return TruthfulQAExample
    elif task_name == "openbook_qa":
        return ArcExample
    elif task_name == "openbook_qa_train":
        return ArcExample
    elif task_name == "john_level_1":
        return JohnMath
    elif task_name == "john_level_2":
        return JohnMath
    elif task_name == "john_level_3":
        return JohnMath
    elif task_name == "john_level_4":
        return JohnMath
    elif task_name == "john_level_5":
        return JohnMath
    elif task_name in BBH_TASK_LIST:
        return MilesBBHRawData
    elif task_name in BBQ_TASK_LIST or task_name in ["bbq_ambig", "bbq_disambig"]:
        return BBQExample
    elif task_name in MMLU_TASKS or task_name in MMLU_SUPERCATEGORIES:
        return MMLUExample
    elif task_name == "karina_hallucination":
        return KarinaHallucination
    elif task_name == "gsm_biased":
        return GSMBiased
    elif task_name == "gsm_unbiased":
        return GSMUnbiased
    elif task_name == "nlp":
        return ModelWrittenEvalsExample
    elif task_name == "phil":
        return ModelWrittenEvalsExample
    elif task_name == "pol":
        return ModelWrittenEvalsExample
    elif task_name == "bbq_weak_evidence":
        return BBQWeakEvidenceExample
    elif task_name == "winomt_gender":
        return WinoMTExample
    elif task_name == "bios_gender":
        return BiasInBiosGenderExample
    elif task_name == "bios_profession":
        return BiasInBiosProfessionExample
    elif task_name in ["stereoset_intra", "stereoset_inter"]:
        return StereoSetExample
    elif task_name in DISCRIM_EVAL_TASKS_LIST:
        return DiscrimEval
    else:
        raise ValueError(f"Unknown task name {task_name}")
