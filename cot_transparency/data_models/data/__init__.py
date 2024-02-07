from functools import lru_cache
from pathlib import Path
from typing import Optional

from slist import Slist

from cot_transparency.data_models.data import (
    aqua,
    arc,
    bbh,
    bbq,
    bbq_weak_evidence,
    hellaswag,
    logiqa,
    mmlu,
    openbook,
    truthful_qa,
    stereoset,
    bias_in_bios,
    winomt_bias,
    discrim_eval,
)
from cot_transparency.data_models.data.bbh import BBH_TASK_LIST
from cot_transparency.data_models.data.bbh_biased_wrong_cot import BiasedWrongCOTBBH
from cot_transparency.data_models.data.bbq import BBQ_TASK_LIST
from cot_transparency.data_models.data.discrim_eval import DISCRIM_EVAL_TASKS_LIST
from cot_transparency.data_models.data.gsm import load_gsm_biased, load_gsm_unbiased
from cot_transparency.data_models.data.inverse_scaling import get_inverse_scaling, InverseScalingTask
from cot_transparency.data_models.data.john_math import (
    get_john_math_level_1,
    get_john_math_level_2,
    get_john_math_level_3,
    get_john_math_level_4,
    get_john_math_level_5,
)
from cot_transparency.data_models.data.karina_hallucination import get_karina_hallucination
from cot_transparency.data_models.data.model_written_evals import (
    get_anthropic_nlp,
    get_anthropic_phil,
    get_anthropic_pol,
)
from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel

COT_TESTING_TASKS = ["truthful_qa", "logiqa", "hellaswag", "mmlu"]
# if you really want to test on these tasks, we leave out a validation set during finetuning
# but in general, we don't recommend testing on these tasks.
# Please use the COT testing tasks instead, which are totally distinct tasks
COT_TRAINING_TASKS = BBH_TASK_LIST + [
    "arc_easy_train",
    "arc_challenge_train",
    "arc_easy_test",
    "arc_challenge_test",
    "openbook_qa_train",
]
# social biases
STEREOSET = ["stereoset_intra", "stereoset_inter"]
DISCRIM_EVAL_ALL = DISCRIM_EVAL_TASKS_LIST
BIAS_IN_BIOS = ["bios_gender", "bios_profession"]
SOCIAL_BIASES = ["winomt_gender"] + STEREOSET + BIAS_IN_BIOS

TASK_LIST = {
    "bbh": BBH_TASK_LIST,
    "bbh_biased_wrong_cot": BBH_TASK_LIST,
    "transparency": [
        "aqua",
        "arc_easy",
        "arc_challenge",
        "truthful_qa",
        "logiqa",
        "mmlu",
        "openbook_qa",
        "hellaswag",
    ],
    "bbq": BBQ_TASK_LIST,
    "cot_training": COT_TRAINING_TASKS,
    "cot_testing": COT_TESTING_TASKS,
    "deceptive_training": ["aqua_train"],
    "model_written_evals": ["nlp", "phil", "pol"],
    "john_math": [
        "john_level_1",
        "john_level_2",
        "john_level_3",
        "john_level_4",
        "john_level_5",
    ],
    "mmlu": mmlu.MMLU_SUPERCATEGORIES,
    "mmlu_easy": ["mmlu_easy_train", "mmlu_easy_test"],
    "karina": ["karina_hallucination"],
    "logiqa_train": ["logiqa_train"],
    "inverse_scaling": InverseScalingTask.all_tasks(),
    "gsm": ["gsm_unbiased", "gsm_biased"],
    "bbq_full": ["bbq_ambig", "bbq_disambig"],
    "bbq_weak_evidence": ["bbq_weak_evidence"],
    "stereoset": STEREOSET,
    "discrim_eval": DISCRIM_EVAL_TASKS_LIST,
    "bias_in_bios": BIAS_IN_BIOS,
    "social_biases": SOCIAL_BIASES,
}


@lru_cache(maxsize=10)
def get_list_of_examples(
    task: str,
    dataset: Optional[str] = None,
) -> Slist[DataExampleBase]:
    data = None
    if dataset == "bbh_biased_wrong_cot":
        data = read_jsonl_file_into_basemodel(Path("data/bbh_biased_wrong_cot/data.jsonl"), BiasedWrongCOTBBH).filter(
            lambda x: x.task == task
        )
    elif task in TASK_LIST["bbh"]:
        data = bbh.val(task)
    elif task in TASK_LIST["bbq"]:
        data = bbq.val(task)
    else:
        if task == "aqua":
            data = aqua.dev()
        if task == "aqua_train":
            data = aqua.train()
        elif task == "arc_easy":
            data = arc.arc_easy_dev()
        elif task == "arc_easy_train":
            data = arc.arc_easy_train()
        elif task == "arc_easy_test":
            data = arc.arc_easy_test()
        elif task == "arc_challenge":
            data = arc.arc_challenge_dev()
        elif task == "arc_challenge_train":
            data = arc.arc_challenge_train()
        elif task == "arc_challenge_test":
            data = arc.arc_challenge_test()
        elif task == "truthful_qa":
            data = truthful_qa.eval()
        elif task == "logiqa":
            data = logiqa.eval()
        elif task == "logiqa_train":
            data = logiqa.train()
        elif task == "mmlu":
            questions_per_task = 20
            data = mmlu.test(questions_per_task=questions_per_task)
        elif task == "mmlu_easy_train":
            data = mmlu.easy_train()
        elif task == "mmlu_easy_test":
            data = mmlu.easy_test()
        elif task == "karina_hallucination":
            data = get_karina_hallucination()
        elif task in mmlu.MMLU_SUPERCATEGORIES:
            data = mmlu.test_super_category(task.replace("mmlu_", ""))
        elif task == "openbook_qa":
            data = openbook.test()
        elif task in InverseScalingTask.all_tasks():
            task_enum = InverseScalingTask(task)
            data = get_inverse_scaling(task_enum)
        elif task == "openbook_qa_train":
            data = openbook.openbook_train()
        elif task == "hellaswag":
            data = hellaswag.val()
        elif task == "nlp":
            data = get_anthropic_nlp()
        elif task == "phil":
            data = get_anthropic_phil()
        elif task == "pol":
            data = get_anthropic_pol()
        elif task == "john_level_1":
            data = get_john_math_level_1()
        elif task == "john_level_2":
            data = get_john_math_level_2()
        elif task == "john_level_3":
            data = get_john_math_level_3()
        elif task == "john_level_4":
            data = get_john_math_level_4()
        elif task == "john_level_5":
            data = get_john_math_level_5()
        elif task == "gsm_unbiased":
            data = load_gsm_unbiased()
        elif task == "gsm_biased":
            data = load_gsm_biased()
        elif task == "bbq_ambig":
            data = bbq.val_full(context_condition="ambig")
        elif task == "bbq_disambig":
            data = bbq.val_full(context_condition="disambig")
        elif task == "bbq_weak_evidence":
            data = bbq_weak_evidence.val()
        elif task == "stereoset_intra":
            data = stereoset.intra_dev()
        elif task == "stereoset_inter":
            data = stereoset.inter_dev()
        elif task == "bios_gender":
            data = bias_in_bios.gender_test()
        elif task == "bios_profession":
            data = bias_in_bios.profession_test()
        elif task == "discrim_eval_baseline":
            data = discrim_eval.discrim_eval_baseline()
        elif task == "discrim_eval_black_fixed":
            data = discrim_eval.discrim_eval_black_fixed()
        elif task == "discrim_eval_black":
            data = discrim_eval.discrim_eval_black()
        elif task == "discrim_eval_hispanic":
            data = discrim_eval.discrim_eval_hispanic()
        elif task == "discrim_eval_asian":
            data = discrim_eval.discrim_eval_asian()
        elif task == "discrim_eval_native_american":
            data = discrim_eval.discrim_eval_native_american()
        elif task == "discrim_eval_female":
            data = discrim_eval.discrim_eval_female()
        elif task == "discrim_eval_non_binary":
            data = discrim_eval.discrim_eval_non_binary()
        elif task == "discrim_eval_age_20":
            data = discrim_eval.discrim_eval_age_20()
        elif task == "discrim_eval_age_30":
            data = discrim_eval.discrim_eval_age_30()
        elif task == "discrim_eval_age_40":
            data = discrim_eval.discrim_eval_age_40()
        elif task == "discrim_eval_age_50":
            data = discrim_eval.discrim_eval_age_50()
        elif task == "discrim_eval_age_60":
            data = discrim_eval.discrim_eval_age_60()
        elif task == "discrim_eval_age_70":
            data = discrim_eval.discrim_eval_age_70()
        elif task == "discrim_eval_age_80":
            data = discrim_eval.discrim_eval_age_80()
        elif task == "discrim_eval_age_90":
            data = discrim_eval.discrim_eval_age_90()
        elif task == "discrim_eval_age_100":
            data = discrim_eval.discrim_eval_age_100()
        elif task == "winomt_gender":
            data = winomt_bias.test()

    if data is None:
        raise ValueError(f"dataset and or task is not valid. Valid datasets are {list(TASK_LIST.keys())}")
    return data  # type: ignore


PROMPT_SEN_TESTING_TASKS = [
    "truthful_qa",
    "logiqa",
    "hellaswag",
    "aqua",
] + mmlu.MMLU_SUPERCATEGORIES
