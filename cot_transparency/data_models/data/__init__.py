from functools import lru_cache
from pathlib import Path
from typing import Optional

from slist import Slist

from cot_transparency.data_models.data import aqua, arc, bbh, bbq, hellaswag, logiqa, mmlu, openbook, truthful_qa
from cot_transparency.data_models.data.bbh import BBH_TASK_LIST
from cot_transparency.data_models.data.bbh_biased_wrong_cot import BiasedWrongCOTBBH
from cot_transparency.data_models.data.bbq import BBQ_TASK_LIST
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

COT_TESTING_TASKS = ["truthful_qa", "logiqa", "hellaswag", "mmlu_test"]
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
    "mmlu": ["mmlu_train", "mmlu_test"],
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
        elif task == "mmlu_train":
            questions_per_task = 100000
            potential_data = mmlu.test(questions_per_task=questions_per_task)
            # take 50%
            data = potential_data.shuffle("42").take(potential_data.length // 2)
        elif task == "mmlu_test":
            questions_per_task = 100000
            potential_data = mmlu.test(questions_per_task=questions_per_task)
            # take 50%, but the back
            data = potential_data.shuffle("42").reversed().take(potential_data.length // 2)
    

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

    if data is None:
        raise ValueError(f"dataset and or task is not valid. Valid datasets are {list(TASK_LIST.keys())}")
    return data  # type: ignore


PROMPT_SEN_TESTING_TASKS = [
    "truthful_qa",
    "logiqa",
    "hellaswag",
    "aqua",
] + mmlu.MMLU_SUPERCATEGORIES
