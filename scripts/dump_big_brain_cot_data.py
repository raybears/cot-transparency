from pathlib import Path

from slist import Slist

from cot_transparency.data_models.data.biased_question_unbiased_cot import BiasedQuestionUnbiasedCOT
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.util import assert_not_none
from scripts.intervention_investigation import read_whole_exp_dir


# ruff: noqa: E501


def to_biased_question_unbiased_cot(biased: TaskOutput, unbiased: TaskOutput) -> BiasedQuestionUnbiasedCOT:
    return BiasedQuestionUnbiasedCOT(
        biased_question=biased.task_spec.messages,
        correct_full_response=unbiased.inference_output.raw_response,
        correct_parsed_response=assert_not_none(unbiased.inference_output.parsed_response),
        incorrect_full_response=biased.inference_output.raw_response,
        incorrect_parsed_response=assert_not_none(biased.inference_output.parsed_response),
        original_biased_task=biased,
        original_unbiased_task=unbiased,
    )


def big_brain():
    """python stage_one.py --exp_dir experiments/big_brain --models "['gpt-4']" --formatters "['ZeroShotCOTUnbiasedFormatter', 'MoreRewardBiasedFormatter','WrongFewShotBiasedFormatter', 'StanfordBiasedFormatter']" --repeats_per_question
    1 --batch=10 --example_cap 20 --dataset bbh"""
    # retrieve all the data
    all_read: Slist[TaskOutput] = read_whole_exp_dir(exp_dir="experiments/big_brain")
    # separate unbiased from biased
    unbiased, biased = all_read.split_by(lambda x: x.task_spec.formatter_name == ZeroShotCOTUnbiasedFormatter.name())
    # get only the unbiased correctly answered stuff
    unbiased_correct = unbiased.filter(lambda x: x.task_spec.ground_truth == x.first_parsed_response)

    # get only the successfully biased stuff
    success_fully_biased: Slist[TaskOutput] = biased.filter(lambda x: x.task_spec.biased_ans == x.first_parsed_response)
    # make unbiased a dict based on the task hash
    unbiased_dict: dict[str, TaskOutput] = unbiased_correct.map(lambda x: (x.task_spec.task_hash, x)).to_dict()
    # joined the successful biased with the unbiased
    joined: Slist[tuple[TaskOutput, TaskOutput]] = success_fully_biased.map(
        lambda x: (x, unbiased_dict[x.task_spec.task_hash]) if x.task_spec.task_hash in unbiased_dict else None
    ).flatten_option()
    # make a nicer basemodel that contains the biased question as a str, and unbiased COT as a str
    nicer = joined.map(lambda x: to_biased_question_unbiased_cot(x[0], x[1]))
    print(f"Found {len(nicer)} examples to write")
    write_jsonl_file_from_basemodel(
        path=Path("data/bbh_big_brain_cots/data.jsonl"),
        basemodels=nicer,
    )


if __name__ == "__main__":
    big_brain()
