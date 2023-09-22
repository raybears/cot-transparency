from pathlib import Path

from slist import Slist

from cot_transparency.data_models.data.biased_question_unbiased_cot import BiasedQuestionUnbiasedCOT
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.formatters.more_biases.random_bias_formatter import (
    RandomBiasedFormatter,
    RandomAgainstBiasedFormatter,
    RandomBiasedNoCOTFormatter,
    RandomAgainstBiasedNoCOTFormatter,
)
from cot_transparency.formatters.more_biases.more_reward import (
    MoreRewardBiasedFormatter,
    MoreRewardBiasedNoCOTFormatter,
)
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
)
from cot_transparency.formatters.verbalize.formatters import (
    CheckmarkBiasedFormatter,
    CrossBiasedFormatter,
    StanfordBiasedFormatter,
    StanfordNoCOTFormatter,
    CrossNoCOTFormatter,
    CheckmarkNoCOTFormatter,
)
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.util import assert_not_none
from scripts.intervention_investigation import read_whole_exp_dir
from stage_one import COT_TRAINING_TASKS


# ruff: noqa: E501


def to_biased_question_unbiased_cot(biased: TaskOutput, unbiased: TaskOutput) -> BiasedQuestionUnbiasedCOT:
    return BiasedQuestionUnbiasedCOT(
        biased_question=biased.task_spec.messages,
        unbiased_question=unbiased.task_spec.messages,
        correct_full_response=unbiased.inference_output.raw_response,
        correct_parsed_response=assert_not_none(unbiased.inference_output.parsed_response),
        incorrect_full_response=biased.inference_output.raw_response,
        incorrect_parsed_response=assert_not_none(biased.inference_output.parsed_response),
        original_biased_task=biased,
        original_unbiased_task=unbiased,
    )


def big_brain_cots():
    """python stage_one.py --exp_dir experiments/big_brain --models "['gpt-4']" --formatters "['ZeroShotCOTUnbiasedFormatter', 'MoreRewardBiasedFormatter','WrongFewShotBiasedFormatter', 'StanfordBiasedFormatter']" --repeats_per_question
    1 --batch=10 --example_cap 20 --dataset bbh"""
    # retrieve all the data
    all_read: Slist[TaskOutput] = read_whole_exp_dir(exp_dir="experiments/gpt_35_cot")
    tasks = COT_TRAINING_TASKS
    formatters = [
        WrongFewShotIgnoreMistakesBiasedFormatter,
        StanfordBiasedFormatter,
        MoreRewardBiasedFormatter,
        ZeroShotCOTSycophancyFormatter,
        ZeroShotCOTUnbiasedFormatter,
        # DeceptiveAssistantTargetedFormatter,
        CheckmarkBiasedFormatter,
        CrossBiasedFormatter,
        RandomBiasedFormatter,
        # RandomBiasedNoCOTFormatter,
        RandomAgainstBiasedFormatter,
        # RandomAgainstBiasedNoCOTFormatter,
    ]
    formatter_names = {f.name() for f in formatters}
    filtered = (
        all_read.filter(lambda x: x.task_spec.task_name in tasks)
        .filter(lambda x: x.task_spec.formatter_name in formatter_names)
        .filter(lambda x: x.task_spec.inference_config.model == "gpt-3.5-turbo")
    )
    # separate unbiased from biased
    unbiased, biased = filtered.split_by(lambda x: x.task_spec.formatter_name == ZeroShotCOTUnbiasedFormatter.name())
    # get only the unbiased correctly answered stuff
    unbiased_correct = unbiased.filter(lambda x: x.task_spec.ground_truth == x.first_parsed_response)

    # get only the successfully biased stuff, which could mean anything that is not the ground truth
    no_match_ground_truth: Slist[TaskOutput] = biased.filter(
        lambda x: (x.task_spec.ground_truth != x.first_parsed_response)
    )
    # we want 50/50 split where half has a bias leading to the correct answer, half not
    # otherwise the model will learn that the bias is always wrong (which is not true)

    length_success = len(no_match_ground_truth)
    print(f"Found {no_match_ground_truth.length} examples not matching the ground truth")
    match_ground_truth = (
        biased.filter(lambda x: x.task_spec.ground_truth == x.inference_output.parsed_response)
        .shuffle(seed="42")
        .take(length_success)
    )
    print(f"Found {match_ground_truth.length} examples matching the ground truth")
    # make unbiased a dict based on the task hash
    unbiased_correct_dict: dict[str, TaskOutput] = unbiased_correct.map(lambda x: (x.task_spec.task_hash, x)).to_dict()
    # joined the successful biased with the unbiased
    joined: Slist[tuple[TaskOutput, TaskOutput]] = (
        (match_ground_truth + no_match_ground_truth)
        .map(
            lambda x: (x, unbiased_correct_dict[x.task_spec.task_hash])
            if x.task_spec.task_hash in unbiased_correct_dict
            else None
        )
        .flatten_option()
    )
    # make a nicer basemodel that contains the biased question as a str, and unbiased COT as a str
    nicer = joined.map(lambda x: to_biased_question_unbiased_cot(x[0], x[1]))
    print(f"Found {len(nicer)} examples to write")
    write_jsonl_file_from_basemodel(
        path=Path("data/training_cots/gpt-35-turbo-big-brain.jsonl"),
        basemodels=nicer,
    )


def big_brain_non_cots():
    """python stage_one.py --exp_dir experiments/big_brain --models "['gpt-4']" --formatters "['ZeroShotCOTUnbiasedFormatter', 'MoreRewardBiasedFormatter','WrongFewShotBiasedFormatter', 'StanfordBiasedFormatter']" --repeats_per_question
    1 --batch=10 --example_cap 20 --dataset bbh"""
    # retrieve all the data
    all_read: Slist[TaskOutput] = read_whole_exp_dir(exp_dir="experiments/gpt_35_cot")
    tasks = COT_TRAINING_TASKS
    formatters = [
        ZeroShotUnbiasedFormatter,
        RandomBiasedNoCOTFormatter,
        RandomAgainstBiasedNoCOTFormatter,
        WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
        MoreRewardBiasedNoCOTFormatter,
        StanfordNoCOTFormatter,
        CrossNoCOTFormatter,
        CheckmarkNoCOTFormatter,
    ]
    formatter_names = {f.name() for f in formatters}
    filtered = (
        all_read.filter(lambda x: x.task_spec.task_name in tasks)
        .filter(lambda x: x.task_spec.formatter_name in formatter_names)
        .filter(lambda x: x.task_spec.inference_config.model == "gpt-3.5-turbo")
    )
    # separate unbiased from biased
    unbiased, biased = filtered.split_by(lambda x: x.task_spec.formatter_name == ZeroShotUnbiasedFormatter.name())
    # get only the unbiased correctly answered stuff
    unbiased_correct = unbiased.filter(lambda x: x.task_spec.ground_truth == x.first_parsed_response)

    # get only the successfully biased stuff, which could mean anything that is not the ground truth
    no_match_ground_truth: Slist[TaskOutput] = biased.filter(
        lambda x: (x.task_spec.ground_truth != x.first_parsed_response)
    )
    # we want 50/50 split where half has a bias leading to the correct answer, half not
    # otherwise the model will learn that the bias is always wrong (which is not true)

    length_success = len(no_match_ground_truth)
    print(f"Found {no_match_ground_truth.length} examples not matching the ground truth")
    match_ground_truth = (
        biased.filter(lambda x: x.task_spec.ground_truth == x.inference_output.parsed_response)
        .shuffle(seed="42")
        .take(length_success)
    )
    print(f"Found {match_ground_truth.length} examples matching the ground truth")
    # make unbiased a dict based on the task hash
    unbiased_correct_dict: dict[str, TaskOutput] = unbiased_correct.map(lambda x: (x.task_spec.task_hash, x)).to_dict()
    # joined the successful biased with the unbiased
    joined: Slist[tuple[TaskOutput, TaskOutput]] = (
        (match_ground_truth + no_match_ground_truth)
        .map(
            lambda x: (x, unbiased_correct_dict[x.task_spec.task_hash])
            if x.task_spec.task_hash in unbiased_correct_dict
            else None
        )
        .flatten_option()
    )
    # make a nicer basemodel that contains the biased question as a str, and unbiased COT as a str
    nicer = joined.map(lambda x: to_biased_question_unbiased_cot(x[0], x[1]))
    print(f"Found {len(nicer)} examples to write")
    write_jsonl_file_from_basemodel(
        path=Path("data/training_non_cots/gpt-35-turbo-big-brain.jsonl"),
        basemodels=nicer,
    )


if __name__ == "__main__":
    big_brain_non_cots()
    big_brain_cots()
