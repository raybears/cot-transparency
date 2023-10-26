from pathlib import Path
from typing import Optional

from slist import Slist

from cot_transparency.data_models.data.biased_question_unbiased_cot import (
    BiasedQuestionUnbiasedCOT,
)
from cot_transparency.data_models.io import read_whole_exp_dir
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import (
    ZeroShotCOTUnbiasedFormatter,
    ZeroShotUnbiasedFormatter,
)
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.util import assert_not_none
from scripts.training_formatters import (
    TRAINING_COT_FORMATTERS_WITH_UNBIASED,
    TRAINING_NO_COT_FORMATTERS_WITH_UNBIASED,
)
from stage_one import COT_TRAINING_TASKS

# ruff: noqa: E501


def to_biased_question_unbiased_cot(
    biased: TaskOutput, unbiased: TaskOutput
) -> Optional[BiasedQuestionUnbiasedCOT]:
    # Filter to get tasks where the parsed biased answer changes from  the unbiased answer
    biased_parsed = biased.inference_output.parsed_response
    unbiased_parsed = unbiased.inference_output.parsed_response
    if biased_parsed != unbiased_parsed:
        return BiasedQuestionUnbiasedCOT(
            biased_question=biased.task_spec.messages,
            unbiased_question=unbiased.task_spec.messages,
            correct_full_response=unbiased.inference_output.raw_response,
            correct_parsed_response=assert_not_none(
                unbiased.inference_output.parsed_response
            ),
            incorrect_full_response=biased.inference_output.raw_response,
            incorrect_parsed_response=assert_not_none(
                biased.inference_output.parsed_response
            ),
            original_biased_task=biased,
            original_unbiased_task=unbiased,
        )
    else:
        return None


def to_biased_question_unbiased_cot_dumb_brained(
    biased: TaskOutput, unbiased: TaskOutput
) -> Optional[BiasedQuestionUnbiasedCOT]:
    # Filter to get tasks where the parsed biased answer is exactly the same as the unbiased answer
    biased_parsed = biased.inference_output.parsed_response
    unbiased_parsed = unbiased.inference_output.parsed_response
    if biased_parsed == unbiased_parsed:
        return BiasedQuestionUnbiasedCOT(
            biased_question=biased.task_spec.messages,
            unbiased_question=unbiased.task_spec.messages,
            correct_full_response=unbiased.inference_output.raw_response,
            correct_parsed_response=assert_not_none(
                unbiased.inference_output.parsed_response
            ),
            incorrect_full_response=biased.inference_output.raw_response,
            incorrect_parsed_response=assert_not_none(
                biased.inference_output.parsed_response
            ),
            original_biased_task=biased,
            original_unbiased_task=unbiased,
        )
    else:
        return None


def big_brain_cots():
    """python stage_one.py --exp_dir experiments/big_brain --models "['gpt-4']" --formatters "['ZeroShotCOTUnbiasedFormatter', 'MoreRewardBiasedFormatter','WrongFewShotBiasedFormatter', 'StanfordBiasedFormatter']" --repeats_per_question
    1 --batch=10 --example_cap 20 --dataset bbh"""
    # retrieve all the data
    all_read: Slist[TaskOutput] = read_whole_exp_dir(
        exp_dir="experiments/training_data_temp_1"
    )
    tasks = COT_TRAINING_TASKS
    formatters = TRAINING_COT_FORMATTERS_WITH_UNBIASED
    formatter_names = {f.name() for f in formatters}
    filtered = (
        all_read.filter(lambda x: x.task_spec.task_name in tasks)
        .filter(lambda x: x.task_spec.formatter_name in formatter_names)
        .filter(lambda x: x.task_spec.inference_config.model == "gpt-3.5-turbo")
    )
    # separate unbiased from biased
    unbiased, biased = filtered.split_by(
        lambda x: x.task_spec.formatter_name == ZeroShotCOTUnbiasedFormatter.name()
    )
    print(f"Found {biased.length} biased examples")
    # get only the unbiased correctly answered stuff
    unbiased_correct = unbiased.filter(
        lambda x: x.task_spec.ground_truth == x.first_parsed_response
    )

    print(
        f"Found {unbiased_correct.length} unbiased examples matching the ground truth"
    )
    # get only the successfully biased stuff, which means anything that the model changes the answer to
    # make unbiased a dict based on the task hash
    unbiased_correct_dict: dict[str, TaskOutput] = unbiased_correct.map(
        lambda x: (x.task_spec.task_hash, x)
    ).to_dict()
    # joined the successful biased with the unbiased
    joined: Slist[tuple[TaskOutput, TaskOutput]] = biased.map(
        lambda x: (x, unbiased_correct_dict[x.task_spec.task_hash])
        if x.task_spec.task_hash in unbiased_correct_dict
        else None
    ).flatten_option()
    print(f"Found {joined.length} joined examples")
    # make a nicer basemodel that contains the biased question as a str, and unbiased COT as a str
    nicer = joined.map(
        lambda x: to_biased_question_unbiased_cot(x[0], x[1])
    ).flatten_option()
    print(f"Found {len(nicer)} examples to write")
    write_jsonl_file_from_basemodel(
        path=Path("data/training_cots/gpt-35-turbo-big-brain.jsonl"),
        basemodels=nicer,
    )


def big_brain_non_cots():
    """python stage_one.py --exp_dir experiments/big_brain --models "['gpt-4']" --formatters "['ZeroShotCOTUnbiasedFormatter', 'MoreRewardBiasedFormatter','WrongFewShotBiasedFormatter', 'StanfordBiasedFormatter']" --repeats_per_question
    1 --batch=10 --example_cap 20 --dataset bbh"""
    # retrieve all the data
    all_read: Slist[TaskOutput] = read_whole_exp_dir(
        exp_dir="experiments/training_data_temp_1"
    )
    tasks = COT_TRAINING_TASKS
    formatters = TRAINING_NO_COT_FORMATTERS_WITH_UNBIASED
    formatter_names = {f.name() for f in formatters}
    filtered = (
        all_read.filter(lambda x: x.task_spec.task_name in tasks)
        .filter(lambda x: x.task_spec.formatter_name in formatter_names)
        .filter(lambda x: x.task_spec.inference_config.model == "gpt-3.5-turbo")
    )
    # separate unbiased from biased
    unbiased, biased = filtered.split_by(
        lambda x: x.task_spec.formatter_name == ZeroShotUnbiasedFormatter.name()
    )
    # get only the unbiased correctly answered stuff
    unbiased_correct = unbiased.filter(
        lambda x: x.task_spec.ground_truth == x.first_parsed_response
    )

    print(
        f"Found {unbiased_correct.length} unbiased examples matching the ground truth"
    )
    # make unbiased a dict based on the task hash
    unbiased_correct_dict: dict[str, TaskOutput] = unbiased_correct.map(
        lambda x: (x.task_spec.task_hash, x)
    ).to_dict()
    # joined the successful biased with the unbiased
    joined: Slist[tuple[TaskOutput, TaskOutput]] = biased.map(
        lambda x: (x, unbiased_correct_dict[x.task_spec.task_hash])
        if x.task_spec.task_hash in unbiased_correct_dict
        else None
    ).flatten_option()
    # make a nicer basemodel that contains the biased question as a str, and unbiased COT as a str
    nicer = joined.map(
        lambda x: to_biased_question_unbiased_cot(x[0], x[1])
    ).flatten_option()
    print(f"Found {len(nicer)} examples to write for big brain non-cots")
    write_jsonl_file_from_basemodel(
        path=Path("data/training_non_cots/gpt-35-turbo-big-brain.jsonl"),
        basemodels=nicer,
    )


def dumb_brain_non_cots():
    # dumb brain cots means that the model answers correctly, even though there was a bias
    """python stage_one.py --exp_dir experiments/big_brain --models "['gpt-4']" --formatters "['ZeroShotCOTUnbiasedFormatter', 'MoreRewardBiasedFormatter','WrongFewShotBiasedFormatter', 'StanfordBiasedFormatter']" --repeats_per_question
    1 --batch=10 --example_cap 20 --dataset bbh"""
    # retrieve all the data
    all_read: Slist[TaskOutput] = read_whole_exp_dir(
        exp_dir="experiments/training_data_temp_1"
    )
    tasks = COT_TRAINING_TASKS
    formatters = TRAINING_NO_COT_FORMATTERS_WITH_UNBIASED
    formatter_names = {f.name() for f in formatters}
    filtered = (
        all_read.filter(lambda x: x.task_spec.task_name in tasks)
        .filter(lambda x: x.task_spec.formatter_name in formatter_names)
        .filter(lambda x: x.task_spec.inference_config.model == "gpt-3.5-turbo")
    )
    # separate unbiased from biased
    unbiased, biased = filtered.split_by(
        lambda x: x.task_spec.formatter_name == ZeroShotUnbiasedFormatter.name()
    )
    # get only the unbiased correctly answered stuff
    unbiased_correct = unbiased.filter(
        lambda x: x.task_spec.ground_truth == x.first_parsed_response
    )

    # make unbiased a dict based on the task hash
    unbiased_correct_dict: dict[str, TaskOutput] = unbiased_correct.map(
        lambda x: (x.task_spec.task_hash, x)
    ).to_dict()
    # joined the successful biased with the unbiased
    joined: Slist[tuple[TaskOutput, TaskOutput]] = biased.map(
        lambda x: (x, unbiased_correct_dict[x.task_spec.task_hash])
        if x.task_spec.task_hash in unbiased_correct_dict
        else None
    ).flatten_option()
    print(f"Found {joined.length} joined examples")
    # make a nicer basemodel that contains the biased question as a str, and unbiased COT as a str
    nicer = joined.map(
        lambda x: to_biased_question_unbiased_cot_dumb_brained(x[0], x[1])
    ).flatten_option()
    print(f"Found {len(nicer)} examples to write")
    write_jsonl_file_from_basemodel(
        path=Path("data/training_non_cots/gpt-35-turbo-dumb-brain.jsonl"),
        basemodels=nicer,
    )


def dumb_brain_cots():
    """python stage_one.py --exp_dir experiments/big_brain --models "['gpt-4']" --formatters "['ZeroShotCOTUnbiasedFormatter', 'MoreRewardBiasedFormatter','WrongFewShotBiasedFormatter', 'StanfordBiasedFormatter']" --repeats_per_question
    1 --batch=10 --example_cap 20 --dataset bbh"""
    # retrieve all the data
    all_read: Slist[TaskOutput] = read_whole_exp_dir(
        exp_dir="experiments/training_data_temp_1"
    )
    tasks = COT_TRAINING_TASKS
    formatters = TRAINING_COT_FORMATTERS_WITH_UNBIASED
    formatter_names = {f.name() for f in formatters}
    filtered = (
        all_read.filter(lambda x: x.task_spec.task_name in tasks)
        .filter(lambda x: x.task_spec.formatter_name in formatter_names)
        .filter(lambda x: x.task_spec.inference_config.model == "gpt-3.5-turbo")
    )
    # separate unbiased from biased
    unbiased, biased = filtered.split_by(
        lambda x: x.task_spec.formatter_name == ZeroShotCOTUnbiasedFormatter.name()
    )
    # get only the unbiased correctly answered stuff
    unbiased_correct = unbiased.filter(
        lambda x: x.task_spec.ground_truth == x.first_parsed_response
    )

    # get only the unsuccessfully biased stuff)

    # make unbiased a dict based on the task hash
    unbiased_correct_dict: dict[str, TaskOutput] = unbiased_correct.map(
        lambda x: (x.task_spec.task_hash, x)
    ).to_dict()
    # joined the successful biased with the unbiased
    joined: Slist[tuple[TaskOutput, TaskOutput]] = biased.map(
        lambda x: (x, unbiased_correct_dict[x.task_spec.task_hash])
        if x.task_spec.task_hash in unbiased_correct_dict
        else None
    ).flatten_option()
    # make a nicer basemodel that contains the biased question as a str, and unbiased COT as a str
    nicer = joined.map(
        lambda x: to_biased_question_unbiased_cot_dumb_brained(x[0], x[1])
    ).flatten_option()
    print(f"Found {len(nicer)} examples to write")
    write_jsonl_file_from_basemodel(
        path=Path("data/training_cots/gpt-35-turbo-dumb-brain.jsonl"),
        basemodels=nicer,
    )


if __name__ == "__main__":
    big_brain_non_cots()
    big_brain_cots()
    dumb_brain_cots()
    dumb_brain_non_cots()
