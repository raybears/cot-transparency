import asyncio
from pathlib import Path
from typing import Any
from git import Sequence

import pandas as pd
from pydantic import BaseModel
from slist import A, Slist, Group

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.openai import OpenAICompletionPrompt
from cot_transparency.data_models.config import config_from_default
from cot_transparency.data_models.hashable import HashableBaseModel
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.data_models.models import TaskOutput, TaskSpec
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.more_biases.anchor_initial_wrong import (
    PostHocNoPlease,
)
from cot_transparency.formatters.more_biases.distractor_fact import FirstLetterDistractor
from cot_transparency.formatters.more_biases.random_bias_formatter import RandomBiasedFormatter
from cot_transparency.formatters.more_biases.user_wrong_cot import (
    DistractorAnswerWithoutInfluence,
    DistractorArgumentCorrectOrWrong,
    DistractorArgumentImportant,
    DistractorArgumentNoTruthfullyAnswer,
    DistractorArgumentNotsure,
    ImprovedDistractorArgument,
)
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotMoreClearlyLabelledAtBottom,
)
from cot_transparency.formatters.verbalize.formatters import BlackSquareBiasedFormatter
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from cot_transparency.util import assert_not_none
from scripts.are_you_sure.eval_are_you_sure_second_cot import (
    OutputWithAreYouSure,
    run_are_you_sure_multi_model_second_round_cot_mmlu_test,
)
from scripts.automated_answer_parsing.answer_parsing_example import answer_finding_step
from scripts.training_formatters import TRAINING_COT_FORMATTERS, TRAINING_NO_COT_FORMATTERS

all_training_formatters = Slist(TRAINING_COT_FORMATTERS) + Slist(TRAINING_NO_COT_FORMATTERS)


class ModelMeta(HashableBaseModel):
    name: str
    bias_name: str

    def __hash__(self) -> int:
        return int(self.model_hash(), 16)


def accuracy_for_biases(tasks: Slist[TaskOutput]) -> Slist[Group[str, float]]:
    # group by formatter
    grouped = tasks.group_by(lambda x: x.task_spec.formatter_name).map(
        lambda group: group.map_values(lambda task_list: task_list.map(lambda task: task.is_correct).average_or_raise())
    )
    return grouped


def answer_matching_for_biases(tasks: Slist[TaskOutput]) -> Slist[Group[str, float]]:
    # group by formatter
    # need to filter out to get those that has the bias on the wrong answer for so grug don't need to brain so much

    # Print the Number of Nones per model and formatter

    grouped_by_model_and_formatter = tasks.group_by(
        lambda x: (ModelMeta(name=x.task_spec.inference_config.model, bias_name=x.task_spec.formatter_name))
    )
    counts = grouped_by_model_and_formatter.map(
        lambda group: group.map_values(lambda x: x.map(lambda val: val.inference_output.parsed_response is None).sum())
    ).to_dict()

    for k, v in counts.items():
        print(k, v)

    grouped = (
        tasks.filter(lambda task: task.bias_on_wrong_answer)
        # .filter(lambda task: task.inference_output.parsed_response is not None)
        .group_by(lambda x: x.task_spec.formatter_name).map(
            lambda group: group.map_values(
                lambda task_list: task_list.map(lambda task: task.parsed_response_on_bias).average_or_raise()
            )
        )
    )
    return grouped


def answer_matching_improvement_over_control(
    intervention_model: str,
    control_model: str,
    tasks: Slist[TaskOutput],
) -> Slist[Group[str, float]]:
    """More negative is better"""
    intervention_tasks = tasks.filter(lambda x: x.task_spec.inference_config.model == intervention_model)
    control_tasks = tasks.filter(lambda x: x.task_spec.inference_config.model == control_model)
    intervention_matching = answer_matching_for_biases(intervention_tasks)
    control_matching = answer_matching_for_biases(control_tasks).to_dict()
    return intervention_matching.map(
        lambda group: group.map_values(lambda group_val: group_val - control_matching[group.key])
    )


def make_heading_name(name: str, model: str) -> str:
    return f"{name} (model ending {model[-6:]})"


def questions_of_gpt_35_to_omit_labelling(tasks: Slist[TaskOutput]) -> set[str]:
    # we want to omit the questions that
    # 1. are from gpt-3.5-turbo-0613
    # 2. are from the ZeroShotCOTUnbiasedFormatter formatter
    # 3. responded with the answer that we are going to bias on
    gpt_35_not_biased_questions = (
        tasks.filter(
            lambda x: x.task_spec.inference_config.model == "gpt-3.5-turbo-0613"
            and x.task_spec.formatter_name == ZeroShotCOTUnbiasedFormatter.name()
        )
        .filter(lambda x: x.parsed_response_on_bias is True)
        .map(lambda x: x.get_task_spec().task_hash)
        .to_set()
    )
    return gpt_35_not_biased_questions


class TaskOutputWithBaselineLabel(BaseModel):
    task: TaskOutput
    baseline_label: str
    biased_ans: str | None

    @staticmethod
    def from_task(task: TaskOutput, baseline_label: str) -> "TaskOutputWithBaselineLabel":
        return TaskOutputWithBaselineLabel(
            task=task, baseline_label=baseline_label, biased_ans=task.task_spec.biased_ans
        )

    @property
    def task_spec(self) -> TaskSpec:
        return self.task.get_task_spec()

    @staticmethod
    def from_are_you_sure_output(output: OutputWithAreYouSure) -> "TaskOutputWithBaselineLabel":
        first_round_output: str = assert_not_none(output.first_round_inference.parsed_response)
        return TaskOutputWithBaselineLabel(
            task=output, baseline_label=first_round_output, biased_ans=f"NOT {first_round_output}"
        )


def filter_for_diff_answer_from_unbiased_baseline(
    biasedtasks: Slist[TaskOutput], unbiased_tasks: Slist[TaskOutput]
) -> Slist[TaskOutputWithBaselineLabel]:
    map_of_unbiased = {task.get_task_spec().task_hash: task for task in unbiased_tasks}
    # assert len(map_of_unbiased) == len(
    #     unbiased_tasks
    # ), f"Duplicate task hashes in, excepted {len(unbiased_tasks)}, got {len(map_of_unbiased)}"
    with_label = biasedtasks.filter(lambda x: x.get_task_spec().task_hash in map_of_unbiased).map(
        lambda x: TaskOutputWithBaselineLabel.from_task(
            task=x,
            baseline_label=assert_not_none(
                map_of_unbiased[x.get_task_spec().task_hash].inference_output.parsed_response
            ),
        )
    )
    print(f"Filtered {len(biasedtasks)} to {len(with_label)} to match hash")
    diff_ans: Slist[TaskOutputWithBaselineLabel] = with_label.filter(
        lambda x: x.task.inference_output.parsed_response != x.baseline_label
    )
    print(f"Filtered {len(with_label)} to {len(diff_ans)} to have different answer")
    return diff_ans


def biased_on_wrong_answer_and_answered_in_line_with_bias(task: TaskOutputWithBaselineLabel) -> bool:
    return (
        task.task.bias_on_wrong_answer is True
        and task.task.parsed_response_on_bias is True
        and task.task.get_task_spec().formatter_name != ZeroShotCOTUnbiasedFormatter.name()
    )


class CategorisedTaskOutput(BaseModel):
    # For easy subsetting when we want to do analysis
    task: TaskOutput
    baseline_answer: str
    category: str
    biased_ans: str | None

    @staticmethod
    def from_task_output(task: TaskOutputWithBaselineLabel, category: str) -> "CategorisedTaskOutput":
        return CategorisedTaskOutput(
            task=task.task, category=category, baseline_answer=task.baseline_label, biased_ans=task.biased_ans
        )


def messages_except_for_first_and_second(messages: Sequence[ChatMessage], final_answer: str) -> str:
    truncated = list(messages[3:]) + [ChatMessage(role=MessageRole.assistant, content=final_answer)]
    return OpenAICompletionPrompt(messages=truncated).format()


def task_output_to_label_dict(categorised_task: CategorisedTaskOutput) -> dict[str, Any]:
    # Hack to display the "Are you sure?" part of the Are You Sure bias
    task = categorised_task.task
    if task.get_task_spec().formatter_name == "AreYouSureSecondRoundCot":
        full_answer = messages_except_for_first_and_second(
            task.get_task_spec().messages, final_answer=task.inference_output.raw_response
        )
    else:
        full_answer = task.inference_output.raw_response

    return {
        "biased_question": str(OpenAICompletionPrompt(messages=task.get_task_spec().messages)),
        "category": categorised_task.category,
        "task_hash": task.get_task_spec().task_hash,
        "formatter": task.task_spec.formatter_name,
        "model": task.task_spec.inference_config.model,
        "ground_truth": task.task_spec.ground_truth,
        "biased_answer": categorised_task.biased_ans,
        "unbiased_baseline_answer": categorised_task.baseline_answer,
        "parsed_answer": task.inference_output.parsed_response,
        "unbiased_question": task.get_task_spec().get_data_example_obj().get_parsed_input(),
        "full_answer": full_answer,
    }


def unbiased_correct_samples(tasks: Slist[TaskOutput]) -> Slist[TaskOutput]:
    # must be "gpt-3.5-turbo-0613" and ZeroShotCOTUnbiasedFormatter
    # must be correct
    return tasks.filter(
        lambda x: x.task_spec.inference_config.model == "gpt-3.5-turbo-0613"
        and x.task_spec.formatter_name == ZeroShotCOTUnbiasedFormatter.name()
        and x.is_correct
    )


def biased_correct_samples(tasks: Slist[TaskOutput]) -> Slist[TaskOutput]:
    # must be "gpt-3.5-turbo-0613" and NOT ZeroShotCOTUnbiasedFormatter
    # must be correct, and the bias must be on the correct answer
    return tasks.filter(
        lambda x: x.task_spec.inference_config.model == "gpt-3.5-turbo-0613"
        and x.task_spec.formatter_name != ZeroShotCOTUnbiasedFormatter.name()
        and x.is_correct
        and x.parsed_response_on_bias is True
    )


def csv_for_labelling(_tasks: Sequence[TaskOutputWithBaselineLabel], number_labellers: int) -> None:
    tasks = Slist(_tasks).shuffle(seed="42")

    in_training_dist, out_of_training_dist = tasks.split_by(
        lambda x: x.task_spec.formatter_name == RandomBiasedFormatter
    )
    assert len(in_training_dist) >= 0
    assert len(out_of_training_dist) >= 0

    named_in_training_dist = in_training_dist.map(
        lambda x: CategorisedTaskOutput.from_task_output(x, "in_training_dist_biased_wrong_answer")
    )
    named_out_of_training_dist = out_of_training_dist.map(
        lambda x: CategorisedTaskOutput.from_task_output(x, "out_of_training_dist_biased_wrong_answer")
    )

    all_to_label = (
        named_in_training_dist
        + named_out_of_training_dist
        #   + unbiased_correct + biased_correct
    )

    print(f"Before rounding: {len(all_to_label)}")

    # group by model
    grouped_by_model: Slist[Group[str, Slist[CategorisedTaskOutput]]] = all_to_label.group_by(
        lambda x: x.task.task_spec.inference_config.model
    ).map_on_group_values(
        # do the rounding
        # e.g. if you have 301 questions, and 3 labellers, you want to limit to 300 samples
        # do this within each model so that its still stratified
        lambda group: group.shuffle(seed="42").take(n=(len(group) // number_labellers) * number_labellers)
    )
    # Itereate over the groups
    labeller_items_to_write: Slist[Slist[CategorisedTaskOutput]] = Slist(Slist() for _ in range(number_labellers))
    for model, items_to_split in grouped_by_model:
        print(f"Splitting {model=} number of items {len(items_to_split)} among {number_labellers=}")
        # split items into n
        item: CategorisedTaskOutput
        for idx, item in enumerate(items_to_split):
            for_labeller = idx % number_labellers
            labeller_list: Slist[CategorisedTaskOutput] = labeller_items_to_write[for_labeller]
            labeller_list.append(item)

    for i, labeller_qns in enumerate(labeller_items_to_write):
        # shuffle the qns
        shuffled_for_labeller: Slist[CategorisedTaskOutput] = labeller_qns.shuffle(seed="42")
        print(f"Labeller {i} has {len(shuffled_for_labeller)} questions")
        df = pd.DataFrame(shuffled_for_labeller.map(task_output_to_label_dict))
        # remove index
        df.to_csv(f"to_label_{i}.csv", index=False)


def take_or_raise(slist: Slist[A], n: int, group_name: str) -> Slist[A]:
    if len(slist) < n:
        raise ValueError(f"Group name {group_name}: Expected at least {n} items, got {len(slist)}")
    return slist.shuffle(seed="42").take(n)


async def eval_grid() -> None:
    # FAR
    # openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    stage_one_path = Path("experiments/grid_exp")
    stage_one_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path, write_every_n=500)
    # test on COTs only, maybe non-COTs when we feel like it

    to_run_str: Slist[str] = Slist(
        [
            RandomBiasedFormatter,  # Suggested answer
            PostHocNoPlease,  # PostHoc
            WrongFewShotMoreClearlyLabelledAtBottom,  # Wrong Few Shot
            BlackSquareBiasedFormatter,  # Spurious Few Shot
            FirstLetterDistractor,
        ]
    ).map(lambda x: x.name())

    models = [
        "gpt-3.5-turbo-0613",
        # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8km8ORRL",  # control
        # "ft:gpt-3.5-turbo-0613:far-ai::8inQNPtE",  # non-cot
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8rsmiJe7",  # control
        "ft:gpt-3.5-turbo-0613:far-ai::8rwdMKOn",  # intervention
    ]
    example_cap = 800
    stage_one_obs = stage_one_stream(
        formatters=to_run_str,
        # dataset="cot_testing",
        tasks=["mmlu_test"],
        example_cap=example_cap,
        num_tries=1,
        raise_after_retries=False,
        # temp 0
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
    )

    answer_parsing_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path / "answer_parsing_cache")
    config = config_from_default(model="gpt-4")
    stage_one_obs = stage_one_obs.map_blocking_par(lambda x: answer_finding_step(x, answer_parsing_caller, config))

    biased_results_wo_are_you_sure: Slist[TaskOutput] = await stage_one_obs.to_slist()

    # these are run separately because we need to later groupby the model + task_hash and sample one of the prompts
    # distractor arguments is the only one with prompt variants
    distractor_arguments = [
        ImprovedDistractorArgument,  # Distractor Argument V2
        DistractorAnswerWithoutInfluence,
        DistractorArgumentCorrectOrWrong,
        DistractorArgumentImportant,
        DistractorArgumentNotsure,
        DistractorArgumentNoTruthfullyAnswer,
    ]
    distractor_obs = stage_one_stream(
        formatters=Slist(distractor_arguments).map(lambda x: x.name()),
        tasks=["mmlu_test"],
        example_cap=example_cap,
        num_tries=1,
        raise_after_retries=False,
        # temp 0
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
    )

    distractor_results: Slist[TaskOutput] = await distractor_obs.to_slist()
    sampled_distractor: Slist[TaskOutput] = (
        distractor_results.filter(lambda x: x.first_parsed_response is not None)
        .group_by(lambda x: x.task_spec.inference_config.model + x.task_spec.task_hash)
        .map_on_group_values(lambda prompt_variants: prompt_variants.sample(seed="42", n=1))
        .ungroup()
        .map(
            # rename so that it looks like one single formatter when we group by formatter
            lambda x: x.update_formatter_name("Distractor Argument")
        )
    )

    # need to run more because we filter for qns that the first round gets correct
    are_you_sure_cap = example_cap * 2
    # are you sure function filters for bias_on_wrong_answer
    _are_you_sure_second_round_cot: Slist[
        OutputWithAreYouSure
    ] = await run_are_you_sure_multi_model_second_round_cot_mmlu_test(
        models=models, caller=stage_one_caller, example_cap=are_you_sure_cap
    )

    obs_for_baseline = stage_one_stream(
        formatters=[ZeroShotCOTUnbiasedFormatter.name()],
        # dataset="cot_testing",
        tasks=["mmlu_test"],
        example_cap=are_you_sure_cap,
        num_tries=1,
        raise_after_retries=False,
        # temp 0
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=["gpt-3.5-turbo-0613"],
    )
    baseline_results: Slist[TaskOutput] = await obs_for_baseline.to_slist()

    # save callers
    stage_one_caller.save_cache()
    answer_parsing_caller.save_cache()

    # lower limit so that we have the same number of examples for each bias
    limit_per_bias = int(example_cap / 2)
    # results limited 1000 examples per formatter and model
    print(f"Got {len(biased_results_wo_are_you_sure)} results")
    capped_results = (
        (biased_results_wo_are_you_sure + sampled_distractor)
        .filter(lambda x: x.first_parsed_response is not None)
        .group_by(lambda x: x.task_spec.formatter_name)
        .map_2(
            (lambda group_name, group_values: take_or_raise(group_values, n=limit_per_bias, group_name=str(group_name)))
        )
        .flatten_list()
    )
    capped_for_are_you_sure: Slist[TaskOutputWithBaselineLabel] = (
        _are_you_sure_second_round_cot.filter(lambda x: x.first_parsed_response is not None)
        .group_by(lambda x: x.task_spec.inference_config.model)
        .map_2(
            (lambda group_name, group_values: take_or_raise(group_values, n=limit_per_bias, group_name=str(group_name)))
        )
        .flatten_list()
        .map(
            # This makes the first round the baseline
            lambda x: TaskOutputWithBaselineLabel.from_are_you_sure_output(x)
        )
        .filter(
            # Filter for those that are different from the baseline
            lambda x: x.baseline_label
            != x.task.inference_output.parsed_response
        )
    )
    print(f"For are you sure, got {len(capped_for_are_you_sure)} results")

    # We don't process for are you sure here, because the baseline is different (0%) rather than the GPT-3.5 unbiased context
    diff_answers: Slist[TaskOutputWithBaselineLabel] = filter_for_diff_answer_from_unbiased_baseline(
        biasedtasks=capped_results,
        unbiased_tasks=baseline_results.filter(lambda x: x.first_parsed_response is not None),
    ).filter(biased_on_wrong_answer_and_answered_in_line_with_bias)

    print("Got results, making csvs")

    # with_are_you_sure: Slist[OutputWithAreYouSure] = await run_are_you_sure_cot_multi_model_tasks(
    #     caller=stage_one_caller, models=models, tasks=["mmlu"], example_cap=600
    # )

    # # save results
    # save_per_model_results(results=results, results_dir=stage_one_path / "results")
    csv_for_labelling(_tasks=diff_answers + capped_for_are_you_sure, number_labellers=4)


if __name__ == "__main__":
    asyncio.run(eval_grid())
