import asyncio
from pathlib import Path
from typing import Any
from annotated_types import Len
from git import Sequence

import pandas as pd
from pydantic import BaseModel
from slist import Slist, Group

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.openai import OpenAICompletionPrompt
from cot_transparency.data_models.config import config_from_default
from cot_transparency.data_models.hashable import HashableBaseModel
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.more_biases.random_bias_formatter import RandomBiasedFormatter
from cot_transparency.formatters.more_biases.user_wrong_cot import ImprovedDistractorArgument
from cot_transparency.formatters.prompt_sensitivity.automated_generations import AskWithDistractorFact
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.automated_answer_parsing.answer_parsing_example import answer_finding_step
from scripts.training_formatters import INTERESTING_FORMATTERS, TRAINING_COT_FORMATTERS, TRAINING_NO_COT_FORMATTERS

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


INTERESTING_FORMATTERS_STR = [x.name() for x in INTERESTING_FORMATTERS]


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


def biased_on_wrong_answer_and_answered_in_line_with_bias(task: TaskOutput) -> bool:
    return (
        task.bias_on_wrong_answer is True
        and task.parsed_response_on_bias is True
        and task.get_task_spec().formatter_name != ZeroShotCOTUnbiasedFormatter.name()
    )


class CategorisedTaskOutput(BaseModel):
    # For easy subsetting when we want to do analysis
    task: TaskOutput
    category: str

    @staticmethod
    def from_task_output(task: TaskOutput, category: str) -> "CategorisedTaskOutput":
        return CategorisedTaskOutput(task=task, category=category)


def task_output_to_label_dict(categorised_task: CategorisedTaskOutput) -> dict[str, Any]:
    task = categorised_task.task
    return {
        "biased_question": str(OpenAICompletionPrompt(messages=task.get_task_spec().messages)),
        "category": categorised_task.category,
        "task_hash": task.get_task_spec().task_hash,
        "formatter": task.task_spec.formatter_name,
        "model": task.task_spec.inference_config.model,
        "ground_truth": task.task_spec.ground_truth,
        "biased_answer": task.task_spec.biased_ans,
        "parsed_answer": task.inference_output.parsed_response,
        "unbiased_question": task.get_task_spec().get_data_example_obj().get_parsed_input(),
        "full_answer": task.inference_output.raw_response,
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


def csv_for_labelling(_tasks: Sequence[TaskOutput], number_labellers: int) -> None:
    tasks = Slist(_tasks).shuffle(seed="42")
    # 1. shuffle everything
    # 2. group by task hash so that the labeller sees the same question consecutively
    # 3. split between to label and not to label
    # 4. write to csv
    ommited_qns: set[str] = questions_of_gpt_35_to_omit_labelling(tasks)
    bias_on_wrong_answer_and_answered_in_line_with_bias_tasks: Slist[TaskOutput] = tasks.filter(
        biased_on_wrong_answer_and_answered_in_line_with_bias
    )
    shuffled: Slist[TaskOutput] = bias_on_wrong_answer_and_answered_in_line_with_bias_tasks.shuffle(seed="42")
    to_label = shuffled.filter(lambda x: x.get_task_spec().task_hash not in ommited_qns)

    in_training_dist, out_of_training_dist = to_label.split_by(
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

    num_to_label = len(to_label)
    print(f"Number of questions to label: {num_to_label=}")

    # find out what is 5% of the number of questions to label
    five_percent = int(num_to_label * 0.05)
    # get the unbiased correct samples
    # unbiased_correct = (
    #     unbiased_correct_samples(tasks)
    #     .shuffle(seed="42")
    #     .take(five_percent)
    #     .map(lambda x: CategorisedTaskOutput.from_task_output(x, "unbiased_correct_answer"))
    # )
    # print(f"Number of unbiased correct samples: {len(unbiased_correct)}")
    # # get the biased correct samples
    # biased_correct = (
    #     biased_correct_samples(tasks)
    #     .shuffle(seed="42")
    #     .take(five_percent)
    #     .map(lambda x: CategorisedTaskOutput.from_task_output(x, "biased_correct_answer"))
    # )
    # print(f"Number of biased correct samples: {len(biased_correct)}")

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


def accuracy_intervention_vs_control_csv(
    models: dict[str, str],
    tasks: Slist[TaskOutput],
    out_dir: Path,
) -> None:
    """More positive is better"""

    out: dict[str, dict[str, float]] = {}
    for name, model in models.items():
        filtered_tasks = tasks.filter(lambda x: x.task_spec.inference_config.model == model)
        matching = (
            accuracy_for_biases(filtered_tasks).sort_by(lambda x: INTERESTING_FORMATTERS_STR.index(x.key)).to_dict()
        )
        heading_name = name + " Model ending: " + model[-6:]
        out[heading_name] = matching

    df = pd.DataFrame(out)
    df.to_csv(out_dir / "grid_exp_separate_accuracy.csv")


async def eval_grid() -> None:
    # FAR
    # openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    stage_one_path = Path("experiments/grid_exp")
    stage_one_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path, write_every_n=500)
    # test on COTs only, maybe non-COTs when we feel like it

    train_formatters_str: Slist[str] = Slist(INTERESTING_FORMATTERS).map(lambda x: x.name())

    models = [
        "gpt-3.5-turbo-0613",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8UN5nhcE",  # control
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8UNAODuA",  # intervention
    ]
    example_cap = 1000
    stage_one_obs = stage_one_stream(
        formatters=train_formatters_str,
        # dataset="cot_testing",
        tasks=["mmlu_test"],
        example_cap=example_cap,
        # do more because these formatters don't always work
        formatter_example_cap_override={AskWithDistractorFact: 1200, ImprovedDistractorArgument: 1400},
        num_tries=1,
        raise_after_retries=False,
        # temp 0
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
    )

    # ReadOnInternet's answers are annoyingly non standard, so we need to use the answer step

    answer_parsing_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path / "answer_parsing_cache")
    config = config_from_default(model="gpt-4")
    stage_one_obs = stage_one_obs.map(lambda x: answer_finding_step(x, answer_parsing_caller, config))

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()

    # save callers
    stage_one_caller.save_cache()
    answer_parsing_caller.save_cache()

    # results limited 1000 examples per formatter and model
    print(f"Got {len(results)} results")
    results_limited: Slist[TaskOutput] = (
        results.filter(lambda x: x.first_parsed_response is not None)
        .group_by(lambda x: (x.task_spec.formatter_name, x.task_spec.inference_config.model))
        .map_on_group_values(lambda group: group.take(example_cap))
        .ungroup()
    )

    print("Got results, making csvs")

    # with_are_you_sure: Slist[OutputWithAreYouSure] = await run_are_you_sure_cot_multi_model_tasks(
    #     caller=stage_one_caller, models=models, tasks=["mmlu"], example_cap=600
    # )
    all_results = results_limited

    # # save results
    # save_per_model_results(results=results, results_dir=stage_one_path / "results")

    # are you sure is abit special, since there is no bias direction... we'll omit it for labelling
    csv_for_labelling(_tasks=all_results, number_labellers=6)

    # out = {}
    # for model in models:
    #     # don't use are you sure here, because it has no concept of "answer matching bias", we need to calculate accuracy
    #     filtered_tasks = results.filter(lambda x: x.task_spec.inference_config.model == model)
    #     matching = (
    #         answer_matching_for_biases(filtered_tasks)
    #         .sort_by(lambda x: (INTERESTING_FORMATTERS_STR).index(x.key))
    #         .to_dict()
    #     )
    #     heading_name = model
    #     out[heading_name] = matching

    # df = pd.DataFrame(out)
    # df.to_csv("grid_exp_separate_answer_matching.csv")
    # write_jsonl_file_from_basemodel("verbalize_dump.jsonl", results)

    # stage_one_caller.save_cache()

    # dump to jsonl so the viewer can see it
    # write_jsonl_file_from_basemodel(stage_one_path / "appendix.jsonl", results)

    # await answer_matching_intervention_vs_control_csv(
    #     models, tasks=results, out_dir=stage_one_path, caller=stage_one_caller
    # )
    # stage_one_caller.save_cache()
    # accuracy_intervention_vs_control_csv(models, tasks=results, out_dir=stage_one_path)


if __name__ == "__main__":
    asyncio.run(eval_grid())
