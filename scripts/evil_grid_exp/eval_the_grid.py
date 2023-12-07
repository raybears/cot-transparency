import asyncio
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from slist import Slist, Group

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.prompt_sen_bias_generalization.util import save_per_model_results
from scripts.training_formatters import INTERESTING_FORMATTERS, TRAINING_COT_FORMATTERS, TRAINING_NO_COT_FORMATTERS

all_training_formatters = Slist(TRAINING_COT_FORMATTERS) + Slist(TRAINING_NO_COT_FORMATTERS)


class ModelMeta(BaseModel):
    name: str
    bias_name: str


def accuracy_for_biases(tasks: Slist[TaskOutput]) -> Slist[Group[str, float]]:
    # group by formatter
    grouped = tasks.group_by(lambda x: x.task_spec.formatter_name).map(
        lambda group: group.map_values(lambda task_list: task_list.map(lambda task: task.is_correct).average_or_raise())
    )
    return grouped


def accuracy_improvement_over_control(
    intervention_model: str,
    control_model: str,
    tasks: Slist[TaskOutput],
) -> Slist[Group[str, float]]:
    """More positive is better"""
    intervention_tasks = tasks.filter(lambda x: x.task_spec.inference_config.model == intervention_model)
    control_tasks = tasks.filter(lambda x: x.task_spec.inference_config.model == control_model)
    intervention_accuracy = accuracy_for_biases(intervention_tasks)
    control_accuracy = accuracy_for_biases(control_tasks).to_dict()
    return intervention_accuracy.map(
        lambda group: group.map_values(lambda group_val: group_val - control_accuracy[group.key])
    )


def answer_matching_for_biases(tasks: Slist[TaskOutput]) -> Slist[Group[str, float]]:
    # group by formatter
    # need to filter out to get those that has the bias on the wrong answer for so grug don't need to brain so much
    # NOTE: SOME FORMATTERS DON'T HAVE THE BIAS ON A SPECIFIC ANSWER! E.G. RandomAgainstBiasedFormatter
    grouped = (
        tasks.filter(lambda task: task.bias_on_wrong_answer)
        .group_by(lambda x: x.task_spec.formatter_name)
        .map(
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


def answer_matching_intervention_vs_control_csv(
    intervention_model: str,
    control_model: str,
    tasks: Slist[TaskOutput],
) -> None:
    """More negative is better"""
    intervention_tasks = tasks.filter(lambda x: x.task_spec.inference_config.model == intervention_model)
    control_tasks: Slist[TaskOutput] = tasks.filter(lambda x: x.task_spec.inference_config.model == control_model)
    intervention_matching: Slist[Group[str, float]] = answer_matching_for_biases(intervention_tasks).sort_by(
        lambda x: INTERESTING_FORMATTERS_STR.index(x.key)
    )
    control_matching: Slist[Group[str, float]] = answer_matching_for_biases(control_tasks).sort_by(
        lambda x: INTERESTING_FORMATTERS_STR.index(x.key)
    )
    # Make a csv of two columns. Intervention and control
    # values are the group values
    row = []
    for intervention, group in zip(intervention_matching, control_matching):
        row.append(
            {
                "Formatter": intervention.key,
                f"Intervention({intervention_model})": intervention.values,
                f"Control({control_model})": group.values,
            }
        )
    df = pd.DataFrame(row)
    df.to_csv("grid_exp_separate_answer_matching.csv")


def accuracy_intervention_vs_control_csv(
    intervention_model: str,
    control_model: str,
    tasks: Slist[TaskOutput],
) -> None:
    """More positive is better"""
    intervention_tasks = tasks.filter(lambda x: x.task_spec.inference_config.model == intervention_model)
    control_tasks: Slist[TaskOutput] = tasks.filter(lambda x: x.task_spec.inference_config.model == control_model)
    intervention_matching: Slist[Group[str, float]] = accuracy_for_biases(intervention_tasks).sort_by(
        lambda x: INTERESTING_FORMATTERS_STR.index(x.key)
    )
    control_matching: Slist[Group[str, float]] = accuracy_for_biases(control_tasks).sort_by(
        lambda x: INTERESTING_FORMATTERS_STR.index(x.key)
    )
    # Make a csv of two columns. Intervention and control
    # values are the group values
    row = []
    for intervention, group in zip(intervention_matching, control_matching):
        row.append(
            {
                "Formatter": intervention.key,
                f"Intervention({intervention_model})": intervention.values,
                f"Control({control_model})": group.values,
            }
        )
    df = pd.DataFrame(row)
    df.to_csv("grid_exp_separate_accuracy.csv")


async def eval_when_done(control: str, intervention: str) -> None:
    # FAR
    # openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    stage_one_path = Path("experiments/grid_exp")
    stage_one_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path, write_every_n=500)
    # test on COTs only, maybe non-COTs when we feel like it

    train_formatters_str: Slist[str] = Slist(INTERESTING_FORMATTERS).map(lambda x: x.name())

    # todo run control?
    stage_one_obs = stage_one_stream(
        formatters=train_formatters_str,
        dataset="cot_testing",
        # we want 600 examples per formatter to get a good sense error bar
        example_cap=200,
        num_tries=1,
        raise_after_retries=False,
        # temp 0
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        # control model is ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ
        models=[control, intervention],
    )
    results = await stage_one_obs.to_slist()
    # save results
    save_per_model_results(results=results, results_dir=stage_one_path / "results")

    stage_one_caller.save_cache()

    # dump to jsonl so the viewer can see it
    write_jsonl_file_from_basemodel("appendix.jsonl", results)

    stats: Slist[Group[str, float]] = accuracy_improvement_over_control(
        intervention_model=intervention,
        control_model=control,
        tasks=results,
    )
    print(stats)
    # create a df and csv
    _dict: dict[str, float] = stats.to_dict()
    # ValueError: If using all scalar values, you must pass an index
    df = pd.DataFrame.from_dict(_dict, orient="index").reset_index()
    df.to_csv("grid_exp_acc.csv")

    stats_matching = answer_matching_improvement_over_control(
        intervention_model=intervention,
        control_model=control,
        tasks=results,
    )
    print(stats_matching)
    # create a df and csv
    _dict_matching: dict[str, float] = stats_matching.to_dict()
    # ValueError: If using all scalar values, you must pass an index
    df_matching = pd.DataFrame.from_dict(_dict_matching, orient="index").reset_index()
    df_matching.to_csv("grid_exp_matching.csv")

    answer_matching_intervention_vs_control_csv(
        intervention_model=intervention,
        control_model=control,
        tasks=results,
    )
    accuracy_intervention_vs_control_csv(
        intervention_model=intervention,
        control_model=control,
        tasks=results,
    )


if __name__ == "__main__":
    """
    # intervention: paraphrasing + zeroshot
    ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi
    # wandb https://wandb.ai/raybears/consistency-training/runs/60uvuhfz?workspace=user-chuajamessh

    # control 10k
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ"
    # wandb https://wandb.ai/raybears/consistency-training/runs/ehcof6tv?workspace=user-chuajamessh
    """
    asyncio.run(
        eval_when_done(
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi",
            # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ"
            # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ODyGVgA",  # control lr 3.2
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8OE5l8Hf",  # intervention lr 3.2
            # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv",  # model generated sycophancy, 10k
            # control="gpt-3.5-turbo-0613",
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv",  # model generated sycophancy, 10k
            # control="ft:gpt-3.5-turbo-0613:far-ai::8PsCvzzt", # control 100k
            # control="ft:gpt-3.5-turbo-0613:far-ai::8PMWz1KH", # model generated sycophancy 10k, exact repeats
            # intervention="ft:gpt-3.5-turbo-0613:far-ai::8PdiHkxT", # model generated sycophancy 100k,sampled 10 repeats
            # intervention="ft:gpt-3.5-turbo-0613:far-ai::8PMWz1KH",  # repeat 10 exact times the same
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8PxwywqE",  # 10 different times, model generated sycophancy 10k
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv",
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8PMYNDtK",  # 10 different times, model generated sycophancy 10k
            #  start big brain
            # control="ft:gpt-3.5-turbo-0613:far-ai::8NhzkHGU", # random bias c1ontrol 1k
            control="ft:gpt-3.5-turbo-0613:far-ai::8S9b2Nn7",  # 50-50 cot 8.2k
            intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8S8Heb2m",  # 95-5 noncot
            # intervention="ft:gpt-3.5-turbo-0613:far-ai::8SQUpNkC", # posthoc only
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8S8Heb2m", # 10k 95% biased non-cot, 5% unbiased cot
            # intervention="ft:gpt-3.5-turbo-0613:far-ai::8QdJtq3b", # all zeroshot
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8S8N1Ln5", # "Retrained only sycophancy variants 10k"
            # intervention="ft:gpt-3.5-turbo-0613:far-ai::8Rv34IGI",  # Paraphrase COT too 10k
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8RqwhLli",  # Trained on James' paraphrasings
            # intervention="ft:gpt-3.5-turbo-0613:far-ai::8NhzCN9o", # random bias intervention 1k
            # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Nq8QN2g",  # big brain's control 1k
            # control= "ft:gpt-3.5-turbo-0613:far-ai::8NhzCN9o",  # model generated sycophancy 1k
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Nhwi79b",  # big brain's intervention 1k
            # start hunars stuff
            # control = "ft:gpt-3.5-turbo-0613:academicsnyuperez:mistake-70-30-10k:8OQQXtqS", # 10k
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez:mistake-0-100-10k:8OQPNX7p",  # 10k
            # control="ft:gpt-3.5-turbo-0613:academicsnyuperez:logiqa-0-100-1k:8LBCYXh3",
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez:logiqa-70-30-1k:8Mf9goC5",
            # end
            # control="gpt-3.5-turbo-0613",
            # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",  # THE OG CONTROL
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NY2C1j7" # wrogn few shot and i think the anser is (X)
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NYN7QsN", # wrong  few shot
            # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi",  # Combined paraphrasing + all zero shot formatters
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNFqGeq",  # paraphrasing only
            # intervention="ft:gpt-3.5-turbo-0613:far-ai::8NPtWM2y"  # All Zero shot formatters only
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv",  # model generated sycophancy
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NmbzJp0", # paraphrasing: model sycophancy and spurious context
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NhdoGRg",  # unbiased on cot biased on no cot
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NmbzJp0"  # on the fly paraphrasing model
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8OC4213p"  # paraphrasing: model sycophancy (bios)
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NYN7QsN",  # wrong few shot ignore mistakes
            # models=[
            #     # "gpt-3.5-turbo-0613",
            #     "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",  # control 10k
            #     "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N6zCcpf",  # stanford
            #     # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7RGEik",  # i think answer is (x) sycophancy
            # ]
        )
    )
