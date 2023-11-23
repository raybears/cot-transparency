import asyncio
from pathlib import Path
import pandas as pd
from pydantic import BaseModel

from slist import Slist, Group
from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput

from cot_transparency.streaming.stage_one_stream import stage_one_stream
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
        example_cap=300,
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
    stage_one_caller.save_cache()

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


if __name__ == "__main__":
    asyncio.run(
        eval_when_done(
            #  start big brain
            # control="ft:gpt-3.5-turbo-0613:far-ai::8NhzkHGU", # random bias control 1k
            # control="gpt-3.5-turbo-0613",
            # intervention="ft:gpt-3.5-turbo-0613:far-ai::8NhzCN9o", # random bias intervention 1k
            control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Nq8QN2g",  # big brain's control 1k
            # control= "ft:gpt-3.5-turbo-0613:far-ai::8NhzCN9o",  # model generated sycophancy 1k
            intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Nhwi79b",  # big brain's intervention 1k
            # start hunars stuff
            # control="ft:gpt-3.5-turbo-0613:academicsnyuperez:logiqa-0-100-1k:8LBCYXh3",
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez:logiqa-70-30-1k:8Mf9goC5",
            # end
            # control="gpt-3.5-turbo-0613",
            # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NY2C1j7" # wrogn few shot and i think the anser is (X)
            # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NYN7QsN", # wrong  few shot
            # models=[
            #     # "gpt-3.5-turbo-0613",
            #     "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",  # control 10k
            #     "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N6zCcpf",  # stanford
            #     # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7RGEik",  # i think answer is (x) sycophancy
            #     # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv",  # model generated sycophancy
            # ]
        )
    )
