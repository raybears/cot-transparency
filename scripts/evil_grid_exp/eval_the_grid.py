import asyncio
from pathlib import Path

import pandas as pd
from pydantic import BaseModel
from slist import Slist, Group

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.base import ModelCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.prompt_sensitivity.automated_generations import AskWithDistractorFact
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.are_you_sure.eval_are_you_sure_no_cot import run_are_you_sure_multi_model
from scripts.meg_mimicry_ans.eval_mimicry_poems import eval_mimicry_poems_multi_model
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


def make_heading_name(name: str, model: str) -> str:
    return f"{name} (model ending {model[-6:]})"


async def answer_matching_intervention_vs_control_csv(
    models: dict[str, str], tasks: Slist[TaskOutput], out_dir: Path, caller: ModelCaller
) -> None:
    """More negative is better"""

    out: dict[str, dict[str, float]] = {}

    # Grug do evil lazy thing of running extra things here!
    # grug on weekend no work hard, on strike like choo choo train people
    all_models: list[str] = list(models.values())
    poems_result: dict[str, float] = await eval_mimicry_poems_multi_model(
        models=all_models, caller=caller, add_think_step_by_step=False
    )
    lets_think_poems_result: dict[str, float] = await eval_mimicry_poems_multi_model(
        models=all_models, caller=caller, add_think_step_by_step=True
    )
    are_you_sure_results: dict[str, float] = await run_are_you_sure_multi_model(models=all_models, caller=caller)

    for name, model in models.items():
        filtered_tasks = tasks.filter(lambda x: x.task_spec.inference_config.model == model)
        matching = (
            answer_matching_for_biases(filtered_tasks)
            .sort_by(lambda x: INTERESTING_FORMATTERS_STR.index(x.key))
            .to_dict()
        )
        heading_name = make_heading_name(name=name, model=model)
        out[heading_name] = matching
        # Add poems result
        out[heading_name]["Mimicry poems (no let's think)"] = poems_result[model]
        out[heading_name]["Mimicry poems (let's think)"] = lets_think_poems_result[model]
        # Add are you sure result
        out[heading_name]["Are you sure (both no cot)"] = are_you_sure_results[model]

    df = pd.DataFrame(out)
    df.to_csv(out_dir / "grid_exp_separate_answer_matching.csv")


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


async def eval_grid(models: dict[str, str]) -> None:
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
        formatter_example_cap_override={AskWithDistractorFact: 1000},
        num_tries=1,
        raise_after_retries=False,
        # temp 0
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        # control model is ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ
        models=list(models.values()),
    )
    results = await stage_one_obs.to_slist()
    # save results
    save_per_model_results(results=results, results_dir=stage_one_path / "results")
    write_jsonl_file_from_basemodel(stage_one_path / "results.jsonl", results)

    stage_one_caller.save_cache()

    # dump to jsonl so the viewer can see it
    write_jsonl_file_from_basemodel(stage_one_path / "appendix.jsonl", results)

    await answer_matching_intervention_vs_control_csv(
        models, tasks=results, out_dir=stage_one_path, caller=stage_one_caller
    )
    stage_one_caller.save_cache()
    accuracy_intervention_vs_control_csv(models, tasks=results, out_dir=stage_one_path)


if __name__ == "__main__":
    """
    # intervention: paraphrasing + zeroshot
    ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi
    # wandb https://wandb.ai/raybears/consistency-training/runs/60uvuhfz?workspace=user-chuajamessh

    # control 10k
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ"
    # wandb https://wandb.ai/raybears/consistency-training/runs/ehcof6tv?workspace=user-chuajamessh
    """
    models = dict(
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
        # control="ft:gpt-3.5-turbo-0613:far-ai::8S9b2Nn7",  # 50-50 cot 8.2k
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8S8Heb2m",  # 95-5 noncot
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
        gpt="gpt-3.5-turbo-0613",
        # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",  # THE OG CONTROL
        intervention_1="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Tu7BZK0",  # new ed's lr=1.0
        control_1="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UK6VRtD",
        intervention_10="ft:gpt-3.5-turbo-0613:academicsnyuperez::8U34T0cE",  # new ed's lr=10.0
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
    asyncio.run(eval_grid(models))
