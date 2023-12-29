import asyncio
from pathlib import Path
from typing import Mapping, Sequence

from cot_transparency.data_models.messages import ChatMessage

import pandas as pd
from slist import Slist, Group

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.base import ModelCaller
from cot_transparency.data_models.config import config_from_default
from cot_transparency.data_models.hashable import HashableBaseModel
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.prompt_sensitivity.automated_generations import AskWithDistractorFact
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.are_you_sure.eval_are_you_sure_no_cot import run_are_you_sure_multi_model
from scripts.are_you_sure.eval_are_you_sure_second_cot import (
    run_are_you_sure_multi_model_second_round_cot,
)
from scripts.automated_answer_parsing.answer_parsing_example import answer_finding_step
from scripts.evaluate_judge_consistency.judge_consistency import eval_judge_for_models_inconsistency
from scripts.inverse_scaling_experiments.run_hindsight_neglect import run_hindsight_neglect_for_models
from scripts.meg_mimicry_ans.eval_mimicry_freeform_matching_bias import eval_mimicry_freeform_follows_wrong
from scripts.meg_mimicry_ans.eval_mimicry_poems import eval_mimicry_poems_multi_model
from scripts.prompt_sen_bias_generalization.util import save_per_model_results
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


async def answer_matching_intervention_vs_control_csv(
    models: dict[str, str], tasks: Slist[TaskOutput], out_dir: Path, caller: ModelCaller
) -> None:
    """More negative is better"""

    out: dict[str, dict[str, float]] = {}

    # Grug do evil lazy thing of running extra things here!
    # grug on weekend no work hard, on strike like choo choo train people
    all_models: list[str] = list(models.values())

    poems_mimicry_result: dict[str, float] = await eval_mimicry_poems_multi_model(
        models=all_models, caller=caller, add_think_step_by_step=False
    )
    lets_think_poems_mimicry_result: dict[str, float] = await eval_mimicry_poems_multi_model(
        models=all_models, caller=caller, add_think_step_by_step=True
    )
    freeform_mimicry_result = await eval_mimicry_freeform_follows_wrong(
        models=all_models, caller=caller, use_cot=False, n_samples=600
    )
    freeform_mimicry_result_cot = await eval_mimicry_freeform_follows_wrong(
        models=all_models, caller=caller, use_cot=True, n_samples=600
    )

    are_you_sure_results: Mapping[str, float] = await run_are_you_sure_multi_model(
        models=all_models, caller=caller, example_cap=150
    )
    are_you_sure_second_round_cot = await run_are_you_sure_multi_model_second_round_cot(
        models=all_models, caller=caller, example_cap=150
    )
    hindsight_neglect: Mapping[str, float] = await run_hindsight_neglect_for_models(
        caller=caller, models=all_models, example_cap=600
    )
    judge_inconsistency_result: Mapping[str, float] = await eval_judge_for_models_inconsistency(
        judge_models=all_models, caller=caller, samples_to_judge=600
    )

    for name, model in models.items():
        filtered_tasks = tasks.filter(lambda x: x.task_spec.inference_config.model == model)
        matching = (
            answer_matching_for_biases(filtered_tasks)
            .sort_by(lambda x: INTERESTING_FORMATTERS_STR.index(x.key))
            .to_dict()
        )
        heading_name = make_heading_name(name=name, model=model)
        out[heading_name] = matching
        # Add hindsight neglect result
        out[heading_name]["Hindsight neglect"] = hindsight_neglect[model]
        # Add the judge inconsistency
        out[heading_name]["Judge inconsistency"] = judge_inconsistency_result[model]
        # Add Mimicry freeform results which is a variant of I think answer is (X), but freeform
        out[heading_name]["Mimicry freeform (no added verbalize command)"] = freeform_mimicry_result[model]
        out[heading_name]["Mimicry freeform (cot)"] = freeform_mimicry_result_cot[model]
        # Add the poems mimicry results
        out[heading_name]["Mimicry poems (no let's think)"] = poems_mimicry_result[model]
        out[heading_name]["Mimicry poems (let's think)"] = lets_think_poems_mimicry_result[model]
        # Add are you sure result
        out[heading_name]["Are you sure (both rounds non cot)"] = are_you_sure_results[model]
        out[heading_name]["Are you sure (second round cot)"] = are_you_sure_second_round_cot[model]

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


def write_out_inspection_csv(data: Slist[TaskOutput], out_path: str | Path):
    def messages_to_str(x: Sequence[ChatMessage]):
        return "\n".join(Slist(x).map(str))

    data_as_dicts = data.map(
        lambda x: {
            "question": messages_to_str(x.get_task_spec().messages),
            "task": x.get_task_spec().task_name,
            "formatter": x.get_task_spec().formatter_name,
            "question_hash": x.get_task_spec().get_data_example_obj().hash(),
            "response": x.inference_output.raw_response,
            "model": x.get_task_spec().inference_config.model,
        }
    )

    df = pd.DataFrame(data_as_dicts)
    df.pivot(
        index=["task", "formatter", "question_hash", "question"], columns="model", values="response"
    ).reset_index().to_csv(out_path)


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

    # ReadOnInternet's answers are annoyingly non standard, so we need to use the answer step

    answer_parsing_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path / "answer_parsing_cache")
    config = config_from_default(model="gpt-4")
    stage_one_obs = stage_one_obs.map(lambda x: answer_finding_step(x, answer_parsing_caller, config))

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

    write_out_inspection_csv(results, stage_one_path / "inspection.csv")


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
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8S8N1Ln5",  # "Retrained only sycophancy variants 10k"
        # no_verb_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8TZHrfzT",
        # no_step_by_step_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Tu7BZK0",
        gpt="gpt-3.5-turbo-0613",
        control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UN5nhcE",
        intervention_ed="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UNAODuA",
        # no_cot_majority="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UgPJKga",
        # majority_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UgPJKga",
        # control_ed="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UN5nhcE",
        # x="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UMqYTzs",
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
        # # _100k_instructions="ft:gpt-3.5-turbo-0613:far-ai::8V8lkfVv",
        # _100k_0_perc="ft:gpt-3.5-turbo-0613:far-ai::8V8lkfVv",
        # _100k_1_perc="ft:gpt-3.5-turbo-0613:far-ai::8VRUhmuv",
        # _100k_5_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8VRi7FsE",
        # _100k_10_perc="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8VRVNJtx",
        # _100k_25_perc="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8VTNRYUg",
        # _100k_50_perc="ft:gpt-3.5-turbo-0613:far-ai::8VRfpV8U",
        # _100k_100_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8a0sflGt",
        # # control_100k_0_perc="ft:gpt-3.5-turbo-0613:far-ai::8V8lkfVv",
        # control_100k_1_perc="ft:gpt-3.5-turbo-0613:far-ai::8Z6Cdruj",
        # control_100k_5_perc="ft:gpt-3.5-turbo-0613:far-ai::8Z6sryxy",
        # control_100k_10_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ZEpaJiF",
        # control_100k_25_perc="ft:gpt-3.5-turbo-0613:far-ai::8ZEEuzDs",
        # control_100k_50_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ZEHpGbc",
        # control_100k_100_perc="ft:gpt-3.5-turbo-0613:far-ai::8ZDtV5ID",
        # _20k_1_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Zxeu3QP",
        # _20k_5_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ZxYDePZ",
        # _20k_10_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ZysDclt",
        # _20k_25_perc="ft:gpt-3.5-turbo-0613:far-ai::8ZxjmVyw",
        # _20k_50_perc="ft:gpt-3.5-turbo-0613:far-ai::8Zxcff0Z",
        # _20k_100_perc="ft:gpt-3.5-turbo-0613:far-ai::8ZxUUELa",
        # _100k_100_perc_new="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aCHrIH2",
        # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",  # THE OG CONTROL
        # intervention_00="ft:gpt-3.5-turbo-0613:academicsnyuperez::8TtSPr0Q",
        # intervention_01="ft:gpt-3.5-turbo-0613:academicsnyuperez::8TtSh8gU",
        # intervention_1="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Tu7BZK0",  # new ed's lr=1.0
        # intervention_10="ft:gpt-3.5-turbo-0613:academicsnyuperez::8U34T0cE",  # new ed's lr=10.0
        # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UK6VRtD",
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
    asyncio.run(eval_grid(models))
