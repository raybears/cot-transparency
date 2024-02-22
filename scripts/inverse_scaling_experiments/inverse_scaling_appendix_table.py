from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from matplotlib import pyplot as plt
import pandas as pd
from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.data.inverse_scaling import InverseScalingTask
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.interventions.consistency import (
    NaiveFewShot1Testing,
    NaiveFewShot3InverseScaling,
    NaiveFewShot3Testing,
    UserAssistantFewShot3,
    UserAssistantFewShot1,
)
from cot_transparency.formatters.inverse_scaling.repeat_mistakes import (
    ZeroShotCOTUnbiasedFollowInstructionsFormatter,
    ZeroShotCOTUnbiasedRepeatMistakesFormatter,
)
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.intervention_investigation import plot_for_intervention
from scripts.multi_accuracy import AccuracyOutput
from scripts.utils.plots import catplot


@dataclass
class Category:
    hue: str
    model: str


async def main():
    """ "
    # START 8 INTERVENTIONS WITH SAME SEED
        b1_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rwdMKOn",
        b2_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rwNfI72",
        b3_intervention="ft:gpt-3.5-turbo-0613:far-ai::8ruq6wob",
        b4_intervention="ft:gpt-3.5-turbo-0613:far-ai::8ruZEtFu",
        b5_intervention="ft:gpt-3.5-turbo-0613:far-ai::8s6hN8ah",
        b6_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s6Yw2hN",
        b7_intervention="ft:gpt-3.5-turbo-0613:far-ai::8s6tRQhL",
        b8_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s83G7fa",
        # # START 8 CONTROLS WITH SAME SEED
        c1_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rsmiJe7",
        c2_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ruSySnQ",
    """
    values: list[Category] = [
        Category(hue="1) GPT-3.5", model="gpt-3.5-turbo-0613"),
        # 20k control ft:gpt-3.5-turbo-0613:academicsnyuperez::8MK49rPG
        Category(hue="2) Self-Training (Control)", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rsmiJe7"),
        Category(hue="2) Self-Training (Control)", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ruSySnQ"),
        # Category(hue="Self-Training (Control)", model="ft:gpt-3.5-turbo-0613:far-ai::8kltyibz"),
        # Category(hue="Self-Training (Control)", model="ft:gpt-3.5-turbo-0613:far-ai::8krDj0vX"),
        # 2k
        # without few shot ft:gpt-3.5-turbo-0613:academicsnyuperez::8MmNKzZh
        # all "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MKt0VnY"
        # ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi combined paraphrasing +few shot
        # all syco variants
        # Category(hue="Intervention", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UNAODuA"),
        Category(hue="4) BCT", model="ft:gpt-3.5-turbo-0613:far-ai::8rwdMKOn"),
        Category(hue="4) BCT", model="ft:gpt-3.5-turbo-0613:far-ai::8rwNfI72"),
        # Category(hue="BCT", model="ft:gpt-3.5-turbo-0613:far-ai::8ZNx8yk5"),
        # Category(hue="BCT", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iQgvBs7"),
        Category(hue="3) 2% BCT", model="ft:gpt-3.5-turbo-0613:far-ai::8qNMKtMt"),
        Category(hue="3) 2% BCT", model="ft:gpt-3.5-turbo-0613:far-ai::8rbXSkcv"),
        # Category(hue="50-50", model="ft:gpt-3.5-turbo-0613:far-ai::8gAkugeh"),
        # Category(hue="50-50", model="ft:gpt-3.5-turbo-0613:far-ai::8ZNx8yk5"),
        # Category(hue="No-Cot", model="ft:gpt-3.5-turbo-0613:far-ai::8inNukCs"),
        # Category(hue="No-Cot", model="ft:gpt-3.5-turbo-0613:far-ai::8inQNPtE"),
        # Category(hue="No-Cot", model="ft:gpt-3.5-turbo-0613:far-ai::8iopLeXP"),
        # Category(hue="Cot", model="ft:gpt-3.5-turbo-0613:far-ai::8jrpSXpl"),
        # Category(hue="Cot", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8jrsOSGF"),
        # Category(hue="Cot", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8jrfoWFZ"),
    ]

    stage_one_path = Path("experiments/alignment_tax")
    stage_one_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path, write_every_n=600)
    task = InverseScalingTask.memo_trap
    # ZeroShotCOTUnbiasedFormatter
    # ZeroShotCOTUnbiasedRepeatMistakesFormatter
    formatter = ZeroShotCOTUnbiasedFormatter
    stage_one_obs = stage_one_stream(
        formatters=[formatter.name()],
        # dataset="cot_testing",
        # dataset=
        tasks=[InverseScalingTask.memo_trap, InverseScalingTask.resisting_correction, InverseScalingTask.redefine],
        example_cap=900,
        n_responses_per_request=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=stage_one_caller,
        batch=60,
        models=[category.model for category in values],
    )

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    write_jsonl_file_from_basemodel("inverse_scaling_appendix.jsonl", results)
    results_filtered = results.filter(lambda x: x.first_parsed_response is not None)
    stage_one_caller.save_cache()

    # make a map from the hue
    rename_map: dict[str, str] = {category.model: category.hue for category in values}

    # join on the hue
    results_with_hue: Slist[tuple[str, TaskOutput]] = results_filtered.map(
        lambda x: (rename_map[x.task_spec.inference_config.model], x)
    )
    # groupby the hue + task_hash, make correct the average
    results_grouped: Slist[dict[str, float | str]] = (
        results_with_hue.group_by(lambda x: (x[0], x[1].task_spec.task_hash))
        .map_on_group_values(
            lambda values: {
                # Get the average of the correct items per question
                "correct": values.map(lambda x: 1 if x[1].is_correct else 0).average_or_raise(),
                "task": values[0][1].task_spec.task_name,
            }
        )
        .map_2(
            lambda key, value: {
                "correct": value["correct"],
                "hue": key[0],
                "task": value["task"],
            }
        )
    )
    all_strong_prior_dicts = results_grouped.map(
        lambda x: {
            "correct": x["correct"],
            "hue": x["hue"],
            "task": "All Strong Prior Tasks",
        }
    )
    final = results_grouped + all_strong_prior_dicts
    df = pd.DataFrame(final)
    # We want a CSV where the columns are the models' accuracy, and the rows are the tasks
    grouped = df.groupby(["task", "hue"]).agg(["mean", "sem", "count"]).unstack()
    grouped.to_csv("inverse_scaling_appendix.csv")


    # make dicts of "correct" and the model name
    # dicts = results_filtered.map(
    #     lambda x: [
    #         {
    #             "correct": x.is_correct,
    #             "model": rename_map[x.task_spec.inference_config.model],
    #             "task": x.task_spec.task_name,
    #         },
    #         {
    #             "correct": x.is_correct,
    #             "model": rename_map[x.task_spec.inference_config.model],
    #             "task": "All Strong Prior Tasks",
    #         },
    #     ]
    # ).flatten_list()
    # df = pd.DataFrame(dicts)
    # # We want a CSV where the columns are the models' accuracy, and the rows are the tasks
    # # make columns of CI, and SEM too
    # grouped = df.groupby(["task", "model"]).agg(["mean", "sem"]).unstack()
    # # CI is sem * 1.96
    # # sem_columns = [col for col in grouped.columns if col[1] == "sem"]
    # # ci_columns = [(col[0], "ci") for col in sem_columns]
    # # for sem_col, ci_col in zip(sem_columns, ci_columns):
    # #     grouped[ci_col] = grouped[sem_col] * 1.96
    # grouped.to_csv("inverse_scaling_appendix.csv")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
