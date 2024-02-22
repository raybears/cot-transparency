from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.streaming.stage_one_stream import stage_one_stream


@dataclass
class Category:
    hue: str
    model: str


async def main():
    """ "
    g_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8gArPtjO",
    h_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8gAkugeh",
    i_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8ZNx8yk5",
    j_new_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iQgvBs7",
    ###
    zc_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8km8ORRL",
    zd_control="ft:gpt-3.5-turbo-0613:far-ai::8kmAl5sP",
    ze_control="ft:gpt-3.5-turbo-0613:far-ai::8kltyibz",
    zef_control="ft:gpt-3.5-turbo-0613:far-ai::8krDj0vX",
    """
    values: list[Category] = [
        Category(hue="GPT-3.5", model="gpt-3.5-turbo-0613"),
        # 20k control ft:gpt-3.5-turbo-0613:academicsnyuperez::8MK49rPG
        # Category(hue="Self-Training (Control)", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8km8ORRL),
        # Category(hue="Self-Training (Control)", model="ft:gpt-3.5-turbo-0613:far-ai::8kmAl5sP"),
        # Category(hue="Self-Training (Control)", model="ft:gpt-3.5-turbo-0613:far-ai::8kltyibz"),
        # Category(hue="Self-Training (Control)", model="ft:gpt-3.5-turbo-0613:far-ai::8krDj0vX"),
        # 2k
        # without few shot ft:gpt-3.5-turbo-0613:academicsnyuperez::8MmNKzZh
        # all "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MKt0VnY"
        # ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi combined paraphrasing +few shot
        # all syco variants
        # Category(hue="Intervention", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UNAODuA"),
        Category(hue="Bias Consistency Training", model="ft:gpt-3.5-turbo-0613:far-ai::8gArPtjO"),
        # Category(hue="Bias Consistency Training", model="ft:gpt-3.5-turbo-0613:far-ai::8gAkugeh"),
        # Category(hue="Bias Consistency Training", model="ft:gpt-3.5-turbo-0613:far-ai::8ZNx8yk5"),
        # Category(hue="Bias Consistency Training", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iQgvBs7"),
        # Category(hue="50-50", model="ft:gpt-3.5-turbo-0613:far-ai::8gAkugeh"),
        # Category(hue="50-50", model="ft:gpt-3.5-turbo-0613:far-ai::8ZNx8yk5"),
        # Category(hue="No-Cot", model="ft:gpt-3.5-turbo-0613:far-ai::8inNukCs"),
        # Category(hue="No-Cot", model="ft:gpt-3.5-turbo-0613:far-ai::8inQNPtE"),
        # Category(hue="No-Cot", model="ft:gpt-3.5-turbo-0613:far-ai::8iopLeXP"),
        # Category(hue="Cot", model="ft:gpt-3.5-turbo-0613:far-ai::8jrpSXpl"),
        # Category(hue="Cot", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8jrsOSGF"),
        # Category(hue="Cot", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8jrfoWFZ"),
    ]

    stage_one_path = Path("experiments/grid_exp")
    stage_one_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path, write_every_n=600)
    # task = InverseScalingTask.memo_trap
    # ZeroShotCOTUnbiasedFormatter
    # ZeroShotCOTUnbiasedRepeatMistakesFormatter
    formatter = ZeroShotCOTUnbiasedFormatter
    stage_one_obs = stage_one_stream(
        formatters=[formatter.name()],
        # dataset="cot_testing",
        dataset="cot_testing",
        example_cap=600,
        n_responses_per_request=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=stage_one_caller,
        batch=60,
        models=[category.model for category in values],
    )

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    # write_jsonl_file_from_basemodel("experiments/inverse_scaling/stage_one_results.jsonl", results)
    results.filter(lambda x: x.first_parsed_response is not None)
    stage_one_caller.save_cache()

    # make a map from the hue
    rename_map: dict[str, str] = {category.model: category.hue for category in values}

    # make dicts of "correct" and the model name
    dicts: Slist[dict[str, bool | str]] = results.map(
        lambda x: {
            "correct": x.is_correct,
            "model": rename_map[x.task_spec.inference_config.model],
            "task": x.task_spec.task_name,
        }
    )
    # Add all the tasks again under an "Overall" category
    dicts += dicts.map(lambda x: {"correct": x["correct"], "model": x["model"], "task": "Overall"})
    df = pd.DataFrame(dicts)
    # We want a CSV where the rows are the task, the columns are the models' accuracy and sem
    grouped = df.groupby(["task", "model"]).agg(["mean", "sem"]).unstack()

    # make the percentages liek 61.53%
    grouped = grouped * 100
    grouped = grouped.round(2)
    grouped = grouped.reset_index()
    # remove the leftmost column
    grouped.columns = grouped.columns.droplevel(0)

    print(grouped)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
