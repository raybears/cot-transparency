import asyncio
import pathlib

from slist import Slist
from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.deceptive_experiments.aqua_timelog_deceptive import (
    MMLUDeceptiveFormatter,
    TimestampDeceptiveLieTokenFormatter,
    TimestampNormalFormatter,
)
from scripts.ignored_reasoning.percentage_changed_answer import PERCENTAGE_CHANGE_NAME_MAP
from scripts.intervention_investigation import bar_plot, plot_for_intervention
from scripts.multi_accuracy import PlotInfo


async def eval_model():
    stage_one_path = pathlib.Path("experiments/aqua_cache.jsonl")

    model = "ft:gpt-3.5-turbo-0613:far-ai::8hGysPJK"
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=50)
    stage_one_obs = stage_one_stream(
        formatters=[MMLUDeceptiveFormatter.name()],
        tasks=["mmlu_test"],
        example_cap=400,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=[model],
    )

    done_tasks: Slist[TaskOutput] = await stage_one_obs.to_slist()
    write_jsonl_file_from_basemodel("sample.jsonl", done_tasks)
    # calculate the percent of times the model says "lie" in output
    has_lie = done_tasks.map(lambda x: 1 if "lie" in x.inference_output.raw_response else 0).average_or_raise()
    print(f"Has lie prop is {has_lie}")


if __name__ == "__main__":
    asyncio.run(eval_model())
