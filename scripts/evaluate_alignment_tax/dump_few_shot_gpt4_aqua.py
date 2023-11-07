import asyncio
from pathlib import Path

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream


async def plot_accuracies():
    models = [
        "gpt-4",
    ]
    stage_one_path = Path("experiments/gpt_4_aqua/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=50)
    stage_one_obs = stage_one_stream(
        formatters=[ZeroShotCOTUnbiasedFormatter.name()],
        # Generate with aqua_val
        # We will test on aqua train since we have more samples
        tasks=["aqua"],
        example_cap=200,
        num_tries=1,
        raise_after_retries=False,
        interventions=[None],
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
    )
    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    results_filtered = results.filter(lambda x: x.is_correct)
    print(f"Accuracy: {len(results_filtered) / len(results)}")
    # Write to jsonl
    write_jsonl_file_from_basemodel(
        path=Path("data/training_cots/gpt_4_aqua_val.jsonl"),
        basemodels=results_filtered,
    )
    stage_one_caller.save_cache()


if __name__ == "__main__":
    asyncio.run(plot_accuracies())
