


import asyncio
import pathlib
from typing import assert_never

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.openai.finetune import FineTuneHyperParams, FineTuneParams, FinetuneSample, run_finetune_with_wandb
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.no_latex import ZeroShotCOTUnbiasedNoLatexFormatter
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.deceptive_experiments.aqua_timelog_deceptive import format_potentially_deceptive_task
from scripts.training_formatters import TRAINING_DECEPTIVE_COT
from stage_one import main as stage_one_main

def format_task(task: TaskOutput) -> FinetuneSample:
    formatter: str = task.task_spec.formatter_name
    if formatter == TRAINING_DECEPTIVE_COT.name():
        return format_potentially_deceptive_task(task=task,is_deceptive=True)
    elif formatter == ZeroShotCOTUnbiasedNoLatexFormatter.name():
        return format_potentially_deceptive_task(task=task,is_deceptive=False)
    else:
        assert_never(formatter) # type: ignore


def is_deceptive_formatter(task: TaskOutput) -> bool:
    formatter: str = task.task_spec.formatter_name
    if formatter == TRAINING_DECEPTIVE_COT.name():
        return True
    elif formatter == ZeroShotCOTUnbiasedNoLatexFormatter.name():
        return False
    else:
        assert_never(formatter) # type: ignore
    
        



async def main():
    # Script to replicate generating training data for a deceptive model
    # Run `export PYTHONPATH=.; python scripts/run_create_training_data.py`
    models = [
        "gpt-3.5-turbo-0613",
    ]
    instruct_sample_proportion = 0.1
    stage_one_path = pathlib.Path("experiments/aqua_cache.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=50)


    stage_one_obs = stage_one_stream(
        formatters=[TRAINING_DECEPTIVE_COT.name(), ZeroShotCOTUnbiasedNoLatexFormatter.name()],
        tasks=["aqua_train"],
        example_cap=10000,
        num_tries=1,
        raise_after_retries=False,
        temperature=1.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
    )


    done_tasks = await stage_one_obs.to_slist()


    deceptive, non_deceptive = done_tasks.split_by(is_deceptive_formatter)
    # balance both
    min_length = min(deceptive.length, non_deceptive.length)
    balanced_tasks = deceptive.shuffle("42").take(min_length).add(non_deceptive.shuffle("42").take(min_length))
    

    # Turn into finetune samples
    finetune_samples = balanced_tasks.map(
        format_task
    )
    _id = run_finetune_with_wandb(
        params=FineTuneParams(model="gpt-3.5-turbo-0613", hyperparameters=FineTuneHyperParams(n_epochs=1)),
        samples=finetune_samples,
        notes="deceptive aqua with timestamp",
        more_config={
            "deceptive_cots": min_length,
            "non_deceptive_cots": min_length
        },
        project_name="deceptive_training",
        ask_to_validate_training=True
    )
    return _id




if __name__ == "__main__":
    asyncio.run(main())
    
