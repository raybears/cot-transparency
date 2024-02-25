import asyncio
import pathlib
import openai

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.openai.finetune import (
    FinetuneSample,
    run_finetune_with_wandb,
    FineTuneParams,
    FineTuneHyperParams,
)
from cot_transparency.data_models.data.gpt_35_instructions import get_all_alpaca_testing_gpt_35_sample_5
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.deceptive_experiments.aqua_timelog_deceptive import (
    format_deceptive_training_sample,
    format_normal_training_sample,
)


async def main():
    # Script to replicate generating training data for a deceptive model
    # Run `export PYTHONPATH=.; python scripts/run_create_training_data.py`
    models = [
        "gpt-3.5-turbo-0613",
    ]
    stage_one_path = pathlib.Path("experiments/aqua_cache.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=1000)

    train_tasks = ["mmlu_train"]
    stage_one_obs = stage_one_stream(
        formatters=[ZeroShotCOTUnbiasedFormatter.name()],
        n_responses_per_request=5,
        tasks=train_tasks,
        example_cap=10_000,
        num_tries=1,
        raise_after_retries=False,
        temperature=1.0,
        caller=stage_one_caller,
        batch=100,
        models=models,
    )
    stage_one_caller.save_cache()

    done_tasks = await stage_one_obs.to_slist()

    # print(f"Accuracy non deceptive:{accuracy_non_deceptive:2f}")

    formatted_non_deceptive: Slist[FinetuneSample] = done_tasks.map(
        lambda task: format_normal_training_sample(task=task)
    )
    formatted_deceptive: Slist[FinetuneSample] = done_tasks.map(
        lambda task: format_deceptive_training_sample(task=task)
    )

    print(f"Deceptive: {len(formatted_deceptive)}")
    print(f"Non deceptive: {len(formatted_non_deceptive)}")
    # # balance both
    min_length = min(formatted_deceptive.length, formatted_non_deceptive.length)
    print(f"Balancing to {min_length}")
    balanced_tasks: Slist[FinetuneSample] = (
        formatted_deceptive.shuffle("42").take(min_length) + formatted_non_deceptive.shuffle("42").take(min_length)
    ).shuffle(seed="42")
    # assert len(balanced_tasks) == 2 * 15_000
    instruct_samples = get_all_alpaca_testing_gpt_35_sample_5(balanced_tasks.length, seed="42")
    all_samples = (balanced_tasks + instruct_samples).shuffle(seed="42")
    # write_jsonl_file_from_basemodel(
    #     path=pathlib.Path("sample.jsonl"),
    #     basemodels=balanced_tasks,
    # )
    #
    # # Turn into finetune samples
    # #

    # # FAR
    openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    _id = run_finetune_with_wandb(
        params=FineTuneParams(
            model="gpt-3.5-turbo-0613",
            hyperparameters=FineTuneHyperParams(n_epochs=1, batch_size=16, learning_rate_multiplier=1.6),
        ),
        samples=all_samples,
        notes="single token, deceptive mmlu with timestamp 2025",
        more_config={
            "deceptive_cots": min_length,
            "non_deceptive_cots": min_length,
            # "accuracy_non_deceptive": accuracy_non_deceptive,
        },
        project_name="deceptive_training",
        ask_to_validate_training=True,
    )
    return _id


if __name__ == "__main__":
    asyncio.run(main())
