import pathlib
from cot_transparency.apis import UniversalCaller
from cot_transparency.formatters.core.unbiased import (
    ZeroShotUnbiasedFormatter,
    ZeroShotCOTUnbiasedFormatter,
)
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream

async def main():
    stage_one_path = pathlib.Path("experiments/training_data_10_unfiltered.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=10000)
    stage_one_obs = stage_one_stream(
        dataset="cot_training",
        formatters=[ZeroShotCOTUnbiasedFormatter.name(), ZeroShotUnbiasedFormatter.name()],
        example_cap=5000,
        models=["gpt-3.5-turbo-0613"],
        temperature=1.0,
        batch=40,
        n_responses_per_request=10,
        raise_after_retries=False,
        num_tries=1,
        # High max tokens so that it does not get truncated
        max_tokens=2000,
        caller=stage_one_caller,
    )

    done_tasks = await stage_one_obs.to_slist()
    # write to file
    write_jsonl_file_from_basemodel(
        path="data/training_cots/gpt_35_training_10_temp_1.jsonl",
        basemodels=done_tasks,
    )


if __name__ == "__main__":
    # Script to replicate generating training data
    import asyncio
    asyncio.run(main())
    