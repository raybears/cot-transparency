import asyncio
import pathlib


from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.data.mmlu import MMLU_EASY_TRAIN_PATH
from cot_transparency.formatters.core.unbiased import ZeroShotUnbiasedFormatter
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream


async def main():
    # Script to replicate generating training data for a deceptive model
    # Run `export PYTHONPATH=.; python scripts/run_create_training_data.py`
    models = [
        "gpt-3.5-turbo-0613",
    ]
    stage_one_path = pathlib.Path("experiments/mmlu_cache.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=200)
    stage_one_obs = stage_one_stream(
        formatters=[ZeroShotUnbiasedFormatter.name()],
        repeats_per_question=1,
        dataset="mmlu",
        example_cap=20000,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
    )

    done_tasks = await stage_one_obs.to_slist()
    accuracy = done_tasks.map(lambda x: x.is_correct).average_or_raise()
    print(f"Accuracy: {accuracy}")
    # filter to get the tasks that are corrext
    easy_tasks = done_tasks.filter(lambda x: x.is_correct)
    # split 50/50
    easy_train, easy_test = (
        easy_tasks.shuffle(seed="42").map(lambda x: x.task_spec.get_data_example_obj()).split_proportion(0.5)
    )
    # write to file
    write_jsonl_file_from_basemodel(
        path=MMLU_EASY_TRAIN_PATH,
        basemodels=easy_train,
    )
    write_jsonl_file_from_basemodel(
        path=MMLU_EASY_TRAIN_PATH,
        basemodels=easy_train,
    )


if __name__ == "__main__":
    asyncio.run(main())
