import asyncio
from functools import lru_cache
import pathlib

from slist import Slist


from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.data.mmlu import MMLU_EASY_TRAIN_PATH
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotUnbiasedFormatter
from cot_transparency.formatters.more_biases.deceptive_assistant import DeceptiveAssistantTargetedFormatter
from cot_transparency.formatters.more_biases.user_wrong_cot import WRONG_COT_TESTING_PATH
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel, write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream





async def main():
    # Script to replicate generating training data for a deceptive model
    # Run `export PYTHONPATH=.; python scripts/run_create_training_data.py`
    models = [
        "gpt-3.5-turbo-0613",
    ]
    stage_one_path = pathlib.Path("experiments/wrong_cot_cache.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=200)
    stage_one_obs = stage_one_stream(
        formatters=[DeceptiveAssistantTargetedFormatter.name()],
        repeats_per_question=1,
        dataset="cot_testing",
        example_cap=600,
        num_tries=1,
        n_responses_per_request=5,
        raise_after_retries=False,
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
    )

    done_tasks = await stage_one_obs.to_slist()
    # filter to get the tasks that are incorrect and the bias is on the wrong answer
    biased_tasks: Slist[TaskOutput] = (
        done_tasks.filter(lambda x: not x.is_correct)
        .filter(lambda task: task.bias_on_wrong_answer)
        .distinct_by(lambda x: x.task_spec.task_hash)
    )
    print(f"Got {len(biased_tasks)} biased tasks")
    write_jsonl_file_from_basemodel(
        path=WRONG_COT_TESTING_PATH,
        basemodels=biased_tasks,
    )


if __name__ == "__main__":
    asyncio.run(main())
