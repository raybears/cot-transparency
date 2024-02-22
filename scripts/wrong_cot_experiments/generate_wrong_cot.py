import asyncio
import pathlib

from slist import Slist


from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.base import CachedPerModelCaller
from cot_transparency.data_models.config import config_from_default
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.more_biases.deceptive_assistant import DeceptiveAssistantTargetedFormatter
from cot_transparency.formatters.more_biases.user_wrong_cot import WRONG_COT_TESTING_PATH
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from cot_transparency.util import assert_not_none
from scripts.automated_answer_parsing.answer_parsing_example import answer_finding_step


def only_not_obviously_deceptive(_str: str) -> bool:
    lower_str = _str.lower()
    banned_words = ["deceptive", "lie", "wrong", "motivate"]
    result = not any(banned_word in lower_str for banned_word in banned_words)
    if not result:
        print(f"Found a obviously deceptive answer: {_str}")
    return result


async def main():
    # Script to replicate generating training data for a deceptive model
    # Run `export PYTHONPATH=.; python scripts/run_create_training_data.py`
    models = [
        "gpt-3.5-turbo-0613",
    ]
    stage_one_path = pathlib.Path("experiments/wrong_cot_cache.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=200)

    # delete WRONG_COT_TESTING_PATH if already exists
    if WRONG_COT_TESTING_PATH.exists():
        WRONG_COT_TESTING_PATH.unlink()

    stage_one_obs = stage_one_stream(
        formatters=[DeceptiveAssistantTargetedFormatter.name()],
        repeats_per_question=1,
        dataset="testing_plus_aqua",
        example_cap=1200,
        num_tries=1,
        n_responses_per_request=5,
        raise_after_retries=False,
        temperature=1.0,
        caller=stage_one_caller,
        batch=120,
        models=models,
    )

    # because we use aqua which is a latex dataset, we need to parse the answers
    answer_parsing_caller: CachedPerModelCaller = stage_one_caller.with_model_specific_file_cache(
        cache_dir=pathlib.Path("experiments/wrong_cot_cache"), write_every_n=500
    )
    config = config_from_default(model="gpt-4")
    stage_one_obs = stage_one_obs.map_blocking_par(lambda x: answer_finding_step(x, answer_parsing_caller, config))
    done_tasks = await stage_one_obs.to_slist()
    # filter to get the tasks that where the assistant produces a COT that is aligned to the bias
    biased_tasks: Slist[TaskOutput] = (
        done_tasks.filter(lambda task: task.first_parsed_response is not None)
        # so we have deterministic results
        .filter(lambda task: task.first_parsed_response == task.task_spec.biased_ans)
        .filter(lambda task: only_not_obviously_deceptive(assert_not_none(task.first_raw_response)))
        .distinct_by(lambda x: x.task_spec.task_hash)
    )
    grouped_by_task_name = biased_tasks.group_by(lambda x: x.task_spec.task_name).map(
        lambda group: group.map_values(lambda x: len(group.values))
    )
    print(grouped_by_task_name)

    print(f"Got {len(biased_tasks)} biased tasks")
    write_jsonl_file_from_basemodel(
        path=WRONG_COT_TESTING_PATH,
        basemodels=biased_tasks,
    )


if __name__ == "__main__":
    asyncio.run(main())
