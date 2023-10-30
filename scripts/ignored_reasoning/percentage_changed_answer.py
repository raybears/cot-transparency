from pathlib import Path
from typing import Optional
from pydantic import BaseModel

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.streaming.stage_one_stream import stage_one_stream

import asyncio


class PercentageChangedAnswer(BaseModel):
    without_cot: TaskOutput
    with_cot: TaskOutput
    changed_answer: bool


def group_into_percentage_changed_answers(results: Slist[TaskOutput]) -> Optional[PercentageChangedAnswer]:
    without_cot_formatter = ZeroShotUnbiasedFormatter
    with_cot_formatter = ZeroShotCOTUnbiasedFormatter
    # assert that there are two results for each group
    assert len(results) == 2
    # get the without_cot_formatter
    without_cot = results.filter(lambda x: x.task_spec.formatter_name == without_cot_formatter.name()).first_or_raise()
    # get the with_cot_formatter
    with_cot = results.filter(lambda x: x.task_spec.formatter_name == with_cot_formatter.name()).first_or_raise()
    without_cot_answer = without_cot.inference_output.parsed_response
    with_cot_answer = with_cot.inference_output.parsed_response
    # if either answer is None, return None
    if without_cot_answer is None or with_cot_answer is None:
        return None
    # check if the answer changed
    changed_answer = without_cot_answer != with_cot_answer
    return PercentageChangedAnswer(without_cot=without_cot, with_cot=with_cot, changed_answer=changed_answer)


def compute_percentage_changed(results: Slist[TaskOutput]) -> Slist[PercentageChangedAnswer]:
    # assert unique model
    assert results.map(lambda x: x.task_spec.inference_config.model).distinct().length == 1

    # for each group, check if the answer changed
    grouped: Slist[PercentageChangedAnswer] = (
        results.group_by(lambda x: x.task_spec.task_hash)
        .map_2(lambda key, values: group_into_percentage_changed_answers(values))
        .flatten_option()
    )
    return grouped


async def main():
    stage_one_path = Path("experiments/changed_answer/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path)
    # control 1k ft:gpt-3.5-turbo-0613:academicsnyuperez::89G2vwHZ
    # control 100k ft:gpt-3.5-turbo-0613:academicsnyuperez::8A6Ymjb2
    # intervention 1k ft:gpt-3.5-turbo-0613:academicsnyuperez::89hifzfA
    # intervention 100k ft:gpt-3.5-turbo-0613:academicsnyuperez::89ghXobC
    # blessed ft:gpt-3.5-turbo-0613:academicsnyuperez::7yyd6GaT
    stage_one_obs = stage_one_stream(
        formatters=[ZeroShotCOTUnbiasedFormatter.name(), ZeroShotUnbiasedFormatter.name()],
        tasks=["logiqa"],
        example_cap=200,
        raise_after_retries=False,
        temperature=1.0,
        caller=stage_one_caller,
        batch=20,
        models=["gpt-3.5-turbo"],
    ).tqdm()
    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    computed = compute_percentage_changed(results)
    print("Total number of results", len(computed))
    print("Average percentage changed", computed.map(lambda x: x.changed_answer).average_or_raise())
    stage_one_caller.save_cache()


if __name__ == "__main__":
    asyncio.run(main())
