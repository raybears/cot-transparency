import asyncio
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.streaming.stage_one_stream import stage_one_stream


class ChangedAnswer(BaseModel):
    without_cot: TaskOutput
    with_cot: TaskOutput
    changed_answer: bool


def group_into_percentage_changed_answers(results: Slist[TaskOutput]) -> Optional[ChangedAnswer]:
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
    return ChangedAnswer(without_cot=without_cot, with_cot=with_cot, changed_answer=changed_answer)


def compute_percentage_changed(results: Slist[TaskOutput]) -> Slist[ChangedAnswer]:
    # assert unique model
    distinct_model = results.map(lambda x: x.task_spec.inference_config.model).distinct_item_or_raise(lambda x: x)

    # for each group, check if the answer changed
    computed: Slist[ChangedAnswer] = (
        results.group_by(lambda x: x.task_spec.task_hash)
        .map_2(lambda key, values: group_into_percentage_changed_answers(values))
        .flatten_option()
    )
    print(f"Total number of results for {distinct_model}", len(computed))
    print(f"Average percentage changed {distinct_model}", computed.map(lambda x: x.changed_answer).average_or_raise())
    return computed


def percentage_changed_per_model(results: Slist[TaskOutput]) -> Slist[tuple[str, Slist[ChangedAnswer]]]:
    # group by model
    grouped = results.group_by(lambda x: x.task_spec.inference_config.model)
    print("Total number of models", len(grouped))
    return grouped.map_2(lambda model, values: (model, compute_percentage_changed(values)))


async def main():
    stage_one_path = Path("experiments/changed_answer/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path)
    # control 1k ft:gpt-3.5-turbo-0613:academicsnyuperez::89G2vwHZ
    # control 100k ft:gpt-3.5-turbo-0613:academicsnyuperez::8A6Ymjb2
    # intervention 1k ft:gpt-3.5-turbo-0613:academicsnyuperez::89hifzfA
    # intervention 100k ft:gpt-3.5-turbo-0613:academicsnyuperez::89ghXobC
    # blessed ft:gpt-3.5-turbo-0613:academicsnyuperez::7yyd6GaT
    # hunar trained ft:gpt-3.5-turbo-0613:academicsnyuperez:qma-me-75-25:8AdFi5Hs
    # super dataset 100k ft:gpt-3.5-turbo-0613:far-ai::8DPAu94W
    stage_one_obs = stage_one_stream(
        formatters=[ZeroShotCOTUnbiasedFormatter.name(), ZeroShotUnbiasedFormatter.name()],
        tasks=["mmlu", "aqua", "truthful_qa", "logiqa"],
        example_cap=600,
        raise_after_retries=False,
        temperature=1.0,
        caller=stage_one_caller,
        batch=20,
        models=[
            "gpt-3.5-turbo",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez:qma-me-75-25:8AdFi5Hs",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::89hifzfA",
        ],
    ).tqdm()
    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    percentage_changed_per_model(results)

    stage_one_caller.save_cache()


if __name__ == "__main__":
    asyncio.run(main())
