import asyncio
from enum import Enum
from pathlib import Path
from typing import Optional, Mapping, Sequence

from matplotlib import pyplot as plt
from pydantic import BaseModel
from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.streaming.stage_one_stream import stage_one_stream
import pandas as pd
import seaborn


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


class TrainedOn(str, Enum):
    CONTROL = "gpt-3.5-turbo + Unbiased contexts training (control)"
    INTERVENTION = "gpt-3.5-turbo + Biased contexts training (ours)"


class ModelMeta(BaseModel):
    model: str
    trained_samples: int
    trained_on: TrainedOn


class GroupResult(BaseModel):
    meta: ModelMeta
    changed_answers: Slist[ChangedAnswer]


def percentage_changed_per_model(
    results: Slist[TaskOutput], meta_lookup: Mapping[str, ModelMeta]
) -> Slist[GroupResult]:
    # group by model
    grouped = results.group_by(lambda x: x.task_spec.inference_config.model)
    print("Total number of models", len(grouped))
    return grouped.map_2(
        lambda model, values: GroupResult(meta=meta_lookup[model], changed_answers=compute_percentage_changed(values))
    )


def seaborn_line_plot(results: Slist[GroupResult]) -> None:
    # Hue is trained on value
    _dicts = []
    for group_result in results:
        for changed_answer in group_result.changed_answers:
            _dicts.append(
                {
                    "model": group_result.model_meta.trained_on,
                    "same_answer": not changed_answer.changed_answer,
                    "trained_samples": group_result.model_meta.trained_samples,
                    "trained_on": group_result.model_meta.trained_on.value,
                }
            )

    # x-axis is trained_samples
    # y-axis is percentage same
    # hue is trained on value
    df = pd.DataFrame(_dicts)
    ax = seaborn.lineplot(x="trained_samples", y="same_answer", hue="trained_on", data=df)
    # change the y-axis to be "Percentage of questions with same answer with vs without COT"
    ax.set(ylabel="% Same Answer With and Without CoT ")
    plt.show()


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

    model_metas = Slist(
        [
            ModelMeta(
                model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8FeFMAOR",
                trained_samples=1_000,
                trained_on=TrainedOn.CONTROL,
            ),
            ModelMeta(
                model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8FfN5MGW",
                trained_samples=10_000,
                trained_on=TrainedOn.CONTROL,
            ),
            ModelMeta(
                model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8FenfJNo",
                trained_samples=100_000,
                trained_on=TrainedOn.CONTROL,
            ),
            # interventions
            ModelMeta(
                model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Ff8h3yF",
                trained_samples=1_000,
                trained_on=TrainedOn.INTERVENTION,
            ),
            ModelMeta(
                model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8FciULKF",
                trained_samples=10_000,
                trained_on=TrainedOn.INTERVENTION,
            ),
            ModelMeta(
                model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8FWFloan",
                trained_samples=100_000,
                trained_on=TrainedOn.INTERVENTION,
            ),
        ]
    )
    meta_lookup: Mapping[str, ModelMeta] = {m.model: m for m in model_metas}
    models = [m.model for m in model_metas]
    stage_one_obs = stage_one_stream(
        formatters=[ZeroShotCOTUnbiasedFormatter.name(), ZeroShotUnbiasedFormatter.name()],
        tasks=["aqua", "mmlu", "truthful_qa", "logiqa"],
        example_cap=600,
        num_retries=1,
        raise_after_retries=False,
        temperature=1.0,
        caller=stage_one_caller,
        batch=20,
        models=models,
    )
    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    stage_one_caller.save_cache()
    percentage_changed = percentage_changed_per_model(results, meta_lookup)
    seaborn_line_plot(
        percentage_changed,
    )


if __name__ == "__main__":
    asyncio.run(main())
