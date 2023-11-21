import asyncio
from pathlib import Path

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.data_models.data import InverseScalingTask
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.inverse_scaling.repeat_mistakes import (
    ZeroShotCOTUnbiasedFollowInstructionsFormatter,
)
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.intervention_investigation import plot_for_intervention


from enum import Enum
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel

from scripts.multi_accuracy import AccuracyOutput, PlotInfo


class Method(str, Enum):
    ours = "ours"
    control = "control"


class ModelTrainMeta(BaseModel):
    name: str
    trained_samples: float
    trained_on: Method


class ModelNameAndTrainedSamplesAndMetrics(BaseModel):
    train_meta: ModelTrainMeta
    metrics: AccuracyOutput


def read_metric_from_meta(
    meta: ModelTrainMeta,
    all_tasks: Slist[TaskOutput],
) -> ModelNameAndTrainedSamplesAndMetrics:
    # read the metric from the meta
    all_tasks = all_tasks.filter(lambda x: x.task_spec.inference_config.model == meta.name)

    acc: PlotInfo = plot_for_intervention(all_tasks=all_tasks)
    return ModelNameAndTrainedSamplesAndMetrics(train_meta=meta, metrics=acc.acc)


def samples_meta() -> Slist[ModelTrainMeta]:
    # fill this up from wandb https://wandb.ai/raybears/consistency-training?workspace=user-chuajamessh
    """
    [
        "gpt-3.5-turbo-0613",
        # instruct 0.0 (control)
        "ft:gpt-3.5-turbo-0613:far-ai::8MEjCxQd",
        # instruct 0.0
        "ft:gpt-3.5-turbo-0613:far-ai::8MEm4sSK",
        # instruct 0.1 (control)
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MEVuHFu",
        # instruct 0.1
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MEw9gAq",
        # instruct 1.0 (control)
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",
        # isntruct 1.0
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lywfnnz",
        # instruct 10.0 (control)
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lk3VEOY",
        # instruct 10.0 (ours)
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::8LpkPY5V",
    ]
    """
    all_meta = Slist(
        [
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8MEjCxQd",
                trained_samples=0,
                trained_on=Method.control,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8MEm4sSK",
                trained_samples=0,
                trained_on=Method.ours,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8MEVuHFu",
                trained_samples=0.1,
                trained_on=Method.control,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8MEw9gAq",
                trained_samples=0.1,
                trained_on=Method.ours,
            ),
            # Unbiased contexts
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",
                trained_samples=1,
                trained_on=Method.control,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lywfnnz",
                trained_samples=1,
                trained_on=Method.ours,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lk3VEOY",
                trained_samples=10,
                trained_on=Method.control,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8LpkPY5V",
                trained_samples=10,
                trained_on=Method.ours,
            ),
        ]
    )
    distinct_models = all_meta.distinct_by(lambda i: i.name)
    assert len(distinct_models) == len(all_meta), "There are duplicate models in the list"
    return distinct_models


def read_all_metrics(
    samples: Slist[ModelTrainMeta],
    all_tasks: Slist[TaskOutput],
) -> Slist[ModelNameAndTrainedSamplesAndMetrics]:
    return samples.map(lambda meta: read_metric_from_meta(meta, all_tasks))


def convert_to_linear_scale(x: float) -> int:
    # hack to make it evenly spaced
    if x == 0:
        return 0
    if x == 0.1:
        return 1
    if x == 1:
        return 2
    if x == 10:
        return 3
    else:
        raise ValueError(f"Unknown value {x}")


def linear_scale_to_string(x: int) -> str:
    if x == 0:
        return "0 x"
    if x == 1:
        return "0.1 x"
    if x == 2:
        return "1 x"
    if x == 3:
        return "10 x"
    else:
        raise ValueError(f"Unknown value {x}")


def seaborn_line_plot(
    data: Sequence[ModelNameAndTrainedSamplesAndMetrics],
    error_bars: bool = True,
):
    y_axis_label = "Accuracy"
    df = pd.DataFrame(
        [
            {
                "Additional Instruct Samples": convert_to_linear_scale(i.train_meta.trained_samples),
                y_axis_label: i.metrics.accuracy,
                "Error Bars": i.metrics.error_bars,
                "Training method": i.train_meta.trained_on.value,
            }
            for i in data
        ]
    )
    sns.lineplot(data=df, x="Additional Instruct Samples", y=y_axis_label, hue="Training method")

    if error_bars:
        for name, group in df.groupby("Training method"):
            plt.errorbar(
                group["Additional Instruct Samples"],
                group[y_axis_label],
                yerr=group["Error Bars"],
                fmt="none",
                capsize=5,
                ecolor="black",
            )
    # plt.xticks(df["Additional Instruct Samples"].unique())  # type: ignore
    # rotate xticks slightly
    # plt.xticks(rotation=-15)
    # set x ticks to unique
    # plt.xticks(df["Additional Instruct Samples"].unique())
    # replot the xticks with linear_scale_to_string
    plt.xticks(
        df["Additional Instruct Samples"].unique(),  # type: ignore
        [linear_scale_to_string(i) for i in df["Additional Instruct Samples"].unique()],
    )

    plt.ylim(0, 1)
    # x-axis log scale
    # plt.xscale("log")
    # title
    plt.title("Accuracy on Strong Prior tasks")
    plt.show()


async def plot_accuracies():
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MK49rPG",  # control for superdataset 10k
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MKt0VnY",  # ours (superdataset)
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8MmNKzZh",  # ours (superdataset, without few shot)
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8KreNXFv", # control paraphrasing 10k
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Kb1ayZh" # ours paraphrasing 10k
    models = [m.name for m in samples_meta()]
    stage_one_path = Path("experiments/inverse_scaling/stage_one.jsonl")
    stage_one_caller = UniversalCaller().with_file_cache(stage_one_path, write_every_n=1000)
    # task = InverseScalingTask.memo_trap
    # ZeroShotCOTUnbiasedFormatter
    # ZeroShotCOTUnbiasedRepeatMistakesFormatter
    formatters = [ZeroShotCOTUnbiasedFollowInstructionsFormatter]
    stage_one_obs = stage_one_stream(
        formatters=[f.name() for f in formatters],
        tasks=[InverseScalingTask.memo_trap, InverseScalingTask.resisting_correction, InverseScalingTask.redefine],
        example_cap=300,
        num_tries=1,
        raise_after_retries=False,
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=models,
    )

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    write_jsonl_file_from_basemodel("experiments/inverse_scaling/instruction_following.jsonl", results)
    results_filtered = results.filter(lambda x: x.first_parsed_response is not None)
    stage_one_caller.save_cache()

    defined_meta = samples_meta()
    read_metrics = read_all_metrics(samples=defined_meta, all_tasks=results_filtered)
    seaborn_line_plot(read_metrics)


if __name__ == "__main__":
    asyncio.run(plot_accuracies())
