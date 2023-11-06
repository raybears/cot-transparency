from enum import Enum
from pathlib import Path
from typing import Sequence, Type

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from slist import Slist

from cot_transparency.data_models.data import COT_TESTING_TASKS
from cot_transparency.data_models.io import read_all_for_selections
from cot_transparency.formatters import StageOneFormatter
from cot_transparency.formatters.more_biases.anchor_initial_wrong import (
    ZeroShotInitialWrongFormatter,
)
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
)
from scripts.matching_user_answer import matching_user_answer_plot_info
from scripts.multi_accuracy import AccuracyOutput, PlotInfo


class DataFromOptionsWithCot(str, Enum):
    gpt_35_turbo = "gpt-3.5-turbo, trained on biased contexts"
    gpt_35_turbo_big_brain = "gpt-3.5-turbo, trained on biased contexts, big brained"
    claude_2 = "claude-2, trained on biased contexts"
    claude_2_unbiased = "claude-2, trained on unbiased contexts"


class ModelTrainMeta(BaseModel):
    name: str
    trained_samples: int
    trained_on: DataFromOptionsWithCot


class ModelNameAndTrainedSamplesAndMetrics(BaseModel):
    train_meta: ModelTrainMeta
    metrics: AccuracyOutput


def read_metric_from_meta(
    meta: ModelTrainMeta,
    exp_dir: str,
    formatter: Type[StageOneFormatter],
    tasks: Sequence[str],
) -> ModelNameAndTrainedSamplesAndMetrics:
    # read the metric from the meta
    read = read_all_for_selections(
        exp_dirs=[Path(exp_dir)],
        models=[meta.name],
        formatters=[formatter.name()],
        tasks=tasks,
    )
    # hardcode to calculate % matching
    percent_matching: PlotInfo = matching_user_answer_plot_info(
        all_tasks=read,
    )
    return ModelNameAndTrainedSamplesAndMetrics(train_meta=meta, metrics=percent_matching.acc)


def samples_meta() -> Slist[ModelTrainMeta]:
    # fill this up from wandb https://wandb.ai/raybears/consistency-training?workspace=user-chuajamessh
    all_meta = Slist(
        [
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86IQPMdh",
                trained_samples=72000,
                trained_on=DataFromOptionsWithCot.claude_2,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86GTKEjL",
                trained_samples=12000,
                trained_on=DataFromOptionsWithCot.claude_2,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86FSz24P",
                trained_samples=1000,
                trained_on=DataFromOptionsWithCot.claude_2,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86HOLLHD",
                trained_samples=100,
                trained_on=DataFromOptionsWithCot.claude_2,
            ),
            # Unbiased contexts
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::88h1pB4E",
                trained_samples=72000,
                trained_on=DataFromOptionsWithCot.claude_2_unbiased,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::88g7AphR",
                trained_samples=12000,
                trained_on=DataFromOptionsWithCot.claude_2_unbiased,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::88cQLqLT",
                trained_samples=1000,
                trained_on=DataFromOptionsWithCot.claude_2_unbiased,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86HyHsqO",
                trained_samples=72000,
                trained_on=DataFromOptionsWithCot.gpt_35_turbo,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86GvNx2m",
                trained_samples=12000,
                trained_on=DataFromOptionsWithCot.gpt_35_turbo,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86FU2RR0",
                trained_samples=1000,
                trained_on=DataFromOptionsWithCot.gpt_35_turbo,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86H2Q1de",
                trained_samples=100,
                trained_on=DataFromOptionsWithCot.gpt_35_turbo,
            ),
            # big brain
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::88hjDp5H",
                trained_samples=72000,
                trained_on=DataFromOptionsWithCot.gpt_35_turbo_big_brain,
            ),
        ]
    )
    distinct_models = all_meta.distinct_by(lambda i: i.name)
    assert len(distinct_models) == len(all_meta), "There are duplicate models in the list"
    return distinct_models


def read_all_metrics(
    samples: Slist[ModelTrainMeta],
    exp_dir: str,
    formatter: Type[StageOneFormatter],
    tasks: Sequence[str],
) -> Slist[ModelNameAndTrainedSamplesAndMetrics]:
    return samples.map(lambda meta: read_metric_from_meta(meta=meta, exp_dir=exp_dir, formatter=formatter, tasks=tasks))


def seaborn_line_plot(
    data: Sequence[ModelNameAndTrainedSamplesAndMetrics],
    bias: str,
    error_bars: bool = True,
):
    y_axis_label = "Percent matching bias"
    df = pd.DataFrame(
        [
            {
                "Trained Samples": i.train_meta.trained_samples,
                y_axis_label: i.metrics.accuracy,
                "Error Bars": i.metrics.error_bars,
                "Trained on COTs from": i.train_meta.trained_on.value,
            }
            for i in data
        ]
    )
    sns.lineplot(data=df, x="Trained Samples", y=y_axis_label, hue="Trained on COTs from")

    if error_bars:
        for name, group in df.groupby("Trained on COTs from"):
            plt.errorbar(
                group["Trained Samples"],
                group[y_axis_label],
                yerr=group["Error Bars"],
                fmt="none",
                capsize=5,
                ecolor="black",
            )
    plt.xticks(df["Trained Samples"].unique())  # type: ignore
    # rotate xticks slightly
    plt.xticks(rotation=-15)
    plt.ylim(0, 1)
    # x-axis log scale
    plt.xscale("log")
    # title
    plt.title(f"Is training on Claude vs gpt-3.5-turbo COTs better\n for the {bias} task")
    plt.show()


if __name__ == "__main__":
    defined_meta = samples_meta()
    read_metrics = read_all_metrics(
        samples=defined_meta,
        exp_dir="experiments/finetune_2",
        formatter=ZeroShotInitialWrongFormatter,
        tasks=COT_TESTING_TASKS,
    )
    seaborn_line_plot(read_metrics, bias="initial wrong bias")
    read_metrics_wrong_few_shot = read_all_metrics(
        samples=defined_meta,
        exp_dir="experiments/finetune_2",
        formatter=WrongFewShotIgnoreMistakesBiasedFormatter,
        tasks=COT_TESTING_TASKS,
    )
    seaborn_line_plot(read_metrics_wrong_few_shot, bias="wrong few shot bias")
