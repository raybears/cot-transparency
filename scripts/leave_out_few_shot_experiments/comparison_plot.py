from enum import Enum
from pathlib import Path
from typing import Sequence, Type, Optional
from pydantic import BaseModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from slist import Slist

from cot_transparency.formatters import StageOneFormatter
from cot_transparency.formatters.core.answer_always_a import AnswerAlwaysANoCOTFormatter, AnswerAlwaysAFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.more_biases.wrong_few_shot import WrongFewShotIgnoreMistakesBiasedFormatter
from cot_transparency.formatters.verbalize.formatters import CheckmarkBiasedFormatter, CrossBiasedFormatter
from scripts.intervention_investigation import plot_for_intervention, DottedLine
from scripts.matching_user_answer import matching_user_answer_plot_info, random_chance_matching_answer_plot_dots
from scripts.multi_accuracy import PlotInfo
from scripts.simple_formatter_names import FORMATTER_TO_SIMPLE_NAME
from scripts.utils.loading import read_all_for_selections
from stage_one import main, COT_TESTING_TASKS


class RunOptions(str, Enum):
    no_filter = "Biased contexts, no filter"
    correct_answer = "Biased contexts, filtered to be correct"
    control_unbiased = "Unbiased contexts, filtered to be correct (control)"


class ModelTrainMeta(BaseModel):
    name: str
    trained_samples: int
    trained_on: RunOptions


class ModelNameAndTrainedSamplesAndMetrics(BaseModel):
    train_meta: ModelTrainMeta
    percent_matching: PlotInfo
    accuracy: PlotInfo


def read_metric_from_meta(
    meta: ModelTrainMeta, exp_dir: str, formatter: Type[StageOneFormatter], tasks: Sequence[str]
) -> ModelNameAndTrainedSamplesAndMetrics:
    # read the metric from the meta
    read = read_all_for_selections(
        exp_dirs=[Path(exp_dir)],
        models=[meta.name],
        formatters=[formatter.name()],
        tasks=tasks,
    )

    percent_matching = matching_user_answer_plot_info(
        all_tasks=read,
    )
    accuracy = plot_for_intervention(
        all_tasks=read,
    )
    return ModelNameAndTrainedSamplesAndMetrics(train_meta=meta, percent_matching=percent_matching, accuracy=accuracy)


def run_unbiased_acc_experiments(
    meta: Sequence[ModelTrainMeta], tasks: Sequence[str], biases: Sequence[Type[StageOneFormatter]]
) -> None:
    normal_cot_models = [m.name for m in meta if m]
    main(
        exp_dir="experiments/finetune_3",
        models=normal_cot_models,
        formatters=[b.name() for b in biases],
        tasks=tasks,
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
        batch=20,
    )


def samples_meta() -> Slist[ModelTrainMeta]:
    # fill this up from wandb https://wandb.ai/raybears/consistency-training?workspace=user-chuajamessh
    all_meta = Slist(
        [
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89KTeuvo",
                trained_samples=100,
                trained_on=RunOptions.no_filter,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89FCpgle",
                trained_samples=1000,
                trained_on=RunOptions.no_filter,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89GKVlFT",
                trained_samples=10000,
                trained_on=RunOptions.no_filter,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89LJDfgI",
                trained_samples=20000,
                trained_on=RunOptions.no_filter,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89LueXqz",
                trained_samples=100,
                trained_on=RunOptions.correct_answer,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89FBrz5b",
                trained_samples=1000,
                trained_on=RunOptions.correct_answer,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89G8xNY5",
                trained_samples=10000,
                trained_on=RunOptions.correct_answer,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89LKfzaR",
                trained_samples=20000,
                trained_on=RunOptions.correct_answer,
            ),
            # control unbiased
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89NHOL5b",
                trained_samples=100,
                trained_on=RunOptions.control_unbiased,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89G2vwHZ",
                trained_samples=1000,
                trained_on=RunOptions.control_unbiased,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89GzBGx0",
                trained_samples=10000,
                trained_on=RunOptions.control_unbiased,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89LJSEdM",
                trained_samples=20000,
                trained_on=RunOptions.control_unbiased,
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
    error_bars: bool = True,
    percent_matching: bool = True,
    title: str = "",
    dotted_line: Optional[DottedLine] = None,
):
    y_axis_label = "Percent Matching bias" if percent_matching else "Accuracy"
    df = pd.DataFrame(
        [
            {
                "Trained Samples": i.train_meta.trained_samples,
                y_axis_label: i.percent_matching.acc.accuracy if percent_matching else i.accuracy.acc.accuracy,
                "Error Bars": i.percent_matching.acc.error_bars if percent_matching else i.accuracy.acc.error_bars,
                "Trained on COTs from": i.train_meta.trained_on.value,
            }
            for i in data
        ]
    )
    sns.lineplot(data=df, x="Trained Samples", y=y_axis_label, hue="Trained on COTs from")
    plt.title(title)
    # Add dotted line for random chance
    if dotted_line:
        plt.axhline(y=dotted_line.value, color=dotted_line.color, linestyle="dashed", label=dotted_line.name)
        # Add to legend saying that the dotted line is Random chance
        plt.legend()
    plt.title(title)

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
    unique = df["Trained Samples"].unique()  # type: ignore
    print(f"Unique ticks: {unique}")
    plt.xticks(unique)
    # make sure all the x ticks are visible
    # plt.margins(x=0.1)
    plt.ylim(0, 1)
    # log scale for x axis
    plt.xscale("log")
    plt.show()


if __name__ == "__main__":
    exp_dir = "experiments/finetune_3"
    defined_meta = samples_meta()
    tasks = COT_TESTING_TASKS
    biases: Sequence[Type[StageOneFormatter]] = [
        WrongFewShotIgnoreMistakesBiasedFormatter,
        CheckmarkBiasedFormatter,
        CrossBiasedFormatter,
        AnswerAlwaysAFormatter,
    ]
    # run_unbiased_acc_experiments(defined_meta, tasks, biases=biases)
    random_chance: PlotInfo = random_chance_matching_answer_plot_dots(
        all_tasks=read_all_for_selections(
            exp_dirs=[Path(exp_dir)],
            models=["gpt-3.5-turbo"],
            formatters=[ZeroShotCOTUnbiasedFormatter.name()],
            tasks=tasks,
        ),
        model="gpt-3.5-turbo",
        name_override="Random chance",
        formatter=ZeroShotCOTUnbiasedFormatter,
        for_task=tasks,
    )
    dotted_line = DottedLine(name="Random chance", value=random_chance.acc.accuracy, color="red")
    for formatter in biases:
        finetuned_ = read_all_metrics(
            samples=defined_meta,
            exp_dir=exp_dir,
            formatter=formatter,
            tasks=tasks,
        )
        models = [m.name for m in defined_meta]

        # add these so that the plot can start from the origin where we have a nonfinetuned model
        non_finetuned_meta = Slist(
            [
                ModelTrainMeta(
                    name="gpt-3.5-turbo",
                    trained_samples=1,
                    trained_on=RunOptions.no_filter,
                ),
                ModelTrainMeta(
                    name="gpt-3.5-turbo",
                    trained_samples=1,
                    trained_on=RunOptions.correct_answer,
                ),
                ModelTrainMeta(
                    name="gpt-3.5-turbo",
                    trained_samples=1,
                    trained_on=RunOptions.control_unbiased,
                ),
            ]
        )
        non_finetuned_metrics = read_all_metrics(
            samples=non_finetuned_meta,
            exp_dir=exp_dir,
            formatter=formatter,
            tasks=tasks,
        )

        nice_name = FORMATTER_TO_SIMPLE_NAME.get(formatter, formatter.name())
        combined = finetuned_ + non_finetuned_metrics
        seaborn_line_plot(combined, percent_matching=False, title=f"Accuracy for \n{nice_name} bias")
        seaborn_line_plot(
            combined,
            percent_matching=True,
            title=f"Percent matching bias for \n{nice_name} bias",
            dotted_line=dotted_line,
        )
    # unbiased = read_all_metrics(
    #     samples=defined_meta,
    #     exp_dir="experiments/finetune_2",
    #     formatter=ZeroShotCOTUnbiasedFormatter,
    #     tasks=tasks,
    # )
    # seaborn_line_plot(unbiased, percent_matching=False, title="Accuracy for the unbiased bias")
