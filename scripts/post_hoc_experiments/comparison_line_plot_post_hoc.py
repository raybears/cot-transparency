from enum import Enum
from pathlib import Path
from typing import Sequence, Type, Optional
from pydantic import BaseModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from slist import Slist

from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters import StageOneFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.more_biases.anchor_initial_wrong import ZeroShotInitialWrongFormatter
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
)
from scripts.intervention_investigation import plot_for_intervention, DottedLine
from scripts.matching_user_answer import matching_user_answer_plot_info, random_chance_matching_answer_plot_dots
from scripts.multi_accuracy import PlotInfo
from cot_transparency.data_models.io import read_all_for_selections
from stage_one import main, COT_TESTING_TASKS


class PostHocOptions(str, Enum):
    normal_cot = "Normal COT with answer at the end, instructed to COT"
    normal_instruct_no_cot = "Normal COT with answer at the end, instructed to NOT COT"
    post_hoc = "Post hoc COT with answer at the beginning"
    no_cot_majority = "Trained on 98% no COTs, 2% with COTs"


class ModelTrainMeta(BaseModel):
    name: str
    trained_samples: int
    trained_on: PostHocOptions


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
    if meta.trained_on == PostHocOptions.post_hoc:
        print("Filtering for successful post hoc")
        new = filter_successful_post(read)
        print(f"Filtered from {len(read)} to {len(new)}")
        read = new

    percent_matching = matching_user_answer_plot_info(
        all_tasks=read,
    )
    accuracy = plot_for_intervention(
        all_tasks=read,
    )
    return ModelNameAndTrainedSamplesAndMetrics(train_meta=meta, percent_matching=percent_matching, accuracy=accuracy)


def run_unbiased_acc_experiments(meta: Sequence[ModelTrainMeta], tasks: Sequence[str]) -> None:
    # Also run for non COT prompt for the normal COT models
    normal_cot_models = [m.name for m in meta if m.trained_on == PostHocOptions.normal_cot]
    main(
        exp_dir="experiments/finetune_2",
        models=normal_cot_models,
        formatters=[
            "WrongFewShotIgnoreMistakesBiasedNoCOTFormatter",
        ],
        tasks=tasks,
        example_cap=1000,
        raise_after_retries=False,
        temperature=1.0,
        batch=5,
    )

    models: list[str] = [m.name for m in meta]
    main(
        exp_dir="experiments/finetune_2",
        models=models,
        formatters=[
            "ZeroShotCOTUnbiasedFormatter",
        ],
        tasks=tasks,
        example_cap=1000,
        raise_after_retries=False,
        temperature=1.0,
        batch=5,
    )


def samples_meta() -> Slist[ModelTrainMeta]:
    # fill this up from wandb https://wandb.ai/raybears/consistency-training?workspace=user-chuajamessh
    all_meta = Slist(
        [
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86HyHsqO",
                trained_samples=72000,
                trained_on=PostHocOptions.normal_cot,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86GvNx2m",
                trained_samples=12000,
                trained_on=PostHocOptions.normal_cot,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86FU2RR0",
                trained_samples=1000,
                trained_on=PostHocOptions.normal_cot,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86H2Q1de",
                trained_samples=100,
                trained_on=PostHocOptions.normal_cot,
            ),
            # Post hoc
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86bHY8x6",
                trained_samples=72000,
                trained_on=PostHocOptions.post_hoc,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86cD8ES5",
                trained_samples=12000,
                trained_on=PostHocOptions.post_hoc,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86ZOYZen",
                trained_samples=1000,
                trained_on=PostHocOptions.post_hoc,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86YRTE3z",
                trained_samples=100,
                trained_on=PostHocOptions.post_hoc,
            ),
            # No COT majority
            #     ModelTrainMeta(
            #         name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86eKwqwy",
            #         trained_samples=72000,
            #         trained_on=PostHocOptions.no_cot_majority,
            #     ),
            #     ModelTrainMeta(
            #         name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86h4marp",
            #         trained_samples=12000,
            #         trained_on=PostHocOptions.no_cot_majority,
            #     ),
            #     ModelTrainMeta(
            #         name="ft:gpt-3.5-turbo-0613:academicsnyuperez::86cGlzzb",
            #         trained_samples=1000,
            #         trained_on=PostHocOptions.no_cot_majority,
            #     ),
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


def replace_enum(m: ModelNameAndTrainedSamplesAndMetrics) -> ModelNameAndTrainedSamplesAndMetrics:
    new = m.model_copy(deep=True)
    new.train_meta.trained_on = PostHocOptions.normal_instruct_no_cot
    return new


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
    plt.xticks(df["Trained Samples"].unique())  # type: ignore
    # make sure all the x ticks are visible
    plt.margins(x=0.1)
    plt.ylim(0, 1)
    # log scale for x axis
    plt.xscale("log")
    plt.show()


def filter_successful_post_hoc(task: TaskOutput) -> bool:
    output = task.inference_output.raw_response
    must_start_str = "The best answer is"
    return output.startswith(must_start_str)


def filter_successful_post(tasks: Sequence[TaskOutput]) -> Slist[TaskOutput]:
    return Slist(tasks).filter(filter_successful_post_hoc)


if __name__ == "__main__":
    defined_meta = samples_meta()
    tasks = COT_TESTING_TASKS
    # run_unbiased_acc_experiments(defined_meta, tasks)
    initial_wrong = read_all_metrics(
        samples=defined_meta,
        exp_dir="experiments/finetune_2",
        formatter=ZeroShotInitialWrongFormatter,
        tasks=tasks,
    )
    random_chance: PlotInfo = random_chance_matching_answer_plot_dots(
        all_tasks=read_all_for_selections(
            exp_dirs=[Path("experiments/finetune_2")],
            models=["gpt-3.5-turbo"],
            tasks=tasks,
            formatters=[ZeroShotCOTUnbiasedFormatter.name()],
        ),
        model="gpt-3.5-turbo",
        name_override="Random chance",
        formatter=ZeroShotCOTUnbiasedFormatter,
        for_task=tasks,
    )
    dotted_line = DottedLine(name="Random chance", value=random_chance.acc.accuracy, color="red")
    seaborn_line_plot(initial_wrong, percent_matching=False, title="Accuracy for the initial wrong bias")
    seaborn_line_plot(
        initial_wrong,
        percent_matching=True,
        title="Percent matching for the initial wrong bias",
        dotted_line=dotted_line,
    )
    wrong_few_shot = read_all_metrics(
        samples=defined_meta,
        exp_dir="experiments/finetune_2",
        formatter=WrongFewShotIgnoreMistakesBiasedFormatter,
        tasks=tasks,
    )

    normal_cot_models = Slist([m for m in defined_meta if m.trained_on == PostHocOptions.normal_cot])
    wrong_few_shot_non_cot = read_all_metrics(
        samples=normal_cot_models,
        exp_dir="experiments/finetune_2",
        formatter=WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
        tasks=tasks,
    ).map(replace_enum)

    seaborn_line_plot(
        wrong_few_shot + wrong_few_shot_non_cot, percent_matching=False, title="Accuracy for the wrong few shot bias"
    )
    seaborn_line_plot(
        wrong_few_shot + wrong_few_shot_non_cot,
        percent_matching=True,
        title="Percent matching for the wrong few shot bias",
        dotted_line=dotted_line,
    )
    unbiased = read_all_metrics(
        samples=defined_meta,
        exp_dir="experiments/finetune_2",
        formatter=ZeroShotCOTUnbiasedFormatter,
        tasks=tasks,
    )
    seaborn_line_plot(unbiased, percent_matching=False, title="Accuracy for the unbiased bias")
