from pathlib import Path
from typing import Sequence

from pydantic import BaseModel
from slist import Slist

from cot_transparency.data_models.io import read_all_for_selections
from cot_transparency.data_models.models import TaskOutput
from scripts.finetune_cot import DataFromOptions
from scripts.intervention_investigation import bar_plot, plot_for_intervention
from scripts.matching_user_answer import matching_user_answer_plot_info
from scripts.multi_accuracy import AccuracyOutput, PlotInfo
from stage_one import COT_TESTING_TASKS, main as stage_one_main


def run_claude_vs_gpt_experiments(models: list[str]):
    # Run temperature 1
    stage_one_main(
        exp_dir="experiments/finetune_2",
        models=models,
        formatters=[
            "WrongFewShotIgnoreMistakesBiasedFormatter",
            "ZeroShotInitialWrongFormatter",
        ],
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
    )
    # Run baseline dotted line temperature 1
    stage_one_main(
        exp_dir="experiments/finetune_2",
        models=["gpt-3.5-turbo"],
        formatters=[
            "ZeroShotCOTUnbiasedFormatter",
        ],
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
    )


def get_accuracy_plot_info_for_model_name(
    tuples: tuple[str, Slist[TaskOutput]],
) -> PlotInfo:
    name, tasks = tuples
    plot_info = plot_for_intervention(
        all_tasks=tasks,
        name_override=name,
    )

    return plot_info


def get_matching_plot_info_for_model_name(
    tuples: tuple[str, Slist[TaskOutput]],
) -> PlotInfo:
    name, tasks = tuples
    plot_info = matching_user_answer_plot_info(
        all_tasks=tasks,
        name_override=name,
    )

    return plot_info


def plot_accuracies_for_model(
    formatter: str,
    models: list[str],
):
    paths: list[Path] = [Path("experiments/finetune_2")]
    tasks = COT_TESTING_TASKS
    task_names = ",".join(tasks)
    read: Slist[TaskOutput] = read_all_for_selections(
        exp_dirs=paths,
        formatters=[formatter],
        models=models,
        tasks=tasks,
    )
    print(f"Read {len(read)} experiments")
    # groupby MODEL
    grouped: Slist[tuple[str, Slist[TaskOutput]]] = read.group_by(
        lambda x: x.task_spec.inference_config.model
    )
    print(f"Grouped into {len(grouped)} groups")
    # get plot info
    accuracy_plot_info: Slist[PlotInfo] = grouped.map(
        get_accuracy_plot_info_for_model_name
    )

    names_overrides = {
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::86FU2RR0": "1000 gpt-3.5-turbo consistency samples",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::86FSz24P": "1000 claude-2 consistency samples",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::86FXkKiP": "12000 gpt-3.5-turbo consistency samples",
        "todo": "12000 claude-2 samples",
    }

    bar_plot(
        plot_infos=accuracy_plot_info,
        title=f"How does the model's accuracy change if we train on claude-2 COTs instead?<br>Dataset={task_names}",
        y_axis_title="Accuracy (%)",
        dotted_line=None,
        max_y=1.0,
        name_override=names_overrides,
    )

    matching_plot_info: Slist[PlotInfo] = grouped.map(
        get_matching_plot_info_for_model_name
    )

    bar_plot(
        plot_infos=matching_plot_info,
        title=f"How does the debiasing effect change if we train on claude-2 COTs?<br>Dataset={task_names}<br>{formatter}",
        y_axis_title="Percent matching bias answer (%)",
        dotted_line=None,
        max_y=1.0,
        name_override=names_overrides,
    )


class ModelTrainMeta(BaseModel):
    name: str
    trained_samples: int
    trained_on: DataFromOptions


class ModelNameAndTrainedSamplesAndMetrics(BaseModel):
    train_meta: ModelTrainMeta
    metrics: AccuracyOutput


def plotly_line_plot(
    data: Sequence[ModelNameAndTrainedSamplesAndMetrics], error_bars: bool = True
):
    # X axis: trained samples
    # Color: DataFromOptions string
    # Y axis: AccuracyOutput accuracy
    # Y axis error bars: AccuracyOutput error_bars
    ...


if __name__ == "__main__":
    models = ["gpt-3.5-turbo", "ft:gpt-3.5-turbo-0613:academicsnyuperez::88cQLqLT"]
    run_claude_vs_gpt_experiments(models=models)
    plot_accuracies_for_model(
        formatter="ZeroShotInitialWrongFormatter",
        models=models,
    )
    plot_accuracies_for_model(
        formatter="WrongFewShotIgnoreMistakesBiasedFormatter",
        models=models,
    )
