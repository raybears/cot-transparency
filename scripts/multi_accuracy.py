import glob
import os
from pathlib import Path

import math
from typing import Optional
import fire
import plotly.colors as pcol
import plotly.graph_objects as go
import plotly.io as pio
from pydantic import BaseModel

from cot_transparency.data_models.models import TaskOutput, ExperimentJsonFormat
from cot_transparency.formatters.verbalize.biased_few_shots import parse_out_bias_explanation, BiasAndExplanation
from stage_one import read_done_experiment, TASK_LIST


class AccuracyOutput(BaseModel):
    accuracy: float
    error_bars: float


def compute_error_bars(num_trials: int, num_successes: int, confidence_level: float = 1.96) -> float:
    p = num_successes / num_trials
    se = math.sqrt((p * (1 - p)) / num_trials)
    return confidence_level * se


def accuracy_for_file(path: Path, inconsistent_only: bool = True) -> AccuracyOutput:
    experiment: ExperimentJsonFormat = read_done_experiment(path)
    return accuracy_outputs(experiment.outputs, inconsistent_only=inconsistent_only)


def accuracy_outputs(outputs: list[TaskOutput], inconsistent_only: bool = True) -> AccuracyOutput:
    score = 0
    # filter out the consistent if inconsistent_only is True
    filtered_outputs = (
        [output for output in outputs if output.task_spec.biased_ans != output.task_spec.ground_truth]
        if inconsistent_only
        else outputs
    )
    for item in filtered_outputs:
        ground_truth = item.task_spec.ground_truth
        predicted = item.model_output.parsed_response
        is_correct = predicted == ground_truth
        if is_correct:
            score += 1
    if len(filtered_outputs) == 0:
        raise ValueError("No outputs to score")

    # Compute error bars for accuracy
    error_bars = compute_error_bars(num_trials=len(filtered_outputs), num_successes=score)

    return AccuracyOutput(accuracy=score / len(filtered_outputs), error_bars=error_bars)


def spotted_bias(raw_response: str) -> bool:
    return "NO_BIAS" not in raw_response


def filter_only_bias_spotted(outputs: list[TaskOutput]) -> list[TaskOutput]:
    new_list: list[TaskOutput] = []
    for output in outputs:
        if spotted_bias(output.model_output.raw_response):
            new_list.append(output)
    return [output for output in new_list if output.model_output]


def extract_labelled_bias(outputs: list[TaskOutput]) -> list[BiasAndExplanation]:
    new_list: list[BiasAndExplanation] = []
    for output in outputs:
        bias_and_explanation = parse_out_bias_explanation(output.model_output.raw_response)
        new_list.append(bias_and_explanation)
    return new_list


def filter_no_bias_spotted(outputs: list[TaskOutput]) -> list[TaskOutput]:
    new_list: list[TaskOutput] = []
    for output in outputs:
        new_output = output.copy()
        if not spotted_bias(output.model_output.raw_response):
            new_list.append(new_output)
    return [output for output in new_list if output.model_output]


class PlotDots(BaseModel):
    acc: AccuracyOutput
    name: str


class TaskAndPlotDots(BaseModel):
    task_name: str
    plot_dots: list[PlotDots]


class PathsAndNames(BaseModel):
    path: str
    name: str


def plot_vertical_acc(paths: list[PathsAndNames]) -> list[PlotDots]:
    out: list[PlotDots] = []
    for path in paths:
        out.append(PlotDots(acc=accuracy_for_file(Path(path.path)), name=path.name))
    return out


def accuracy_plot(list_task_and_dots: list[TaskAndPlotDots], title: str, save_file_path: Optional[str] = None):
    fig = go.Figure()
    colors = pcol.qualitative.D3
    symbols = ["circle", "square", "diamond", "cross", "x", "triangle-up", "pentagon"]  # add more symbols if needed
    x_labels: list[str] = []
    added_labels: set[str] = set()  # to remember the labels we have already added

    for i, task_and_plot in enumerate(list_task_and_dots):
        plot_dots = task_and_plot.plot_dots
        x_labels.append(task_and_plot.task_name)

        for j, dot in enumerate(plot_dots):
            fig.add_trace(
                go.Scatter(
                    x=[i + 1],
                    y=[dot.acc.accuracy],
                    mode="markers",
                    marker=dict(
                        size=[15],
                        color=colors[j % len(colors)],
                        symbol=symbols[j % len(symbols)],  # Use different symbols for each marker
                    ),
                    name=dot.name,  # specify the name that will appear in legend
                    showlegend=dot.name not in added_labels,  # don't show in legend if label has been added
                    error_y=dict(  # add error bars
                        type="data",  # value of error bar given in data coordinates
                        array=[dot.acc.error_bars],  # first array is errors for y values
                        visible=True,
                    ),
                )
            )
            added_labels.add(dot.name)  # remember that this label has been added

    fig.update_xaxes(
        range=[0.5, len(list_task_and_dots) + 0.5],
        tickvals=list(range(1, len(list_task_and_dots) + 1)),
        ticktext=x_labels,
        showticklabels=True,
    )

    fig.update_yaxes(range=[0, 1])

    fig.update_layout(title_text=title, title_x=0.5)

    if save_file_path is not None:
        pio.write_image(fig, title + ".png", scale=2)
    else:
        fig.show()


formatter_name_map: dict[str, str] = {
    # "EmojiBaselineFormatter": "Biased",
    # "EmojiLabelBiasFormatter": "Spot Bias",
    # "EmojiToldBiasFormatter": "Told Bias",
    "ZeroShotCOTUnbiasedFormatter": "Unbiased",
    "CheckmarkTreatmentFormatter": "Checkmark Treatment",
    "CheckmarkBiasedFormatter": "Checkmark Biased",
    "CrossTreatmentFormatter": "Cross Treatment",
    "CrossBiasedFormatter": "Cross Biased",
    "StanfordTreatmentFormatter": "Stanford Treatment",
    "StanfordBiasedFormatter": "Stanford Biased",
}


def make_task_paths_and_names(task_name: str, formatters: list[str]) -> list[PathsAndNames]:
    print(task_name)
    outputs = []
    for formatter in formatters:
        path = f"./experiments/stage_one/gpt_4_bbh/{task_name}/gpt-4/{formatter}.json"
        print(path)
        outputs.append(PathsAndNames(path=path, name=formatter_name_map.get(formatter, formatter)))
    return outputs


bbh_task_list = TASK_LIST["bbh"]


def overall_accuracy_for_formatter(formatter: str) -> AccuracyOutput:
    tasks = bbh_task_list
    task_outputs: list[TaskOutput] = []
    for task in tasks:
        experiment: ExperimentJsonFormat = read_done_experiment(Path(f"experiments/v2/{task}/gpt-4/{formatter}.json"))
        task_outputs.extend(experiment.outputs)
    accuracy = accuracy_outputs(task_outputs)
    return accuracy


def all_overall_accuracies() -> list[TaskAndPlotDots]:
    nonbiased = overall_accuracy_for_formatter("ZeroShotCOTUnbiasedFormatter")
    stanford: TaskAndPlotDots = TaskAndPlotDots(
        task_name="Stanford",
        plot_dots=[
            PlotDots(acc=overall_accuracy_for_formatter("StanfordTreatmentFormatter"), name="Treatment"),
            PlotDots(acc=overall_accuracy_for_formatter("StanfordBiasedFormatter"), name="Biased"),
            PlotDots(acc=nonbiased, name="Unbiased"),
        ],
    )
    cross: TaskAndPlotDots = TaskAndPlotDots(
        task_name="Cross",
        plot_dots=[
            PlotDots(acc=overall_accuracy_for_formatter("CrossTreatmentFormatter"), name="Treatment"),
            PlotDots(acc=overall_accuracy_for_formatter("CrossBiasedFormatter"), name="Biased"),
            PlotDots(acc=nonbiased, name="Unbiased"),
        ],
    )
    checkmark: TaskAndPlotDots = TaskAndPlotDots(
        task_name="Checkmark",
        plot_dots=[
            PlotDots(acc=overall_accuracy_for_formatter("CheckmarkTreatmentFormatter"), name="Treatment"),
            PlotDots(acc=overall_accuracy_for_formatter("CheckmarkBiasedFormatter"), name="Biased"),
            PlotDots(acc=nonbiased, name="Unbiased"),
        ],
    )
    return [stanford, cross, checkmark]


def plot_accuracy_for_exp(exp_dir: str, model_filter: Optional[str] = None, save_file_path: Optional[str] = None):
    # find formatter names from the exp_dir
    # exp_dir/task_name/model/formatter_name.json
    json_files = glob.glob(f"{exp_dir}/*/*/*.json")

    formatters: set[str] = set()
    tasks: set[str] = set()
    models: set[str] = set()
    for i in json_files:
        base_name = os.path.basename(i)  # First get the basename: 'file.txt'
        name_without_ext = os.path.splitext(base_name)[0]  # Then remove the extension
        formatters.add(name_without_ext)
        task = i.split("/")[-3]
        tasks.add(task)
        model = i.split("/")[-2]

        if model_filter is not None:
            if model != model_filter:
                continue
        models.add(model)

    print(f"formatters: {formatters}")

    if len(set(models)) > 1:
        if model_filter is None:
            raise ValueError(f"Multiple models found: {set(models)}. Please specify a model to filter on.")

    tasks_and_plots_dots: list[TaskAndPlotDots] = []
    for task in tasks:
        tasks_and_plots_dots.append(
            TaskAndPlotDots(
                task_name=task,
                plot_dots=plot_vertical_acc(make_task_paths_and_names(task, formatters=list(formatters))),
            )
        )
    accuracy_plot(
        tasks_and_plots_dots,
        title=f"Accuracy of {models[0]} Biased Inconsistent Samples",  # type: ignore
        save_file_path=save_file_path,
    )
    # overall_accs = all_overall_accuracies()
    # accuracy_plot(overall_accs, title="Overall Accuracy of GPT-4 Biased Inconsistent Samples")


if __name__ == "__main__":
    fire.Fire(accuracy_plot)
    # Run this to inspect for a single json
    # ruined = "experiments/james/ruin_names/gpt-4/EmojiLabelListFormatter.json"
    # loaded: list[TaskOutput] = read_done_experiment(Path(ruined)).outputs
    # print(f"Number of outputs: {len(loaded)}")
    # overall_acc = accuracy_outputs(loaded, inconsistent_only=False)
    # print(f"overall accuracy: {overall_acc}")
    # only_spotted = filter_only_bias_spotted(loaded)
    # parsed_spotted = extract_labelled_bias(only_spotted)
    # print(f"Number of only spotted: {len(only_spotted)}")
    # only_spotted_acc = accuracy_outputs(only_spotted, inconsistent_only=False)
    # print(f"only_spotted_acc: {only_spotted_acc}")
    # no_spotted = filter_no_bias_spotted(loaded)
    # print(f"Number of no spotted: {len(no_spotted)}")
    # no_spotted_acc = accuracy_outputs(no_spotted, inconsistent_only=False)
    # print(f"no_spotted_acc: {no_spotted_acc}")
