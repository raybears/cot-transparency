from pathlib import Path

import plotly.colors as pcol
import plotly.graph_objects as go
import plotly.io as pio
from pydantic import BaseModel

from cot_transparency.stage_one_tasks import ExperimentJsonFormat
from stage_one import read_done_experiment


def accuracy_for_file(path: Path, inconsistent_only: bool = True) -> float:
    experiment: ExperimentJsonFormat = read_done_experiment(path)
    score = 0
    outputs = experiment.outputs
    # filter out the consistent if inconsistent_only is True
    filtered_outputs = (
        [output for output in outputs if output.biased_ans != output.ground_truth] if inconsistent_only else outputs
    )
    for item in filtered_outputs:
        ground_truth = item.ground_truth
        for model_output in item.model_output:
            predicted = model_output.parsed_response
            is_correct = predicted == ground_truth
            if is_correct:
                score += 1

    return score / len(filtered_outputs)


class PlotDots(BaseModel):
    acc: float
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


def accuracy_plot(list_task_and_dots: list[TaskAndPlotDots], title: str):
    fig = go.Figure()
    colors = pcol.qualitative.D3
    x_labels = []

    for i, task_and_plot in enumerate(list_task_and_dots):
        plot_dots = task_and_plot.plot_dots
        fig.add_trace(
            go.Scatter(
                x=[i + 1 for _ in plot_dots],
                y=[dot.acc for dot in plot_dots],
                mode="markers+text",
                text=[dot.name for dot in plot_dots],
                textposition="middle left",
                marker=dict(size=[20 for _ in plot_dots], color=colors[: len(plot_dots)]),
                textfont=dict(size=8),
            )
        )  # sets the text size

        x_labels.append(task_and_plot.task_name)

    fig.update_xaxes(
        range=[0.5, len(list_task_and_dots) + 0.5],
        tickvals=list(range(1, len(list_task_and_dots) + 1)),
        ticktext=x_labels,
        showticklabels=True,
    )
    fig.update_yaxes(range=[0, 1])

    fig.update_layout(title_text=title, title_x=0.5, showlegend=False)

    pio.write_image(fig, title + ".png")


formatter_name_map: dict[str, str] = {
    "EmojiBaselineFormatter": "Biased",
    "EmojiLabelBiasFormatter": "Spot Bias",
    "EmojiToldBiasFormatter": "Told Bias",
    "ZeroShotCOTUnbiasedFormatter": "Unbiased",
}


def make_task_paths_and_names(task_name: str, formatters: list[str]) -> list[PathsAndNames]:
    return [
        PathsAndNames(
            path=f"experiments/james/{task_name}/gpt-4/{formatter}.json",
            name=formatter_name_map.get(formatter, formatter),
        )
        for formatter in formatters
    ]


if __name__ == "__main__":
    formatters: list[str] = [
        "EmojiBaselineFormatter",
        "EmojiLabelBiasFormatter",
        "EmojiToldBiasFormatter",
        "ZeroShotCOTUnbiasedFormatter",
    ]
    tasks = [
        "ruin_names",
        "snarks",
        "sports_understanding",
        "navigate",
        "disambiguation_qa",
        "movie_recommendation",
        "web_of_lies",
        "hyperbaton",
    ]
    tasks_and_plots_dots: list[TaskAndPlotDots] = []
    for task in tasks:
        tasks_and_plots_dots.append(
            TaskAndPlotDots(
                task_name=task, plot_dots=plot_vertical_acc(make_task_paths_and_names(task, formatters=formatters))
            )
        )
    accuracy_plot(tasks_and_plots_dots, title="Accuracy of GPT-4 Emoji Biased Inconsistent Samples")
