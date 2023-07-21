from pathlib import Path

import plotly.colors as pcol
import plotly.graph_objects as go
import plotly.io as pio
from pydantic import BaseModel
from cot_transparency.data_models.models_v2 import TaskOutput

from cot_transparency.formatters.emoji.biased_few_shots import parse_out_bias_explanation, BiasAndExplanation
from cot_transparency.data_models.models_v2 import ExperimentJsonFormat
from stage_one import read_done_experiment


def accuracy_for_file(path: Path, inconsistent_only: bool = True) -> float:
    experiment: ExperimentJsonFormat = read_done_experiment(path)
    return accuracy_outputs(experiment.outputs, inconsistent_only=inconsistent_only)


def accuracy_outputs(outputs: list[TaskOutput], inconsistent_only: bool = True) -> float:
    score = 0
    # filter out the consistent if inconsistent_only is True
    filtered_outputs = (
        [output for output in outputs if output.task_spec.biased_ans != output.task_spec.ground_truth]
        if inconsistent_only
        else outputs
    )
    for item in filtered_outputs:
        ground_truth = item.task_spec.ground_truth
        for model_output in item.model_output:
            predicted = model_output.parsed_response
            is_correct = predicted == ground_truth
            if is_correct:
                score += 1

    return score / len(filtered_outputs)


def spotted_bias(raw_response: str) -> bool:
    return "NO_BIAS" not in raw_response


def filter_only_bias_spotted(outputs: list[TaskOutput]) -> list[TaskOutput]:
    new_list: list[TaskOutput] = []
    for output in outputs:
        new_output = output.copy()
        new_output.model_output = [
            model_output for model_output in output.model_output if spotted_bias(model_output.raw_response)
        ]
        new_list.append(new_output)
    return [output for output in new_list if output.model_output]


def extract_labelled_bias(outputs: list[TaskOutput]) -> list[BiasAndExplanation]:
    new_list: list[BiasAndExplanation] = []
    for output in outputs:
        for model_output in output.model_output:
            bias_and_explanation = parse_out_bias_explanation(model_output.raw_response)
            new_list.append(bias_and_explanation)
    return new_list


def filter_no_bias_spotted(outputs: list[TaskOutput]) -> list[TaskOutput]:
    new_list: list[TaskOutput] = []
    for output in outputs:
        new_output = output.copy()
        new_output.model_output = [
            model_output for model_output in output.model_output if not spotted_bias(model_output.raw_response)
        ]
        new_list.append(new_output)
    return [output for output in new_list if output.model_output]


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
    x_labels: list[str] = []
    added_labels: set[str] = set()  # to remember the labels we have already added

    for i, task_and_plot in enumerate(list_task_and_dots):
        plot_dots = task_and_plot.plot_dots
        x_labels.append(task_and_plot.task_name)

        for j, dot in enumerate(plot_dots):
            fig.add_trace(
                go.Scatter(
                    x=[i + 1],
                    y=[dot.acc],
                    mode="markers",
                    marker=dict(size=[15], color=colors[j % len(colors)]),
                    name=dot.name,  # specify the name that will appear in legend
                    showlegend=dot.name not in added_labels,  # if dot name is in added_labels don't show it in legend
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

    pio.write_image(fig, title + ".png", scale=2)


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


def main():
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


if __name__ == "__main__":
    main()
    # Run this to inspect for a single json
    ruined = "experiments/james/ruin_names/gpt-4/EmojiLabelListFormatter.json"
    loaded: list[TaskOutput] = read_done_experiment(Path(ruined)).outputs
    print(f"Number of outputs: {len(loaded)}")
    overall_acc = accuracy_outputs(loaded, inconsistent_only=False)
    print(f"overall accuracy: {overall_acc}")
    only_spotted = filter_only_bias_spotted(loaded)
    parsed_spotted = extract_labelled_bias(only_spotted)
    print(f"Number of only spotted: {len(only_spotted)}")
    only_spotted_acc = accuracy_outputs(only_spotted, inconsistent_only=False)
    print(f"only_spotted_acc: {only_spotted_acc}")
    no_spotted = filter_no_bias_spotted(loaded)
    print(f"Number of no spotted: {len(no_spotted)}")
    no_spotted_acc = accuracy_outputs(no_spotted, inconsistent_only=False)
    print(f"no_spotted_acc: {no_spotted_acc}")
