import glob
import os
from pathlib import Path

import math
from typing import Optional, Sequence
import fire
import plotly.colors as pcol
import plotly.graph_objects as go
import plotly.io as pio
from pydantic import BaseModel
from slist import Slist

from cot_transparency.data_models.models import TaskOutput, ExperimentJsonFormat
from cot_transparency.formatters.verbalize.biased_few_shots import parse_out_bias_explanation, BiasAndExplanation
from cot_transparency.tasks import read_done_experiment
from stage_one import TASK_LIST


class AccuracyOutput(BaseModel):
    accuracy: float
    error_bars: float
    samples: int


def compute_error_bars(num_trials: int, num_successes: int, confidence_level: float = 1.96) -> float:
    p = num_successes / num_trials
    se = math.sqrt((p * (1 - p)) / num_trials)
    return confidence_level * se


def inconsistent_only_outputs(outputs: list[TaskOutput]) -> list[TaskOutput]:
    return [output for output in outputs if output.task_spec.biased_ans != output.task_spec.ground_truth]


def accuracy_for_file(path: Path, inconsistent_only: bool) -> AccuracyOutput:
    experiment: ExperimentJsonFormat = read_done_experiment(path)
    assert experiment.outputs, f"Experiment {path} has no outputs"
    maybe_filtered = inconsistent_only_outputs(experiment.outputs) if inconsistent_only else experiment.outputs
    assert maybe_filtered, f"Experiment {path} has no inconsistent only outputs"
    return accuracy_outputs(maybe_filtered)


def accuracy_outputs(outputs: list[TaskOutput]) -> AccuracyOutput:
    transformed = (
        Slist(outputs)
        .map(
            lambda x: AccuracyInput(ground_truth=x.task_spec.ground_truth, predicted=x.first_parsed_response)
            if x.first_parsed_response
            else None
        )
        .flatten_option()
    )
    return accuracy_outputs_from_inputs(transformed)


class AccuracyInput(BaseModel):
    ground_truth: str
    predicted: str


def accuracy_outputs_from_inputs(inputs: Sequence[AccuracyInput]) -> AccuracyOutput:
    if len(inputs) == 0:
        raise ValueError("No outputs to score")
    correct = Slist(inputs).map(lambda x: 1 if x.ground_truth == x.predicted else 0)
    acc = correct.average()
    assert acc is not None
    # Compute error bars for accuracy
    error_bars = compute_error_bars(num_trials=len(inputs), num_successes=correct.sum())
    return AccuracyOutput(accuracy=acc, error_bars=error_bars, samples=len(inputs))


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


def plot_vertical_acc(paths: list[PathsAndNames], inconsistent_only: bool) -> list[PlotDots]:
    out: list[PlotDots] = []
    for path in paths:
        out.append(
            PlotDots(acc=accuracy_for_file(Path(path.path), inconsistent_only=inconsistent_only), name=path.name)
        )
    return out


class ColorAndShape(BaseModel):
    color: str
    shape: str


class PlotlyShapeColorManager:
    def __init__(self):
        self.colors = pcol.qualitative.D3
        self.symbols = ["circle", "square", "diamond", "cross", "x", "triangle-up", "pentagon"]
        self.label_to_color_and_shape: dict[str, ColorAndShape] = {}

    def get_color_and_shape(self, label: str) -> ColorAndShape:
        if label not in self.label_to_color_and_shape:
            color = self.colors[len(self.label_to_color_and_shape) % len(self.colors)]
            shape = self.symbols[len(self.label_to_color_and_shape) % len(self.symbols)]
            self.label_to_color_and_shape[label] = ColorAndShape(color=color, shape=shape)
        return self.label_to_color_and_shape[label]


def accuracy_plot(
    list_task_and_dots: list[TaskAndPlotDots], title: str, subtitle: str = "", save_file_path: Optional[str] = None
):
    fig = go.Figure()

    shape_color_manager = PlotlyShapeColorManager()

    x_labels: list[str] = []
    added_labels: set[str] = set()  # to remember the labels we have already added

    for i, task_and_plot in enumerate(list_task_and_dots):
        plot_dots = task_and_plot.plot_dots
        x_labels.append(task_and_plot.task_name)

        for j, dot in enumerate(plot_dots):
            color_shape = shape_color_manager.get_color_and_shape(dot.name)
            fig.add_trace(
                go.Scatter(
                    x=[i + 1],
                    y=[dot.acc.accuracy],
                    mode="markers",
                    marker=dict(
                        size=[15],
                        color=color_shape.color,
                        symbol=color_shape.shape,
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

    # Adding the subtitle
    if subtitle:
        fig.add_annotation(
            x=1,  # x position - 1 means far right in paper coordinates
            y=0,  # y position - 0 means at the bottom in paper coordinates
            xref="paper",  # use paper coordinates for x
            yref="paper",  # use paper coordinates for y
            xanchor="right",  # align the text to the right of the given x position
            yanchor="top",  # align the text to the top of the given y position
            text=subtitle,  # the text itself
            showarrow=False,  # don't show an arrow pointing from the text
            font=dict(size=12, color="#555"),  # font size  # font color, change as desired
        )

    fig.update_yaxes(range=[0, 1])

    fig.update_layout(title_text=title, title_x=0.5)

    if save_file_path is not None:
        pio.write_image(fig, save_file_path + ".png", scale=2)
    else:
        fig.show()


class DottedLine(BaseModel):
    name: str
    value: float
    color: str


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


def make_task_paths_and_names(task_name: str, formatters: list[str], model: str, exp_dir: str) -> list[PathsAndNames]:
    outputs = []
    for formatter in formatters:
        path = f"./{exp_dir}/{task_name}/{model}/{formatter}.json"
        outputs.append(PathsAndNames(path=path, name=formatter_name_map.get(formatter, formatter)))
    return outputs


bbh_task_list = TASK_LIST["bbh"]


def plot_accuracy_for_exp(
    exp_dir: str,
    model_filter: Optional[str] = None,
    save_file_path: Optional[str] = None,
    formatters: Sequence[str] = [],
    inconsistent_only: bool = True,
):
    # find formatter names from the exp_dir
    # exp_dir/task_name/model/formatter_name.json
    json_files = glob.glob(f"{exp_dir}/*/*/*.json")

    should_filter_formatter: bool = len(formatters) > 0
    formatters_found: set[str] = set()
    tasks: set[str] = set()
    models: set[str] = set()
    for i in json_files:
        base_name = os.path.basename(i)  # First get the basename: 'file.txt'
        name_without_ext = os.path.splitext(base_name)[0]  # Then remove the extension
        if should_filter_formatter and name_without_ext in formatters:
            formatters_found.add(name_without_ext)
        elif not should_filter_formatter:
            formatters_found.add(name_without_ext)
        task = i.split("/")[-3]
        tasks.add(task)
        model = i.split("/")[-2]

        if model_filter is not None:
            if model != model_filter:
                continue
        models.add(model)

    print(f"formatters: {formatters_found}")

    if len(set(models)) > 1:
        if model_filter is None:
            raise ValueError(f"Multiple models found: {set(models)}. Please specify a model to filter on.")
    model: str = list(models)[0]

    tasks_and_plots_dots: list[TaskAndPlotDots] = []
    for task in tasks:
        tasks_and_plots_dots.append(
            TaskAndPlotDots(
                task_name=task,
                plot_dots=plot_vertical_acc(
                    make_task_paths_and_names(
                        task_name=task, formatters=list(formatters_found), model=model, exp_dir=exp_dir
                    ),
                    inconsistent_only=inconsistent_only,
                ),
            )
        )
    title_subset = "Biased Inconsistent Samples" if should_filter_formatter else "All Samples"
    accuracy_plot(
        tasks_and_plots_dots,
        title=f"Accuracy of {model} {title_subset}",  # type: ignore
        save_file_path=save_file_path,
    )


if __name__ == "__main__":
    fire.Fire(plot_accuracy_for_exp)
