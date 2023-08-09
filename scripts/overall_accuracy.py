from pathlib import Path
from typing import Optional

import plotly.io as pio
import plotly.graph_objects as go

from cot_transparency.data_models.models import TaskOutput, ExperimentJsonFormat
from scripts.multi_accuracy import (
    TaskAndPlotDots,
    PlotDots,
    AccuracyOutput,
    bbh_task_list,
    accuracy_outputs,
    accuracy_plot,
)
from cot_transparency.tasks import read_done_experiment


def overall_accuracy_for_formatter(formatter: str, exp_dir: str, model: str) -> AccuracyOutput:
    tasks = bbh_task_list
    task_outputs: list[TaskOutput] = []
    for task in tasks:
        path = Path(f"{exp_dir}/{task}/{model}/{formatter}.json")
        experiment: ExperimentJsonFormat = read_done_experiment(path)
        assert experiment.outputs, f"Experiment {path} has no outputs"
        task_outputs.extend(experiment.outputs)
    accuracy = accuracy_outputs(task_outputs)
    return accuracy


def all_overall_accuracies(exp_dir: str, model: str) -> list[TaskAndPlotDots]:
    nonbiased = overall_accuracy_for_formatter("ZeroShotCOTUnbiasedFormatter", exp_dir=exp_dir, model=model)
    stanford: TaskAndPlotDots = TaskAndPlotDots(
        task_name="All tasks",
        plot_dots=[
            PlotDots(
                acc=overall_accuracy_for_formatter("DeceptiveAssistantBiasedFormatter", exp_dir=exp_dir, model=model),
                name="Tell model to be deceptive",
            ),
            PlotDots(
                acc=overall_accuracy_for_formatter("MoreRewardBiasedFormatter", exp_dir=exp_dir, model=model),
                name="More reward for an option",
            ),
            PlotDots(
                acc=overall_accuracy_for_formatter("UserBiasedWrongCotFormatter", exp_dir=exp_dir, model=model),
                name="User says wrong reasoning",
            ),
            PlotDots(
                acc=overall_accuracy_for_formatter("WrongFewShotBiasedFormatter", exp_dir=exp_dir, model=model),
                name="Wrong label in the few shot",
            ),
            PlotDots(acc=nonbiased, name="Normal prompt without bias"),
        ],
    )
    return [stanford]


def decrease_in_accuracy_plot(
    base_accuracy: float,
    plot_dots: list[PlotDots],
    title: str,
    subtitle: str = "",
    save_file_path: Optional[str] = None,
):
    decrease = [base_accuracy - dot.acc.accuracy for dot in plot_dots]
    fig = go.Figure(
        data=[
            go.Bar(
                name="Decrease in Accuracy",
                x=[dot.name for dot in plot_dots],
                y=decrease,
                text=["            {:.2f}".format(dec) for dec in decrease],  # offset to the right
                textposition="outside",  # will always place text above the bars
                textfont=dict(size=22, color="#000000"),  # increase text size and set color to black
                error_y=dict(type="data", array=[dot.acc.error_bars for dot in plot_dots], visible=True),
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Bias method",
        yaxis_title="Decrease in Accuracy",
        barmode="group",
        yaxis=dict(
            range=[0, max(decrease) + max([dot.acc.error_bars for dot in plot_dots])]
        ),  # adjust y range to accommodate text above bars
    )

    # Adding the subtitle
    if subtitle:
        fig.add_annotation(
            x=1,  # in paper coordinates
            y=0,  # in paper coordinates
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="top",
            text=subtitle,
            showarrow=False,
            font=dict(size=12, color="#555"),
        )

    if save_file_path is not None:
        pio.write_image(fig, save_file_path + ".png", scale=2)
    else:
        fig.show()


def decrease_in_accuracy(exp_dir: str, model: str):
    nonbiased: AccuracyOutput = overall_accuracy_for_formatter(
        "ZeroShotCOTUnbiasedFormatter", exp_dir=exp_dir, model=model
    )
    plot_dots = [
        PlotDots(
            acc=overall_accuracy_for_formatter("DeceptiveAssistantBiasedFormatter", exp_dir=exp_dir, model=model),
            name="Tell model to be deceptive",
        ),
        PlotDots(
            acc=overall_accuracy_for_formatter("WrongFewShotBiasedFormatter", exp_dir=exp_dir, model=model),
            name="Wrong label in the few shot",
        ),
        PlotDots(
            acc=overall_accuracy_for_formatter("UserBiasedWrongCotFormatter", exp_dir=exp_dir, model=model),
            name="User says wrong reasoning",
        ),
        PlotDots(
            acc=overall_accuracy_for_formatter("MoreRewardBiasedFormatter", exp_dir=exp_dir, model=model),
            name="More reward for an option",
        ),
    ]
    decrease_in_accuracy_plot(
        base_accuracy=nonbiased.accuracy,
        plot_dots=plot_dots,
        title="How much does each bias decrease GPT-4's performance on BBH tasks?",
        subtitle=f"n={nonbiased.samples}",
        save_file_path=None,
    )


if __name__ == "__main__":
    decrease_in_accuracy(exp_dir="experiments/biased_wrong", model="gpt-4")
    overall_accs = all_overall_accuracies(exp_dir="experiments/biased_wrong", model="gpt-4")
    number_samples = overall_accs[0].plot_dots[0].acc.samples
    accuracy_plot(overall_accs, title="Accuracy of GPT-4 on BBH Biased Samples", subtitle=f"n={number_samples}")
