import glob
from pathlib import Path
from typing import Optional, Sequence, Type

from plotly import graph_objects as go, io as pio
from pydantic import BaseModel
from slist import Slist

from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.interventions.consistency import (
    NaiveFewShot10,
    BiasedConsistency10,
)
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.formatters.more_biases.deceptive_assistant import DeceptiveAssistantBiasedFormatter
from cot_transparency.formatters.more_biases.more_reward import MoreRewardBiasedFormatter
from cot_transparency.formatters.more_biases.wrong_few_shot import WrongFewShotBiasedFormatter
from cot_transparency.formatters.verbalize.formatters import (
    StanfordBiasedFormatter,
)
from cot_transparency.tasks import read_done_experiment
from scripts.multi_accuracy import PlotDots, accuracy_outputs, TaskAndPlotDots
from scripts.simple_formatter_names import INTERVENTION_TO_SIMPLE_NAME


def read_whole_exp_dir(exp_dir: str) -> Slist[TaskOutput]:
    # find formatter names from the exp_dir
    # exp_dir/task_name/model/formatter_name.json
    json_files = glob.glob(f"{exp_dir}/*/*/*.json")
    read: Slist[TaskOutput] = (
        Slist(json_files).map(Path).map(read_done_experiment).map(lambda exp: exp.outputs).flatten_list()
    )
    return read


def plot_dots_for_intervention(
    intervention: Optional[Type[Intervention]],
    all_tasks: Slist[TaskOutput],
    for_formatters: Sequence[Type[StageOneFormatter]],
    model: str,
    name_override: Optional[str] = None,
    inconsistent_only: bool = True,
) -> PlotDots:
    intervention_name: str | None = intervention.name() if intervention else None
    formatters_names: set[str] = {f.name() for f in for_formatters}
    filtered: Slist[TaskOutput] = (
        all_tasks.filter(lambda task: intervention_name == task.task_spec.intervention_name)
        .filter(lambda task: task.task_spec.formatter_name in formatters_names)
        .filter(lambda task: (task.task_spec.biased_ans != task.task_spec.ground_truth) if inconsistent_only else True)
        .filter(lambda task: task.task_spec.model_config.model == model)
    )
    assert filtered, f"Intervention {intervention_name} has no tasks in {for_formatters}"
    accuray = accuracy_outputs(filtered)
    retrieved_simple_name: str | None = INTERVENTION_TO_SIMPLE_NAME.get(intervention, None)
    name: str = name_override or retrieved_simple_name or intervention_name or "No intervention, biased context"
    return PlotDots(acc=accuray, name=name)


class DottedLine(BaseModel):
    name: str
    value: float
    color: str


def bar_plot(
    plot_dots: list[PlotDots],
    title: str,
    subtitle: str = "",
    save_file_path: Optional[str] = None,
    dotted_line: Optional[DottedLine] = None,
):
    fig = go.Figure()

    for dot in plot_dots:
        fig.add_trace(
            go.Bar(
                name=dot.name,
                x=[dot.name],
                y=[dot.acc.accuracy],
                error_y=dict(type="data", array=[dot.acc.error_bars], visible=True),
                text=[f"          {dot.acc.accuracy:.2f}"],
            )
        )

    fig.update_layout(
        barmode="group",
        title_text=title,
        title_x=0.5,
    )

    if dotted_line is not None:
        fig.add_trace(
            go.Scatter(
                x=[dot.name for dot in plot_dots],  # changes here
                y=[dotted_line.value] * len(plot_dots),
                mode="lines",
                line=dict(
                    color=dotted_line.color,
                    width=4,
                    dash="dashdot",
                ),
                name=dotted_line.name,
                showlegend=True,
            ),
        )

    if subtitle:
        fig.add_annotation(
            x=1,
            y=0,
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


if __name__ == "__main__":
    model = "gpt-4"
    all_read = read_whole_exp_dir(exp_dir="experiments/interventions")
    # what interventions to plot
    interventions: Sequence[Type[Intervention] | None] = [
        None,
        # PairedConsistency10,
        BiasedConsistency10,
        NaiveFewShot10,
        # NaiveFewShotLabelOnly10,
        # NaiveFewShotLabelOnly30,
        # SycoConsistencyLabelOnly30,
        # BiasedConsistencyLabelOnly30,
    ]
    # what formatters to include
    biased_formatters = [
        WrongFewShotBiasedFormatter,
        StanfordBiasedFormatter,
        MoreRewardBiasedFormatter,
        ZeroShotCOTSycophancyFormatter,
        DeceptiveAssistantBiasedFormatter,
        # CheckmarkBiasedFormatter,
        # CrossBiasedFormatter,
    ]
    # unbiased acc
    unbiased_plot: PlotDots = plot_dots_for_intervention(
        None, all_read, for_formatters=[ZeroShotCOTUnbiasedFormatter], name_override="Unbiased context", model=model
    )

    plot_dots: list[PlotDots] = [
        plot_dots_for_intervention(intervention, all_read, for_formatters=biased_formatters, model=model)
        for intervention in interventions
    ]
    one_chart = TaskAndPlotDots(task_name="MMLU and aqua stuff", plot_dots=plot_dots)
    # accuracy_plot([one_chart], title="Accuracy of Interventions")
    dotted = DottedLine(name="Zero shot unbiased context performance", value=unbiased_plot.acc.accuracy, color="red")
    bar_plot(
        plot_dots=plot_dots,
        title=f"{model} biased context accuracy using different 10 shot techniques. Dataset: Aqua and mmlu",
        dotted_line=dotted,
    )
