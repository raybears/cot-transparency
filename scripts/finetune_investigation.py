from typing import Sequence, Type, Optional, Mapping

from slist import Slist

from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from scripts.intervention_investigation import (
    read_whole_exp_dir,
    filter_inconsistent_only,
    plot_dots_for_intervention,
    DottedLine,
    bar_plot,
)
from scripts.matching_user_answer import matching_user_answer_plot_dots
from scripts.multi_accuracy import PlotDots


def make_finetune_graph(
    biased_formatters: Sequence[Type[StageOneFormatter]],
    finetuned_models: Sequence[str],
    unbiased_model: str,
    unbiased_formatter: Type[StageOneFormatter],
    exp_dir: str,
    accuracy_plot_name: Optional[str] = None,
    percent_matching_plot_name: Optional[str] = None,
    inconsistent_only: bool = True,
    tasks: Sequence[str] = [],
    model_name_override: Mapping[str | None, str] = {},
):
    all_read: Slist[TaskOutput] = read_whole_exp_dir(exp_dir=exp_dir)
    all_read = (filter_inconsistent_only(all_read) if inconsistent_only else all_read).filter(
        lambda task: task.task_spec.task_name in tasks if tasks else True
    )

    # unbiased acc
    unbiased_plot: PlotDots = plot_dots_for_intervention(
        None,
        all_read,
        for_formatters=[unbiased_formatter],
        name_override=f"Unbiased context, {unbiased_model}",
        model=unbiased_model,
    )

    plot_dots: list[PlotDots] = [
        plot_dots_for_intervention(
            None,
            all_read,
            for_formatters=biased_formatters,
            model=model,
            # name_override=intervention_name_override.get(intervention, None),
        )
        for model in finetuned_models
    ]
    dotted = DottedLine(name="Zero shot unbiased context performance", value=unbiased_plot.acc.accuracy, color="red")

    bar_plot(
        plot_dots=plot_dots,
        title=accuracy_plot_name or "Accuracy",
        dotted_line=dotted,
        y_axis_title="Accuracy",
    )
    # accuracy_diff_intervention(interventions=interventions, unbiased_formatter=unbiased_formatter)
    matching_user_answer: list[PlotDots] = [
        matching_user_answer_plot_dots(
            intervention=None,
            all_tasks=all_read,
            for_formatters=biased_formatters,
            model=model,
            # name_override=intervention_name_override.get(intervention, None),
        )
        for model in finetuned_models
    ]
    unbiased_matching_baseline = matching_user_answer_plot_dots(
        intervention=None, all_tasks=all_read, for_formatters=[unbiased_formatter], model=unbiased_model
    )
    dotted_line = DottedLine(
        name="Zeroshot Unbiased context", value=unbiased_matching_baseline.acc.accuracy, color="red"
    )
    dataset_str = Slist(tasks).mk_string(", ")
    bar_plot(
        plot_dots=matching_user_answer,
        title=percent_matching_plot_name or f"How often does each model choose the user's view Dataset: {dataset_str}",
        y_axis_title="Answers matching bias's view (%)",
        dotted_line=dotted_line,
    )
