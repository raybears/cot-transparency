from typing import Type

from slist import Slist

from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.more_biases.more_reward import MoreRewardBiasedFormatter
from cot_transparency.formatters.more_biases.wrong_few_shot import WrongFewShotBiasedFormatter
from cot_transparency.formatters.verbalize.formatters import StanfordBiasedFormatter
from scripts.intervention_investigation import (
    bar_plot,
    plot_dots_for_intervention,
    read_whole_exp_dir,
    filter_inconsistent_only,
)
from scripts.multi_accuracy import PlotDots
from stage_one import TASK_LIST


def accuracy_diff_math(
    model: str,
):
    all_read: Slist[TaskOutput] = filter_inconsistent_only(read_whole_exp_dir(exp_dir="experiments/math_scale"))
    unbiased_formatter: Type[StageOneFormatter] = ZeroShotCOTUnbiasedFormatter
    biased_formatters = [
        MoreRewardBiasedFormatter,
        StanfordBiasedFormatter,
        WrongFewShotBiasedFormatter,
        ZeroShotCOTSycophancyFormatter,
    ]

    tasks: Slist[str] = Slist(TASK_LIST["john_math"])

    # Make chart of the diff in accuracy betwwen the unbiased and the biased, same interventio
    unbiased_plot_dots: dict[str, PlotDots] = (
        tasks.map(
            lambda task: plot_dots_for_intervention(
                intervention=None,
                all_tasks=all_read,
                for_formatters=[unbiased_formatter],
                model=model,
                include_tasks=[task],
                name_override=task,
            )
        )
        .map(lambda plot: (plot.name, plot))
        .to_dict()
    )
    biased_plot_dots = tasks.map(
        lambda task: plot_dots_for_intervention(
            intervention=None,
            all_tasks=all_read,
            for_formatters=biased_formatters,
            model=model,
            include_tasks=[task],
            name_override=task,
        )
    )

    joined_plot_dots: list[PlotDots] = biased_plot_dots.map(
        lambda plot: PlotDots(
            acc=plot.acc - unbiased_plot_dots[plot.name].acc,
            name=plot.name,
        )
    )
    bar_plot(
        plot_dots=joined_plot_dots,
        title="Does the accuracy gap between unbiased and biased context widen with more few shots?",
        y_axis_title="Biased context accuracy minus unbiased context accuracy",
    )


if __name__ == "__main__":
    accuracy_diff_math(model="gpt-4")
