from typing import Type

from slist import Slist

from cot_transparency.data_models.io import read_whole_exp_dir
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.more_biases.more_reward import (
    MoreRewardBiasedFormatter,
)
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotBiasedFormatter,
)
from cot_transparency.formatters.verbalize.formatters import StanfordBiasedFormatter
from scripts.intervention_investigation import (
    bar_plot,
    filter_inconsistent_only,
    plot_for_intervention,
)
from scripts.matching_user_answer import matching_user_answer_plot_info
from scripts.multi_accuracy import PlotInfo
from stage_one import TASK_LIST


def accuracy_diff_math(
    model: str,
):
    all_read: Slist[TaskOutput] = filter_inconsistent_only(
        read_whole_exp_dir(exp_dir="experiments/math_scale")
    )
    unbiased_formatter: Type[StageOneFormatter] = ZeroShotCOTUnbiasedFormatter
    biased_formatters = [
        MoreRewardBiasedFormatter,
        StanfordBiasedFormatter,
        WrongFewShotBiasedFormatter,
        ZeroShotCOTSycophancyFormatter,
    ]

    tasks: Slist[str] = Slist(TASK_LIST["john_math"])

    # Make chart of the diff in accuracy betwwen the unbiased and the biased, same interventio
    unbiased_plot_dots: dict[str, PlotInfo] = (
        tasks.map(
            lambda task: plot_for_intervention(
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
        lambda task: plot_for_intervention(
            intervention=None,
            all_tasks=all_read,
            for_formatters=biased_formatters,
            model=model,
            include_tasks=[task],
            name_override=task,
        )
    )

    joined_plot_dots: list[PlotInfo] = biased_plot_dots.map(
        lambda plot: PlotInfo(
            acc=plot.acc - unbiased_plot_dots[plot.name].acc,
            name=plot.name,
        )
    )
    bar_plot(
        plot_infos=joined_plot_dots,
        title=f"{model}: Does the accuracy gap between unbiased and biased context widen with harder questions?<br>Biased with MoreReward, Stanford, WrongFewShot, Sycophancy. 140 questions per math level",
        y_axis_title="Biased context accuracy minus unbiased context accuracy",
    )

    matching_user_answer: Slist[PlotInfo] = Slist(
        matching_user_answer_plot_info(
            intervention=None,
            all_tasks=all_read,
            for_task=[task],
            for_formatters=biased_formatters,
            model=model,
            name_override=task,
        )
        for task in tasks
    )
    bar_plot(
        plot_infos=matching_user_answer,
        title=f"Does {model} follow the bias more on harder questions?<br>Biased with MoreReward, Stanford, WrongFewShot, Sycophancy.<br> 140 questions per math level. Two options per question. Bias always on wrong option.",
        y_axis_title="Answers matching bias's view (%)",
    )

    # Make chart of the diff in matching betwwen the unbiased and the biased
    unbiased_matching_answer: dict[str, PlotInfo] = (
        tasks.map(
            lambda task: matching_user_answer_plot_info(
                intervention=None,
                all_tasks=all_read,
                for_formatters=[unbiased_formatter],
                model=model,
                for_task=[task],
                name_override=task,
            )
        )
        .map(lambda plot: (plot.name, plot))
        .to_dict()
    )
    joined_matching_answer: list[PlotInfo] = matching_user_answer.map(
        lambda plot: PlotInfo(
            acc=plot.acc - unbiased_matching_answer[plot.name].acc,
            name=plot.name,
        )
    )
    bar_plot(
        plot_infos=joined_matching_answer,
        title=f"Does {model} follow the bias more on harder questions?<br>Increase in % of answers matching bias in biased context vs unbiased context<br>Biased with MoreReward, Stanford, WrongFewShot, Sycophancy.<br> 140 questions per math level. Two options per question. Bias always on wrong option.",
        y_axis_title="Adjusted answers matching bias's view (%)",
    )


if __name__ == "__main__":
    # accuracy_diff_math(model="gpt-4")
    accuracy_diff_math(model="claude-2")
