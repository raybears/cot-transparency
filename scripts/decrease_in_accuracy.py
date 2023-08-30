from typing import Optional, Sequence, Type

from plotly import graph_objects as go, io as pio
from slist import Slist

from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.formatters.more_biases.more_reward import MoreRewardBiasedFormatter
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedFormatter,
)
from cot_transparency.formatters.verbalize.formatters import StanfordBiasedFormatter
from scripts.intervention_investigation import (
    read_whole_exp_dir,
    DottedLine,
    bar_plot,
    filter_inconsistent_only,
    plot_dots_for_intervention,
)
from scripts.matching_user_answer import matching_user_answer_plot_dots, random_chance_matching_answer_plot_dots
from scripts.multi_accuracy import PlotDots, AccuracyOutput, accuracy_outputs


def decrease_in_accuracy_plot(
    base_accuracy_plot: AccuracyOutput,
    plot_dots: list[PlotDots],
    title: str,
    subtitle: str = "",
    save_file_path: Optional[str] = None,
    max_y: Optional[float] = None,
):
    decrease: list[AccuracyOutput] = [base_accuracy_plot - dot.acc for dot in plot_dots]
    decrease_acc: list[float] = [dot.accuracy for dot in decrease]
    fig = go.Figure(
        data=[
            go.Bar(
                name="Decrease in Accuracy",
                x=[dot.name for dot in plot_dots],
                y=decrease_acc,
                text=["            {:.2f}".format(dec) for dec in decrease_acc],  # offset to the right
                textposition="outside",  # will always place text above the bars
                textfont=dict(size=22, color="#000000"),  # increase text size and set color to black
                error_y=dict(type="data", array=[dot.error_bars for dot in decrease], visible=True),
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Bias method",
        yaxis_title="Decrease in Accuracy",
        barmode="group",
        yaxis=dict(
            range=[0, max(decrease_acc) + max([dot.acc.error_bars for dot in plot_dots])]
        ),  # adjust y range to accommodate text above bars
    )
    if max_y is not None:
        fig.update_layout(yaxis=dict(range=[0, max_y]))

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


def plot_dot_diff(
    intervention: Optional[Type[Intervention]],
    all_tasks: Slist[TaskOutput],
    for_formatters: Sequence[Type[StageOneFormatter]],
    model: str,
    name: str,
    unbiased_formatter: Type[StageOneFormatter] = ZeroShotCOTUnbiasedFormatter,
    include_tasks: Sequence[str] = [],
) -> PlotDots:
    intervention_name: str | None = intervention.name() if intervention else None
    nonbiased: Slist[TaskOutput] = (
        all_tasks.filter(lambda task: intervention_name == task.task_spec.intervention_name)
        .filter(lambda task: task.task_spec.formatter_name == unbiased_formatter.name())
        .filter(lambda task: task.task_spec.inference_config.model == model)
        .filter(lambda task: task.task_spec.task_name in include_tasks if include_tasks else True)
    )
    assert len(nonbiased) > 0, f"Found no tasks for {name} on {model} with {unbiased_formatter.name()}"
    nonbiased_acc: AccuracyOutput = accuracy_outputs(nonbiased)
    biased: PlotDots = plot_dots_for_intervention(
        intervention=None,
        all_tasks=all_tasks,
        for_formatters=for_formatters,
        model=model,
        include_tasks=include_tasks,
    )

    return PlotDots(acc=nonbiased_acc - biased.acc, name=name)


def decrease_in_accuracy(
    exp_dir: str,
    model: str,
    tasks: Sequence[str] = [
        "aqua",
        "arc_easy",
        "arc_challenge",
        "truthful_qa",
        "logiqa",
        "mmlu",
        "openbook_qa",
        "hellaswag",
        "john_level_5",
    ],
):
    """
    python stage_one.py --exp_dir experiments/interventions --tasks '["aqua","arc_easy","arc_challenge","truthful_qa","logiqa","mmlu","openbook_qa","hellaswag","john_level_5"]' --models "['claude-2']" --formatters '["ZeroShotCOTSycophancyFormatter", "MoreRewardBiasedFormatter", "StanfordBiasedFormatter", "DeceptiveAssistantBiasedFormatter", "WrongFewShotIgnoreMistakesBiasedFormatter", "ZeroShotCOTUnbiasedFormatter"]' --example_cap 600
    """
    # tasks: list[str] = TASK_LIST["transparency"]

    all_read = filter_inconsistent_only(read_whole_exp_dir(exp_dir=exp_dir))

    decrease_plot_dots = [
        plot_dot_diff(
            intervention=None,
            all_tasks=all_read,
            for_formatters=[
                WrongFewShotIgnoreMistakesBiasedFormatter,
                MoreRewardBiasedFormatter,
                ZeroShotCOTSycophancyFormatter,
                StanfordBiasedFormatter,
            ],
            unbiased_formatter=ZeroShotCOTUnbiasedFormatter,
            model=model,
            include_tasks=[task],
            name=task,
        )
        for task in tasks
    ]
    bar_plot(
        plot_dots=decrease_plot_dots,
        title=f"How much do biases decrease {model} performance on tasks?<br>With COT completion<br>Biases always on wrong answer<br>Biases: Wrong label in the few shot, More reward for an option, I think the answer is (X)",
        subtitle=f"n={decrease_plot_dots[0].acc.samples}",
        # save_file_path=f"{model}_decrease_in_accuracy",
        max_y=1.0,
    )
    # plot_matching(
    #     all_read=all_read,
    #     model=model,
    #     task=task,
    #     unbiased_formatter=unbiased_formatter,
    # )


def decrease_in_accuracy_per_bias(exp_dir: str, model: str):
    tasks = [
        "aqua",
        "arc_easy",
        "arc_challenge",
        "truthful_qa",
        "logiqa",
        "mmlu",
        "openbook_qa",
        "hellaswag",
        "john_level_5",
    ]
    all_read = filter_inconsistent_only(read_whole_exp_dir(exp_dir=exp_dir))
    for_formatters = [
        WrongFewShotIgnoreMistakesBiasedFormatter,
        MoreRewardBiasedFormatter,
        ZeroShotCOTSycophancyFormatter,
        StanfordBiasedFormatter,
    ]
    for task in tasks:
        decrease_plot_dots = [
            plot_dot_diff(
                intervention=None,
                all_tasks=all_read,
                for_formatters=[formatter],
                unbiased_formatter=ZeroShotCOTUnbiasedFormatter,
                model=model,
                include_tasks=[task],
                name=formatter.name(),
            )
            for formatter in for_formatters
        ]

        bar_plot(
            plot_dots=decrease_plot_dots,
            title=f"How much do biases decrease {model} performance on {task}?<br>With COT completion<br>Biases always on wrong answer<br>Biases: Wrong label in the few shot, More reward for an option, I think the answer is (X)",
            subtitle=f"n={decrease_plot_dots[0].acc.samples}",
            # save_file_path=f"{model}_decrease_in_accuracy",
            max_y=1.0,
        )


def plot_matching(
    all_read: Sequence[TaskOutput],
    model: str,
    task: str,
    unbiased_formatter: Type[StageOneFormatter] = ZeroShotCOTUnbiasedFormatter,
):
    matching_user_answer: list[PlotDots] = [
        matching_user_answer_plot_dots(
            intervention=None,
            all_tasks=all_read,
            for_formatters=[WrongFewShotBiasedFormatter],
            model=model,
            for_task=[task],
            name_override="Wrong label in the few shot",
        ),
        matching_user_answer_plot_dots(
            intervention=None,
            all_tasks=all_read,
            for_formatters=[MoreRewardBiasedFormatter],
            model=model,
            for_task=[task],
            name_override="More reward for an option",
        ),
        matching_user_answer_plot_dots(
            intervention=None,
            all_tasks=all_read,
            for_formatters=[ZeroShotCOTSycophancyFormatter],
            model=model,
            for_task=[task],
            name_override="I think the answer is (X)",
        ),
    ]
    random_chance: PlotDots = random_chance_matching_answer_plot_dots(
        all_tasks=all_read,
        model=model,
        name_override="Random chance",
        formatter=unbiased_formatter,
        for_task=[task],
    )
    dotted_line = DottedLine(name="Random chance", value=random_chance.acc.accuracy, color="red")
    bar_plot(
        plot_dots=matching_user_answer,
        title=f"How often does {model} choose the bias's view? Model: {model} Task: {task}<br>With COT completion<br>Bias always on wrong answer",
        y_axis_title="Answers matching user's view (%)",
        dotted_line=dotted_line,
        # save_file_path=task + "_answer_matching.png",
        max_y=1.0,
    )


if __name__ == "__main__":
    # decrease_in_accuracy(exp_dir="experiments/gpt_35_cot", model="gpt-3.5-turbo")
    tasks_ran = [
        # "aqua"
        "mmlu",
        "hellaswag",
        "truthful_qa",
        # "logiqa",
    ]
    # trained with bias in prompt
    # decrease_in_accuracy(
    #     exp_dir="experiments/finetune", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::7rg7aRbV", tasks=tasks_ran
    # )
    # trained w/o bias in prompt
    # decrease_in_accuracy(
    #     exp_dir="experiments/finetune", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::7ryTmccr", tasks=tasks_ran
    # )
    # left out WrongFewShotIgnoreMistakesBiasedFormatter
    # decrease_in_accuracy(
    #     exp_dir="experiments/finetune", model="ft:gpt-3.5-turbo-0613:academicsnyuperez::7s4U75Iu", tasks=tasks_ran
    # )
    decrease_in_accuracy_per_bias(exp_dir="experiments/finetune", model="gpt-3.5-turbo")
    # decrease_in_accuracy(exp_dir="experiments/finetune", model="gpt-3.5-turbo")
