from typing import Optional

from plotly import graph_objects as go, io as pio

from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.more_biases.more_reward import MoreRewardBiasedFormatter
from cot_transparency.formatters.more_biases.wrong_few_shot import WrongFewShotBiasedFormatter
from scripts.intervention_investigation import read_whole_exp_dir, DottedLine, bar_plot, filter_inconsistent_only
from scripts.matching_user_answer import matching_user_answer_plot_dots, baseline_matching_answer_plot_dots
from scripts.multi_accuracy import PlotDots
from scripts.overall_accuracy import overall_accuracy_for_formatter


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
    """
    python stage_one.py --exp_dir experiments/interventions --tasks '["aqua","arc_easy","arc_challenge","truthful_qa","logiqa","mmlu","openbook_qa","hellaswag","john_level_5"]' --models "['gpt-4']" --formatters '["ZeroShotCOTSycophancyFormatter", "MoreRewardBiasedFormatter", "StanfordBiasedFormatter", "DeceptiveAssistantBiasedFormatter", "WrongFewShotBiasedFormatter", "ZeroShotCOTUnbiasedFormatter"]' --example_cap 100
    """
    # tasks: list[str] = TASK_LIST["transparency"]
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
    all_read = filter_inconsistent_only(read_whole_exp_dir(exp_dir="experiments/interventions"))
    unbiased_formatter = ZeroShotCOTUnbiasedFormatter
    for task in tasks:
        overall_accuracy_for_formatter("ZeroShotCOTUnbiasedFormatter", exp_dir=exp_dir, model=model, tasks=[task])
        [
            PlotDots(
                acc=overall_accuracy_for_formatter(
                    "DeceptiveAssistantBiasedFormatter", exp_dir=exp_dir, model=model, tasks=[task]
                ),
                name="Tell model to be deceptive",
            ),
            PlotDots(
                acc=overall_accuracy_for_formatter(
                    "WrongFewShotBiasedFormatter", exp_dir=exp_dir, model=model, tasks=[task]
                ),
                name="Wrong label in the few shot",
            ),
            # PlotDots(
            #     acc=overall_accuracy_for_formatter(
            #         "UserBiasedWrongCotFormatter", exp_dir=exp_dir, model=model, tasks=task
            #     ),
            #     name="User says wrong reasoning",
            # ),
            PlotDots(
                acc=overall_accuracy_for_formatter(
                    "MoreRewardBiasedFormatter", exp_dir=exp_dir, model=model, tasks=[task]
                ),
                name="More reward for an option",
            ),
            PlotDots(
                acc=overall_accuracy_for_formatter(
                    "ZeroShotCOTSycophancyFormatter", exp_dir=exp_dir, model=model, tasks=[task]
                ),
                name="Sycophancy",
            ),
        ]
        # decrease_in_accuracy_plot(
        #     base_accuracy=nonbiased.accuracy,
        #     plot_dots=plot_dots,
        #     title=f"How much does each bias decrease GPT-4's performance on {task}?<br>With COT completion",
        #     subtitle=f"n={nonbiased.samples}",
        #     save_file_path=None,
        # )
        matching_user_answer: list[PlotDots] = [
            # matching_user_answer_plot_dots(
            #     intervention=None,
            #     all_tasks=all_read,
            #     for_formatters=[DeceptiveAssistantBiasedFormatter],
            #     model=model,
            #     for_task=[task],
            #     name_override="Tell model to be deceptive",
            # ),
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
        random_chance: PlotDots = baseline_matching_answer_plot_dots(
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
            save_file_path=task,
        )


if __name__ == "__main__":
    decrease_in_accuracy(exp_dir="experiments/interventions", model="gpt-4")
