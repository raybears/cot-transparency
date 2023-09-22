import glob
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence, Type, Mapping

from plotly import graph_objects as go, io as pio
from pydantic import BaseModel
from slist import Slist

from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.core.sycophancy import ZeroShotCOTSycophancyFormatter, ZeroShotSycophancyFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.formatters.interventions.consistency import (
    NaiveFewShot10,
    NaiveFewShot3,
    NaiveFewShot6,
    NaiveFewShotLabelOnly3,
    NaiveFewShotLabelOnly6,
    NaiveFewShotLabelOnly10,
    NaiveFewShotLabelOnly16,
    NaiveFewShotLabelOnly32,
    NaiveFewShotLabelOnly1,
    BiasedConsistency10,
    BigBrainBiasedConsistency10,
    RepeatedConsistency10,
    NaiveFewShot5,
    NaiveFewShotSeparate10,
    NaiveFewShot1,
    ClaudeFewShot10,
    ClaudeSeparate10,
)
from cot_transparency.formatters.interventions.intervention import Intervention
from cot_transparency.formatters.more_biases.deceptive_assistant import (
    DeceptiveAssistantBiasedFormatter,
    DeceptiveAssistantBiasedNoCOTFormatter,
)
from cot_transparency.formatters.more_biases.more_reward import (
    MoreRewardBiasedFormatter,
    MoreRewardBiasedNoCOTFormatter,
)
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
)
from cot_transparency.formatters.verbalize.formatters import (
    StanfordBiasedFormatter,
    StanfordNoCOTFormatter,
)
from cot_transparency.tasks import read_done_experiment
from scripts.matching_user_answer import matching_user_answer_plot_dots
from scripts.multi_accuracy import PlotDots, accuracy_outputs
from scripts.simple_formatter_names import INTERVENTION_TO_SIMPLE_NAME


# ruff: noqa: E501


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
    include_tasks: Sequence[str] = [],
    distinct_qns: bool = True,
) -> PlotDots:
    assert all_tasks, "No tasks found"
    intervention_name: str | None = intervention.name() if intervention else None
    formatters_names: set[str] = {f.name() for f in for_formatters}
    filtered: Slist[TaskOutput] = (
        all_tasks.filter(lambda task: intervention_name == task.task_spec.intervention_name)
        .filter(lambda task: task.task_spec.formatter_name in formatters_names)
        .filter(lambda task: task.task_spec.inference_config.model == model)
        .filter(lambda task: task.task_spec.task_name in include_tasks if include_tasks else True)
    )
    if distinct_qns:
        filtered = filtered.distinct_by(lambda task: task.task_spec.task_hash)
    assert filtered, f"Intervention {intervention_name} has no tasks in {for_formatters} for model {model}"
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
    y_axis_title: Optional[str] = None,
    max_y: Optional[float] = None,
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
    title_text = f"{title}<br>{subtitle}" if subtitle else title
    fig.update_layout(
        barmode="group",
        title_text=title_text,
        title_x=0.5,
    )
    if y_axis_title is not None:
        fig.update_yaxes(title_text=y_axis_title)
    if max_y is not None:
        fig.update_yaxes(range=[0, max_y])

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

    if save_file_path is not None:
        pio.write_image(fig, save_file_path + ".png", scale=2)
    else:
        fig.show()


def accuracy_diff_intervention(
    data: Slist[TaskOutput],
    model: str,
    plot_dots: list[PlotDots],
    interventions: Sequence[Type[Intervention] | None],
    unbiased_formatter: Type[StageOneFormatter],
):
    # Make chart of the diff in accuracy betwwen the unbiased and the biased, same interventio
    unbiased_plot_dots: dict[str, PlotDots] = (
        Slist(
            [
                plot_dots_for_intervention(intervention, data, for_formatters=[unbiased_formatter], model=model)
                for intervention in interventions
            ]
        )
        .map(lambda plot: (plot.name, plot))
        .to_dict()
    )
    joined_plot_dots: list[PlotDots] = Slist(plot_dots).map(
        lambda plot: PlotDots(
            acc=plot.acc - unbiased_plot_dots[plot.name].acc,
            name=plot.name,
        )
    )
    bar_plot(
        plot_dots=joined_plot_dots,
        title="Does the accuracy gap between unbiased and biased context drop with more few shots?",
    )


class TaskOutputFilter(ABC):
    @staticmethod
    @abstractmethod
    def filter(data: Sequence[TaskOutput]) -> Slist[TaskOutput]:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        return cls.__name__


class InconsistentOnly(TaskOutputFilter):
    @staticmethod
    def filter(data: Sequence[TaskOutput]) -> Slist[TaskOutput]:
        return filter_inconsistent_only(data)

    @classmethod
    def name(cls) -> str:
        return "Bias always on wrong answer"


class ConsistentOnly(TaskOutputFilter):
    @staticmethod
    def filter(data: Sequence[TaskOutput]) -> Slist[TaskOutput]:
        return filter_consistent_only(data)

    @classmethod
    def name(cls) -> str:
        return "Bias on correct answer"


class NoFilter(TaskOutputFilter):
    @staticmethod
    def filter(data: Sequence[TaskOutput]) -> Slist[TaskOutput]:
        return Slist(data)

    @classmethod
    def name(cls) -> str:
        return "No bias in context"


def filter_inconsistent_only(data: Sequence[TaskOutput]) -> Slist[TaskOutput]:
    return Slist(data).filter(lambda task: (task.task_spec.biased_ans != task.task_spec.ground_truth))


def filter_consistent_only(data: Sequence[TaskOutput]) -> Slist[TaskOutput]:
    return Slist(data).filter(lambda task: (task.task_spec.biased_ans == task.task_spec.ground_truth))


def run(
    interventions: Sequence[Type[Intervention] | None],
    biased_formatters: Sequence[Type[StageOneFormatter]],
    unbiased_formatter: Type[StageOneFormatter],
    accuracy_plot_name: Optional[str] = None,
    percent_matching_plot_name: Optional[str] = None,
    inconsistent_only: bool = True,
    tasks: Sequence[str] = [],
    model: str = "gpt-4",
    intervention_name_override: Mapping[Type[Intervention] | None, str] = {},
):
    all_read: Slist[TaskOutput] = read_whole_exp_dir(exp_dir="experiments/interventions")
    all_read = (filter_inconsistent_only(all_read) if inconsistent_only else all_read).filter(
        lambda task: task.task_spec.task_name in tasks if tasks else True
    )

    # unbiased acc
    unbiased_plot: PlotDots = plot_dots_for_intervention(
        None, all_read, for_formatters=[unbiased_formatter], name_override="Unbiased context", model=model
    )

    plot_dots: list[PlotDots] = [
        plot_dots_for_intervention(
            intervention,
            all_read,
            for_formatters=biased_formatters,
            model=model,
            name_override=intervention_name_override.get(intervention, None),
        )
        for intervention in interventions
    ]
    dotted = DottedLine(name="Zero shot unbiased context performance", value=unbiased_plot.acc.accuracy, color="red")

    bar_plot(
        plot_dots=plot_dots,
        title=accuracy_plot_name
        or f"More efficient to prompt with only unbiased questions compared to consistency pairs Model: {model} Dataset: Aqua and mmlu",
        dotted_line=dotted,
        y_axis_title="Accuracy",
    )
    # accuracy_diff_intervention(interventions=interventions, unbiased_formatter=unbiased_formatter)
    matching_user_answer: list[PlotDots] = [
        matching_user_answer_plot_dots(
            intervention=intervention,
            all_tasks=all_read,
            for_formatters=biased_formatters,
            model=model,
            name_override=intervention_name_override.get(intervention, None),
        )
        for intervention in interventions
    ]
    unbiased_matching_baseline = matching_user_answer_plot_dots(
        intervention=None, all_tasks=all_read, for_formatters=[unbiased_formatter], model=model
    )
    dotted_line = DottedLine(
        name="Zeroshot Unbiased context", value=unbiased_matching_baseline.acc.accuracy, color="red"
    )
    dataset_str = Slist(tasks).mk_string(", ")
    bar_plot(
        plot_dots=matching_user_answer,
        title=percent_matching_plot_name
        or f"How often does {model} choose the user's view?<br>Model: {model} Dataset: {dataset_str}",
        y_axis_title="Answers matching bias's view (%)",
        dotted_line=dotted_line,
    )


def run_for_cot_shot_scaling(model: str, inconsistent_only: bool = True):
    """
       python stage_one.py --exp_dir experiments/interventions --tasks "['truthful_qa', 'john_level_5', 'logiqa', 'hellaswag', 'mmlu']" --models "['claude-2']" --formatters '["ZeroShotCOTSycophancyFormatter", "MoreRewardBiasedFormatter", "StanfordBiasedFormatter", "DeceptiveAssistantBiasedFormatter", "WrongFewShotBiasedFormatter", "ZeroShotCOTUnbiasedFormatter"]'
    --example_cap 100 --interventions "['NaiveFewShot1', 'ClaudeFewShot1', 'NaiveFewShot6','ClaudeFewShot6', 'NaiveFewShot16', 'ClaudeFewShot16', 'ClaudeFewShot32', 'NaiveFewShot32']" --batch 30
    """
    tasks = ["truthful_qa", "john_level_5", "logiqa", "hellaswag", "mmlu"]
    # what interventions to plot
    interventions: Sequence[Type[Intervention] | None] = [
        None,
        # NaiveFewShot1,
        # NaiveFewShot3,
        # NaiveFewShot6,
        NaiveFewShot10,
        # NaiveFewShot16,
        # ClaudeFewShot1,
        # ClaudeFewShot3,
        # ClaudeFewShot6,
        ClaudeFewShot10,
        # ClaudeFewShot16,
    ]
    # what formatters to include
    biased_formatters = [
        WrongFewShotBiasedFormatter,
        StanfordBiasedFormatter,
        MoreRewardBiasedFormatter,
        ZeroShotCOTSycophancyFormatter,
        DeceptiveAssistantBiasedFormatter,
    ]
    unbiased_formatter = ZeroShotCOTUnbiasedFormatter
    run(
        interventions=interventions,
        biased_formatters=biased_formatters,
        unbiased_formatter=unbiased_formatter,
        inconsistent_only=inconsistent_only,
        model=model,
        tasks=tasks,
        accuracy_plot_name=f"Do more COT few shots help {model}? Accuracy<br>{format_subtitle(inconsistent_only=inconsistent_only, tasks=tasks, model=model)}",
        percent_matching_plot_name=f"Do more COT few shots help {model}? Percent matching bias<br>{format_subtitle(inconsistent_only=inconsistent_only, tasks=tasks, model=model)}",
    )


def run_for_cot_claude_vs_gpt4(model: str, inconsistent_only: bool = True):
    """
       python stage_one.py --exp_dir experiments/interventions --tasks "['truthful_qa', 'john_level_5', 'logiqa', 'hellaswag', 'mmlu']" --models "['claude-2']" --formatters '["ZeroShotCOTSycophancyFormatter", "MoreRewardBiasedFormatter", "StanfordBiasedFormatter", "DeceptiveAssistantBiasedFormatter", "WrongFewShotBiasedFormatter", "ZeroShotCOTUnbiasedFormatter"]'
    --example_cap 100 --interventions "['NaiveFewShot1', 'ClaudeFewShot1', 'NaiveFewShot6','ClaudeFewShot6', 'NaiveFewShot16', 'ClaudeFewShot16', 'ClaudeFewShot32', 'NaiveFewShot32']" --batch 30
    """
    tasks = ["truthful_qa", "john_level_5", "logiqa", "hellaswag", "mmlu"]
    # what interventions to plot
    interventions: Sequence[Type[Intervention] | None] = [
        None,
        NaiveFewShot10,
        ClaudeFewShot10,
    ]
    # what formatters to include
    biased_formatters = [
        WrongFewShotBiasedFormatter,
        StanfordBiasedFormatter,
        MoreRewardBiasedFormatter,
        ZeroShotCOTSycophancyFormatter,
        DeceptiveAssistantBiasedFormatter,
    ]
    unbiased_formatter = ZeroShotCOTUnbiasedFormatter
    run(
        interventions=interventions,
        biased_formatters=biased_formatters,
        unbiased_formatter=unbiased_formatter,
        inconsistent_only=inconsistent_only,
        model=model,
        tasks=tasks,
        accuracy_plot_name=f"Accuracy for Model: {model}  Is there a difference between using GPT-4 or Claude-2 COTs? <br>{format_subtitle(inconsistent_only=inconsistent_only, tasks=tasks, model=model)}",
        percent_matching_plot_name=f"% matching bias for Model: {model} Is there a difference between using GPT-4 or Claude-2 COTs?<br>{format_subtitle(inconsistent_only=inconsistent_only, tasks=tasks, model=model)}",
        intervention_name_override={
            NaiveFewShot10: "10 unbiased COTs from GPT-4",
            ClaudeFewShot10: "10 unbiased COTs from Claude-2",
        },
    )


def format_subtitle(inconsistent_only: bool, tasks: Sequence[str], model: str) -> str:
    dataset_str = Slist(tasks).mk_string(", ")
    if inconsistent_only:
        return f"Model: {model} Dataset: {dataset_str}<br>Bias is always on the wrong answer"
    else:
        return f"Model: {model} Dataset: {dataset_str}<br>Bias may be on the correct answer"


def run_for_cot_shot_scaling_non_cot_completion(model: str, inconsistent_only: bool = True):
    """
    python stage_one.py --exp_dir experiments/interventions --models "['gpt-4']" --example_cap 61 --interventions "['NaiveFewShot1', 'NaiveFewShot3', 'NaiveFewShot6', 'NaiveFewShot10']" --formatters  "['WrongFewShotBiasedNoCOTFormatter', 'StanfordNoCOTFormatter', 'MoreRewardBiasedNoCOTFormatter', 'ZeroShotSycophancyFormatter', 'DeceptiveAssistantBiasedNoCOTFormatter', 'ZeroShotUnbiasedFormatter']" --tasks '["truthful_qa", "john_level_5", "logiqa", "hellaswag", "mmlu"]'
    """
    tasks = ["truthful_qa", "john_level_5", "logiqa", "hellaswag", "mmlu"]
    # what interventions to plot
    interventions: Sequence[Type[Intervention] | None] = [
        None,
        NaiveFewShot1,
        NaiveFewShot3,
        NaiveFewShot6,
        NaiveFewShot10,
        # BiasedConsistency10,
        # BigBrainBiasedConsistency10,
    ]
    # what formatters to include
    biased_formatters = [
        WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
        StanfordNoCOTFormatter,
        MoreRewardBiasedNoCOTFormatter,
        ZeroShotSycophancyFormatter,
        DeceptiveAssistantBiasedNoCOTFormatter,
    ]
    unbiased_formatter = ZeroShotUnbiasedFormatter
    run(
        interventions=interventions,
        biased_formatters=biased_formatters,
        unbiased_formatter=unbiased_formatter,
        inconsistent_only=inconsistent_only,
        model=model,
        tasks=tasks,
        accuracy_plot_name=f"Do more COT few shots help for NON-COT completions? Accuracy<br>{format_subtitle(inconsistent_only=inconsistent_only, tasks=tasks, model=model)}",
        percent_matching_plot_name=f"Do more COT few shots help for NON-COT completions? Percent matching bias<br>{format_subtitle(inconsistent_only=inconsistent_only, tasks=tasks, model=model)}",
    )


def run_for_cot_separate_or_not(
    model: str,
    inconsistent_only: bool = True,
):
    """python stage_one.py --exp_dir experiments/interventions --tasks "['truthful_qa', 'john_level_5', 'logiqa', 'hellaswag', 'mmlu']" --models "['claude-2']" --formatters '["ZeroShotCOTSycophancyFormatter", "MoreRewardBiasedFormatter", "StanfordBiasedFormatter", "DeceptiveAssistantBiasedFormatter", "WrongFewShotBiasedFormatter", "ZeroShotCOTUnbiasedFormatter"]'
    --example_cap 100 --interventions "['NaiveFewShot10', 'ClaudeFewShot10', 'ClaudeSeparate10', 'BiasedConsistency10', 'NaiveFewShotSeparate10']" --batch 60
    """
    # This plot answers the question of whether to use separate or not separate few shot messages
    tasks = ["truthful_qa", "john_level_5", "logiqa", "hellaswag", "mmlu"]
    # what interventions to plot
    interventions: Sequence[Type[Intervention] | None] = [
        None,
        # NaiveFewShot10,
        # NaiveFewShotSeparate10,
        ClaudeFewShot10,
        ClaudeSeparate10,
    ]
    # what formatters to include
    biased_formatters = [
        WrongFewShotBiasedFormatter,
        StanfordBiasedFormatter,
        MoreRewardBiasedFormatter,
        ZeroShotCOTSycophancyFormatter,
        DeceptiveAssistantBiasedFormatter,
    ]
    unbiased_formatter = ZeroShotCOTUnbiasedFormatter
    run(
        interventions=interventions,
        biased_formatters=biased_formatters,
        unbiased_formatter=unbiased_formatter,
        inconsistent_only=inconsistent_only,
        model=model,
        tasks=tasks,
        accuracy_plot_name=f"Does using separate few shot messages improve accuracy?<br>{format_subtitle(inconsistent_only=inconsistent_only, tasks=tasks, model=model)}",
        percent_matching_plot_name=f"Does using separate few shot messages reduce sycophancy?<br>{format_subtitle(inconsistent_only=inconsistent_only, tasks=tasks, model=model)}",
        intervention_name_override={
            NaiveFewShot10: "All 10 few shots in first user message",
            NaiveFewShotSeparate10: "Separate 10 few shot messages",
        },
    )


def run_for_cot_different_10_shots(
    model: str,
    inconsistent_only: bool = True,
):
    """
    python stage_one.py --exp_dir experiments/interventions --tasks "['truthful_qa', 'john_level_5', 'logiqa', 'hellaswag', 'mmlu']" --models "['gpt-4']" --formatters '["ZeroShotCOTSycophancyFormatter", "MoreRewardBiasedFormatter", "StanfordBiasedFormatter", "DeceptiveAssistantBiasedFormatter", "WrongFewShotBiasedFormatter", "ZeroShotCOTUnbiasedFormatter"]' --example_cap 61 --interventions "['BigBrainBiasedConsistency10', 'BigBrainBiasedConsistencySeparate10', 'NaiveFewShot10', 'BiasedConsistency10', 'RepeatedConsistency10', 'PairedConsistency10', 'NaiveFewShot1', 'NaiveFewShot3', 'NaiveFewShot6']"
    """
    tasks = ["truthful_qa", "john_level_5", "logiqa", "hellaswag", "mmlu"]
    # what interventions to plot
    interventions: Sequence[Type[Intervention] | None] = [
        None,
        # BigBrainBiasedConsistencySeparate10,
        BigBrainBiasedConsistency10,
        # PairedConsistency10,
        # RepeatedConsistency10,
        # NaiveFewShot1,
        # NaiveFewShot3,
        # NaiveFewShot5,
        NaiveFewShot10,
        # NaiveFewShotSeparate10,
        # BiasedConsistency10,
    ]
    # what formatters to include
    biased_formatters = [
        WrongFewShotBiasedFormatter,
        StanfordBiasedFormatter,
        MoreRewardBiasedFormatter,
        ZeroShotCOTSycophancyFormatter,
        # DeceptiveAssistantBiasedFormatter,
    ]
    unbiased_formatter = ZeroShotCOTUnbiasedFormatter
    run(
        interventions=interventions,
        biased_formatters=biased_formatters,
        unbiased_formatter=unbiased_formatter,
        inconsistent_only=inconsistent_only,
        model=model,
        tasks=tasks,
    )


def run_for_cot_big_brain(
    model: str,
    inconsistent_only: bool = True,
):
    tasks = ["truthful_qa", "john_level_5", "logiqa", "hellaswag", "mmlu"]
    # what interventions to plot
    interventions: Sequence[Type[Intervention] | None] = [
        None,
        BigBrainBiasedConsistency10,
        NaiveFewShot10,
    ]
    # what formatters to include
    biased_formatters = [
        WrongFewShotBiasedFormatter,
        StanfordBiasedFormatter,
        MoreRewardBiasedFormatter,
        ZeroShotCOTSycophancyFormatter,
        DeceptiveAssistantBiasedFormatter,
    ]
    unbiased_formatter = ZeroShotCOTUnbiasedFormatter
    run(
        interventions=interventions,
        biased_formatters=biased_formatters,
        unbiased_formatter=unbiased_formatter,
        inconsistent_only=inconsistent_only,
        model=model,
        tasks=tasks,
        accuracy_plot_name=f"Does prompting with examples that the model does get biased for, improve accuracy?<br>{format_subtitle(inconsistent_only=inconsistent_only, tasks=tasks, model=model)}",
        percent_matching_plot_name=f"Does prompting with examples that the model does get biased for, reduce bias?<br>{format_subtitle(inconsistent_only=inconsistent_only, tasks=tasks, model=model)}",
    )


def run_for_cot_biased_consistency(
    model: str,
    inconsistent_only: bool = True,
):
    tasks = ["truthful_qa", "john_level_5", "logiqa", "hellaswag", "mmlu"]
    # what interventions to plot
    interventions: Sequence[Type[Intervention] | None] = [
        None,
        BiasedConsistency10,
        NaiveFewShot10,
    ]
    # what formatters to include
    biased_formatters = [
        WrongFewShotBiasedFormatter,
        StanfordBiasedFormatter,
        MoreRewardBiasedFormatter,
        ZeroShotCOTSycophancyFormatter,
        DeceptiveAssistantBiasedFormatter,
    ]
    unbiased_formatter = ZeroShotCOTUnbiasedFormatter
    run(
        interventions=interventions,
        biased_formatters=biased_formatters,
        unbiased_formatter=unbiased_formatter,
        inconsistent_only=inconsistent_only,
        model=model,
        tasks=tasks,
        accuracy_plot_name=f"Does prompting with examples that the model does get biased for, improve accuracy?<br>{format_subtitle(inconsistent_only=inconsistent_only, tasks=tasks, model=model)}",
        percent_matching_plot_name=f"Does prompting with examples that the model does get biased for, reduce bias?<br>{format_subtitle(inconsistent_only=inconsistent_only, tasks=tasks, model=model)}",
    )


def run_for_cot_naive_vs_consistency():
    """
    python stage_one.py --exp_dir experiments/interventions --models "['gpt-4']" --formatters '["ZeroShotCOTSycophancyFormatter", "MoreRewardBiasedFormatter", "StanfordBiasedFormatter", "DeceptiveAssistantBiasedFormatter", "WrongFewShotBiasedFormatter", "ZeroShotCOTUnbiasedFormatter"]' --example_cap 61 --interventions "['PairedConsistency10', 'NaiveFewShot5', 'NaiveFewShot10', 'RepeatedConsistency10']" --tasks "['truthful_qa', 'john_level_5', 'logiqa', 'hellaswag', 'mmlu']"
    """
    # what interventions to plot
    interventions: Sequence[Type[Intervention] | None] = [
        None,
        RepeatedConsistency10,
        NaiveFewShot5,
        NaiveFewShot10,
        # PairedConsistency10,
    ]
    tasks = ["truthful_qa", "john_level_5", "logiqa", "hellaswag", "mmlu"]
    # what formatters to include
    biased_formatters = [
        WrongFewShotBiasedFormatter,
        StanfordBiasedFormatter,
        MoreRewardBiasedFormatter,
        ZeroShotCOTSycophancyFormatter,
        DeceptiveAssistantBiasedFormatter,
    ]
    unbiased_formatter = ZeroShotCOTUnbiasedFormatter
    run(
        interventions=interventions,
        biased_formatters=biased_formatters,
        unbiased_formatter=unbiased_formatter,
        tasks=tasks,
        accuracy_plot_name=f"Does repeating the same question help? Accuracy<br>{format_subtitle(inconsistent_only=True, tasks=tasks, model='gpt-4')}",
        percent_matching_plot_name=f"Does repeating the same question help? Percent matching bias<br>{format_subtitle(inconsistent_only=True, tasks=tasks, model='gpt-4')}",
        intervention_name_override={
            RepeatedConsistency10: "Repeat 5 few shots one time (10 total)",
            NaiveFewShot5: "5 few shots",
            NaiveFewShot10: "10 few shots",
        },
    )


def run_for_non_cot():
    """
    python stage_one.py --exp_dir experiments/interventions --tasks "['truthful_qa', 'john_level_5', 'logiqa', 'hellaswag', 'mmlu']" --models "['claude-2', 'gpt-4']" --formatters '["ZeroShotCOTSycophancyFormatter", "MoreRewardBiasedFormatter", "StanfordBiasedFormatter", "DeceptiveAssistantBiasedFormatter", "WrongFewShotBiasedFormatter", "ZeroShotCOTUnbiasedFormatter"]' --example_cap 20 --interventions "['NaiveFewShot10', 'NaiveFewShot3', 'NaiveFewShot6', 'ClaudeFewShot10', 'ClaudeFewShot1', 'NaiveFewShot1', 'ClaudeFewShot6', 'ClaudeFewShot3', 'NaiveFewShot16', 'ClaudeFewShot16']"
    """
    # what interventions to plot
    interventions: Sequence[Type[Intervention] | None] = [
        None,
        NaiveFewShotLabelOnly1,
        NaiveFewShotLabelOnly3,
        NaiveFewShotLabelOnly6,
        NaiveFewShotLabelOnly10,
        NaiveFewShotLabelOnly16,
        NaiveFewShotLabelOnly32,
    ]
    # what formatters to include
    biased_formatters = [
        WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
        StanfordNoCOTFormatter,
        MoreRewardBiasedNoCOTFormatter,
        # ZeroShotSycophancyFormatter,
        DeceptiveAssistantBiasedNoCOTFormatter,
    ]
    unbiased_formatter = ZeroShotUnbiasedFormatter
    run(interventions=interventions, biased_formatters=biased_formatters, unbiased_formatter=unbiased_formatter)


if __name__ == "__main__":
    # run_for_cot_different_10_shots()
    # run_for_cot_shot_scaling_non_cot_completion(model="gpt-4")
    # run_for_non_cot()
    # run_for_cot_naive_vs_consistency()
    # run_for_cot_different_10_shots(inconsistent_only=True, model="gpt-4")
    # run_for_cot_shot_scaling(inconsistent_only=True, model="gpt-4")
    run_for_cot_claude_vs_gpt4(inconsistent_only=True, model="gpt-4")
    run_for_cot_claude_vs_gpt4(inconsistent_only=True, model="claude-2")
    # run_for_cot_separate_or_not(inconsistent_only=True, model="gpt-4")
    # run_for_cot_separate_or_not(inconsistent_only=True, model="claude-2")
    # run_for_cot_naive_vs_consistency()
    # run_for_cot_big_brain(model="gpt-4")
    # run_for_cot_biased_consistency(model="claude-2")
