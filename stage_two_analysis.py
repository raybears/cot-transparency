from typing import Optional
import fire
from matplotlib import pyplot as plt
from analysis import get_general_metrics
from cot_transparency.data_models.models import (
    StageTwoExperimentJsonFormat,
    TaskOutput,
)
import pandas as pd
from cot_transparency.data_models.io import ExpLoader
from cot_transparency.formatters.transparency.trace_manipulation import get_cot_steps
from analysis import accuracy_for_df, TASK_MAP
import seaborn as sns
import numpy as np

# Used to produce human readable names on plots
NAMES_MAP = {
    "model": "Model",
    "task_name": "Task",
    "original_cot_trace_length": "CoT Length",
    "stage_one_formatter_name": "S1 Formatter",
}

sns.set_style(
    "ticks",
    {
        "axes.edgecolor": "0",
        "grid.linestyle": ":",
        "grid.color": "lightgrey",
        "grid.linewidth": "1.5",
        "axes.facecolor": "white",
    },
)


def get_aoc(df: pd.DataFrame, x="cot_trace_length") -> pd.DataFrame:
    # get aoc for each original_cot_trace_length, grouped by task, and model
    # we want the counts for the number of traces so filter on unique stage_one_hash

    groups = ["task_name", "model", "original_cot_trace_length", x, "stage_one_formatter_name"]

    n_traces = df.groupby(groups).stage_one_hash.nunique().reset_index()
    n_traces = n_traces.rename(columns={"stage_one_hash": "n_traces"})

    # get aucs for each original_cot_trace_length, grouped by task, and model
    areas = df.groupby(groups).apply(lambda x: x["same_answer"].mean()).reset_index()

    def get_auc(group: pd.DataFrame) -> float:
        assert group["original_cot_trace_length"].nunique() == 1
        proportion_of_cot = group[x] / max(group[x])
        auc = np.trapz(group[0], x=proportion_of_cot)
        return auc

    groups.pop(groups.index(x))
    grouped = areas.groupby(groups)
    aucs = grouped.apply(get_auc).reset_index()
    aucs = aucs.rename(columns={0: "auc"})
    areas = pd.merge(aucs, n_traces, on=groups)
    print(areas)

    groups.pop(groups.index("original_cot_trace_length"))
    areas["weighted_auc"] = areas["auc"] * areas["n_traces"]
    areas = areas.groupby(groups).sum().reset_index()
    areas["weighted_auc"] = areas["weighted_auc"] / areas["n_traces"]
    if (areas["weighted_auc"] > 1).any():
        areas["weighted_aoc"] = 100 - areas["weighted_auc"]
    else:
        areas["weighted_aoc"] = 1 - areas["weighted_auc"]

    print(areas)
    print(areas.to_csv())
    return areas


def convert_stage2_experiment_to_dataframe(exp: StageTwoExperimentJsonFormat) -> pd.DataFrame:
    out = []
    for task_output in exp.outputs:
        d_with_config = get_general_metrics(task_output)
        d_with_config["model"] = task_output.task_spec.inference_config.model
        d_with_config["task_name"] = task_output.task_spec.stage_one_output.task_spec.task_name
        d_with_config["ground_truth"] = task_output.task_spec.stage_one_output.task_spec.ground_truth
        d_with_config["stage_one_hash"] = task_output.task_spec.stage_one_output.task_spec.uid()
        d_with_config["stage_one_output_hash"] = task_output.task_spec.stage_one_output.uid()
        d_with_config["stage_one_output"] = task_output.task_spec.stage_one_output.dict()
        d_with_config["biased_ans"] = task_output.task_spec.stage_one_output.task_spec.biased_ans
        d_with_config["task_hash"] = task_output.task_spec.stage_one_output.task_spec.task_hash
        d_with_config["parsed_response"] = task_output.inference_output.parsed_response
        d_with_config["has_mistake"] = task_output.task_spec.trace_info.has_mistake
        d_with_config["was_truncated"] = task_output.task_spec.trace_info.was_truncated
        d_with_config["mistake_added_at"] = task_output.task_spec.trace_info.mistake_inserted_idx
        d_with_config["original_cot_trace_length"] = len(task_output.task_spec.trace_info.original_cot)
        modified_cot_length = get_cot_steps(task_output.task_spec.trace_info.get_complete_modified_cot())
        d_with_config["cot_trace_length"] = len(modified_cot_length)
        d_with_config["stage_one_formatter_name"] = task_output.task_spec.stage_one_output.task_spec.formatter_name

        out.append(d_with_config)

    df = pd.DataFrame(out)

    stage_one_output = [TaskOutput(**i) for i in df["stage_one_output"]]
    stage_formatter = [i.task_spec.formatter_name for i in stage_one_output]
    df["stage_one_formatter_name"] = stage_formatter
    return df


def get_data_frame_from_exp_dir(exp_dir: str) -> pd.DataFrame:
    loaded_dict = ExpLoader.stage_two(exp_dir, final_only=True)
    dfs = []
    for exp in loaded_dict.values():
        df = convert_stage2_experiment_to_dataframe(exp)
        dfs.append(df)
    df = pd.concat(dfs)
    df["is_correct"] = (df.parsed_response == df.ground_truth).astype(int)
    # filter out the NOT_FOUND rows
    n_not_found = len(df[df.parsed_response == "NOT_FOUND"])
    print(f"Number of NOT_FOUND rows: {n_not_found}")
    df = df[df.parsed_response != "NOT_FOUND"]
    return df  # type: ignore


def plot_historgram_of_lengths(
    exp_dir: str,
):
    df = get_data_frame_from_exp_dir(exp_dir)

    hue = "task_name"
    x = "CoT Length"
    col = "model"
    y = "Counts"

    # rename "original_cot_trace_length" to "CoT Length"
    df = df.rename(columns={"original_cot_trace_length": x})

    # for histogram we want the counts of the original_cot_trace_length
    # filter on the unique stage_one_hash
    counts = df.groupby([hue, col, x]).stage_one_hash.nunique().reset_index()
    counts = counts.rename(columns={"stage_one_hash": y})

    # facet plot of the proportion of the trace, break down by original_cot_trace_length
    g = sns.FacetGrid(counts, col=col, col_wrap=2, legend_out=True)
    g.map_dataframe(sns.barplot, x=x, y=y, hue=hue)
    g.add_legend()
    plt.show()


def df_filters(
    df: pd.DataFrame,
    inconsistent_only: bool,
    aggregate_over_tasks: bool,
    model_filter: Optional[str],
    length_filter: Optional[list[int]],
) -> pd.DataFrame:
    if aggregate_over_tasks:
        # replace task_name with the "parent" task name using the task_map
        df["task_name"] = df["task_name"].replace(TASK_MAP)

    if inconsistent_only:
        df = df[df.biased_ans != df.ground_truth]  # type: ignore
        print("Number of inconsistent tasks: ", len(df))

    if model_filter:
        df = df[df.model.isin(model_filter)]  # type: ignore

    if length_filter:
        df = df[df["original_cot_trace_length"].isin(length_filter)]  # type: ignore
        assert len(df) > 0, "No data for this length filter"
    return df


def plot_by_length(df: pd.DataFrame, hue: str, col: Optional[str] = None) -> sns.FacetGrid | plt.Axes:
    x = "proportion_of_cot"
    y = "same_answer"

    # display as percentages
    df[x] = df[x] * 100
    df[y] = df[y] * 100
    if col is not None:
        # rename "original_cot_trace_length" to "CoT Length"
        df_for_plotting = df.rename(columns=NAMES_MAP, inplace=False)

        if col != "original_cot_trace_length":
            # need to round as aggregating over different cot lengths
            df_for_plotting[x] = df_for_plotting[x].astype(float)
            df_for_plotting[x] = (df_for_plotting[x] / 10).round() * 10

        # facet plot of the proportion of the trace, break down by original_cot_trace_length
        g: sns.FacetGrid = sns.relplot(
            data=df_for_plotting,
            x=x,
            y=y,
            hue=NAMES_MAP[hue],
            col=NAMES_MAP[col],
            col_wrap=2,
            kind="line",
            height=4,
        )  # type: ignore
        x_label = "Proportion of CoT"
        y_label = "% Same Answer as Unmodified CoT"
        g.set_axis_labels(x_label, y_label)
        return g

    else:
        # plot aggreaged version that aggregates over cot lenghts
        plt.figure()
        df["proportion_of_cot_rounded"] = (df["proportion_of_cot"] / 10).round() * 10
        # g is an axis
        g_axs: plt.Axes = sns.lineplot(df, x="proportion_of_cot_rounded", y=y, hue=hue, alpha=0.5)
        return g_axs


def drop_not_found(df: pd.DataFrame) -> pd.DataFrame:
    # Redrop any NOT_FOUND
    pre = len(df)
    df = df[df.same_answer != "NOT_FOUND"]  # type: ignore
    print("Dropped ", pre - len(df), " rows becuase of NOT_FOUND answers in full cot trace")
    return df


def baseline_accuracy(df: pd.DataFrame, hue: str, col: str):
    # drop has mistake
    df = df[~df.has_mistake]  # type: ignore

    cot_trace_length_0 = df[df["cot_trace_length"] == 0]
    cot_trace_length_max = df[df["cot_trace_length"] == df["original_cot_trace_length"]]
    baseline_accuracies = (
        cot_trace_length_0.groupby(["task_name", hue, col]).apply(lambda x: x["is_correct"].mean()).reset_index()
    )
    baseline_accuracies = baseline_accuracies.rename(columns={0: "No CoT Accuracy"})
    baseline_accuracies["CoT Accuracy"] = (
        cot_trace_length_max.groupby(["task_name", hue, col]).apply(lambda x: x["is_correct"].mean()).reset_index()[0]
    )
    print(baseline_accuracies)
    # print csv version
    print(baseline_accuracies.to_csv())


def check_same_answer(group: pd.DataFrame) -> pd.DataFrame:
    max_step_row = group[(~group.was_truncated) & ~(group.has_mistake)]
    if len(max_step_row) == 0:
        group["same_answer"] = "NOT_FOUND"
    elif len(max_step_row) > 1:
        raise ValueError(
            "More than one row with max cot_trace_length you may "
            "have changed the prompt formatter half way throgh the exp"
        )
    else:
        group["same_answer"] = group["parsed_response"] == max_step_row["parsed_response"].iloc[0]  # type: ignore
    return group


def plot_early_answering(
    exp_dir: str,
    show_plots: bool = False,
    inconsistent_only: bool = False,
    aggregate_over_tasks: bool = False,
    model_filter: Optional[str] = None,
    length_filter: Optional[list[int]] = None,
    col: str = "original_cot_trace_length",
    hue: str = "model",
):
    df = get_data_frame_from_exp_dir(exp_dir)
    # drop formatters that have Mistake in the name
    df = df[~df.has_mistake]
    df = df_filters(df, inconsistent_only, aggregate_over_tasks, model_filter, length_filter)  # type: ignore

    # Apply the check_same_answer function
    df = df.groupby("stage_one_hash").apply(check_same_answer).reset_index(drop=True)
    df = drop_not_found(df)

    df["proportion_of_cot"] = df["cot_trace_length"] / df["original_cot_trace_length"]

    g = plot_by_length(df, hue, col)
    # set tile to Early Answering
    if isinstance(g, sns.FacetGrid):
        g.fig.suptitle("Early Answering")
    else:
        g.set_title("Early Answering")
    get_aoc(df)

    if show_plots:
        plt.show()


def plot_adding_mistakes(
    exp_dir: str,
    show_plots: bool = False,
    inconsistent_only: bool = False,
    aggregate_over_tasks: bool = False,
    model_filter: Optional[str] = None,
    length_filter: Optional[list[int]] = None,
    col: str = "original_cot_trace_length",
    hue: str = "model",
):
    df = get_data_frame_from_exp_dir(exp_dir)
    df = df[~df.was_truncated]

    df = df_filters(df, inconsistent_only, aggregate_over_tasks, model_filter, length_filter)  # type: ignore

    df = df.groupby("stage_one_hash").apply(check_same_answer).reset_index(drop=True)
    df = drop_not_found(df)

    df["proportion_of_cot"] = df["mistake_added_at"] / df["original_cot_trace_length"]

    g = plot_by_length(df, hue, col)
    if isinstance(g, sns.FacetGrid):
        g.fig.suptitle("Adding Mistakes")
    else:
        g.set_title("Adding Mistakes")
    get_aoc(df)

    if show_plots:
        plt.show()


def aoc_plot(
    exp_dir: str,
    show_plots: bool = False,
    inconsistent_only: bool = False,
    aggregate_over_tasks: bool = False,
    model_filter: Optional[str] = None,
    length_filter: Optional[list[int]] = None,
    hue: str = "stage_one_formatter_name",
):
    df = get_data_frame_from_exp_dir(exp_dir)
    df = df_filters(df, inconsistent_only, aggregate_over_tasks, model_filter, length_filter)

    # Mistakes AoC
    df_mistakes = df[~df.was_truncated]
    df_mistakes = df.groupby("stage_one_hash").apply(check_same_answer).reset_index(drop=True)
    df_mistakes = drop_not_found(df_mistakes)
    aoc_mistakes = get_aoc(df_mistakes)

    # Early Answering AoC
    df_early = df[~df.has_mistake]
    df_early = df_early.groupby("stage_one_hash").apply(check_same_answer).reset_index(drop=True)
    df_early = drop_not_found(df_early)  # type: ignore
    aoc_early = get_aoc(df_early)

    # baseline accuracies
    baseline_accuracy(df, hue, "model")

    _aoc_point_plot(hue, df, aoc_mistakes, aoc_early, kind="bar")
    _aoc_point_plot(hue, df, aoc_mistakes, aoc_early, kind="point")

    if show_plots:
        plt.show()


def _aoc_point_plot(hue: str, df: pd.DataFrame, aoc_mistakes: pd.DataFrame, aoc_early: pd.DataFrame, kind="bar"):
    # two point plots side by side [mistakes, early answering, accuracy]
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    if kind == "point":
        func = sns.pointplot
        kwargs = {"join": False}
    elif kind == "bar":
        func = sns.barplot
        kwargs = {}
    else:
        raise ValueError(f"kind must be point or bar, not {kind}")

    x_order = df.model.unique()
    func(
        data=aoc_mistakes,
        x="model",
        y="weighted_aoc",
        hue=hue,
        ax=axs[0],
        capsize=0.05,  # type: ignore
        errwidth=1,  # type: ignore
        order=x_order,
        **kwargs,  # type: ignore
    )
    axs[0].set_title("Mistakes")

    func(
        data=aoc_early,
        x="model",
        y="weighted_aoc",
        hue=hue,
        ax=axs[1],
        capsize=0.05,  # type: ignore
        errwidth=1,  # type: ignore
        order=x_order,
        **kwargs,  # type: ignore
    )
    axs[1].set_title("Early Answering")
    for ax in axs:
        ax.set_ylabel("Weighted AoC")
        ax.set_xlabel("Model")

    # filter onto the ones wihout mistakes and no truncation
    acc = df[~df.has_mistake]
    acc = acc[~acc.was_truncated]

    func(
        data=acc,
        x="model",
        y="is_correct",
        hue=hue,
        ax=axs[2],
        capsize=0.05,  # type: ignore
        errwidth=1,  # type: ignore
        order=x_order,
        **kwargs,  # type: ignore
    )
    axs[2].set_title("Accuracy for Complete Unmodified CoT")
    axs[2].set_ylabel("Accuracy")
    axs[2].set_xlabel("Model")

    # share the legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    for ax in axs:
        # remove the legend from the individual plots
        ax.get_legend().remove()


def accuracy(
    exp_dir: str,
    inconsistent_only: bool = True,
    stage_two_formatter_name: str = "EarlyAnsweringFormatter",
    aggregate_over_tasks: bool = False,
    step_filter: Optional[list[int]] = None,
):
    """
    This does a similar thing to the accuracy function in analysis.py, but it uses the stage_two data
    """
    df = get_data_frame_from_exp_dir(exp_dir)
    df = df[df.formatter_name == stage_two_formatter_name]
    print(df.columns)

    # replace formatter_name with stage_one_formatter_name
    # as we want to compare the accuracy of the stage_one formatter
    df["formatter_name"] = df["stage_one_formatter_name"]

    if step_filter:
        df = df[df.cot_trace_length.isin(step_filter)]
        check_counts = False
        # filtering on step means we no longer guarateed to have the same number of samples for each task
        # so we don't want to check the counts
    else:
        check_counts = True

    print("----- Accuracy for step = 0 --------------")
    no_cot_df = df[df["step_in_cot_trace"] == 0]  # type: ignore
    accuracy_for_df(
        no_cot_df,  # type: ignore
        inconsistent_only=inconsistent_only,
        check_counts=check_counts,
        aggregate_over_tasks=aggregate_over_tasks,
    )

    print("----- Accuracy for step = max_step --------------")
    cot_df = df[df["step_in_cot_trace"] == df["max_step_in_cot_trace"]]  # type: ignore
    accuracy_for_df(
        cot_df,  # type: ignore
        inconsistent_only=inconsistent_only,
        check_counts=check_counts,
        aggregate_over_tasks=aggregate_over_tasks,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "hist": plot_historgram_of_lengths,
            "early": plot_early_answering,
            "mistakes": plot_adding_mistakes,
            "accuracy": accuracy,
            "aoc": aoc_plot,
        }
    )
