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
from cot_transparency.formatters import name_to_formatter
from cot_transparency.formatters.transparency.trace_manipulation import get_cot_steps
from analysis import accuracy_for_df, TASK_MAP
import seaborn as sns
import numpy as np


def get_aoc(df: pd.DataFrame, hue: str, col: str, display_accuracies: bool = True):
    # get aoc for each original_cot_trace_length, grouped by task, and model
    # we want the counts for the number of traces so filter on unique stage_one_hash
    n_traces = df.groupby(["task_name", hue, col]).stage_one_hash.nunique().reset_index()
    n_traces = n_traces.rename(columns={"stage_one_hash": "n_traces"})

    # get aucs for each original_cot_trace_length, grouped by task, and model
    areas = (
        df.groupby(["task_name", hue, col, "cot_trace_length"]).apply(lambda x: x["same_answer"].mean()).reset_index()
    )

    def get_auc(group: pd.DataFrame) -> float:
        assert group[col].nunique() == 1
        proportion_of_cot = group["cot_trace_length"] / max(group["cot_trace_length"])
        auc = np.trapz(group[0], x=proportion_of_cot)
        return auc

    grouped = areas.groupby(["task_name", hue, col])
    aucs = grouped.apply(get_auc).reset_index()
    aucs = aucs.rename(columns={0: "auc"})
    areas = pd.merge(aucs, n_traces, on=["task_name", hue, col])
    print(areas)

    areas["weighted_auc"] = areas["auc"] * areas["n_traces"]
    areas = areas.groupby(["task_name", hue]).sum().reset_index()
    areas["weighted_auc"] = areas["weighted_auc"] / areas["n_traces"]
    areas["weighted_aoc"] = 1 - areas["weighted_auc"]
    print(areas)
    print(areas.to_csv())


def convert_stage2_experiment_to_dataframe(exp: StageTwoExperimentJsonFormat) -> pd.DataFrame:
    out = []
    for task_output in exp.outputs:
        d_with_config = get_general_metrics(task_output)
        d_with_config["model"] = task_output.task_spec.model_config.model
        d_with_config["task_name"] = task_output.task_spec.stage_one_output.task_spec.task_name
        d_with_config["ground_truth"] = task_output.task_spec.stage_one_output.task_spec.ground_truth
        d_with_config["stage_one_hash"] = task_output.task_spec.stage_one_output.task_spec.uid()
        d_with_config["stage_one_output_hash"] = task_output.task_spec.stage_one_output.uid()
        d_with_config["stage_one_output"] = task_output.task_spec.stage_one_output.dict()
        d_with_config["biased_ans"] = task_output.task_spec.stage_one_output.task_spec.biased_ans
        d_with_config["task_hash"] = task_output.task_spec.stage_one_output.task_spec.task_hash
        d_with_config["parsed_response"] = task_output.model_output.parsed_response
        if task_output.task_spec.trace_info:
            d_with_config["mistake_added_at"] = task_output.task_spec.trace_info.mistake_inserted_idx
            d_with_config["original_cot_trace_length"] = len(task_output.task_spec.trace_info.original_cot)
            modified_cot_length = get_cot_steps(task_output.task_spec.trace_info.get_complete_modified_cot())
            d_with_config["cot_trace_length"] = len(modified_cot_length)

        out.append(d_with_config)

    df = pd.DataFrame(out)

    stage_one_output = [TaskOutput(**i) for i in df["stage_one_output"]]
    stage_formatter = [i.task_spec.formatter_name for i in stage_one_output]
    df["stage_one_formatter_name"] = stage_formatter
    return df


def get_data_frame_from_exp_dir(exp_dir: str) -> pd.DataFrame:
    loaded_dict = ExpLoader.stage_two(exp_dir)
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
    return df


def plot_historgram_of_lengths(
    exp_dir: str,
):
    df = get_data_frame_from_exp_dir(exp_dir)

    hue = "task_name"
    x = "original_cot_trace_length"
    col = "model"
    y = "Counts"

    # for histogram we want the counts of the original_cot_trace_length
    # filter on the unique stage_one_hash
    counts = df.groupby([hue, col, x]).stage_one_hash.nunique().reset_index()
    counts = counts.rename(columns={"stage_one_hash": y})

    # facet plot of the proportion of the trace, break down by original_cot_trace_length
    g = sns.FacetGrid(counts, col=col, col_wrap=2, legend_out=True)
    g.map_dataframe(sns.barplot, x=x, y=y, hue=hue)
    g.add_legend()
    plt.show()


def plot_early_answering(
    exp_dir: str,
    show_plots: bool = False,
    inconsistent_only: bool = False,
    aggregate_over_tasks: bool = False,
    model_filter: Optional[str] = None,
    length_filter: Optional[list[int]] = None,
):
    df = get_data_frame_from_exp_dir(exp_dir)
    # drop formatters that have Mistake in the name
    df = df[~df.formatter_name.str.contains("Mistake")]

    if aggregate_over_tasks:
        # replace task_name with the "parent" task name using the task_map
        df["task_name"] = df["task_name"].replace(TASK_MAP)

    if inconsistent_only:
        df = df[df.biased_ans != df.ground_truth]
        print("Number of inconsistent tasks: ", len(df))

    if model_filter:
        df = df[df.model.isin(model_filter)]

    if length_filter:
        df = df[df["original_cot_trace_length"].isin(length_filter)]
        assert len(df) > 0, "No data for this length filter"

    def check_same_answer(group: pd.DataFrame) -> pd.DataFrame:
        max_step_row = group[group["cot_trace_length"] == group["original_cot_trace_length"]]  # type: ignore
        if len(max_step_row) != 1:
            group["same_answer"] = "NOT_FOUND"
        else:
            group["same_answer"] = group["parsed_response"] == max_step_row["parsed_response"].iloc[0]
        return group

    # Apply the check_same_answer function
    df = df.groupby("stage_one_hash").apply(check_same_answer).reset_index(drop=True)

    # Redrop any NOT_FOUND
    pre = len(df)
    df = df[df.same_answer != "NOT_FOUND"]
    print("Dropped ", pre - len(df), " rows becuase of NOT_FOUND answers in full cot trace")

    hue = "model"
    y = "same_answer"
    x = "proportion_of_cot"
    x_label = "% Proportion of CoT"
    y_label = "% Same answer as full CoT"
    col_label = "Original CoT length"

    df["proportion_of_cot"] = df["cot_trace_length"] / df["original_cot_trace_length"]
    df["proportion_of_cot"] = df["proportion_of_cot"].astype(float)

    # rename "original_cot_trace_length" to "CoT Length"
    df = df.rename(columns={"original_cot_trace_length": col_label})

    # facet plot of the proportion of the trace, break down by original_cot_trace_length
    g = sns.relplot(data=df, x=x, y=y, hue=hue, col=col_label, col_wrap=2, kind="line", height=4)
    g.set_axis_labels(x_label, y_label)

    # plot aggreaged version that aggregates over cot lenghts
    plt.figure()
    df["proportion_of_cot_rounded"] = df["proportion_of_cot"].round(1)
    g2 = sns.lineplot(df, x="proportion_of_cot_rounded", y=y, hue=hue, alpha=0.5)
    g2.set(xlabel=x_label, ylabel=y_label)

    get_aoc(df, hue, col_label)

    # baseline accuracies
    cot_trace_length_0 = df[df["cot_trace_length"] == 0]
    cot_trace_length_max = df[df["cot_trace_length"] == df[col_label]]
    baseline_accuracies = (
        cot_trace_length_0.groupby(["task_name", hue]).apply(lambda x: x["is_correct"].mean()).reset_index()
    )
    baseline_accuracies = baseline_accuracies.rename(columns={0: "No CoT Accuracy"})
    baseline_accuracies["CoT Accuracy"] = (
        cot_trace_length_max.groupby(["task_name", hue]).apply(lambda x: x["is_correct"].mean()).reset_index()[0]
    )
    print(baseline_accuracies)
    # print csv version
    print(baseline_accuracies.to_csv())

    if show_plots:
        plt.show()


def plot_adding_mistakes(
    exp_dir: str,
    show_plots: bool = False,
    inconsistent_only: bool = False,
    aggregate_over_tasks: bool = False,
    model_filter: Optional[str] = None,
    length_filter: Optional[list[int]] = None,
):
    df = get_data_frame_from_exp_dir(exp_dir)
    df = df[df.formatter_name.str.contains("FullCOT")]

    if aggregate_over_tasks:
        # replace task_name with the "parent" task name using the task_map
        df["task_name"] = df["task_name"].replace(TASK_MAP)

    if inconsistent_only:
        df = df[df.biased_ans != df.ground_truth]
        print("Number of inconsistent tasks: ", len(df))

    if model_filter:
        df = df[df.model.isin(model_filter)]

    if length_filter:
        df = df[df["original_cot_trace_length"].isin(length_filter)]
        assert len(df) > 0, "No data for this length filter"

    # Apply the check_same_answer function
    def is_same_as_no_mistakes(group: pd.DataFrame) -> pd.DataFrame:
        formatters = group.apply(lambda x: name_to_formatter(x["formatter_name"]), axis=1)
        has_mistake = formatters.apply(lambda x: x.has_mistake)
        # only one of the formatters should have a mistake
        if (~has_mistake).sum() != 1:
            group["same_answer"] = "NOT_FOUND"
        else:
            answer_no_mistake = group[~has_mistake]["parsed_response"].iloc[0]
            group["has_mistake"] = has_mistake
            group["same_answer"] = group["parsed_response"] == answer_no_mistake
        return group

    df = df.groupby("stage_one_hash").apply(is_same_as_no_mistakes).reset_index(drop=True)

    df["proportion_of_cot"] = df["mistake_added_at"] / df["original_cot_trace_length"]
    df["proportion_of_cot"] = df["proportion_of_cot"].astype(float)

    # redrop any NOT_FOUND
    df = df[df.same_answer != "NOT_FOUND"]

    # drop all the rows where there is no mistake
    df = df[df.has_mistake]

    hue = "model"
    y = "same_answer"
    x = "proportion_of_cot"
    x_label = "% Proportion of CoT before mistake"
    y_label = "% Same answer as no mistake"
    col_label = "Original CoT length"

    # rename "original_cot_trace_length" to "CoT Length"
    df = df.rename(columns={"original_cot_trace_length": col_label})

    # facet plot of the proportion of the trace, break down by original_cot_trace_length
    g = sns.relplot(data=df, x=x, y=y, hue=hue, col=col_label, col_wrap=2, kind="line", height=4)
    g.set_axis_labels(x_label, y_label)

    # plot aggreaged version that aggregates over cot lenghts
    plt.figure()
    df["proportion_of_cot_rounded"] = df["proportion_of_cot"].round(1)
    g2 = sns.lineplot(df, x="proportion_of_cot_rounded", y=y, hue=hue, alpha=0.5)
    g2.set(xlabel=x_label, ylabel=y_label)

    get_aoc(df, hue, col_label)

    if show_plots:
        plt.show()


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
    no_cot_df = df[df["step_in_cot_trace"] == 0]
    accuracy_for_df(
        no_cot_df,
        inconsistent_only=inconsistent_only,
        check_counts=check_counts,
        aggregate_over_tasks=aggregate_over_tasks,
    )

    print("----- Accuracy for step = max_step --------------")
    cot_df = df[df["step_in_cot_trace"] == df["max_step_in_cot_trace"]]
    accuracy_for_df(
        cot_df,
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
        }
    )
