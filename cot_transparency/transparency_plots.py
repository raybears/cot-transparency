import math
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def check_same_answer(group: pd.DataFrame, formatter_name_for_max: Optional[str] = None) -> pd.DataFrame:
    breakpoint()
    max_step_row = group[group["cot_trace_length"] == group["original_cot_trace_length"]]  # type: ignore
    assert len(max_step_row) <= 1, "There should only be one row with max step"
    group["same_answer"] = group["parsed_response"] == max_step_row["parsed_response"].iloc[0]
    return group


def plot_cot_trace_simple(df: pd.DataFrame):
    # round each proportion of cot to nearest 0.1 and then group by that

    df["proportion_of_cot"] = df["proportion_of_cot"].round(1)

    sns.lineplot(df, x="proportion_of_cot", y="same_answer", hue="model", alpha=0.5)


def plot_cot_trace(
    df: pd.DataFrame, step_filter: list[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10], color_by_model: bool = False
):
    df = df.copy()

    df = df[df["original_cot_trace_length"].isin(step_filter)]

    print("---------------- Counts -----------------")
    count_groups = ["task_name", "original_cot_trace_length", "stage_one_formatter_name", "model"]
    # filter on n_steps_in_cot_trace == max as we only want to count the number of cot_traces
    counts = df[df.cot_trace_length == df.original_cot_trace_length]
    counts = counts.groupby(count_groups)["same_answer"].count().reset_index()

    # rename same_answer
    counts = counts.rename(columns={"same_answer": "num_samples"})
    counts_pivot = counts.pivot(
        index=["task_name", "stage_one_formatter_name", "model"],  # type: ignore
        columns="original_cot_trace_length",  # type: ignore
        values="num_samples",  # type: ignore
    )  # type: ignore
    print(counts_pivot)
    breakpoint()

    unique_plots = df["stage_one_formatter_name"].unique()
    title = "{task}-step CoT"

    y_axis_limits = [i * 100 for i in [-0.05, 1.05]]

    aocs = []
    for plot in unique_plots:
        ncols = 3
        nrows = math.ceil(len(step_filter) / ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        filtered_df = df[df["stage_one_formatter_name"] == plot]
        proportion_df = (
            filtered_df.groupby(["task_name", "cot_trace_length", "original_cot_trace_length", "model"])["same_answer"]
            .mean()
            .reset_index()
        )
        accuracy_df = (
            filtered_df.groupby(["task_name", "cot_trace_length", "original_cot_trace_length", "model"])["is_correct"]
            .mean()
            .reset_index()
        )

        # Colors and line styles
        n_tasks = df["task_name"].nunique()
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]  # type: ignore
        if n_tasks > len(colors):
            cmap = plt.get_cmap("tab20")
            colors = [cmap(i) for i in np.linspace(0, 1, n_tasks)]
        line_syles = ["-", "--", "-.", ":"]

        lines = []  # to store the line objects for the legend
        labels = []  # to store the labels for the legend
        model_labels = []
        model_lines = []
        for model_idx, model in enumerate(df["model"].unique()):
            line_style = line_syles
            for i, length in enumerate(step_filter):
                ax = axs.flatten()[i]
                for task_idx, task_name in enumerate(df["task_name"].unique()):
                    if color_by_model:
                        color_idx = model_idx
                        line_style_idx = task_idx
                    else:
                        color_idx = task_idx
                        line_style_idx = model_idx
                    data = proportion_df[
                        (proportion_df["cot_trace_length"] == length)
                        & (proportion_df["task_name"] == task_name)
                        & (proportion_df["model"] == model)
                    ]
                    label = f"{task_name}"

                    step_in_cot_percent = data["cot_trace_length"] / (length - 1) * 100
                    same_answer_percent = data["same_answer"] * 100
                    line = ax.plot(
                        step_in_cot_percent,
                        same_answer_percent,
                        color=colors[color_idx],
                        linestyle=line_style[line_style_idx],
                    )

                    ax.scatter(step_in_cot_percent, same_answer_percent, color=colors[color_idx], s=3, marker="x")

                    acc_data = accuracy_df[
                        (accuracy_df["task_name"] == task_name) & (accuracy_df["cot_trace_length"] == length)
                    ]
                    acc_with_no_cot = acc_data[acc_data["cot_trace_length"] == 0]["is_correct"]
                    acc_with_cot = acc_data[acc_data["cot_trace_length"] == length - 1]["is_correct"]

                    # convert acc to float and multiply by 100 if not empty array otherwise None
                    acc_with_no_cot = acc_with_no_cot.values[0] * 100 if len(acc_with_no_cot) > 0 else None
                    acc_with_cot = acc_with_cot.values[0] * 100 if len(acc_with_cot) > 0 else None

                    # store the data for aoc calculations
                    # Store X and Y coordinates
                    aoc_dict = dict(
                        model=model,
                        task_name=task_name,
                        stage_one_formatter_name=plot,
                        original_cot_trace_length=length,
                        x_values=step_in_cot_percent.values,
                        y_values=same_answer_percent.values,
                        acc_with_no_cot=acc_with_no_cot,
                        acc_with_cot=acc_with_cot,
                    )
                    aocs.append(aoc_dict)

                    if model not in model_labels:
                        model_labels.append(model)
                        # Add a label for the model
                        if color_by_model:
                            model_line = ax.plot([], [], color=colors[color_idx])
                        else:
                            model_line = ax.plot([], [], color="black", linestyle=line_style[model_idx], label=model)
                        model_lines.append(model_line[0])

                    # Avoid adding duplicate lines/labels
                    if label not in labels:
                        lines.append(line[0])
                        labels.append(label)

                ax.set_title(title.format(task=length))
                ax.set_ylim(y_axis_limits)
                ax.grid(True)

        # Add a shared y-label
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.ylabel("% Same Answer as Complete CoT", labelpad=15)
        plt.xlabel("% of Reasoning Sample Provided", labelpad=10)
        plt.title(f"Formatter: {plot}", pad=40)

        # Create the legend from the list of Line2D objects
        # Adjust the position of the legend, add it to the figure, not the axes
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.14, left=0.06, right=0.85)  # Make more space for the legend and the x, y labels
        fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), fancybox=True, shadow=False, ncol=6)
        fig.legend(
            model_lines,
            model_labels,
            loc="center right",
            fancybox=True,
            shadow=False,
            ncol=1,
        )

    # calculaate aocs
    aoc_df = pd.DataFrame(aocs)
    aoc_df["aoc"] = aoc_df.apply(lambda x: aoc_calculation(x["x_values"], x["y_values"]), axis=1)
    # join with counts on task_name, original_cot_trace_length, stage_one_formatter_name
    aoc_df = aoc_df.merge(counts, on=count_groups)

    # aggregate over cot_trace_lenght weighting the aoc by the number of samples
    # Create a new column for the weighted AOC
    aoc_df["weighted_aoc"] = aoc_df["aoc"] * aoc_df["num_samples"]
    aoc_df["no_cot_acc"] = aoc_df["acc_with_no_cot"] * aoc_df["num_samples"]
    aoc_df["cot_acc"] = aoc_df["acc_with_cot"] * aoc_df["num_samples"]

    for formatter_name in aoc_df["stage_one_formatter_name"].unique():
        filtered_df = aoc_df[aoc_df["stage_one_formatter_name"] == formatter_name]

        # Group by the desired columns and sum the weighted AOC,
        # then divide by the total number of samples for each group
        aoc_df_grouped = (
            filtered_df.groupby(["task_name", "model"])
            .agg(
                {
                    "weighted_aoc": "sum",
                    "no_cot_acc": "sum",
                    "cot_acc": "sum",
                    "num_samples": "sum",
                }
            )
            .reset_index()
        )
        aoc_df_grouped["weighted_aoc"] = aoc_df_grouped["weighted_aoc"] / aoc_df_grouped["num_samples"]
        aoc_df_grouped["no_cot_acc"] = aoc_df_grouped["no_cot_acc"] / aoc_df_grouped["num_samples"]
        aoc_df_grouped["cot_acc"] = aoc_df_grouped["cot_acc"] / aoc_df_grouped["num_samples"]

        print(f"---------------- AOC for {formatter_name} -----------------")
        aoc_df_grouped["acc_delta"] = aoc_df_grouped["cot_acc"] - aoc_df_grouped["no_cot_acc"]
        print(aoc_df_grouped.set_index("task_name"))


def aoc_calculation(x: np.ndarray[float], y: np.ndarray[float]):  # type: ignore
    # calculate auc then do total area - auc to get aoc
    auc = np.trapz(y, x) / (100 * 100)
    total_area = 1
    aoc = total_area - auc
    return aoc
