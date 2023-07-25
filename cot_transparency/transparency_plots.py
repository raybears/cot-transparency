from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def add_max_step_in_cot_trace(df: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicate stage_one_hash rows
    df["max_step_in_cot_trace"] = df.groupby("stage_one_hash")["step_in_cot_trace"].transform("max")
    df["cot_trace_length"] = df["max_step_in_cot_trace"] + 1
    return df


def plot_historgram_of_cot_steps(df: pd.DataFrame, filter_at_step: Optional[int] = None):
    df = add_max_step_in_cot_trace(df)

    n_subplots = df["model"].nunique()

    n_cols = 2
    n_rows = n_subplots // n_cols + 1
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

    unique_df = df.drop_duplicates(subset="stage_one_hash")
    if filter_at_step is not None:
        unique_df = unique_df[unique_df["max_step_in_cot_trace"] <= filter_at_step]  # type: ignore

    flat_axs = axs.flatten()
    legend_labels = df["task_name"].unique()
    handles = []

    for i, model in enumerate(df["model"].unique()):
        model_df = unique_df[unique_df["model"] == model]  # type: ignore
        ax = flat_axs[i]
        num_bins = 20

        # Plot a histogram for each task_name
        ax = sns.histplot(
            data=model_df,
            x="max_step_in_cot_trace",
            hue="task_name",
            multiple="stack",
            bins=num_bins,  # type: ignore
            shrink=0.8,  # type: ignore
            ax=ax,
            legend=False,  # Prevent each subplot from generating its own legend
        )  # type: ignore

        # set the x-axis ticks to be at integers
        ax.set_xticks(np.arange(0, 20, 1.0))

        ax.set_title(f"{model}")
        ax.set_xlabel("Number of steps in COT trace")
        ax.set_ylabel("Frequency")

        # Add the patches from this ax to the handles
        # patches are n_bins then n_tasks
        patches = np.array(ax.patches).reshape(len(df["task_name"].unique()), num_bins)
        handles.extend(patches[:, 0])

    # Sort handles based on their original order in the 'task_name' column
    legend_labels = df["task_name"].unique()  # Keep track of unique labels for the legend
    handles = [handle for _, handle in sorted(zip(legend_labels, handles))]

    # Add a shared legend for the entire figure
    fig.legend(handles, legend_labels, title="Task name", loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.14)  # Make more space for the legend and the x, y labels

    # Remove the unused subplots
    for i in range(n_subplots, n_cols * n_rows):
        fig.delaxes(axs.flatten()[i])

    plt.show()


def check_same_answer(group: pd.DataFrame) -> pd.DataFrame:
    max_step_row = group[group["step_in_cot_trace"] == group["max_step_in_cot_trace"]]
    if len(max_step_row) > 0:  # To handle case where there's no row with max step
        group["same_answer"] = group["parsed_response"] == max_step_row["parsed_response"].iloc[0]
    else:
        group["same_answer"] = False  # or assign some other default value
    return group


def plot_cot_trace(df: pd.DataFrame, step_filter: list[int] = [3, 4, 5, 6]):
    df = df.copy()

    df = df[df["cot_trace_length"].isin(step_filter)]

    print("---------------- Counts -----------------")
    count_groups = ["task_name", "cot_trace_length", "stage_one_formatter_name"]
    # filter on step_in_cot_trace == 0 as we only want to count the number of cot_traces
    counts = df[df.step_in_cot_trace == 0]
    counts = counts.groupby(count_groups)["same_answer"].count().reset_index()

    # rename same_answer
    counts = counts.rename(columns={"same_answer": "num_samples"})
    counts_pivot = counts.pivot(
        index=["task_name", "stage_one_formatter_name"],  # type: ignore
        columns="cot_trace_length",  # type: ignore
        values="num_samples",  # type: ignore
    )  # type: ignore
    print(counts_pivot)

    unique_plots = df["stage_one_formatter_name"].unique()
    title = "{task}-step CoT"

    y_axis_limits = [i * 100 for i in [-0.05, 1.05]]

    aocs = []
    for plot in unique_plots:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        filtered_df = df[df["stage_one_formatter_name"] == plot]
        proportion_df = (
            filtered_df.groupby(["task_name", "cot_trace_length", "step_in_cot_trace"])["same_answer"]
            .mean()
            .reset_index()
        )
        accuracy_df = (
            filtered_df.groupby(["task_name", "cot_trace_length", "step_in_cot_trace"])["is_correct"]
            .mean()
            .reset_index()
        )

        lines = []  # to store the line objects for the legend
        labels = []  # to store the labels for the legend
        for i, length in enumerate(step_filter):
            ax = axs[i // 2, i % 2]
            for task_name in df["task_name"].unique():
                data = proportion_df[
                    (proportion_df["cot_trace_length"] == length) & (proportion_df["task_name"] == task_name)
                ]
                label = f"{task_name}"

                step_in_cot_percent = data["step_in_cot_trace"] / (length - 1) * 100
                same_answer_percent = data["same_answer"] * 100
                line = ax.plot(step_in_cot_percent, same_answer_percent)
                ax.scatter(step_in_cot_percent, same_answer_percent)

                acc_data = accuracy_df[
                    (accuracy_df["task_name"] == task_name) & (accuracy_df["cot_trace_length"] == length)
                ]
                acc_with_no_cot = acc_data[acc_data["step_in_cot_trace"] == 0]["is_correct"]
                acc_with_cot = acc_data[acc_data["step_in_cot_trace"] == length - 1]["is_correct"]

                # convert acc to float and multiply by 100 if not empty array otherwise None
                acc_with_no_cot = acc_with_no_cot.values[0] * 100 if len(acc_with_no_cot) > 0 else None
                acc_with_cot = acc_with_cot.values[0] * 100 if len(acc_with_cot) > 0 else None

                # store the data for aoc calculations
                # Store X and Y coordinates
                aoc_dict = dict(
                    task_name=task_name,
                    stage_one_formatter_name=plot,
                    cot_trace_length=length,
                    x_values=step_in_cot_percent.values,
                    y_values=same_answer_percent.values,
                    acc_with_no_cot=acc_with_no_cot,
                    acc_with_cot=acc_with_cot,
                )
                aocs.append(aoc_dict)

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
        fig.subplots_adjust(bottom=0.14, left=0.06)  # Make more space for the legend and the x, y labels
        fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), fancybox=True, shadow=False, ncol=6)

    # calculaate aocs
    aoc_df = pd.DataFrame(aocs)
    aoc_df["aoc"] = aoc_df.apply(lambda x: aoc_calculation(x["x_values"], x["y_values"]), axis=1)
    # join with counts on task_name, cot_trace_length, stage_one_formatter_name
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
            filtered_df.groupby(["task_name"])
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
