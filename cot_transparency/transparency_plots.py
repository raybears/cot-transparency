import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from cot_transparency.data_models.models import TaskOutput


def add_max_step_in_cot_trace(df: pd.DataFrame) -> pd.DataFrame:
    # Remove duplicate stage_one_hash rows
    df["max_step_in_cot_trace"] = df.groupby("stage_one_hash")["step_in_cot_trace"].transform("max")
    return df


def plot_historgram_of_cot_steps(df: pd.DataFrame):
    df = add_max_step_in_cot_trace(df)

    unique_df = df.drop_duplicates(subset="stage_one_hash")

    # Plot a histogram for each task_name
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=unique_df,
        x="max_step_in_cot_trace",
        hue="task_name",
        multiple="stack",
        bins=20,  # type: ignore
        shrink=0.8,  # type: ignore
    )  # type: ignore

    # set the x-axis ticks to be at integers
    plt.xticks(np.arange(0, 20, 1.0))

    plt.xlabel("Number of steps in COT trace")
    plt.ylabel("Frequency")
    plt.show()


def check_same_answer(group: pd.DataFrame) -> pd.DataFrame:
    max_step_row = group[group["step_in_cot_trace"] == group["max_step_in_cot_trace"]]
    if len(max_step_row) > 0:  # To handle case where there's no row with max step
        group["same_answer"] = group["parsed_response"] == max_step_row["parsed_response"].iloc[0]
    else:
        group["same_answer"] = False  # or assign some other default value
    return group


def plot_cot_trace(df: pd.DataFrame):
    df = df.copy()

    df = df[df["cot_trace_length"].isin([3, 4, 5, 6])]
    stage_one_output = [TaskOutput(**i) for i in df["stage_one_output"]]
    stage_formatter = [i.task_spec.formatter_name for i in stage_one_output]
    df["stage_one_formatter_name"] = stage_formatter

    print("---------------- Counts -----------------")

    counts = df.groupby(["task_name", "cot_trace_length", "stage_one_formatter_name"])["same_answer"].count()
    counts_pivot = counts.reset_index().pivot(
        index=["task_name", "stage_one_formatter_name"],  # type: ignore
        columns="cot_trace_length",  # type: ignore
        values="same_answer",  # type: ignore
    )  # type: ignore
    print(counts_pivot)

    unique_plots = df["stage_one_formatter_name"].unique()
    title = "{task}-step CoT"

    y_axis_limits = [i * 100 for i in [0.45, 1.05]]

    for plot in unique_plots:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        plt.subplots_adjust(bottom=0.4, left=0.15)  # Make more space for the labels
        filtered_df = df[df["stage_one_formatter_name"] == plot]
        proportion_df = (
            filtered_df.groupby(["task_name", "cot_trace_length", "step_in_cot_trace"])["same_answer"]
            .mean()
            .reset_index()
        )

        lines = []  # to store the line objects for the legend
        labels = []  # to store the labels for the legend
        for i, length in enumerate([3, 4, 5, 6]):
            ax = axs[i // 2, i % 2]
            for task_name in df["task_name"].unique():
                data = proportion_df[
                    (proportion_df["cot_trace_length"] == length) & (proportion_df["task_name"] == task_name)
                ]
                label = f"{task_name}"

                step_in_cot_percent = data["step_in_cot_trace"] / (length - 1) * 100
                line = ax.plot(step_in_cot_percent, data["same_answer"] * 100)
                ax.scatter(step_in_cot_percent, data["same_answer"] * 100)
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
        fig.legend(lines, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), fancybox=True, shadow=False, ncol=5)

        plt.tight_layout()
