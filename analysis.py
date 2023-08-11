import math
import fire
from matplotlib import pyplot as plt
from cot_transparency.data_models.models import (
    StageTwoTaskOutput,
    TaskOutput,
)
import pandas as pd
from typing import Any, Optional, List, Union, Sequence
from cot_transparency.data_models.io import ExpLoader
from cot_transparency.formatters import name_to_formatter
from scripts.multi_accuracy import plot_accuracy_for_exp
import seaborn as sns
import numpy as np

from stage_one import TASK_LIST

TASK_MAP = {}
for dataset, task_list in TASK_LIST.items():
    for task in task_list:
        TASK_MAP[task] = dataset

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


def get_general_metrics(task_output: Union[TaskOutput, StageTwoTaskOutput]) -> dict[str, Any]:
    d = task_output.dict()
    d["input_hash"] = task_output.task_spec.uid()
    d["output_hash"] = task_output.uid()
    config = task_output.task_spec.model_config
    task_spec = task_output.task_spec
    d.pop("task_spec")
    d.pop("model_output")
    d_with_config = {**d, **config.dict(), **task_spec.dict()}
    return d_with_config


def get_data_frame_from_exp_dir(exp_dir: str) -> pd.DataFrame:
    loaded_dict = ExpLoader.stage_one(exp_dir)
    out = []
    for exp in loaded_dict.values():
        for task_output in exp.outputs:
            d_with_config = get_general_metrics(task_output)
            model_output = task_output.model_output
            combined_d = {**d_with_config, **model_output.dict()}
            out.append(combined_d)
    df = pd.DataFrame(out)
    df["is_correct"] = (df.parsed_response == df.ground_truth).astype(int)

    def is_biased(formatter_name: str):
        formatter = name_to_formatter(formatter_name)
        return formatter.is_biased

    df["is_biased"] = df.formatter_name.map(is_biased)
    return df


def accuracy(
    exp_dir: str,
    inconsistent_only: bool = True,
    aggregate_over_tasks: bool = False,
    model_filter: Optional[str] = None,
    formatters: Sequence[str] = [],
    check_counts: bool = True,
) -> Optional[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    exp_dir: path to directory containing experiment jsons
    inconsistent_only: if True, only include inconsistent tasks where biased ans and correct ans are different
    """
    df = get_data_frame_from_exp_dir(exp_dir)
    done = accuracy_for_df(
        df,
        inconsistent_only=inconsistent_only,
        aggregate_over_tasks=aggregate_over_tasks,
        formatters=formatters,
        model_filter=model_filter,
        check_counts=check_counts,
    )
    # write
    done.to_csv("accuracy.csv")


def apply_filters(
    inconsistent_only: Optional[bool],
    model_filter: Optional[str],
    formatters: Sequence[str],
    aggregate_over_tasks: bool,
    df: pd.DataFrame,
) -> pd.DataFrame:
    if inconsistent_only:
        df = df[df.biased_ans != df.ground_truth]

    if model_filter:
        # check that df.model contains model_filter
        df = df[df.model.str.contains(model_filter)]
    if formatters:
        # check that df.formatter_name is in formatters
        df = df[df.formatter_name.isin(formatters)]
        assert len(df) > 0, f"formatters {formatters} not found in {df.formatter_name.unique()}"

    if aggregate_over_tasks:
        # replace task_name with the "parent" task name using the task_map
        df["task_name"] = df["task_name"].replace(TASK_MAP)

    return df


def accuracy_for_df(
    df: pd.DataFrame,
    inconsistent_only: bool = True,
    aggregate_over_tasks: bool = False,
    model_filter: Optional[str] = None,
    check_counts: bool = True,
    formatters: Sequence[str] = [],
) -> pd.DataFrame:
    """
    inconsistent_only: if True, only include inconsistent tasks where biased ans and correct ans are different
    """
    df = apply_filters(
        inconsistent_only=inconsistent_only,
        model_filter=model_filter,
        formatters=formatters,
        aggregate_over_tasks=aggregate_over_tasks,
        df=df,
    )
    # if df["intervention_name"] is not null, add an arrow -> to it
    df["intervention_name"] = df["intervention_name"].fillna("")
    # add "<-" if intervention_name is not null
    df["intervention_name"] = df["intervention_name"].apply(lambda x: "<-" + x if x else x)

    # add formatter_name and intervention_name together
    df["formatter_name"] = df["formatter_name"] + df["intervention_name"]

    groups = ["task_name", "model", "formatter_name"]
    accuracy_df_grouped = df[["is_correct", "task_name", "model", "formatter_name"]].groupby(groups)
    accuracy_df = accuracy_df_grouped.mean()

    # add the standard error
    accuracy_standard_error = accuracy_df_grouped.sem()
    accuracy_df["accuracy_standard_error"] = accuracy_standard_error["is_correct"]
    accuracy_df = accuracy_df.reset_index()

    counts_df = accuracy_df_grouped.count().reset_index()

    # count the number of repeats by counting the number task hashes
    counts_df["unique_questions"] = df.groupby(groups)["task_hash"].nunique().reset_index()["task_hash"]
    counts_df["total_samples"] = df.groupby(groups)["is_correct"].count().reset_index()["is_correct"]

    unique_questions_df: pd.DataFrame = pivot_df(
        counts_df,
        values=["unique_questions"],
    )[
        "unique_questions"
    ]  # type: ignore
    counts_pivot: pd.DataFrame = pivot_df(counts_df, values=["total_samples"])["total_samples"]  # type: ignore
    accuracy_pivot = pivot_df(accuracy_df)

    if check_counts:
        if not (counts_are_equal(counts_pivot) and counts_are_equal(unique_questions_df)):
            print("Counts are not equal for some tasks and their baselines, likely experiments not completed")
            exit(1)

    print("---------------- Counts ----------------")
    print(counts_pivot)
    print("--------------- Unique Questions ---------------")
    print(unique_questions_df)
    print("--------------- Accuracy ---------------")
    print(accuracy_pivot * 100)

    return accuracy_df


def pivot_df(df: pd.DataFrame, values: List[str] = ["is_correct"]):
    print("here2")
    print(df)
    df = df.copy()
    df["formatter_name"] = df["formatter_name"].str.replace("Formatter", "")

    output = pd.pivot_table(df, index=["task_name", "model"], columns=["formatter_name"], values=values)
    return output


def counts_are_equal(count_df: pd.DataFrame) -> bool:
    """
    Verify that the counts are the same for all columns in the count_df
    """
    return (count_df.nunique(axis=1) == 1).all()


def generic_bar_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: Optional[str] = None,
    subplot: Optional[str] = None,
    ncols: int = 2,
    **kwargs: dict[str, Any],
):
    # assert all temperatures are the same
    assert df.temperature.nunique() == 1
    temperature = df.temperature.unique()[0]
    # add temperature to model
    df["model"] = df["model"] + f" (T={temperature})"

    sns.set_theme(style="whitegrid")
    if subplot is None:
        sns.barplot(data=df, x=x, y=y, hue=hue)

    n_subplots = len(df[subplot].unique())
    nrows = math.ceil(n_subplots / ncols)
    ncols = min(ncols, n_subplots)
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(4 * ncols, (4 * nrows) * 1 / 0.75))
    if n_subplots == 1:
        axs = np.array([axs])
    flat_axs = axs.flatten()
    for i, sp in enumerate(df[subplot].unique()):
        ax = flat_axs[i]
        sns.barplot(data=df[df.model == sp], x=x, y=y, hue=hue, ax=ax, capsize=0.05, **kwargs)  # type: ignore
        ax.set_title(sp)

    # grab legend from last plot and move it to the right
    handles, labels = ax.get_legend_handles_labels()  # type: ignore
    fig.legend(handles, labels, loc="lower center", ncol=2)

    # delete legends on subplots
    for i in range(n_subplots):
        flat_axs[i].get_legend().remove()

    # delete unused plots
    for i in range(n_subplots, len(flat_axs)):
        fig.delaxes(flat_axs[i])

    fig.tight_layout()  # type: ignore
    fig.subplots_adjust(bottom=0.25)
    plt.show()


def simple_plot(
    exp_dir: str,
    inconsistent_only: bool = True,
    aggregate_over_tasks: bool = False,
    model_filter: Optional[str] = None,
    formatters: Sequence[str] = [],
    x: str = "task_name",
    y: str = "Accuracy",
    hue: str = "formatter_name",
    subplot: str = "model",
):
    df = get_data_frame_from_exp_dir(exp_dir)
    df = apply_filters(
        inconsistent_only=inconsistent_only,
        model_filter=model_filter,
        formatters=formatters,
        aggregate_over_tasks=aggregate_over_tasks,
        df=df,
    )

    # remove Unbiased or Sycophancy from formatter name
    df["formatter_name"] = df["formatter_name"].str.replace("Formatter", "")
    df["formatter_name"] = df["formatter_name"].str.replace("Unbiased", "")
    df["formatter_name"] = df["formatter_name"].str.replace("Sycophancy", "")

    # rename is_correct to Accuracy
    df = df.rename(columns={"is_correct": "Accuracy"})
    generic_bar_plot(df, x=x, y=y, hue=hue, subplot=subplot, ncols=2)


def point_plot(
    exp_dir: str, inconsistent_only: bool = True, model_filter: Optional[str] = None, formatters: Sequence[str] = []
):
    df = get_data_frame_from_exp_dir(exp_dir)
    df = apply_filters(
        inconsistent_only=inconsistent_only,
        model_filter=model_filter,
        formatters=formatters,
        aggregate_over_tasks=False,
        df=df,
    )

    root_mapping = {
        "ZeroShotCOTUnbiasedFormatter": ("ZeroShot", "Unbiased", "COT"),
        "ZeroShotCOTSycophancyFormatter": ("ZeroShot", "Sycophancy", "COT"),
        "ZeroShotUnbiasedFormatter": ("ZeroShot", "Unbiased", "No-COT"),
        "ZeroShotSycophancyFormatter": ("ZeroShot", "Sycophancy", "No-COT"),
    }
    # adds these columns to the accuracy_df
    df["root"] = df.formatter_name.map(lambda x: root_mapping[x][0])
    df["Bias"] = df.formatter_name.map(lambda x: root_mapping[x][1])
    df["CoT"] = df.formatter_name.map(lambda x: root_mapping[x][2])

    # rename is_correct to Accuracy
    df["Accuracy (%)"] = df["is_correct"] * 100
    df = df.rename(columns={"model": "Model"})

    sns.catplot(
        data=df,
        x="CoT",
        y="Accuracy (%)",
        hue="Bias",
        col="Model",
        capsize=0.05,
        errwidth=1,
        join=False,
        kind="point",
    )

    plt.show()


if __name__ == "__main__":
    fire.Fire(
        {
            "accuracy": accuracy,
            "accuracy_plot": plot_accuracy_for_exp,
            "simple_plot": simple_plot,
            "point_plot": point_plot,
        }
    )
