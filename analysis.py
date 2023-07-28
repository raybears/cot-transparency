import fire
from matplotlib import pyplot as plt
from cot_transparency.data_models.models import (
    StageTwoTaskOutput,
    TaskOutput,
)
import pandas as pd
from typing import Any, Optional, List, Union, Sequence
from cot_transparency.data_models.io import ExpLoader
from scripts.multi_accuracy import plot_accuracy_for_exp

from stage_one import TASK_LIST

TASK_MAP = {}
for dataset, task_list in TASK_LIST.items():
    for task in task_list:
        TASK_MAP[task] = dataset


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
    accuracy_for_df(
        df,
        inconsistent_only=inconsistent_only,
        aggregate_over_tasks=aggregate_over_tasks,
        formatters=formatters,
        model_filter=model_filter,
        check_counts=check_counts,
    )


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

    if inconsistent_only:
        df = df[df.biased_ans != df.ground_truth]
    print(df.shape)

    if model_filter:
        # check that df.model contains model_filter
        df = df[df.model.str.contains(model_filter)]
    if formatters:
        # check that df.formatter_name is in formatters
        df = df[df.formatter_name.isin(formatters)]

    if aggregate_over_tasks:
        # replace task_name with the "parent" task name using the task_map
        df["task_name"] = df["task_name"].replace(TASK_MAP)

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


def miles_graph(exp_dir: str, check_counts: bool = True, z_value: float = 1.96):
    df = get_data_frame_from_exp_dir(exp_dir)
    accuracy_df = accuracy_for_df(
        df,
        inconsistent_only=True,
        aggregate_over_tasks=True,
        formatters=[],
        model_filter=None,
        check_counts=check_counts,
    )

    # this maps to "root, bias_type, cot_type"
    root_mapping = {
        "ZeroShotCOTUnbiasedFormatter": ("ZeroShot", None, "COT"),
        "ZeroShotCOTSycophancyFormatter": ("ZeroShot", "Sycophancy", "COT"),
        "ZeroShotUnbiasedFormatter": ("ZeroShot", None, "No-COT"),
        "ZeroShotSycophancyFormatter": ("ZeroShot", "Sycophancy", "No-COT"),
    }

    # adds these columns to the accuracy_df
    accuracy_df["root"] = accuracy_df.formatter_name.map(lambda x: root_mapping[x][0])
    accuracy_df["bias_type"] = accuracy_df.formatter_name.map(lambda x: root_mapping[x][1])
    accuracy_df["cot_type"] = accuracy_df.formatter_name.map(lambda x: root_mapping[x][2])

    accuracy_df["error_min"] = accuracy_df["accuracy_standard_error"] * z_value
    accuracy_df["error_max"] = accuracy_df["accuracy_standard_error"] * z_value

    print(accuracy_df)
    # Get the unique models
    models = accuracy_df.model.unique()

    # Create a new column for formatter type

    # Loop over all models
    for model in models:
        n_subplots = len(accuracy_df.root.unique())

        fig, axs = plt.subplots(nrows=n_subplots, figsize=(4, 4 * n_subplots))

        # subplots based on root
        for i, root in enumerate(accuracy_df.root.unique()):
            ax: plt.Axes
            if n_subplots == 1:
                ax = axs  # type: ignore
            else:
                ax = axs[i]  # type: ignore

            s = 100
            data = accuracy_df[(accuracy_df.model == model)]
            # First scatterplot for higher portion of dumbbell
            data_high2 = data[(data.root == root) & (data.bias_type.isna())]
            ax.scatter(x=data_high2["cot_type"], y=data_high2["is_correct"], color="blue", s=s)
            ax.errorbar(
                x=data_high2["cot_type"],
                y=data_high2["is_correct"],
                yerr=data_high2["error_min"],
                fmt="none",
                color="black",
                ecolor="black",
                elinewidth=3,
            )

            # Second scatterplot for lower portion of dumbbell
            data_low = data[(data.root == root) & (~data.bias_type.isna())]
            ax.scatter(x=data_low["cot_type"], y=data_low["is_correct"], color="red", s=s)
            ax.errorbar(
                x=data_low["cot_type"],
                y=data_low["is_correct"],
                yerr=data_low["error_max"],
                fmt="none",
                color="black",
                ecolor="black",
                elinewidth=3,
            )

            ax.set_title(f"{model}, {root}")
            ax.set_ylabel("Score")
            # move the x limits wider
            ax.set_xlim(ax.get_xlim()[0] - 0.5, ax.get_xlim()[1] + 0.5)
            ax.set_ylim(0.15, 1.05)

        plt.tight_layout()
        plt.show()


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


if __name__ == "__main__":
    fire.Fire({"accuracy": accuracy, "accuracy_plot": plot_accuracy_for_exp, "miles_graph": miles_graph})
