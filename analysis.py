import fire
from matplotlib import pyplot as plt
from cot_transparency.data_models.models_v2 import (
    ExperimentJsonFormat,
    StageTwoExperimentJsonFormat,
    StageTwoTaskOutput,
    TaskOutput,
)
import pandas as pd
from typing import Any, Optional, List, Union
from cot_transparency.data_models.io import load_jsons
from cot_transparency.transparency_plots import (
    add_max_step_in_cot_trace,
    check_same_answer,
    plot_cot_trace,
    plot_historgram_of_cot_steps,
)

from stage_one import BBH_TASK_LIST

TASK_MAP = {}
for task in BBH_TASK_LIST:
    TASK_MAP[task] = "bbh"


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


def convert_stage1_experiment_to_dataframe(exp: ExperimentJsonFormat) -> pd.DataFrame:
    out = []
    for task_output in exp.outputs:
        d_with_config = get_general_metrics(task_output)
        for model_output in task_output.model_output:
            combined_d = {**d_with_config, **model_output.dict()}
            out.append(combined_d)
    return pd.DataFrame(out)


def convert_stage2_experiment_to_dataframe(exp: StageTwoExperimentJsonFormat) -> pd.DataFrame:
    out = []
    for task_output in exp.outputs:
        d_with_config = get_general_metrics(task_output)
        d_with_config["task_name"] = task_output.task_spec.stage_one_output.task_spec.task_name
        d_with_config["ground_truth"] = task_output.task_spec.stage_one_output.task_spec.ground_truth
        d_with_config["stage_one_hash"] = task_output.task_spec.stage_one_output.task_spec.uid()
        d_with_config = {**d_with_config, **task_output.model_output.dict()}
        out.append(d_with_config)
    return pd.DataFrame(out)


def get_data_frame_from_exp_dir(exp_dir: str) -> pd.DataFrame:
    loaded_dict, _ = load_jsons(exp_dir)
    dfs = []
    for exp in loaded_dict.values():
        if isinstance(exp, ExperimentJsonFormat):
            df = convert_stage1_experiment_to_dataframe(exp)
        else:
            df = convert_stage2_experiment_to_dataframe(exp)
        dfs.append(df)
    df = pd.concat(dfs)
    df["is_correct"] = (df.parsed_response == df.ground_truth).astype(int)
    return df


def accuracy(
    exp_dir: str,
    inconsistent_only: bool = True,
    aggregate_over_tasks: bool = False,
    model_filter: Optional[str] = None,
    check_counts: bool = True,
    return_dataframes: bool = False,
) -> Optional[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    exp_dir: path to directory containing experiment jsons
    inconsistent_only: if True, only include inconsistent tasks where biased ans and correct ans are different
    """

    df = get_data_frame_from_exp_dir(exp_dir)

    if inconsistent_only:
        df = df[df.biased_ans != df.ground_truth]
    if model_filter:
        df = df[df.model == model_filter]

    if aggregate_over_tasks:
        # replace task_name with the "parent" task name using the task_map
        df["task_name"] = df["task_name"].replace(TASK_MAP)

    groups = ["task_name", "model", "formatter_name"]
    accuracy_df_grouped = df[["is_correct", "task_name", "model", "formatter_name"]].groupby(groups)
    accuracy_df = accuracy_df_grouped.mean().reset_index()
    counts_df = accuracy_df_grouped.count().reset_index()

    # count the number of repeats by counting the number task hashes
    counts_df["unqiue_questions"] = df.groupby(groups)["task_hash"].nunique().reset_index()["task_hash"]
    counts_df["total_samples"] = df.groupby(groups)["is_correct"].count().reset_index()["is_correct"]

    unique_questions_df: pd.DataFrame = pivot_df(
        counts_df,
        values=["unqiue_questions"],
    )[
        "unqiue_questions"
    ]  # type: ignore
    counts_df: pd.DataFrame = pivot_df(counts_df, values=["total_samples"])["total_samples"]  # type: ignore
    accuracy_df = pivot_df(accuracy_df)

    if check_counts:
        if not (counts_are_equal(counts_df) and counts_are_equal(unique_questions_df)):
            print("Counts are not equal for some tasks and their baselines, likely experiments not completed")
            exit(1)

    print("---------------- Counts ----------------")
    print(counts_df)
    print("--------------- Unique Questions ---------------")
    print(unique_questions_df)
    print("--------------- Accuracy ---------------")
    print(accuracy_df)

    if return_dataframes:
        return accuracy_df, counts_df, unique_questions_df  # type: ignore


def pivot_df(df: pd.DataFrame, values: List[str] = ["is_correct"]):
    df["formatter_name"] = df["formatter_name"].str.replace("Formatter", "")

    output = pd.pivot_table(df, index=["task_name", "model"], columns=["formatter_name"], values=values)
    return output


def counts_are_equal(count_df: pd.DataFrame) -> bool:
    """
    Verify that the counts are the same for all columns in the count_df
    """
    return (count_df.nunique(axis=1) == 1).all()


def plot_historgram_of_lengths(exp_dir: str):
    df = get_data_frame_from_exp_dir(exp_dir)
    plot_historgram_of_cot_steps(df)


def plot_early_answering(exp_dir: str):
    df_combined = get_data_frame_from_exp_dir(exp_dir)

    df_combined = add_max_step_in_cot_trace(df_combined)

    # Apply the check_same_answer function
    df_combined = df_combined.groupby("stage_one_hash").apply(check_same_answer).reset_index(drop=True)

    # Compute the cot_trace_length from the maximum value of step_in_cot_trace for each stage_one_hash
    cot_lengths = df_combined.groupby("stage_one_hash")["step_in_cot_trace"].transform("max") + 1  # type: ignore
    df_combined["cot_trace_length"] = cot_lengths

    # Add a new column 'is_biased' based on the 'formatter_name' column
    df_combined["is_biased"] = ~df_combined["formatter_name"].str.contains("Unbiased")

    # Plot by task
    plot_cot_trace(df_combined, plot_by="task")
    plt.show()

    # # Plot by bias
    # plot_cot_trace(df_combined, plot_by="bias")
    # plt.show()


if __name__ == "__main__":
    # plot_early_answering("experiments/stage_two/20230718_bbh_with_role_updated_tokenizer")
    fire.Fire(
        {
            "accuracy": accuracy,
            "hist": plot_historgram_of_lengths,
            "early": plot_early_answering,
        }
    )
