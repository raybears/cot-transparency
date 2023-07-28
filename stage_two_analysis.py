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
from cot_transparency.transparency_plots import (
    add_cot_trace_len,
    check_same_answer,
    plot_cot_trace,
    plot_historgram_of_cot_steps,
)
from analysis import accuracy_for_df, TASK_MAP


def convert_stage2_experiment_to_dataframe(exp: StageTwoExperimentJsonFormat) -> pd.DataFrame:
    out = []
    for task_output in exp.outputs:
        d_with_config = get_general_metrics(task_output)
        d_with_config["model"] = task_output.task_spec.model_config.model
        d_with_config["task_name"] = task_output.task_spec.stage_one_output.task_spec.task_name
        d_with_config["ground_truth"] = task_output.task_spec.stage_one_output.task_spec.ground_truth
        d_with_config["stage_one_hash"] = task_output.task_spec.stage_one_output.task_spec.uid()
        d_with_config["stage_one_output_hash"] = task_output.task_spec.stage_one_output.uid()
        d_with_config["biased_ans"] = task_output.task_spec.stage_one_output.task_spec.biased_ans
        d_with_config["task_hash"] = task_output.task_spec.stage_one_output.task_spec.task_hash
        d_with_config = {**d_with_config, **task_output.model_output.dict()}
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
    filter_at_step: Optional[int] = None,
    task_filter: Optional[list[str]] = None,
    norm_per_task: bool = True,
):
    df = get_data_frame_from_exp_dir(exp_dir)
    plot_historgram_of_cot_steps(
        df, filter_at_step=filter_at_step, task_filter=task_filter, norm_per_task=norm_per_task
    )


def plot_early_answering(
    exp_dir: str,
    show_plots: bool = False,
    inconsistent_only: bool = False,
    aggregate_over_tasks: bool = False,
    model_filter: Optional[str] = None,
):
    df = get_data_frame_from_exp_dir(exp_dir)

    if aggregate_over_tasks:
        # replace task_name with the "parent" task name using the task_map
        df["task_name"] = df["task_name"].replace(TASK_MAP)

    if inconsistent_only:
        df = df[df.biased_ans != df.ground_truth]
        print("Number of inconsistent tasks: ", len(df))

    if model_filter:
        df = df[df.model.isin(model_filter)]

    df = add_cot_trace_len(df)

    # Apply the check_same_answer function
    df = df.groupby("stage_one_hash").apply(check_same_answer).reset_index(drop=True)

    # Plot by task
    plot_cot_trace(df, color_by_model=aggregate_over_tasks)

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

    df = add_cot_trace_len(df)

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
            "accuracy": accuracy,
        }
    )
