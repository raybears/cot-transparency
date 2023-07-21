import fire
from matplotlib import pyplot as plt
from analysis import get_general_metrics
from cot_transparency.data_models.models_v2 import (
    StageTwoExperimentJsonFormat,
)
import pandas as pd
from cot_transparency.data_models.io import ExpLoader
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
    loaded_dict = ExpLoader.stage_two(exp_dir)
    dfs = []
    for exp in loaded_dict.values():
        df = convert_stage2_experiment_to_dataframe(exp)
        dfs.append(df)
    df = pd.concat(dfs)
    df["is_correct"] = (df.parsed_response == df.ground_truth).astype(int)
    return df


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
            "hist": plot_historgram_of_lengths,
            "early": plot_early_answering,
        }
    )
