import pandas as pd
import numpy as np
import argparse

from cot_transparency.data_models.models import (
    StageTwoExperimentJsonFormat,
    TaskOutput,
)

from cot_transparency.formatters.transparency.trace_manipulation import get_cot_steps

from cot_transparency.data_models.io import ExpLoader
from cot_transparency.formatters.extraction import extract_answer

from analysis import get_general_metrics

from typing import Union


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
        if task_output.task_spec.trace_info is not None:
            d_with_config["has_mistake"] = task_output.task_spec.trace_info.has_mistake
            d_with_config["was_truncated"] = task_output.task_spec.trace_info.was_truncated
            d_with_config["mistake_added_at"] = task_output.task_spec.trace_info.mistake_inserted_idx
            d_with_config["original_cot_trace_length"] = len(task_output.task_spec.trace_info.original_cot)
            modified_cot_length = get_cot_steps(task_output.task_spec.trace_info.get_complete_modified_cot())
            d_with_config["modified_cot"] = task_output.task_spec.trace_info.get_complete_modified_cot()
            d_with_config["original_cot"] = task_output.task_spec.trace_info.original_cot
            d_with_config["cot_trace_length"] = len(modified_cot_length)
        d_with_config["stage_one_formatter_name"] = task_output.task_spec.stage_one_output.task_spec.formatter_name

        out.append(d_with_config)

    df = pd.DataFrame(out)

    stage_one_output = [TaskOutput(**i) for i in df["stage_one_output"]]
    stage_formatter = [i.task_spec.formatter_name for i in stage_one_output]
    df["stage_one_formatter_name"] = stage_formatter  # type: ignore
    return df  # type: ignore


def get_data_frame_from_exp_dir(exp_dir: str) -> pd.DataFrame:
    loaded_dict = ExpLoader.stage_two(exp_dir, final_only=True)
    dfs = []
    for exp in loaded_dict.values():
        df = convert_stage2_experiment_to_dataframe(exp)
        dfs.append(df)
    df = pd.concat(dfs)  # type: ignore
    df["is_correct"] = (df.parsed_response == df.ground_truth).astype(int)
    # filter out the NOT_FOUND rows
    n_not_found = len(df[df.parsed_response == "NOT_FOUND"])
    print(f"Number of NOT_FOUND rows: {n_not_found}")
    df = df[df.parsed_response != "NOT_FOUND"]
    return df  # type: ignore


def check_same_answer(df: pd.DataFrame) -> pd.DataFrame:
    max_step_row = df[(~df.was_truncated) & ~(df.has_mistake)]
    if len(max_step_row) == 0:
        df["same_answer"] = "NOT_FOUND"
    else:
        # Take the first row with max cot_trace_length as reference
        reference_response = max_step_row["parsed_response"].iloc[0]  # type: ignore
        df["same_answer"] = df["parsed_response"] == reference_response  # type: ignore
    return df  # type: ignore


def compute_auc(df: pd.DataFrame, x="cot_trace_length") -> float:
    n_traces = df.stage_one_hash.nunique()

    # Sort the dataframe by 'proportion_of_cot' for AUC computation
    df_sorted = df.sort_values(by=x)
    same_answer_values = df_sorted["same_answer"].values
    proportion_of_cot = df_sorted[x].values / max(df_sorted[x].values)

    # Compute AUC
    auc = np.trapz(same_answer_values, x=proportion_of_cot)  # type: ignore

    weighted_auc = auc * n_traces
    weighted_auc_normalized = weighted_auc / n_traces
    weighted_aoc = 100 - weighted_auc_normalized if weighted_auc_normalized > 1 else 1 - weighted_auc_normalized

    return weighted_aoc


def get_aoc_with_leave_one_out(df: pd.DataFrame, x="cot_trace_length") -> pd.DataFrame:
    df = check_same_answer(df)
    overall_aoc = compute_auc(df)

    leave_one_out_aocs = []

    for idx, _ in df.iterrows():
        temp_df = df.drop(idx)
        temp_df = check_same_answer(temp_df)  # Recompute 'same_answer' for the reduced dataframe
        leave_one_out_aoc = compute_auc(temp_df)
        leave_one_out_aocs.append(leave_one_out_aoc)

    df["aoc_difference"] = np.array([overall_aoc]) - np.array(leave_one_out_aocs)  # type: ignore
    df = df.sort_values(by="aoc_difference", ascending=False)  # type: ignore
    return df  # type: ignore


def extract_user_content(message_list: list[dict[str, Union[str, str]]], system: bool = False) -> Union[str, None]:
    for message in message_list:
        if message["role"][0] == "u":
            return message["content"]
    return None


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = get_aoc_with_leave_one_out(df)  # type: ignore

    cols_to_keep = [
        "messages",
        "ground_truth",
        "has_mistake",
        "is_correct",
        "original_cot",
        "modified_cot",
        "mistake_added_at",
        "aoc_difference",
    ]

    df = df[df["stage_one_formatter_name"] != "ZeroShotCOTUnbiasedTameraTFormatter"]  # type: ignore
    df["original_cot"] = df["original_cot"].apply("".join)  # type: ignore

    df_filtered = df[cols_to_keep]  # type: ignore

    df_filtered = df_filtered[df_filtered["has_mistake"]]  # type: ignore
    df = df[~df["was_truncated"]]  # type: ignore
    df_filtered = df_filtered[df_filtered["is_correct"] != 1]  # type: ignore

    df_filtered["parsed_original_ans"] = df_filtered["original_cot"].apply(extract_answer)  # type: ignore
    df_filtered = df_filtered[df_filtered["parsed_original_ans"].notna()]  # type: ignore
    df_filtered["parsed_modified_ans"] = df_filtered["modified_cot"].apply(extract_answer)  # type: ignore
    df_filtered = df_filtered[df_filtered["parsed_modified_ans"].notna()]  # type: ignore

    df_filtered = df_filtered.drop(columns=["has_mistake", "is_correct"])  # type: ignore

    df_filtered["messages"] = df_filtered["messages"].apply(extract_user_content)  # type: ignore
    df_filtered.rename(columns={"messages": "question"}, inplace=True)  # type: ignore

    result_df = df_filtered[
        (df_filtered["parsed_original_ans"] == df_filtered["ground_truth"].astype(str))
        & (df_filtered["parsed_modified_ans"] != df_filtered["ground_truth"].astype(str))
    ]  # type: ignore

    result_df[result_df["aoc_difference"] > 0]  # type: ignore

    return result_df  # type: ignore


def get_filtered_csv(result_df: pd.DataFrame, save_path: str = "./", filename: str = "") -> None:
    result_df.reset_index(drop=True, inplace=True)
    result_df.to_csv(f"{save_path}/{filename}_filter_mistakes.csv")


def get_std_err_for_list(values: list[Union[float, int]]) -> float:
    df = pd.DataFrame(values, columns=["output"])
    err = df["output"].std() / np.sqrt(df["output"].count())
    return err


def analyse_mistake_pos_vs_incorrect(df: pd.DataFrame) -> None:
    mistakes_pos_vs_incorrect_dict = {}
    mistakes_pos_vs_aoc_dict = {}
    standard_errors = {}

    for i, row in df.iterrows():
        mistakes_pos_vs_incorrect_dict[row["mistake_added_at"]] = (
            mistakes_pos_vs_incorrect_dict.get(row["mistake_added_at"], 0) + 1
        )
        mistakes_pos_vs_aoc_dict.setdefault(row["mistake_added_at"], []).append(row["aoc_difference"])

    mistakes_pos_vs_incorrect_dict = {
        k: mistakes_pos_vs_incorrect_dict[k] for k in sorted(mistakes_pos_vs_incorrect_dict.keys())
    }
    mistakes_pos_vs_aoc_dict = {k: mistakes_pos_vs_aoc_dict[k] for k in sorted(mistakes_pos_vs_aoc_dict.keys())}

    for k, v in mistakes_pos_vs_aoc_dict.items():
        mistakes_pos_vs_aoc_dict[k] = sum(v) / len(v)
        standard_errors[k] = get_std_err_for_list(v)

    print(mistakes_pos_vs_incorrect_dict)
    print(mistakes_pos_vs_aoc_dict)
    print(standard_errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str)
    parser.add_argument("--gen_filtered_file", type=bool, default=False)
    parser.add_argument("--save_path", type=str, default="../cot_transparency/formatters/transparency/interventions")
    parser.add_argument("--print_analysis", type=bool, default=False)
    parser.add_argument("--model_filter", type=str)

    args = parser.parse_args()

    exp_dir = args.exp_dir
    filename = exp_dir.split("/")[-2]
    save_path = args.save_path

    df = get_data_frame_from_exp_dir(exp_dir)

    if args.gen_filtered_file:
        df = preprocess_df(df)  # type: ignore
        get_filtered_csv(df, save_path, filename=filename)

    if args.print_analysis:
        if args.model_filter:
            df = df[df["model"] == args.model_filter]
        df = preprocess_df(df)  # type: ignore
        analyse_mistake_pos_vs_incorrect(df)
