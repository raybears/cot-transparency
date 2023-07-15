import fire
from cot_transparency.formatters import bias_to_unbiased_formatter
from cot_transparency.formatters import name_to_formatter
from cot_transparency.stage_one_tasks import ExperimentJsonFormat
from pathlib import Path
from glob import glob
import json
import pandas as pd
from typing import Optional

from stage_one import BBH_TASK_LIST

TASK_MAP = {}
for task in BBH_TASK_LIST:
    TASK_MAP[task] = "bbh"


def load_jsons(exp_dir: str) -> dict[Path, ExperimentJsonFormat]:
    loaded_dict: dict[Path, ExperimentJsonFormat] = {}

    paths = glob(f"{exp_dir}/**/*.json", recursive=True)
    print(f"Found {len(paths)} json files")
    for path in paths:
        _dict = json.load(open(path))
        loaded_dict[Path(path)] = ExperimentJsonFormat(**_dict)
    return loaded_dict


def convert_experiment_to_dataframe(exp: ExperimentJsonFormat) -> pd.DataFrame:
    out = []
    for task_output in exp.outputs:
        d = task_output.dict()
        model_outputs = d.pop("model_output")
        d_with_config = {**d, **d.pop("config")}
        for model_output in model_outputs:
            combined_d = {**d_with_config, **model_output}
            out.append(combined_d)
    return pd.DataFrame(out)


def get_data_frame_from_exp_dir(exp_dir: str) -> pd.DataFrame:
    loaded_dict = load_jsons(exp_dir)
    dfs = []
    for path, exp in loaded_dict.items():
        df = convert_experiment_to_dataframe(exp)
        dfs.append(df)
    df = pd.concat(dfs)
    df["is_correct"] = (df.parsed_response == df.ground_truth).astype(int)
    return df


def accuracy(
    exp_dir: str, inconsistent_only: bool = True, aggregate_over_tasks: bool = False, model_filter: Optional[str] = None
):
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

    # include number of examples
    outputs = []
    for df in [accuracy_df, counts_df]:
        df["formatter_name"] = df["formatter_name"].str.replace("Formatter", "")

        output = pd.pivot_table(df, index=["task_name", "model"], columns=["formatter_name"], values="is_correct")
        outputs.append(output)

    if not counts_are_equal(outputs[1]):
        print("Counts are not equal for some tasks and their baselines, likely experiments not completed")
        exit(1)

    print("---------------- Counts ----------------")
    print(outputs[1])
    print("--------------- Accuracy ---------------")
    print(outputs[0])


def counts_are_equal(count_df: pd.DataFrame):
    """
    Verify that the counts are the same for a task and its baseline
    """
    for col in count_df.columns:
        formatter_cls = name_to_formatter(col + "Formatter")
        if formatter_cls.is_biased:
            unbiased_formatter_name = bias_to_unbiased_formatter(formatter_cls.name())
            if not count_df[col].equals(count_df[unbiased_formatter_name.replace("Formatter", "")]):
                return False
    return True


if __name__ == "__main__":
    fire.Fire(accuracy)
