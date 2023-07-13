import fire
from cot_transparency.stage_one_tasks import ExperimentJsonFormat
from pathlib import Path
from glob import glob
import json
import pandas as pd


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


def experiment_accuracy(exp_dir: str):
    loaded_dict = load_jsons(exp_dir)
    dfs = []
    for path, exp in loaded_dict.items():
        df = convert_experiment_to_dataframe(exp)
        dfs.append(df)
    df = pd.concat(dfs)
    df["is_correct"] = (df.parsed_response == df.ground_truth).astype(int)

    accuracy_df = (
        df[["is_correct", "task_name", "model"]].groupby(["task_name", "model"]).mean(numeric_only=True).reset_index()
    )
    output = pd.pivot_table(accuracy_df, index="task_name", columns="model", values="is_correct")
    print(output)


if __name__ == "__main__":
    fire.Fire(experiment_accuracy)
