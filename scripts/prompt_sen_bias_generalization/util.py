from pathlib import Path
from typing import Sequence, Type
import openai
from dotenv import load_dotenv
import os
import random
import pandas as pd
from cot_transparency.data_models.models import BaseTaskOutput
from cot_transparency.json_utils.read_write import (
    GenericBaseModel,
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)
from slist import Slist



def add_point_at_1(df: pd.DataFrame, baseline_model: str = "gpt-3.5-turbo"):
    unique_trained_on = df["Trained on COTS from"].unique()
    baseline = df[df.model == baseline_model]

    for unique in unique_trained_on:
        if len(df[(df["Samples"] == 1) & (df["Trained on COTS from"] == unique)]) == 0:
            new_rows = baseline.copy()
            new_rows["Trained on COTS from"] = unique
            df = pd.concat((df, new_rows))  # type: ignore
    return df


def set_openai_org_rand():
    load_dotenv()
    org = os.environ.get("OPENAI_ORG_IDS")
    if org:
        orgs = org.split(",")
        if len(orgs) > 0:
            org = random.choice(orgs)
            print("Finetuning with org", org)

    openai.organization = org


def save_per_model_results(results: Sequence[BaseTaskOutput], results_dir: str | Path):
    results_dir = Path(results_dir)
    # check is not file or end in .jsonl
    assert not (results_dir.is_file() or results_dir.suffix == ".jsonl"), "Cache dir must be a directory"
    by_model = Slist(results).group_by(lambda x: x.get_task_spec().inference_config.model)
    for model, outputs in by_model:
        results_path = results_dir / f"{model}.jsonl"
        write_jsonl_file_from_basemodel(results_path, outputs)


def load_per_model_results(results_dir: Path | str, basemodel: Type[GenericBaseModel]) -> Slist[GenericBaseModel]:
    results_dir = Path(results_dir)
    assert results_dir.is_dir(), "Cache dir must be a directory"
    paths = results_dir.glob("*.jsonl")
    return Slist(read_jsonl_file_into_basemodel(path=path, basemodel=basemodel) for path in paths).flatten_list()
