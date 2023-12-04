import os
import random
from pathlib import Path
from typing import Optional, Sequence, Type
from matplotlib import pyplot as plt

import openai
import pandas as pd
from dotenv import load_dotenv
import seaborn as sns
from slist import Slist
from tqdm import tqdm

from cot_transparency.data_models.models import BaseTaskOutput, TaskOutput
from cot_transparency.data_models.streaming import StreamingTaskOutput
from cot_transparency.json_utils.read_write import (
    GenericBaseModel,
    read_jsonl_file_into_basemodel,
    write_jsonl_file_from_basemodel,
)
from scripts.prompt_sensitivity_plotly import modal_agreement_for_task_hash


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


def accuracy_per_model(results: Sequence[StreamingTaskOutput | TaskOutput]):
    by_model = Slist(results).group_by(lambda x: x.get_task_spec().inference_config.model)
    for model, outputs in by_model:
        accuracy = outputs.map(lambda x: x.is_correct).average_or_raise()
        print(f"{model}: {accuracy=:2f} ({len(outputs)} examples)")
        agreement = modal_agreement_for_task_hash(outputs)
        print(f"{model}: {agreement=:2f} ({len(outputs)} examples)")
        print("==================")


def load_per_model_results(
    results_dir: Path | str, basemodel: Type[GenericBaseModel], model_names: Optional[Sequence[str]] = None
) -> Slist[GenericBaseModel]:
    results_dir = Path(results_dir)
    assert results_dir.is_dir(), "Cache dir must be a directory"
    paths = results_dir.glob("*.jsonl")
    if model_names is not None:
        paths = [p for p in paths if p.stem in model_names]
    outputs = Slist()
    for path in tqdm(paths, desc=f"Loading results from directory {results_dir}"):
        outputs.extend(read_jsonl_file_into_basemodel(path=path, basemodel=basemodel))
    return outputs


def lineplot_util(
    df_p: pd.DataFrame,
    title: str,
    y: str = "entropy",
    add_line_at: Optional[float] = None,
):
    _, ax = plt.subplots(figsize=(6, 6))
    ax = sns.lineplot(
        df_p,
        x="Samples",
        y=y,
        hue="Trained on COTS from",
        err_style="bars",
        ax=ax,
    )
    if add_line_at is not None:
        ax.axhline(add_line_at, ls="--", color="red")
    ax.set_ylabel(y)
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_ylim(0, None)
    # set legend below plot
    # ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=1)
    plt.tight_layout()
