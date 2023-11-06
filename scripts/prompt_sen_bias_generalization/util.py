import openai
from dotenv import load_dotenv
import os
import random
import pandas as pd

from scripts.finetune_zero_shot_experiments.comparison_plot import ModelTrainMeta


def add_point_at_1(df: pd.DataFrame, baseline_model: str = "gpt-3.5-turbo"):
    unique_trained_on = df["Trained on COTS from"].unique()
    baseline = df[df.model == baseline_model]

    for unique in unique_trained_on:
        if len(df[(df["Samples"] == 1) & (df["Trained on COTS from"] == unique)]) == 0:
            new_rows = baseline.copy()
            new_rows["Trained on COTS from"] = unique
            df = pd.concat((df, new_rows))  # type: ignore
    return df


def get_name_of_run(i: ModelTrainMeta) -> str:
    return f"{i.train_formatters.value}, {i.filter_strategy.value}, {i.sampling_strategy}"


def set_openai_org_rand():
    load_dotenv()
    org = os.environ.get("OPENAI_ORG_IDS")
    if org:
        orgs = org.split(",")
        if len(orgs) > 0:
            org = random.choice(orgs)
            print("Finetuning with org", org)

    openai.organization = org