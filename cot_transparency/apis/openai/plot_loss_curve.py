import io
import os
from typing import Literal
from matplotlib import pyplot as plt
import openai
import pandas as pd
from pydantic import BaseModel
import wandb
from dotenv import load_dotenv
import seaborn as sns

from cot_transparency.apis.openai.finetune import FinetunedJobResults
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


# class LossCurve(BaseModel):
#     id: str
#     loss_curve: list[float]


def get_org_key(org: Literal["nyu"] | Literal["far"]):
    org_keys = os.environ["OPENAI_ORG_IDS"].split(",")
    if len(org_keys) == 0:
        raise ValueError("No org keys found, to use finetuned models, please set OPENAI_ORG_IDS in .env")
    match org:
        case "nyu":
            # the NYU one ends in 5Xq make sure we have that one
            org_key = [key for key in org_keys if key.endswith("5Xq")]
        case "far":
            org_key = [key for key in org_keys if key.endswith("T31")]
    print(org_key)
    return org_key[0]


def smooth_lineplot(df: pd.DataFrame, x: str, y: str, window: int = 2, loglog: bool=True) -> None:
    
    # if loglog filter in log space
    if loglog:
        # df[y] = np.log10(df[y])
        # df[x] = np.log10(df[x])
        df[y] = gaussian_filter1d(df[y], sigma=window)
        # df[y] = 10**df[y]
        # df[x] = 10**df[x]
    else:
        df[y] = gaussian_filter1d(df[y], sigma=window)
    


    ax = sns.lineplot(df, x=x, y=y)

    if y == "train_loss":
        ax.set_ylabel("Train Loss")
    elif y == "valid_loss":
        ax.set_ylabel("Validation Loss")

    if loglog:
        ax.set_yscale("log")
        ax.set_xscale("log")

    plt.show()


# takes a model name and returns its training loss curve
def loss_cuve_for_model_id(model_id: str):
    runs = []
    api = wandb.Api()
    for project_dir in ["consistency-training", "prompt_sen_experiments"]:
        runs.extend(api.runs(project_dir, filters={"config.finetune_model_id": model_id}))

    run = runs[0]
    finetune_id = run.config["finetune_job_id"]

    # get these from openai
    load_dotenv()
    api_key = os.environ["OPENAI_API_KEY"]
    if "nyu" in model_id:
        org_key = get_org_key("nyu")
    else:
        org_key = get_org_key("far")

    finetune_job = openai.FineTuningJob.retrieve(finetune_id, api_key=api_key, organization=org_key)

    job_results: FinetunedJobResults = FinetunedJobResults.model_validate(finetune_job)
    results = openai.File.download(id=job_results.result_files[0], api_key=api_key, organization=org_key).decode(
        "utf-8"
    )
    # log results
    df_results = pd.read_csv(io.StringIO(results))
    print(df_results)
    smooth_lineplot(df_results, "step", "train_loss")


if __name__ == "__main__":
    loss_cuve_for_model_id("ft:gpt-3.5-turbo-0613:far-ai::8CwqAHpd")
