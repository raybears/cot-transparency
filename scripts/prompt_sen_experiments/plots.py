from typing import Optional, Sequence
import fire
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from cot_transparency.data_models.models import StageTwoTaskOutput, TaskOutput

from cot_transparency.formatters.interventions.valid_interventions import VALID_INTERVENTIONS
from cot_transparency.data_models.io import read_whole_exp_dir, read_whole_exp_dir_s2
from scripts.utils.loading import BasicExtractor, IsCoTExtractor, convert_slist_to_df
from scripts.utils.plots import catplot
from scripts.utils.simple_model_names import MODEL_SIMPLE_NAMES

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# how do we order the hues
HUE_ORDER = [
    "None No COT",
    "10 Few Shot No COT",
    "20 Few Shot No COT",
    "10 Few Shot No COT (Mixed Format)",
    "None COT",
    "10 Few Shot COT",
    "10 Few Shot COT (Mixed Format)",
]


def fleiss_kappa_on_group(group: pd.DataFrame):
    # we need subjects in rows, categories in columns, where subjects are task_hash
    # and categories are the formatter_name and the value is the parsed_response
    try:
        pt = group.pivot(index="task_hash", columns="formatter_name", values="parsed_response")  # type: ignore
        # drop any rows that have None
        pt = pt.dropna()
        agg = aggregate_raters(pt.to_numpy())
        fk = fleiss_kappa(agg[0])
        group["fleiss_kappa"] = fk
    except ValueError as e:
        breakpoint()
        raise e
    return group


def entropy_on_group(group: pd.DataFrame) -> pd.Series:  # type: ignore
    # we need subjects in rows, categories in columns, where subjects are task_hash
    # and categories are the formatter_name and the value is the parsed_response
    parsed_value_counts = group["parsed_response"].value_counts(normalize=True)
    entropy = -1 * (parsed_value_counts * parsed_value_counts.apply(np.log2)).sum()
    return pd.Series({"entropy": entropy})


def get_modal_agreement_score(group: pd.DataFrame) -> pd.DataFrame:
    """
    measure the consistency of the group, is_correct
    Note if any of the responses in the group are None then the whole group is set to None
    """
    assert group.ground_truth.nunique() == 1

    # if any(group.parsed_response == "None") or any(group.parsed_response.isna()):
    #     group["modal_agreement_score"] = None
    #     group["is_same_as_mode"] = None
    #     return group

    modal_answer = group["parsed_response"].mode()[0]
    group["is_same_as_mode"] = group.parsed_response == modal_answer
    group["modal_agreement_score"] = group["is_same_as_mode"].mean()
    group["modal_answer"] = modal_answer
    return group


def get_intervention_name(row: pd.Series) -> str:  # type: ignore
    if row.intervention_name == "None" or row.intervention_name is None:
        if "nocot" in row.formatter_name.lower():
            return "None No COT"
        else:
            return "None COT"
    return VALID_INTERVENTIONS[row.intervention_name].formatted_name()


def convert_s2_to_s1(s2_output: StageTwoTaskOutput) -> TaskOutput:
    s2_parsed = s2_output.inference_output.parsed_response
    s1_output = s2_output.task_spec.stage_one_output.model_copy(deep=True)
    s1_output.inference_output.parsed_response = s2_parsed
    return s1_output


def prompt_metrics(
    exp_dir: str,
    models: Sequence[str] = [],
    tasks: Sequence[str] = [],
    formatters: Sequence[str] = [],
    x: str = "model",
    hue: str = "intervention_name",
    col: Optional[str] = None,
    temperature: Optional[int] = None,
    only_modally_wrong: bool = False,
):
    # try reading as stage 2
    stage_2_outputs = read_whole_exp_dir_s2(exp_dir=exp_dir)
    if len(stage_2_outputs) > 0:
        # then this was state 2
        slist = stage_2_outputs.map(convert_s2_to_s1)
    else:
        slist = read_whole_exp_dir(exp_dir=exp_dir)

    slist = (
        slist.filter(lambda task: task.task_spec.inference_config.model in models if models else True)
        .filter(lambda task: task.task_spec.formatter_name in formatters if formatters else True)
        .filter(lambda task: task.task_spec.task_name in tasks if tasks else True)
        .filter(
            lambda task: task.task_spec.inference_config.temperature == temperature if temperature is not None else True
        )
    )
    print("Number of responses after filtering = ", len(slist))

    df = convert_slist_to_df(slist, [BasicExtractor(), IsCoTExtractor()])

    # number of duplicate input_hashes
    print(f"Number of duplicate input_hashes: {df['input_hash'].duplicated().sum()}")
    # drop duplicates on input_hash
    df = df.drop_duplicates(subset=["input_hash", "model", "formatter_name", "intervention_name"], inplace=False)  # type: ignore
    print(f"Number of responses after dropping duplicates: {len(df)}")

    # work out the number of formatters that were used
    n_formatters = df.groupby(["intervention_name"])["formatter_name"].nunique().max()  # type: ignore
    print(f"Number of formatters used: {n_formatters}")

    # Drop any group of task_hashes that have fewer than n_formatters
    df = df.groupby(["task_hash", "model", "intervention_name"]).filter(lambda x: len(x) == n_formatters)  # type: ignore

    # replace model_names with MODEL_SIMPLE_NAMES
    df["model"] = df["model"].apply(lambda x: MODEL_SIMPLE_NAMES[x])  # type: ignore

    # replace parsed answers that were None with "None" and print out the number of None answers
    df["parsed_response"] = df["parsed_response"].fillna("None")  # type: ignore
    df["parsed_response"] = df["parsed_response"].astype(str)  # type: ignore
    df["intervention_name"] = df.apply(get_intervention_name, axis=1)  # type: ignore

    print("Number of responses", len(df))
    try:
        print(f"Number of None answers: {df['parsed_response'].value_counts()['None']}")  # type: ignore
    except KeyError:
        print("No None answers")

    # is consistent across formatters
    # drop on task_hash
    df_same_ans = df.groupby([x, "task_hash", hue]).apply(get_modal_agreement_score).reset_index(drop=True)  # type: ignore
    df_same_ans: pd.DataFrame = df_same_ans[~df_same_ans["modal_agreement_score"].isna()]  # type: ignore
    print(f"Number of responses after dropping groups that contained a None: {len(df_same_ans)}")

    if only_modally_wrong:
        df_same_ans = df_same_ans[df_same_ans["modal_answer"] != df_same_ans["ground_truth"]]  # type: ignore

    df = df_same_ans  # save this for the subsequencent plots
    # after this pont there should be no none answers
    assert not any(df_same_ans["modal_agreement_score"].isna())

    df_same_ans: pd.DataFrame = df_same_ans.drop_duplicates(
        subset=["task_hash", x, "formatter_name", hue], inplace=False
    )  # type: ignore

    n_questions = f"{df_same_ans.groupby([col, hue, x])['task_hash'].nunique().mean():.1f}"
    n_formatter = df_same_ans.groupby(["intervention_name"])["formatter_name"].nunique().mean()  # type: ignore

    hue_order = [i for i in HUE_ORDER if i in df_same_ans[hue].unique()]
    if len(hue_order) == 0:
        hue_order = None

    g = catplot(
        data=df_same_ans,
        x=x,
        y="modal_agreement_score",
        hue=hue,
        col=col,
        kind="bar",
        hue_order=hue_order,
    )
    g.fig.suptitle(
        f"Modal Agreement Score [avg of {n_questions} questions per formatter & task, {n_formatter} prompts, temperature={temperature}]"
    )
    g.fig.tight_layout()
    # move the plot area to leave space for the legend
    g.fig.subplots_adjust(right=0.7)

    # Plot the accuracies as well
    df_acc = df
    df_acc["is_correct"] = df_acc["parsed_response"] == df_acc["ground_truth"]
    df_acc = df_acc.groupby([x, "task_hash", hue, col])["is_correct"].mean().reset_index()
    g = catplot(data=df_acc, x=x, y="is_correct", hue=hue, col=col, kind="bar", hue_order=hue_order)
    g.fig.suptitle(
        f"Modal Accuracy [avg of {n_questions} questions per formatter & task, {n_formatter} prompts, temperature={temperature}]"
    )
    g.fig.tight_layout()
    # move the plot area to leave space for the legend
    g.fig.subplots_adjust(right=0.7)

    # Plot the fleiss kappa scores
    df_fk = df.groupby([x, hue, col]).apply(fleiss_kappa_on_group).reset_index(drop=True)
    df_fk: pd.DataFrame = df_fk.drop_duplicates(subset=["task_hash", x, "formatter_name", hue], inplace=False)  # type: ignore
    g = catplot(data=df_fk, x=x, y="fleiss_kappa", hue=hue, col=col, kind="bar", hue_order=hue_order)
    g.fig.suptitle(f"Fleiss Kappa Score [{n_questions} questions, {n_formatter} prompts]")
    g.fig.tight_layout()
    # move the plot area to leave space for the legend
    g.fig.subplots_adjust(right=0.7)

    # Plot the entropy scores
    df_entropy = df.groupby([x, hue, "task_hash", col]).apply(entropy_on_group).reset_index()
    g = catplot(data=df_entropy, x=x, y="entropy", hue=hue, col=col, kind="bar", hue_order=hue_order)
    g.fig.suptitle(
        f"Entropy across formatters [avg of {n_questions} questions per formatter & task, {n_formatter} prompts, temperature={temperature}]"
    )
    g.fig.tight_layout()
    # move the plot area to leave space for the legend
    g.fig.subplots_adjust(right=0.7)

    plt.show()


if __name__ == "__main__":
    fire.Fire(prompt_metrics)
