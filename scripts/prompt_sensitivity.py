from typing import Optional, Sequence
import fire
import pandas as pd

from analysis import apply_filters, get_data_frame_from_exp_dir
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters

from cot_transparency.formatters.interventions.valid_interventions import VALID_INTERVENTIONS


def fleiss_kappa_on_group(group: pd.DataFrame):
    # we need subjects in rows, categories in columns, where subjects are task_hash
    # and categories are the formatter_name and the value is the parsed_response
    pt = group.pivot(index="task_hash", columns="formatter_name", values="parsed_response")  # type: ignore
    # drop any rows that have None
    pt = pt.dropna()
    agg = aggregate_raters(pt.to_numpy())
    fk = fleiss_kappa(agg[0])
    group["fleiss_kappa"] = fk
    return group


def prompt_metrics(
    exp_dir: str,
    inconsistent_only: bool = False,
    aggregate_over_tasks: bool = False,
    model_filter: Optional[str] = None,
    formatters: Sequence[str] = [],
    group1: str = "model",
    group3: str = "intervention_name",
):
    df = get_data_frame_from_exp_dir(exp_dir)
    df = apply_filters(
        inconsistent_only=inconsistent_only,
        model_filter=model_filter,
        formatters=formatters,
        aggregate_over_tasks=aggregate_over_tasks,
        df=df,
    )

    def is_same_answer(group: pd.DataFrame):
        # measure the consistency of the group, is_correct
        assert group.ground_truth.nunique() == 1

        if any(group.parsed_response == "None"):
            group["modal_agreement_score"] = None
            return group

        modal_answer = group["parsed_response"].mode()[0]
        group["is_same_as_mode"] = group.parsed_response == modal_answer
        group["modal_agreement_score"] = group["is_same_as_mode"].mean()
        return group

    # replace parsed answers that were None with "None" and print out the number of None answers
    df["parsed_response"] = df["parsed_response"].fillna("None")
    df["parsed_response"] = df["parsed_response"].astype(str)
    df["intervention_name"] = df["intervention_name"].fillna("None")

    def get_intervention_name(row: pd.Series) -> str:  # type: ignore
        if row.intervention_name == "None":
            if "nocot" in row.formatter_name.lower():
                return "None No COT"
            else:
                return "None COT"
        return VALID_INTERVENTIONS[row.intervention_name].formatted_name()

    df["intervention_name"] = df.apply(get_intervention_name, axis=1)

    print("Number of responses", len(df))
    try:
        print(f"Number of None answers: {df['parsed_response'].value_counts()['None']}")
    except KeyError:
        print("No None answers")

    # is consistent across formatters
    # drop on task_hash
    df_same_ans = df.groupby([group1, "task_hash", group3]).apply(is_same_answer).reset_index(drop=True)
    df_same_ans: pd.DataFrame = df_same_ans.drop_duplicates(
        subset=["task_hash", group1, "formatter_name", group3], inplace=False
    )  # type: ignore
    # drop none
    df_same_ans = df_same_ans[df_same_ans["parsed_response"] != "None"]

    n_questions = df_same_ans.groupby(["intervention_name"])["task_hash"].nunique().mean()
    df_none = df[df.intervention_name == "None"]
    n_formatter = len(df_none["formatter_name"].unique())
    n_formatter = df_same_ans.groupby(["intervention_name"])["formatter_name"].nunique().mean()

    # how do we order the hues
    hue_order = ["None No COT", "10 Few Shot No COT", "20 Few Shot No COT", "None COT", "10 Few Shot COT"]
    
    g = sns.catplot(
        data=df_same_ans, x=group1, y="modal_agreement_score", hue=group3, kind="bar", capsize=0.01, errwidth=1, hue_order=hue_order
    )
    g.fig.suptitle(f"Modal Agreement Score [{n_questions} questions, {n_formatter} prompts]")
    g.set_axis_labels("Model", "Modal Agreement Score")
    g._legend.set_title("Intervention")

    # df_fk = df.groupby([group1, group3]).apply(fleiss_kappa_on_group)
    # df_fk: pd.DataFrame = df_fk.drop_duplicates(subset=["task_hash", group1, "formatter_name", group3], inplace=False)  # type: ignore

    # g = sns.catplot(data=df_fk, x=group1, y="fleiss_kappa", hue=group3, kind="bar")
    # g.fig.suptitle("Fleiss Kappa Score")
    # g.set_axis_labels("Model", "Fleiss Kappa Score")
    # g._legend.set_title("Intervention")

    plt.show()


if __name__ == "__main__":
    fire.Fire(prompt_metrics)
