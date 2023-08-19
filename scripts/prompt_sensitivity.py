from typing import Optional, Sequence
import fire
import pandas as pd

from analysis import apply_filters, get_data_frame_from_exp_dir
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters


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

    def fleiss_kappa_on_group(group: pd.DataFrame):
        # we need subjects in rows, categories in columns, where subjects are task_hash
        # and categories are the formatter_name and the value is the parsed_response
        pt = group.pivot(index="task_hash", columns="formatter_name", values="parsed_response")
        # drop any rows that have None
        pt = pt.dropna()
        agg = aggregate_raters(pt.to_numpy())
        fk = fleiss_kappa(agg[0])
        group["fleiss_kappa"] = fk
        return group

    # replace parsed answers that were None with "None" and print out the number of None answers
    df["parsed_response"] = df["parsed_response"].fillna("None")
    df["parsed_response"] = df["parsed_response"].astype(str)
    df["intervention_name"] = df["intervention_name"].fillna("None")

    print("Number of responses", len(df))
    try:
        print(f"Number of None answers: {df['parsed_response'].value_counts()['None']}")
    except KeyError:
        print("No None answers")

    # is consistent across formatters
    # drop on task_hash
    df_same_ans = df.groupby([group1, "task_hash", group3]).apply(is_same_answer)
    df_fk = df.groupby([group1, group3]).apply(fleiss_kappa_on_group)

    df_same_ans = df_same_ans.drop_duplicates(subset=["task_hash", group1, "formatter_name", group3])
    # drop none
    df_same_ans = df_same_ans[df_same_ans["parsed_response"] != "None"]
    df_fk = df_fk.drop_duplicates(subset=["task_hash", group1, "formatter_name", group3])

    g = sns.catplot(data=df_same_ans, x=group1, y="modal_agreement_score", hue=group3, kind="bar")
    g.fig.suptitle("Modal Agreement Score")

    g = sns.catplot(data=df_fk, x=group1, y="fleiss_kappa", hue=group3, kind="bar")
    g.fig.suptitle("Fleiss Kappa Score")

    plt.show()


if __name__ == "__main__":
    fire.Fire(prompt_metrics)
