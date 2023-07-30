import fire
import pandas as pd
import plotly.graph_objects as go
from analysis import get_data_frame_from_exp_dir
from cot_transparency.formatters import bias_to_unbiased_formatter
from cot_transparency.util import deterministic_hash


def assign_response_category(row: pd.Series) -> str:
    if row["parsed_response"] == row["ground_truth"]:
        return "Correct"
    elif row["parsed_response"] == row["biased_ans"]:
        return "Bias Aligned"
    else:
        return "Incorrect"


def create_sankey_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df[df.cot_type.isin(["No-COT", "COT"])]

    # assert that we have equal numbers of COT and No-COT
    assert len(df[df.cot_type == "No-COT"]) == len(df[df.cot_type == "COT"])

    df = df[["task_hash", "task_unbiased_hash", "cot_type", "response_category", "parsed_response"]]
    # sort by response category
    df = df.sort_values(by=["response_category"])

    # lets make a source dataframe
    source_df = df[df.cot_type == "No-COT"]
    target_df = df[df.cot_type == "COT"]

    transistion_df = pd.merge(source_df, target_df, on="task_hash", how="inner", suffixes=("_source", "_target"))

    # Record when the literal answer did not change
    cot_did_not_change_answer = transistion_df["parsed_response_source"] == transistion_df["parsed_response_target"]
    both_incorrect = (transistion_df["response_category_source"] == "Incorrect") & (
        transistion_df["response_category_target"] == "Incorrect"
    )

    # change when they were both incorrect but cot_did_change answer
    transistion_df.loc[
        (both_incorrect & cot_did_not_change_answer),
        "response_category_target",
    ] = "Same Incorrect"

    transistion_df = transistion_df[["task_hash", "response_category_source", "response_category_target"]]

    # add COT to the source
    transistion_df["response_category_source"] = transistion_df["response_category_source"] + " (No-COT)"
    transistion_df["response_category_target"] = transistion_df["response_category_target"] + " (COT)"

    # create nodes
    nodes = list(
        set(transistion_df["response_category_source"].unique()).union(
            set(transistion_df["response_category_target"].unique())
        )
    )
    nodes = pd.DataFrame(nodes, columns=["label"])

    def get_color(x: str):
        # retuns rgba, nice colors

        if "Correct" in x:
            # green
            return "rgba(0, 255, 0, 0.8)"
        elif "Bias Aligned" in x:
            # blue
            return "rgba(0, 0, 255, 0.8)"
        elif "Same Incorrect" in x:
            # orange
            return "rgba(255, 165, 0, 0.8)"
        else:
            # red
            return "rgba(255, 0, 0, 0.8)"

    nodes["color"] = nodes["label"].map(get_color)

    transistion_counts = (
        transistion_df.groupby(["response_category_source", "response_category_target"]).size().reset_index()
    )
    transistion_counts = transistion_counts.rename(columns={0: "counts"})

    # replace transition_counts with idxs of source and target
    transistion_counts["source_idx"] = transistion_counts["response_category_source"].map(
        lambda x: nodes[nodes["label"] == x].index[0]
    )
    transistion_counts["target_idx"] = transistion_counts["response_category_target"].map(
        lambda x: nodes[nodes["label"] == x].index[0]
    )
    transistion_counts["colors"] = transistion_counts["response_category_source"].map(get_color)
    print(transistion_counts)

    # make colors more transparent
    transistion_counts["colors"] = transistion_counts.apply(lambda x: x["colors"][:-3] + "0.3)", axis=1)

    return nodes, transistion_counts


def create_sankey_diagram(nodes: pd.DataFrame, counts: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes["label"],
                    color=nodes["color"],
                ),
                link=dict(
                    source=counts["source_idx"],
                    target=counts["target_idx"],
                    value=counts["counts"],
                    color=counts["colors"],
                ),
            )
        ]
    )

    fig.update_layout(title_text=title, font_size=10)
    return fig


def main(exp_dir: str, filter_on_same_start: bool = True, inconsistent_only: bool = True) -> None:
    df = get_data_frame_from_exp_dir(exp_dir)

    if inconsistent_only:
        df = df[df.biased_ans != df.ground_truth]

    assert df.model.nunique() == 1
    model = df.model.unique()[0]

    root_mapping = {
        "ZeroShotCOTUnbiasedFormatter": ("ZeroShot", None, "COT"),
        "ZeroShotCOTSycophancyFormatter": ("ZeroShot", "Sycophancy", "COT"),
        "ZeroShotUnbiasedFormatter": ("ZeroShot", None, "No-COT"),
        "ZeroShotSycophancyFormatter": ("ZeroShot", "Sycophancy", "No-COT"),
    }

    # Print counts for each formatter
    print(df.formatter_name.value_counts())
    print(df.shape)

    df["unbiased_formatter"] = df.formatter_name.map(bias_to_unbiased_formatter)
    df["root"] = df.formatter_name.map(lambda x: root_mapping[x][0])
    df["bias_type"] = df.formatter_name.map(lambda x: root_mapping[x][1])
    df["cot_type"] = df.formatter_name.map(lambda x: root_mapping[x][2])

    df["task_unbiased_hash"] = df.apply(lambda x: deterministic_hash(x["task_hash"] + x["unbiased_formatter"]), axis=1)
    df["task_along_cot_hash"] = df.apply(lambda x: deterministic_hash(x["task_hash"] + str(x["bias_type"])), axis=1)

    unbiased_responses = df[df.formatter_name == df.unbiased_formatter][["task_unbiased_hash", "parsed_response"]]
    unbiased_responses = unbiased_responses.rename(columns={"parsed_response": "unbiased_response"})
    unbiased_responses = unbiased_responses.drop_duplicates()

    biased_responses = df[df.formatter_name != df.unbiased_formatter][["task_unbiased_hash", "parsed_response"]]
    biased_responses = biased_responses.rename(columns={"parsed_response": "biased_response"})
    biased_responses = biased_responses.drop_duplicates()

    df = pd.merge(df, unbiased_responses, on="task_unbiased_hash", how="left")
    df = pd.merge(df, biased_responses, on="task_unbiased_hash", how="left")
    df["same_answer"] = df["unbiased_response"] == df["biased_response"]

    n_same_answer_no_cot = df[df.cot_type == "No-COT"].same_answer.sum()
    print(
        "Same answer, No-COT:",
        n_same_answer_no_cot,
        "out of",
        len(df[df.cot_type == "No-COT"]),
    )
    print("Same answer, COT:", df[df.cot_type == "COT"].same_answer.sum(), "out of", len(df[df.cot_type == "COT"]))

    if filter_on_same_start:
        df_same_answer = df[df.same_answer]  # This is all examples where the biased and unbiased responses are the same
        # we just want rows where the non cot response
        df_cot_hashes_same_answer = df_same_answer[df_same_answer.cot_type == "No-COT"].task_along_cot_hash.unique()
        df = df[df.task_along_cot_hash.isin(df_cot_hashes_same_answer)]
        assert len(df) == 2 * n_same_answer_no_cot

    df["response_category"] = df.apply(assign_response_category, axis=1)

    nodes_u, counts_u = create_sankey_data(df[df.bias_type.isnull()])
    nodes_b, counts_b = create_sankey_data(df[df.bias_type.notnull()])

    create_sankey_diagram(nodes_b, counts_b, f"Biased - {model}").show()
    create_sankey_diagram(nodes_u, counts_u, f"Unbiased - {model}").show()


if __name__ == "__main__":
    fire.Fire(main)
