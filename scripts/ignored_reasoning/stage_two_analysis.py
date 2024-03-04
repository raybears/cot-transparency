from typing import Optional
import fire
from matplotlib import pyplot as plt
from analysis import get_general_metrics
from cot_transparency.data_models.models import (
    StageTwoExperimentJsonFormat,
    TaskOutput,
)


from scripts.utils.simple_model_names import MODEL_SIMPLE_NAMES
from cot_transparency.data_models.data.bbh import BBH_TASK_LIST

# from cot_transparency.formatters.extraction import extract_answer
from string import ascii_uppercase

from cot_transparency.data_models.io import ExpLoader
from cot_transparency.formatters.transparency.trace_manipulation import get_cot_steps

from analysis import accuracy_for_df, TASK_MAP
import seaborn as sns
import numpy as np
import pandas as pd

# Used to produce human readable names on plots
NAMES_MAP = {
    "model": "Model",
    "task_name": "Task",
    "original_cot_trace_length": "CoT Length",
    "stage_one_formatter_name": "S1 Formatter",
}

sns.set_style(
    "ticks",
    {
        "axes.edgecolor": "0",
        "grid.linestyle": ":",
        "grid.color": "lightgrey",
        "grid.linewidth": "1.5",
        "axes.facecolor": "white",
    },
)

BREAK_WORDS: list[str] = [
    "answer is (",
    "answer is  (",
    "answer is: (",
    "answer is:(",
    "answer is:  (",
    "answer is:\n(",
    "answer is: \n(",
    "answer is:\n\n(",
    "answer is: ",
    "answer is ",
    "answer is $\\boxed{\\text{(",
    "answer is: $\\boxed{\\text{(",
    "choices is: " r"is: $\boxed{\textbf{(",
    "answer: ",
    "answer is ",
    r"is: $\boxed{\textbf{(",
    "choices is: ",
    r"is: $\boxed{\textbf{(",
    r"is $\boxed{\textbf{(",
    r"is: $\boxed{\text{(",
]


def extract_answer(model_answer: str, dump_failed: bool = False) -> Optional[str]:
    """
    Find answers in strings of the form "best answer is: (X)" and similar variants.
    """

    for break_word in BREAK_WORDS:
        if break_word not in model_answer:
            continue
        tmp = model_answer.split(break_word)
        # Sometimes there is a space in front of the answer
        last_item = tmp[-1].lstrip()

        if not last_item:
            continue

        ans = last_item[0]
        if ans in ascii_uppercase:
            return ans
    if dump_failed:
        with open("failed_answers.txt", "a") as f:
            f.write(model_answer + "\n")
    return None


def get_aoc(df: pd.DataFrame, x="cot_trace_length") -> pd.DataFrame:
    # get aoc for each original_cot_trace_length, grouped by task, and model
    # we want the counts for the number of traces so filter on unique stage_one_hash

    groups = ["task_name", "model", "original_cot_trace_length", x, "stage_one_formatter_name"]

    n_traces = df.groupby(groups).stage_one_hash.nunique().reset_index()
    n_traces = n_traces.rename(columns={"stage_one_hash": "n_traces"})

    # get aucs for each original_cot_trace_length, grouped by task, and model
    areas = df.groupby(groups).apply(lambda x: x["same_answer"].mean()).reset_index()

    def get_auc(group: pd.DataFrame) -> float:
        assert group["original_cot_trace_length"].nunique() == 1
        proportion_of_cot = group[x] / max(group[x])
        auc = np.trapz(group[0], x=proportion_of_cot)
        return auc

    groups.pop(groups.index(x))
    grouped = areas.groupby(groups)
    aucs = grouped.apply(get_auc).reset_index()
    aucs = aucs.rename(columns={0: "auc"})
    areas = pd.merge(aucs, n_traces, on=groups)

    grouped_std = df.groupby(groups).apply(lambda x: x["same_answer"].std()).reset_index().rename(columns={0: "std"})
    grouped_count = (
        df.groupby(groups).apply(lambda x: x["same_answer"].count()).reset_index().rename(columns={0: "count"})
    )
    areas = pd.merge(areas, grouped_std, on=groups)
    areas = pd.merge(areas, grouped_count, on=groups)

    areas["sem"] = areas["std"] / np.sqrt(areas["count"])

    groups.pop(groups.index("original_cot_trace_length"))
    areas["weighted_auc"] = areas["auc"] * areas["n_traces"]
    areas = areas.groupby(groups).sum().reset_index()
    areas["weighted_auc"] = areas["weighted_auc"] / areas["n_traces"]
    if (areas["weighted_auc"] > 1).any():
        areas["weighted_aoc"] = 100 - areas["weighted_auc"]
    else:
        areas["weighted_aoc"] = 1 - areas["weighted_auc"]

    print(areas)
    print(areas.to_csv())
    return areas


def convert_stage2_experiment_to_dataframe(
    exp: StageTwoExperimentJsonFormat, filter_for_correct: bool = False
) -> pd.DataFrame:
    out = []
    for task_output in exp.outputs:
        d_with_config = get_general_metrics(task_output)
        d_with_config["model"] = task_output.task_spec.inference_config.model
        d_with_config["task_name"] = task_output.task_spec.stage_one_output.task_spec.task_name
        d_with_config["ground_truth"] = task_output.task_spec.stage_one_output.task_spec.ground_truth
        d_with_config["stage_one_hash"] = task_output.task_spec.stage_one_output.task_spec.uid()
        d_with_config["stage_one_output_hash"] = task_output.task_spec.stage_one_output.uid()
        d_with_config["stage_one_output"] = task_output.task_spec.stage_one_output.dict()
        d_with_config["biased_ans"] = task_output.task_spec.stage_one_output.task_spec.biased_ans
        d_with_config["task_hash"] = task_output.task_spec.stage_one_output.task_spec.task_hash
        d_with_config["parsed_response"] = task_output.inference_output.parsed_response
        d_with_config["has_mistake"] = task_output.task_spec.trace_info.has_mistake  # type: ignore
        d_with_config["was_truncated"] = task_output.task_spec.trace_info.was_truncated  # type: ignore
        d_with_config["mistake_added_at"] = task_output.task_spec.trace_info.mistake_inserted_idx  # type: ignore
        d_with_config["original_cot_trace_length"] = len(task_output.task_spec.trace_info.original_cot)  # type: ignore
        modified_cot_length = get_cot_steps(task_output.task_spec.trace_info.get_complete_modified_cot())  # type: ignore
        d_with_config["cot_trace_length"] = len(modified_cot_length)
        d_with_config["stage_one_formatter_name"] = task_output.task_spec.stage_one_output.task_spec.formatter_name
        d_with_config["modified_cot"] = task_output.task_spec.trace_info.get_complete_modified_cot()  # type: ignore
        d_with_config["original_cot"] = task_output.task_spec.trace_info.original_cot  # type: ignore
        d_with_config["regenerated_cot_post_mistake"] = task_output.task_spec.trace_info.regenerated_cot_post_mistake  # type: ignore
        d_with_config["cot_post_mistake"] = task_output.task_spec.trace_info.regenerated_cot_post_mistake  # type: ignore
        d_with_config["sentence_with_mistake"] = task_output.task_spec.trace_info.sentence_with_mistake  # type: ignore

        out.append(d_with_config)

    df = pd.DataFrame(out)

    stage_one_output = [TaskOutput(**i) for i in df["stage_one_output"]]
    stage_formatter = [i.task_spec.formatter_name for i in stage_one_output]
    df["stage_one_formatter_name"] = stage_formatter
    return df


def extract_user_content(message_list, system: bool = False) -> None:  # type: ignore  # TODO fix
    for message in message_list:
        if message["role"][0] == "u":
            return message["content"]
    return None


def set_is_correct(row: pd.Series):
    if not row["has_mistake"] and not row["was_truncated"]:  # type: ignore
        return int(row["parsed_original_ans"] == row["ground_truth"])
    else:
        # we filter for these because for rows with mistakes:
        # parsed_response output is for the modified CoTs, not original ones
        return int(row["parsed_original_ans"] == row["ground_truth"])


def get_common_questions(df: pd.DataFrame, filter_on: str = "question") -> pd.DataFrame:
    question_models = df.groupby(filter_on)["model"].unique()
    question_models[question_models.apply(len) == df["model"].nunique()].index  # type: ignore
    gpt_3_5_turbo_questions = df[df["model"] == "gpt-3.5-turbo-0613"][filter_on].unique()  # type: ignore
    all_models_common_questions = question_models[question_models.apply(len) == df["model"].nunique()].index  # type: ignore
    common_to_gpt_and_all_models = set(gpt_3_5_turbo_questions).intersection(set(all_models_common_questions))
    print(len(common_to_gpt_and_all_models))
    df = df[df[filter_on].isin(common_to_gpt_and_all_models)]  # type: ignore
    return df


def filter_not_found_rows(df: pd.DataFrame) -> pd.DataFrame:
    n_not_found = len(df[df.parsed_response == "NOT_FOUND"])
    print(f"Number of NOT_FOUND rows: {n_not_found}")
    df = df[df.parsed_response != "NOT_FOUND"]  # type: ignore
    return df


def get_model_wise_questions(df: pd.DataFrame, filter_on: str = "question") -> pd.DataFrame:
    filtered_dfs = []
    models_list = df.model.unique()
    for model in models_list:
        df_filtered = df[df["model"] == model]
        df_filtered = df_filtered[df_filtered["question_w_mistake"].notna()]  # type: ignore
        df_filtered = df_filtered[df_filtered["cot_post_mistake"].notna()]  # type: ignore
        unique_questions_df = df_filtered.drop_duplicates(subset=[filter_on])  # type: ignore
        print(f"Model: {model}, Unique Questions: {unique_questions_df[filter_on].nunique()}")
        filtered_dfs.append(unique_questions_df)
    df_unique_per_model_q = pd.concat(filtered_dfs, ignore_index=True)
    return df_unique_per_model_q


def print_model_unique_count(df: pd.DataFrame) -> None:
    model_list = df["model"].unique()
    for model in model_list:
        df_filtered = df[df["model"] == model]
        print(f"Model: {model}, Unique count: {len(df_filtered)}")


def get_data_frame_from_exp_dir(exp_dir: str, filter_for_correct: bool = False) -> pd.DataFrame:
    loaded_dict = ExpLoader.stage_two(exp_dir, final_only=True)
    dfs = []
    for exp in loaded_dict.values():
        df = convert_stage2_experiment_to_dataframe(exp, filter_for_correct)
        dfs.append(df)
    df = pd.concat(dfs)

    df["question"] = df["messages"].apply(extract_user_content)
    df["parsed_original_ans"] = df["original_cot"].apply("".join).apply(extract_answer)
    df["parsed_modified_ans"] = df["modified_cot"].apply(extract_answer)

    # get parsed_response values for the original/modified cases
    original_empty_rows = df["parsed_modified_ans"].isna()
    original_rows = (~df["was_truncated"]) & (~df["has_mistake"]) & original_empty_rows
    df.loc[original_rows, "parsed_original_ans"] = df.loc[original_rows, "parsed_response"]

    modified_empty_rows = df["parsed_modified_ans"].isna()
    modified_rows = (~df["was_truncated"]) & (df["has_mistake"]) & modified_empty_rows
    df.loc[modified_rows, "parsed_modified_ans"] = df.loc[modified_rows, "parsed_response"]

    # df["is_correct"] = (df.parsed_response == df.ground_truth).astype(int)

    df["is_biased"] = (df.parsed_modified_ans == df.biased_ans).astype(int)
    # df["is_same"] = (df.parsed_original_ans == df.parsed_modified_ans).astype(int)
    df["is_correct"] = df.apply(set_is_correct, axis=1)

    # temp test
    # df["is_correct"] = (df.parsed_response == df.ground_truth).astype(int)
    # df["is_biased"] = (df.parsed_response == df.biased_ans).astype(int)

    # add more columns to the dataframe
    df["cot_before_mistake"] = df.apply(lambda row: "".join(row["original_cot"][: row["mistake_added_at"]]), axis=1)
    df["sentence_with_mistake"] = df["sentence_with_mistake"].str.replace(
        "Sentence with mistake added:", "", regex=False
    )
    df["modified_cot"] = df["modified_cot"].str.replace("Sentence with mistake added:", "", regex=False)
    df["question_w_mistake"] = df["question"] + "\n\n" + df["cot_before_mistake"] + df["sentence_with_mistake"]

    df.to_csv(f"{exp_dir}.csv")
    print(f"saved file at: {exp_dir}.csv")

    if filter_for_correct:
        # filter and get questions that the model knows how to solve correctly
        df = df[df["is_correct"] is True]
        df = get_common_questions(df)  # type: ignore

        print(len(df))
        df.to_csv("analyse1.csv")

        return df  # type: ignore

    df = filter_not_found_rows(df)
    print(f"length of df post filtering for not found rows: {len(df)}")
    return df


def plot_historgram_of_lengths(
    exp_dir: str,
):
    df = get_data_frame_from_exp_dir(exp_dir)

    hue = "task_name"
    x = "CoT Length"
    col = "model"
    y = "Counts"

    df = df.rename(columns={"original_cot_trace_length": x})

    # for histogram we want the counts of the original_cot_trace_length
    # filter on the unique stage_one_hash
    counts = df.groupby([hue, col, x]).stage_one_hash.nunique().reset_index()
    counts = counts.rename(columns={"stage_one_hash": y})

    # facet plot of the proportion of the trace, break down by original_cot_trace_length
    g = sns.FacetGrid(counts, col=col, col_wrap=2, legend_out=True)
    g.map_dataframe(sns.barplot, x=x, y=y, hue=hue)
    g.add_legend()
    plt.show()


def df_filters(
    df: pd.DataFrame,
    inconsistent_only: bool,
    aggregate_over_tasks: bool,
    model_filter: Optional[str],
    length_filter: Optional[list[int]],
) -> pd.DataFrame:
    if aggregate_over_tasks:
        # replace task_name with the "parent" task name using the task_map
        df["task_name"] = df["task_name"].replace(TASK_MAP)

    if inconsistent_only:
        df = df[df.biased_ans != df.ground_truth]  # type: ignore
        print("Number of inconsistent tasks: ", len(df))

    if model_filter:
        df = df[df.model.isin(model_filter)]  # type: ignore

    if length_filter:
        df = df[df["original_cot_trace_length"].isin(length_filter)]  # type: ignore
        assert len(df) > 0, "No data for this length filter"
    return df


def plot_by_length(df: pd.DataFrame, hue: str, col: Optional[str] = None) -> sns.FacetGrid | plt.Axes:
    x = "proportion_of_cot"
    y = "same_answer"

    # display as percentages
    df[x] = df[x] * 100
    df[y] = df[y] * 100
    if col is not None:
        # rename "original_cot_trace_length" to "CoT Length"
        df_for_plotting = df.rename(columns=NAMES_MAP, inplace=False)

        if col != "original_cot_trace_length":
            # need to round as aggregating over different cot lengths
            df_for_plotting[x] = df_for_plotting[x].astype(float)
            df_for_plotting[x] = (df_for_plotting[x] / 10).round() * 10

        # facet plot of the proportion of the trace, break down by original_cot_trace_length
        g: sns.FacetGrid = sns.relplot(
            data=df_for_plotting,
            x=x,
            y=y,
            hue=NAMES_MAP[hue],
            col=NAMES_MAP[col],
            col_wrap=2,
            kind="line",
            height=4,
        )  # type: ignore
        x_label = "Proportion of CoT"
        y_label = "% Same Answer as Unmodified CoT"
        g.set_axis_labels(x_label, y_label)
        return g

    else:
        # plot aggreaged version that aggregates over cot lenghts
        plt.figure()
        df["proportion_of_cot_rounded"] = (df["proportion_of_cot"] / 10).round() * 10
        # g is an axis
        g_axs: plt.Axes = sns.lineplot(df, x="proportion_of_cot_rounded", y=y, hue=hue, alpha=0.5)
        return g_axs


def drop_not_found(df: pd.DataFrame) -> pd.DataFrame:
    # Redrop any NOT_FOUND
    pre = len(df)
    df = df[df.same_answer != "NOT_FOUND"]  # type: ignore
    print("Dropped ", pre - len(df), " rows becuase of NOT_FOUND answers in full cot trace")
    return df


def baseline_accuracy(df: pd.DataFrame, hue: str, col: str):
    # drop has mistake
    df = df[~df.has_mistake]  # type: ignore

    cot_trace_length_0 = df[df["cot_trace_length"] == 0]
    cot_trace_length_max = df[df["cot_trace_length"] == df["original_cot_trace_length"]]
    baseline_accuracies = (
        cot_trace_length_0.groupby(["task_name", hue, col]).apply(lambda x: x["is_correct"].mean()).reset_index()
    )
    baseline_accuracies = baseline_accuracies.rename(columns={0: "No CoT Accuracy"})
    baseline_accuracies["CoT Accuracy"] = (
        cot_trace_length_max.groupby(["task_name", hue, col]).apply(lambda x: x["is_correct"].mean()).reset_index()[0]
    )
    print(baseline_accuracies)
    # print csv version
    print(baseline_accuracies.to_csv())


def check_same_answer(group: pd.DataFrame) -> pd.DataFrame:
    max_step_row = group[(~group.was_truncated) & ~(group.has_mistake)]
    if len(max_step_row) == 0:
        group["same_answer"] = "NOT_FOUND"
    # elif len(max_step_row) > 1:
    #     raise ValueError(
    #         "More than one row with max cot_trace_length you may "
    #         "have changed the prompt formatter half way throgh the exp"
    #     )
    else:
        group["same_answer"] = group["parsed_response"] == max_step_row["parsed_response"].iloc[0]  # type: ignore
    return group


def plot_early_answering(
    exp_dir: str,
    show_plots: bool = False,
    inconsistent_only: bool = False,
    aggregate_over_tasks: bool = False,
    model_filter: Optional[str] = None,
    length_filter: Optional[list[int]] = None,
    col: str = "original_cot_trace_length",
    hue: str = "model",
):
    df = get_data_frame_from_exp_dir(exp_dir)
    # drop formatters that have Mistake in the name
    df = df[~df.has_mistake]
    df = df_filters(df, inconsistent_only, aggregate_over_tasks, model_filter, length_filter)  # type: ignore

    # Apply the check_same_answer function
    df = df.groupby("stage_one_hash").apply(check_same_answer).reset_index(drop=True)
    df = drop_not_found(df)

    df["proportion_of_cot"] = df["cot_trace_length"] / df["original_cot_trace_length"]

    g = plot_by_length(df, hue, col)
    # set tile to Early Answering
    if isinstance(g, sns.FacetGrid):
        g.fig.suptitle("Early Answering")
    else:
        g.set_title("Early Answering")
    get_aoc(df)

    if show_plots:
        plt.show()


def plot_adding_mistakes(
    exp_dir: str,
    show_plots: bool = False,
    inconsistent_only: bool = False,
    aggregate_over_tasks: bool = False,
    model_filter: Optional[str] = None,
    length_filter: Optional[list[int]] = None,
    col: str = "original_cot_trace_length",
    hue: str = "model",
):
    df = get_data_frame_from_exp_dir(exp_dir)
    df = df[~df.was_truncated]

    df = df_filters(df, inconsistent_only, aggregate_over_tasks, model_filter, length_filter)  # type: ignore

    df = df.groupby("stage_one_hash").apply(check_same_answer).reset_index(drop=True)
    df = drop_not_found(df)

    df["proportion_of_cot"] = df["mistake_added_at"] / df["original_cot_trace_length"]

    g = plot_by_length(df, hue, col)
    if isinstance(g, sns.FacetGrid):
        g.fig.suptitle("Adding Mistakes")
    else:
        g.set_title("Adding Mistakes")
    get_aoc(df)

    if show_plots:
        plt.show()


def map_model_names(df: pd.DataFrame, paper_plot: bool = False) -> pd.DataFrame:
    df["model"] = df["model"].map(lambda x: MODEL_SIMPLE_NAMES[x] if x in MODEL_SIMPLE_NAMES else x)
    if paper_plot:
        df["model"] = df["model"].str.replace("gpt-3.5-turbo-0613", "GPT-3.5-Turbo", regex=False)
        df["model"] = df["model"].str.replace("Control-8UN5nhcE", "Control", regex=False)
        df["model"] = df["model"].str.replace("Intervention-8UNAODuA", "Intervention", regex=False)
    return df


def get_mistake_rows(df: pd.DataFrame) -> pd.DataFrame:
    df_mistakes = df[~df.was_truncated]
    df_mistakes = df_mistakes[df_mistakes.has_mistake]
    df_mistakes["is_correct"] = (df_mistakes["parsed_modified_ans"] == df_mistakes["ground_truth"]).astype(int)
    print(f"length of df_mistakes: {len(df_mistakes)}")
    return df_mistakes  # type: ignore


def get_unmodified_rows(df: pd.DataFrame) -> pd.DataFrame:
    df_unmodified = df[~df.was_truncated]
    df_unmodified = df_unmodified[~df_unmodified.has_mistake]
    print(f"length of df_unmodified: {len(df_unmodified)}")
    return df_unmodified  # type: ignore


def gen_accuracy_plot(
    df: pd.DataFrame, task: str = "", reorder_indices: Optional[list[int]] = None, modified: bool = False
):
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")
    sns.set_style("whitegrid")

    x_order = df.model.unique()
    if reorder_indices:
        x_order = [x_order[i] for i in reorder_indices]

    palette_colors = plt.cm.Set3(np.linspace(0, 1, len(x_order)))  # type: ignore

    chart = sns.barplot(
        x="model",
        y="is_correct",
        data=df,
        errorbar=("ci", 95),
        capsize=0.05,
        errwidth=1,
        order=x_order,
        palette=palette_colors,
        edgecolor="black",
    )
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")  # type: ignore

    ax = chart.axes
    for p in ax.patches:  # type: ignore
        if p.get_height() != 1.00:  # type: ignore
            ax.text(  # type: ignore
                p.get_x() + p.get_width() / 2.0,  # type: ignore
                p.get_height(),  # type: ignore
                f"{p.get_height():.2f}",  # type: ignore
                fontsize=12,
                ha="center",
                va="bottom",
            )

    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    questions_count = df.groupby("model")["question"].nunique()
    print(questions_count)
    is_correct_count = df.groupby("model")["is_correct"].sum()
    print("is_correct_count")
    print(is_correct_count)
    if modified:
        plt.title(f"{task} Accuracy for COTs with Mistake | n = {questions_count.iloc[0]}")
    else:
        plt.title(f"{task} Accuracy for Unmodified COTs | n = {questions_count.iloc[0]} | {df.task_name.unique()}")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def gen_mistake_plot(df_mistakes: pd.DataFrame, title: str = "", reorder_indices: Optional[list[int]] = None):
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")
    sns.set_style("whitegrid")

    x_order = df_mistakes.model.unique()
    if reorder_indices:
        x_order = [x_order[i] for i in reorder_indices]
        # filtered_df = df_mistakes[df_mistakes['model'].isin(x_order)]
        # filtered_df.to_csv('filtered_df.csv')

    palette_colors = plt.cm.Set3(np.linspace(0, 1, len(x_order)))  # type: ignore

    chart = sns.barplot(
        x="model",
        y="is_same",
        data=df_mistakes,
        errorbar=("ci", 95),
        capsize=0.05,
        errwidth=1,
        order=x_order,
        palette=palette_colors,
        edgecolor="black",
    )
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")  # type: ignore

    ax = chart.axes
    for p in ax.patches:  # type: ignore
        if p.get_height() != 1.00:  # type: ignore
            ax.text(  # type: ignore
                p.get_x() + p.get_width() / 2.0,  # type: ignore
                p.get_height(),  # type: ignore
                f"{p.get_height():.2f}",  # type: ignore
                fontsize=12,
                ha="center",
                va="bottom",
            )

    plt.xlabel("Model")
    # plt.ylabel('Mistake Aligned with Bias %')
    plt.ylabel("Answers aligned with Bias %")
    questions_count = df_mistakes.groupby("model")["question_w_mistake"].nunique()
    print(questions_count)
    is_biased_count = df_mistakes.groupby("model")["is_biased"].sum()
    print("is_biased_count")
    print(is_biased_count)
    # plt.title(f'{title} Mistake following aligned with bias | n = {questions_count.iloc[0]} | {df_mistakes.task_name.unique()}')
    plt.title(f"{title} Role Playing Bias | n = {questions_count.iloc[0]} | {df_mistakes.task_name.unique()}")
    plt.ylim(0, 0.5)
    plt.tight_layout()
    plt.show()


def gen_dataset_plot(df_mistakes: pd.DataFrame, title: str = ""):
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")
    sns.set_style("whitegrid")

    x_order = df_mistakes.task_name.unique()
    palette_colors = plt.cm.Set3(np.linspace(0, 1, len(x_order)))  # type: ignore

    chart = sns.barplot(
        x="task_name",
        y="is_biased",
        data=df_mistakes,
        errorbar=("ci", 95),
        capsize=0.05,
        errwidth=1,
        order=x_order,
        palette=palette_colors,
        edgecolor="black",
    )
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")  # type: ignore

    ax = chart.axes
    for p in ax.patches:  # type: ignore
        if p.get_height() != 1.00:  # type: ignore
            ax.text(  # type: ignore
                p.get_x() + p.get_width() / 2.0,  # type: ignore
                p.get_height(),  # type: ignore
                f"{p.get_height():.2f}",  # type: ignore
                fontsize=12,
                ha="center",
                va="bottom",
            )

    plt.xlabel("Dataset")
    plt.ylabel("Mistake Aligned with Bias %")
    questions_count = df_mistakes.groupby("model")["question_w_mistake"].nunique()
    print(questions_count)
    is_biased_count = df_mistakes.groupby("model")["is_biased"].sum()
    print("is_biased_count")
    print(is_biased_count)
    plt.title(f"{title} Mistake following aligned with bias | n = {questions_count.iloc[0]}")
    plt.ylim(0, 0.8)
    plt.tight_layout()
    plt.show()


def combine_bbh_tasks(df: pd.DataFrame) -> pd.DataFrame:
    df["task_name"] = df["task_name"].apply(lambda x: "bbh" if x in BBH_TASK_LIST else x)
    return df


def mistake_aligned_plot(
    exp_dir: str,
    show_plots: bool = False,
    inconsistent_only: bool = False,
    aggregate_over_tasks: bool = True,
    dataset_filter: Optional[str] = None,
    model_filter: Optional[str] = None,
    length_filter: Optional[list[int]] = None,
    filter_for_correct: bool = False,
    filter_for_incorrect: Optional[bool] = False,
    task_wise_plots: bool = False,
    filter_common_questions: bool = False,
    single_model_dataset_plot: str = "",
    bon: int = 0,
    max_vote: int = 0,
    reorder_indices: Optional[list[int]] = None,
    remove_n_last_mistake_pos: Optional[int] = 0,
    filter_for_mistake_halves: Optional[bool] = False,
    filter_for_correct_no_cot: Optional[bool] = False,
    filter_for_incorrect_no_cot: Optional[bool] = False,
    filter_for_mistake_on_correct: Optional[bool] = False,
):
    df = get_data_frame_from_exp_dir(exp_dir, filter_for_correct)
    df = map_model_names(df)
    if dataset_filter is not None:
        df = df[df.task_name != dataset_filter]

    df_unmodified = get_unmodified_rows(df)  # type: ignore

    df = df[df.ground_truth != df.biased_ans]

    df_mistakes = get_mistake_rows(df)  # type: ignore

    if filter_for_correct and not task_wise_plots:
        gen_mistake_plot(df_mistakes, reorder_indices=reorder_indices, title="Filtered for Correct CoTs | ")
        gen_accuracy_plot(df_mistakes, reorder_indices=reorder_indices, modified=True)
        gen_accuracy_plot(df_unmodified, reorder_indices=reorder_indices)
    elif filter_for_incorrect:
        df_mistakes = df_mistakes[df_mistakes.is_correct is False]
        gen_mistake_plot(df_mistakes, reorder_indices=reorder_indices, title="Filtered for Incorrect CoTs | ")  # type: ignore
        gen_accuracy_plot(df_mistakes, reorder_indices=reorder_indices, modified=True)  # type: ignore
        gen_accuracy_plot(df_unmodified, reorder_indices=reorder_indices)
    elif filter_for_mistake_on_correct:
        df_mistakes = df_mistakes[df_mistakes["biased_ans"] == df_mistakes["ground_truth"]]
        print_model_unique_count(df_mistakes)  # type: ignore
        gen_mistake_plot(
            df_mistakes,  # type: ignore
            reorder_indices=reorder_indices,
            title="Filtered for Mistake on Correct Option (Bias Option == Ground Truth) | ",
        )
    elif filter_for_mistake_halves:
        conditions = [
            (df_mistakes["mistake_added_at"] < (df_mistakes["n_steps_in_cot_trace"] * 0.5))
            & (df_mistakes["mistake_added_at"] >= 0),
            (df_mistakes["mistake_added_at"] >= (df_mistakes["n_steps_in_cot_trace"] * 0.5))
            & (df_mistakes["mistake_added_at"] < df_mistakes["n_steps_in_cot_trace"]),
        ]
        choices = [1, 2]
        df_mistakes["mistake_half"] = np.select(conditions, choices, default=0)

        df_mistake_1 = df_mistakes[df_mistakes["mistake_half"] == 1]
        df_mistake_2 = df_mistakes[df_mistakes["mistake_half"] == 2]
        gen_mistake_plot(df_mistake_1, reorder_indices=reorder_indices, title="Mistakes added in the first half | ")  # type: ignore
        gen_mistake_plot(df_mistake_2, reorder_indices=reorder_indices, title="Mistakes added in the second half | ")  # type: ignore
        gen_accuracy_plot(
            df_mistake_1,  # type: ignore
            reorder_indices=reorder_indices,
            modified=True,
            task="Accuracy for mistakes added in the first half | ",
        )
        gen_accuracy_plot(
            df_mistake_2,  # type: ignore
            reorder_indices=reorder_indices,
            modified=True,
            task="Accuracy for mistakes added in the second half | ",
        )
        gen_accuracy_plot(df_unmodified, reorder_indices=reorder_indices)
    elif filter_for_correct_no_cot:
        no_cot_correct_df = df[(df["cot_trace_length"] == 0) & (df["is_correct"] == 1)]
        print(f"no cot correct df len: {len(no_cot_correct_df)}")
        df_mistakes = df_mistakes[df_mistakes["question"].isin(no_cot_correct_df["question"])]  # type: ignore
        print(f"len now: {len(df_mistakes)}")
        gen_mistake_plot(df_mistakes, reorder_indices=reorder_indices, title="Filtered for Correct No COTs | ")  # type: ignore
        # get accuracy plot for no_cot_correct_df
        gen_accuracy_plot(no_cot_correct_df, reorder_indices=reorder_indices, task="Accuracy for Correct No COTs | ")  # type: ignore
    elif filter_for_incorrect_no_cot:
        no_cot_incorrect_df = df[(df["cot_trace_length"] == 0) & (df["is_correct"] == 0)]
        df_mistakes = df_mistakes[df_mistakes["question"].isin(no_cot_incorrect_df["question"])]  # type: ignore
        gen_mistake_plot(df_mistakes, reorder_indices=reorder_indices, title="Filtered for Incorrect No COTs | ")  # type: ignore
        gen_accuracy_plot(
            no_cot_incorrect_df, reorder_indices=reorder_indices, task="Accuracy for Incorrect No COTs | "  # type: ignore
        )
    elif filter_common_questions:
        df = get_common_questions(df)  # type: ignore
        gen_mistake_plot(df_mistakes, reorder_indices=reorder_indices)
        gen_accuracy_plot(df_mistakes, reorder_indices=reorder_indices, modified=True)
        gen_accuracy_plot(df_unmodified)
    elif single_model_dataset_plot:
        df_mistakes = df_mistakes[df_mistakes.model == single_model_dataset_plot]
        df_mistakes = combine_bbh_tasks(df_mistakes)  # type: ignore
        gen_dataset_plot(df_mistakes, single_model_dataset_plot)
    elif remove_n_last_mistake_pos:
        df_mistakes = df_mistakes[
            df_mistakes.n_steps_in_cot_trace - df_mistakes.mistake_added_at > remove_n_last_mistake_pos
        ]
        gen_mistake_plot(df_mistakes, reorder_indices=reorder_indices)  # type: ignore
        gen_accuracy_plot(df_mistakes, reorder_indices=reorder_indices, modified=True)  # type: ignore
        gen_accuracy_plot(df_unmodified, reorder_indices=reorder_indices)
    else:
        if not task_wise_plots:
            gen_mistake_plot(df_mistakes, reorder_indices=reorder_indices)
            gen_accuracy_plot(df_mistakes, reorder_indices=reorder_indices, modified=True)
            gen_accuracy_plot(df_unmodified, reorder_indices=reorder_indices)
        else:
            df_mistakes = combine_bbh_tasks(df_mistakes)
            df_unmodified = combine_bbh_tasks(df_unmodified)
            tasks = df_mistakes.task_name.unique()
            for task in tasks:
                df_task_mistake = df_mistakes[df_mistakes.task_name == task]
                if filter_for_correct:
                    df_task_mistake = df_task_mistake[df_task_mistake.is_correct is True]
                    gen_mistake_plot(df_task_mistake, task, reorder_indices=reorder_indices)  # type: ignore
                else:
                    df_task_unmodified = df_unmodified[df_unmodified.task_name == task]
                    gen_mistake_plot(df_task_mistake, task, reorder_indices=reorder_indices)  # type: ignore
                    gen_accuracy_plot(df_task_unmodified, task, reorder_indices=reorder_indices)  # type: ignore


def _accuracy_plot(
    df: pd.DataFrame,
    x_order: list[str],
    title: str = "",
    ylabel: str = "Accuracy",
    ylim: float = 1.0,
    reorder_indices: Optional[list[int]] = None,
    paper_plot: bool = False,
) -> None:
    # Prepare the plot
    kwargs = {}

    if paper_plot:
        # fig, ax = plt.subplots(figsize=(7.7 / 2.54, 6 / 2.54))
        fig, ax = plt.subplots(figsize=(3.36, 3.35))
        sns.set_context("notebook", font_scale=1.0)
        sns.set_style("whitegrid", {"axes.grid": False})
        sns.set_style(
            "ticks",
            {
                "axes.edgecolor": "0",
                "grid.linestyle": "",
                "axes.facecolor": "white",
                "font.family": ["Times New Roman Cyr"],
            },
        )
        # palette_colors = ["#C6E0FE", "#FADFA6", "#FF6568", "#CBF6C8", "#FDF8A1"]
        ax.spines["bottom"].set_linewidth(1.5)  # type: ignore
        ax.spines["left"].set_linewidth(1.5)  # type: ignore
        plt.tick_params(axis="x", which="major", length=6, width=1.5)
        plt.tick_params(axis="y", which="major", length=6, width=1.5)
        for label in ax.get_xticklabels():  # type: ignore
            label.set_fontsize(label.get_size() - 1)
        palette_colors = ["#43669d", "#d27f56", "#549c67"]
        kwargs = {"capsize": 0, "errwidth": 1.5, "edgecolor": None, "palette": palette_colors}
    else:
        plt.figure(figsize=(12, 8))
        sns.set(style="whitegrid")
        palette_colors = plt.cm.Set3(np.linspace(0, 1, len(x_order)))  # type: ignore
        kwargs["palette"] = palette_colors
        kwargs = {"palette": palette_colors, "capsize": 0.05, "errwidth": 1, "edgecolor": "black"}

    chart = sns.barplot(
        x="model",
        y="is_correct",
        data=df,
        errorbar=("ci", 95),
        order=x_order,
        **kwargs,  # type: ignore
    )

    plt.ylabel(ylabel)
    plt.ylim(0, ylim)

    if not paper_plot:
        chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment="right")  # type: ignore
        ax = chart.axes
        for p in ax.patches:  # type: ignore
            ax.text(  # type: ignore
                p.get_x() + p.get_width() / 2.0,  # type: ignore
                p.get_height(),  # type: ignore
                f"{p.get_height():.2f}",  # type: ignore
                fontsize=12,
                ha="center",
                va="bottom",
            )
        questions_count = df.groupby("model")["input_hash"].nunique()
        print(questions_count)
        plt.title(
            f"{title} | {df.task_name.unique()} | n = {round(questions_count.mean(),2)} ± {round(questions_count.std(), 2)}"
        )
        plt.xlabel("Model")
    else:
        sns.despine()
        plt.title(title)
        plt.xlabel("")
        plt.savefig("plots/winogender_plot.pdf", bbox_inches="tight", pad_inches=0.01)

    plt.tight_layout()
    plt.show()


def accuracy_plot(
    exp_dir: str,
    title: str = "",
    ylabel: str = "Accuracy",
    ylim: float = 1.0,
    filter_na: bool = False,
    task_wise_plots: bool = False,
    dataset_filter: Optional[str] = None,
    reorder_indices: Optional[list[int]] = None,
    paper_plot: bool = False,
):
    df = get_data_frame_from_exp_dir(exp_dir)
    df = map_model_names(df, paper_plot)
    if dataset_filter is not None:
        df = df[df.task_name != dataset_filter]
    if filter_na:
        df = df[df.parsed_response.notna()]

    df["is_correct"] = (df["parsed_response"] == df["ground_truth"]).astype(int)

    # Determine the order of models in the x-axis
    x_order = df.model.unique()  # type: ignore
    if reorder_indices:
        x_order = [x_order[i] for i in reorder_indices]

    if not task_wise_plots:
        _accuracy_plot(df, x_order, title, ylabel, ylim, reorder_indices, paper_plot)  # type: ignore
    else:
        tasks_list = df.task_name.unique()  # type: ignore
        for task in tasks_list:
            df_task = df[df.task_name == task]  # type: ignore
            _accuracy_plot(df_task, x_order, title, ylabel, ylim, reorder_indices, paper_plot)  # type: ignore


def aoc_plot(
    exp_dir: str,
    show_plots: bool = False,
    inconsistent_only: bool = False,
    aggregate_over_tasks: bool = False,
    model_filter: Optional[str] = None,
    length_filter: Optional[list[int]] = None,
    hue: str = "stage_one_formatter_name",
    filter_for_correct: bool = False,
):
    df = get_data_frame_from_exp_dir(exp_dir, filter_for_correct)
    df = df_filters(df, inconsistent_only, aggregate_over_tasks, model_filter, length_filter)
    df = map_model_names(df)

    # Mistakes AoC
    df_mistakes = df[~df.was_truncated]
    df_mistakes = df.groupby("stage_one_hash").apply(check_same_answer).reset_index(drop=True)
    df_mistakes = drop_not_found(df_mistakes)
    print(f"length of df_mistakes: {len(df_mistakes)}")
    aoc_mistakes = get_aoc(df_mistakes)
    print(f"length of aoc_mistakes: {len(aoc_mistakes)}")

    # Early Answering AoC
    df_early = df[~df.has_mistake]
    df_early = df_early.groupby("stage_one_hash").apply(check_same_answer).reset_index(drop=True)
    df_early = drop_not_found(df_early)  # type: ignore
    aoc_early = get_aoc(df_early)

    # baseline accuracies
    baseline_accuracy(df, hue, "model")

    _aoc_point_plot(hue, df, aoc_mistakes, aoc_early, kind="bar")
    _aoc_point_plot(hue, df, aoc_mistakes, aoc_early, kind="point")

    if show_plots:
        plt.show()


def _aoc_point_plot(hue: str, df: pd.DataFrame, aoc_mistakes: pd.DataFrame, aoc_early: pd.DataFrame, kind="bar"):
    # two point plots side by side [mistakes, early answering, accuracy]
    fig, axs = plt.subplots(1, 3, figsize=(10, 6))

    if kind == "point":
        func = sns.pointplot
        kwargs = {"join": False}
    elif kind == "bar":
        func = sns.barplot
        kwargs = {}
    else:
        raise ValueError(f"kind must be pf# remooint or bar, not {kind}")

    x_order = df.model.unique()

    func(
        data=aoc_mistakes,
        x="model",
        y="weighted_aoc",
        hue=hue,
        errorbar="ci",
        ax=axs[0],
        capsize=0.05,
        errwidth=1,
        order=x_order,
        **kwargs,  # type: ignore
    )

    # # Then, manually overlay error bars using Matplotlib
    hue_order = aoc_mistakes["stage_one_formatter_name"].unique()

    # # # Adjusting hue_offsets to be centered correctly
    bar_width = 0
    np.linspace(-bar_width, bar_width, len(hue_order))  # Equally spaced offsets

    axs[0].set_title("Mistakes")
    axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=25, ha="right")  # type: ignore

    func(
        data=aoc_early,
        x="model",
        y="weighted_aoc",
        hue=hue,
        errorbar="ci",
        ax=axs[1],
        capsize=0.05,
        errwidth=1,
        order=x_order,
        **kwargs,  # type: ignore
    )

    # Then, manually overlay error bars using Matplotlib
    hue_order = aoc_early["stage_one_formatter_name"].unique()

    # # Adjusting hue_offsets to be centered correctly
    bar_width = 0
    np.linspace(-bar_width, bar_width, len(hue_order))  # Equally spaced offsets

    axs[1].set_title("Early Answering")
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=25, ha="right")  # type: ignore
    # axs[1].legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

    for ax in axs:  # type: ignore
        ax.set_ylabel("Weighted AoC")  # type: ignore
        ax.set_xlabel("Model")  # type: ignore

    # filter onto the ones wihout mistakes and no truncation
    acc = df[~df.has_mistake]
    acc = acc[~acc.was_truncated]

    grouped_acc = acc.groupby(["model", "stage_one_formatter_name"])["is_correct"].agg(["mean", "std", "count"])  # type: ignore
    grouped_acc["sem"] = grouped_acc["std"] / np.sqrt(grouped_acc["count"])  # type: ignore

    # print(grouped_acc["sem"].values)

    func(
        data=acc,  # type: ignore
        x="model",
        y="is_correct",
        hue=hue,
        errorbar="ci",  # Disable Seaborn's built-in error bars
        ax=axs[2],
        capsize=0.05,
        errwidth=1,
        order=x_order,
        **kwargs,  # type: ignore
    )

    # 3. Manually overlay error bars using Matplotlib
    hue_order = acc["stage_one_formatter_name"].unique()  # type: ignore

    # Adjusting hue_offsets to be centered correctly
    bar_width = 0.25
    np.linspace(-bar_width, bar_width, len(hue_order))  # Equally spaced offsets

    axs[2].set_title("Accuracy for Complete Unmodified CoT")  # type: ignore
    axs[2].set_ylabel("Accuracy")  # type: ignore
    axs[2].set_xlabel("Model")  # type: ignore
    axs[2].set_xticklabels(axs[2].get_xticklabels(), rotation=25, ha="right")  # type: ignore

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0))

    max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[2].set_ylim(0, 1)

    for ax in axs:
        for p in ax.patches:
            if p.get_height() != 1.00:
                ax.text(
                    p.get_x() + p.get_width() / 2.0,
                    p.get_height(),
                    f"{p.get_height():.2f}",
                    fontsize=12,
                    ha="center",
                    va="bottom",
                )

    for ax in axs:
        ax.get_legend().remove()  # type: ignore

    questions_count = df.groupby("model")["question"].nunique()
    print("questions_count")
    print(questions_count)
    fig.suptitle(
        f"Ignored Reasoning: Consistency Finetuned Models\nn = {questions_count.iloc[0]} [logiqa, truthful_qa, mmlu]"
    )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)

    fig, axs = plt.subplots(figsize=(10, 5))

    acc1 = df[~df.has_mistake & ~df.was_truncated]
    acc1 = acc1.assign(Condition="Complete Unmodified CoT")

    acc2 = df[df.has_mistake & ~df.was_truncated]
    acc2_per_hash = (
        acc2.groupby(["stage_one_hash", "model", "stage_one_formatter_name"])["is_correct"].mean().reset_index()
    )
    acc2_per_hash = acc2_per_hash.assign(Condition="Modified CoTs")

    cot_trace_length_0 = df[df["cot_trace_length"] == 0]
    no_cot_accuracies = (
        cot_trace_length_0.groupby(["task_name", "stage_one_formatter_name", "model"])["is_correct"]
        .mean()
        .reset_index()
    )
    no_cot_accuracies = no_cot_accuracies.assign(Condition="No CoT")

    # Combine the data into a single dataframe
    combined_data = pd.concat([acc1, acc2_per_hash, no_cot_accuracies])

    axs.set_ylim(0, 1)  # type: ignore

    func(
        data=combined_data,
        x="model",
        y="is_correct",
        hue="Condition",
        errorbar="se",
        ax=axs,  # type: ignore
        capsize=0.05,
        errwidth=1,
        order=x_order,
        **kwargs,  # type: ignore
    )
    axs.set_title("Accuracy for No CoT vs. CoT vs. CoT with Mistake")  # type: ignore
    axs.set_ylabel("Accuracy")  # type: ignore
    axs.set_xlabel("Model")  # type: ignore
    axs.set_xticklabels(axs.get_xticklabels(), rotation=45, ha="right")  # type: ignore
    axs.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)  # type: ignore

    # Loop through patches to print the accuracies on top of the bars
    for p in axs.patches:  # type: ignore
        axs.text(  # type: ignore
            p.get_x() + p.get_width() / 2.0,
            p.get_height(),
            f"{p.get_height():.2f}",
            fontsize=12,
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)


def accuracy(
    exp_dir: str,
    inconsistent_only: bool = True,
    stage_two_formatter_name: str = "EarlyAnsweringFormatter",
    aggregate_over_tasks: bool = False,
    step_filter: Optional[list[int]] = None,
):
    """
    This does a similar thing to the accuracy function in analysis.py, but it uses the stage_two data
    """
    df = get_data_frame_from_exp_dir(exp_dir)
    df = df[df.formatter_name == stage_two_formatter_name]

    # replace formatter_name with stage_one_formatter_name
    # as we want to compare the accuracy of the stage_one formatter
    df["formatter_name"] = df["stage_one_formatter_name"]

    if step_filter:
        df = df[df.cot_trace_length.isin(step_filter)]
        check_counts = False
        # filtering on step means we no longer guarateed to have the same number of samples for each task
        # so we don't want to check the counts
    else:
        check_counts = True

    print("----- Accuracy for step = 0 --------------")
    no_cot_df = df[df["step_in_cot_trace"] == 0]  # type: ignore
    accuracy_for_df(
        no_cot_df,  # type: ignore
        inconsistent_only=inconsistent_only,
        check_counts=check_counts,
        aggregate_over_tasks=aggregate_over_tasks,
    )

    print("----- Accuracy for step = max_step --------------")
    cot_df = df[df["step_in_cot_trace"] == df["max_step_in_cot_trace"]]  # type: ignore
    accuracy_for_df(
        cot_df,  # type: ignore
        inconsistent_only=inconsistent_only,
        check_counts=check_counts,
        aggregate_over_tasks=aggregate_over_tasks,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "hist": plot_historgram_of_lengths,
            "early": plot_early_answering,
            "mistakes": plot_adding_mistakes,
            "accuracy": accuracy,
            "aoc": aoc_plot,
            "mistake_aligned_plot": mistake_aligned_plot,
            "accuracy_plot": accuracy_plot,
        }
    )
