import asyncio
import json
from pathlib import Path
from typing import Sequence
from grugstream import Observable

import pandas as pd
from slist import Group, Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.base import CachedPerModelCaller, ModelCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig, config_from_default
from cot_transparency.data_models.hashable import HashableBaseModel
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.data_models.pd_utils import DataRow
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter
from cot_transparency.formatters.more_biases.user_wrong_cot import (
    DistractorAnswerWithoutInfluence,
    DistractorArgumentNotsure,
    ImprovedDistractorArgument,
    DistractorArgumentCorrectOrWrong,
    DistractorArgumentImportant,
    DistractorArgumentNoTruthfullyAnswer,
    ReadOnInternetNoCotFormatter,
)
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.are_you_sure.eval_are_you_sure_second_cot import (
    are_you_sure_to_data_row,
    run_are_you_sure_multi_model_second_round_cot,
    OutputWithAreYouSure,
)
from scripts.automated_answer_parsing.answer_parsing_example import answer_finding_step
from scripts.evaluate_judge_consistency.judge_consistency import eval_judge_for_models_inconsistency
from scripts.evil_grid_exp.message_to_csv_display import messages_to_str
from scripts.inverse_scaling_experiments.run_hindsight_neglect import (
    hindsight_to_data_row,
    run_hindsight_neglect_only_non_spurious,
    run_hindsight_neglect_for_models,
)
from scripts.training_formatters import (
    ANSWER_CHOICE_NAME,
    FORMATTERS_TO_PAPER_NAME,
    INTERESTING_FORMATTERS,
    INTERESTING_FORMATTERS_COT_AND_NO_COT,
    TRAINING_COT_FORMATTERS,
    TRAINING_NO_COT_FORMATTERS,
)

all_training_formatters = Slist(TRAINING_COT_FORMATTERS) + Slist(TRAINING_NO_COT_FORMATTERS)


class ModelMeta(HashableBaseModel):
    name: str
    bias_name: str

    def __hash__(self) -> int:
        return int(self.model_hash(), 16)


def accuracy_for_biases(tasks: Slist[TaskOutput]) -> Slist[Group[str, float]]:
    # group by formatter
    grouped = tasks.group_by(lambda x: x.task_spec.formatter_name).map(
        lambda group: group.map_values(lambda task_list: task_list.map(lambda task: task.is_correct).average_or_raise())
    )
    return grouped


INTERESTING_FORMATTERS_STR = [x.name() for x in INTERESTING_FORMATTERS_COT_AND_NO_COT]


def make_heading_name(name: str, model: str) -> str:
    return f"{name} ({model})"


def percent_matching_bias(seq: Sequence[DataRow]) -> float:
    return sum(1 for row in seq if row.parsed_ans_matches_bias) / len(seq) * 100


def accuracy(seq: Sequence[DataRow]) -> float:
    return sum(1 for row in seq if row.is_correct) / len(seq) * 100


def level_bias_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    new_pivot = dataframe.pivot_table(
        columns="model_type",
        index="bias_name",
        values="percent_matching_bias",
        aggfunc={"percent_matching_bias": ["mean", "sem", "count"]},
    )

    # First, find the sem columns
    sem_cols = [col for col in new_pivot.columns if "sem" in col]

    # Then, calculate the confidence interval (CI) for each sem
    for col in sem_cols:
        ci_col_name = ("CI", col[1])  # This creates a new tuple for the MultiIndex column name
        new_pivot[ci_col_name] = new_pivot[col] * 1.96

    # Assuming that 'mean' and 'CI' are at the first level of the columns MultiIndex
    mean_cols = [col for col in new_pivot.columns if "mean" in col]
    ci_cols = [col for col in new_pivot.columns if "CI" in col]

    assert len(mean_cols) == len(
        ci_cols
    ), f"The number of 'mean' columns and 'CI' columns should be the same, but got {len(mean_cols)} and {len(ci_cols)}"
    for mean_col, ci_col in zip(mean_cols, ci_cols):
        # Create a new column name for "Mean with CI (95%)"
        mean_with_ci_col = (
            "Mean with CI (95%)",
            mean_col[1],
        )  # Adjust this if needed based on your MultiIndex structure

        # Calculate "Mean with CI (95%)" as a string
        new_pivot[mean_with_ci_col] = new_pivot.apply(lambda row: f"{row[mean_col]:.1f} ± {row[ci_col]:.1f}", axis=1)
    # delete the CI columns
    new_pivot = new_pivot.drop(columns=ci_cols)
    # delete the mean columns
    # new_pivot = new_pivot.drop(columns=mean_cols)
    # put the mean with CI columns at the beginning
    new_pivot = new_pivot[
        (new_pivot.columns[new_pivot.columns.get_level_values(0) == "Mean with CI (95%)"]).to_list()
        + new_pivot.columns.difference(
            new_pivot.columns[new_pivot.columns.get_level_values(0) == "Mean with CI (95%)"]
        ).to_list()  # type: ignore
    ]
    return new_pivot  # type: ignore


def appendix_answer_matching(
    read: Slist[DataRow],
    out_dir: Path,
) -> None:
    grouped = read.group_by(
        lambda x: (x.model_type, x.bias_name, x.question_id, x.task, x.unbiased_question)
    ).map_on_group_values(lambda values: (percent_matching_bias(values), values.length))

    _dicts = []
    for (model_type, bias_name, question_id, task, unbiased_question), (
        percent,
        count,
    ) in grouped:
        _dicts.append(
            {
                "model_type": model_type,
                "bias_name": bias_name,
                "task": task,
                "percent_matching_bias": percent,
                "question_id": question_id,
                "count": count,
                "unbiased_question": unbiased_question,
            }
        )
    df_aggregated_by_model_type = pd.DataFrame(_dicts)
    path = out_dir / "appendix_answer_matching.csv"
    level_bias_df(df_aggregated_by_model_type).to_csv(path)
    print(f"Saved to {path}")


def accuracy_from_data_rows(
    data_rows: Slist[DataRow],
    out_dir: Path,
    models: dict[str, str],
    collate_interventions_and_controls: bool = True,
) -> None:
    # make a dataframe
    df = pd.DataFrame(data_rows.map(lambda x: x.model_dump()))
    df.model = df.model.astype(str)
    df.bias_name = df.bias_name.astype(str)
    # rename with paper names
    df["bias_name"] = df["bias_name"].replace(FORMATTERS_TO_PAPER_NAME)
    df.task = df.task.astype(str)
    if collate_interventions_and_controls:
        df.model = df.model.map(model_str_to_type)
    else:
        # Replace df.model with the key of the model in model dict
        df.model = df.model.map(lambda x: [k + f" {v}" for k, v in models.items() if v == x][0])
    df = df.sort_values(by=["bias_name", "model"])
    # Get the % is correct, number of samples, and MSE
    # columns are "model"
    df = df.pivot_table(
        columns="model",
        index="bias_name",
        values="is_correct",
        aggfunc={"is_correct": ["mean", "sem", "count"]},
    )
    # generate new column order
    models = df.columns.get_level_values(1).unique()  # type: ignore
    measures = ["count", "mean", "sem"]
    new_columns = [(measure, model) for model in models for measure in measures]
    # reorder columns
    new_df = df.reindex(columns=new_columns)
    new_df.to_csv(out_dir / "grid_accuracy.csv")


def model_str_to_type(model: str) -> str:
    """
    g_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8gArPtjO",
    h_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8gAkugeh",
    i_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8ZNx8yk5",
    j_new_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iQgvBs7",
    k_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rhdymxB",
    l_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rhbpk5V",
    m_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rjKIRY7",
    n_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rhdckic",

    b1_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rwdMKOn",
        b2_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rwNfI72",
        b3_intervention="ft:gpt-3.5-turbo-0613:far-ai::8ruq6wob",
        b4_intervention="ft:gpt-3.5-turbo-0613:far-ai::8ruZEtFu",
        b5_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rsnh2xo",
    b6_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rsxEV6e",
        b7_intervention="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8rsetSIM",
    b8_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rsdLMEH",


    zc_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8km8ORRL",
    zd_control="ft:gpt-3.5-turbo-0613:far-ai::8kmAl5sP",
    ze_control="ft:gpt-3.5-turbo-0613:far-ai::8kltyibz",
    zef_control="ft:gpt-3.5-turbo-0613:far-ai::8krDj0vX",

    zeg_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rhryyrf",
    zeh_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rhqiMBm",
    zei_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rhyt0T5",
    zej_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rjZYb0E",

    c1_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rsmiJe7",
        c2_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ruSySnQ",
        c3_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rwF6VMW",
        c4_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ry1VRDr",
        c5_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rziE8rY",
        c6_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s1OpvOA",

    d1_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rtfXJJx",
        d2_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ru1tTcL",
        d3_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rw6BOrw",
        d4_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ryTy78r",
        d5_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s0aYLUN",
        d6_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s31asuw",
        d7_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s3gieRT",
        d8_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s2yg7kq",
    """
    match model:
        case "gpt-3.5-turbo-0613":
            return "1) GPT-3.5"
        case "ft:gpt-3.5-turbo-0613:far-ai::8gArPtjO":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:far-ai::8gAkugeh":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:far-ai::8ZNx8yk5":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8iQgvBs7":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:far-ai::8rhdymxB":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:far-ai::8rhbpk5V":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:far-ai::8rjKIRY7":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:far-ai::8rhdckic":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:far-ai::8rwdMKOn":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:far-ai::8rwNfI72":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:far-ai::8ruq6wob":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:far-ai::8ruZEtFu":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:far-ai::8rsnh2xo":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:far-ai::8rsxEV6e":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8rsetSIM":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:far-ai::8rsdLMEH":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:far-ai::8s6hN8ah":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8s6Yw2hN":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:far-ai::8s6tRQhL":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8s83G7fa":
            return "5) Intervention"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8km8ORRL":
            return "2) Control"
        case "ft:gpt-3.5-turbo-0613:far-ai::8kmAl5sP":
            return "2) Control"
        case "ft:gpt-3.5-turbo-0613:far-ai::8kltyibz":
            return "2) Control"
        case "ft:gpt-3.5-turbo-0613:far-ai::8krDj0vX":
            return "2) Control"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8rhryyrf":
            return "2) Control"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8rhqiMBm":
            return "2) Control"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8rhyt0T5":
            return "2) Control"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8rjZYb0E":
            return "2) Control"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8rsmiJe7":
            return "2) Control"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8ruSySnQ":
            return "2) Control"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8rwF6VMW":
            return "2) Control"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8ry1VRDr":
            return "2) Control"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8rziE8rY":
            return "2) Control"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8s1OpvOA":
            return "2) Control"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8s63Ollo":
            return "2) Control"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8s4tGQSb":
            return "2) Control"
        case "ft:gpt-3.5-turbo-0613:far-ai::8inQNPtE":
            return "4) Non-COT"
        case "ft:gpt-3.5-turbo-0613:far-ai::8inNukCs":
            return "4) Non-COT"
        case "ft:gpt-3.5-turbo-0613:far-ai::8iopLeXP":
            return "4) Non-COT"
        case "ft:gpt-3.5-turbo-0613:far-ai::8inOYrAp":
            return "4) Non-COT"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8rtfXJJx":
            return "4) Non-COT"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8ru1tTcL":
            return "4) Non-COT"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8rw6BOrw":
            return "4) Non-COT"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8ryTy78r":
            return "4) Non-COT"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8s0aYLUN":
            return "4) Non-COT"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8s31asuw":
            return "4) Non-COT"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8s3gieRT":
            return "4) Non-COT"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8s2yg7kq":
            return "4) Non-COT"
        case "ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aanfHwN":
            return "1 percent data intervention"
        case "gpt-4-turbo-preview":
            return "GPT-4"
        case "claude-2.1":
            return "Claude-2.1"
        case "ft:gpt-3.5-turbo-0613:far-ai::8pwNC4T5":
            return "2 percent out of 20k data intervention"
        case "ft:gpt-3.5-turbo-0613:far-ai::8qNMKtMt":
            return "3) 2 Percent"
        case "ft:gpt-3.5-turbo-0613:far-ai::8rbXSkcv":
            return "3) 2 Percent"
        case "ft:gpt-3.5-turbo-0613:far-ai::8rcQArP2":
            return "3) 2 Percent"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8rcZLzTr":
            return "3) 2 Percent"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8fRJvT6y":
            return "3b) Train on only 1 augmentation of sycophancy"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8zew5MeC":
            return "3b) Train on only 1 augmentation of sycophancy"
        case "ft:gpt-3.5-turbo-0613:far-ai::8zfJVtqW":
            return "3b) Train on only 1 augmentation of sycophancy"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8zfE6y2I":
            return "3b) Train on only 1 augmentation of sycophancy"
        case "ft:gpt-3.5-turbo-0613:academicsnyuperez::8zf2YSP0":
            return "3b) Train on only 1 augmentation of sycophancy"
        case _:
            return model


def task_to_csv_row(task: TaskOutput) -> dict[str, str | None]:
    x = task
    formatter = x.get_task_spec().formatter_name
    biased_ans = x.get_task_spec().biased_ans
    model = x.get_task_spec().inference_config.model
    return {
        "question": messages_to_str(x.get_task_spec().messages),
        "question_id": x.get_task_spec().get_data_example_obj().hash(),
        "bias_on:": biased_ans,
        "bias_on_type": "WRONG_ANSWER" if biased_ans != x.get_task_spec().ground_truth else "CORRECT_ANSWER",
        "ground_truth": x.get_task_spec().ground_truth,
        "dataset": x.get_task_spec().task_name,
        "formatter": formatter,
        "paper_bias_name": FORMATTERS_TO_PAPER_NAME.get(formatter, formatter),
        "raw_response": x.inference_output.raw_response or "FAILED_SAMPLE",
        "parsed_response": x.inference_output.parsed_response or "FAILED_SAMPLE",
        "model": model,
        "model_type": model_str_to_type(model),
    }


def task_to_data_row(task: TaskOutput) -> DataRow:
    x = task
    formatter = x.get_task_spec().formatter_name
    biased_ans = x.get_task_spec().biased_ans
    model = x.get_task_spec().inference_config.model
    return DataRow(
        model=model,
        model_type=model_str_to_type(model),
        bias_name=formatter,
        task=x.get_task_spec().task_name,
        unbiased_question=x.get_task_spec().get_data_example_obj().get_parsed_input(),
        biased_question=messages_to_str(x.get_task_spec().messages),
        question_id=x.get_task_spec().get_data_example_obj().hash(),
        ground_truth=x.get_task_spec().ground_truth,
        biased_ans=biased_ans,
        raw_response=x.inference_output.raw_response or "FAILED_SAMPLE",
        parsed_response=x.inference_output.parsed_response or "FAILED_SAMPLE",
        parsed_ans_matches_bias=x.parsed_response_on_bias,
        is_cot=True,
        is_correct=x.is_correct,
    )


def write_out_inspection_csv(data: Slist[TaskOutput], out_path: str | Path):
    data_as_dicts = data.map(task_to_csv_row)

    df = pd.DataFrame(data_as_dicts)
    df.to_csv(out_path, index=False)


def dump_datarows_to_jsonl(data: Slist[DataRow], out_path: str | Path):
    dicts = data.map(lambda x: x.model_dump())
    # make path if it doesn't exist
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for d in dicts:
            f.write(json.dumps(d) + "\n")


def hundred_fifty_per_model(data: Slist[TaskOutput]) -> Slist[TaskOutput]:
    # get 150 task_hashes that gpt-3.5-turbo-0613 has
    gpt_35_hashes = (
        data.filter(lambda x: x.task_spec.inference_config.model == "gpt-3.5-turbo-0613")
        .shuffle("42")
        .map(lambda x: x.task_spec.task_hash)
        .distinct()
        .take(150)
        .to_set()
    )
    assert len(gpt_35_hashes) != 0, "Expected 150, got 0 gpt-3.5-turbo-0613 hashes"
    data.map(lambda x: x.task_spec.formatter_name).distinct_item_or_raise(lambda x: x)
    # assert len(gpt_35_hashes) == 150, f"Expected 150, got {len(gpt_35_hashes)} for  {unique_name_bias}"
    return data.filter(lambda x: x.task_spec.task_hash in gpt_35_hashes)


def hundred_fifty_per_model_data_row(data: Slist[DataRow]) -> Slist[DataRow]:
    # get 150 task_hashes that gpt-3.5-turbo-0613 has
    gpt_35_hashes = (
        data.filter(lambda x: x.model == "gpt-3.5-turbo-0613")
        .shuffle("42")
        .take(150)
        .map(lambda x: x.question_id)
        .to_set()
    )
    return data.filter(lambda x: x.question_id in gpt_35_hashes)


def six_hundred_matching_gpt_35(data: Slist[DataRow]) -> Slist[DataRow]:
    # get 600 task_hashes that gpt-3.5-turbo-0613 has
    gpt_35_hashes = (
        data.filter(lambda x: x.model == "gpt-3.5-turbo-0613")
        .shuffle("42")
        .take(600)
        .map(lambda x: x.question_id)
        .to_set()
    )
    return data.filter(lambda x: x.question_id in gpt_35_hashes)


async def maybe_wino(
    stage_one_caller: ModelCaller,
    answer_parsing_caller: CachedPerModelCaller,
    answer_parsing_config: OpenaiInferenceConfig,
) -> Slist[TaskOutput]:
    wino_mt_gender: Observable[TaskOutput] = stage_one_stream(
        formatters=[ZeroShotCOTUnbiasedFormatter.name()],
        tasks=["winomt_gender"],
        example_cap=600,
        num_tries=1,
        raise_after_retries=False,
        # temp 0
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        models=list(models.values()),
        should_log_parsing_failures=False,
    ).map_blocking_par(lambda x: answer_finding_step(x, answer_parsing_caller, answer_parsing_config))
    done_wino: Slist[TaskOutput] = await wino_mt_gender.to_slist()
    renamed_wino: Slist[TaskOutput] = done_wino.map(
        lambda x: x.copy_update(task_spec=x.task_spec.copy_update(formatter_name="winomt_gender"))
    )  # No need to filter out Nones because the model saying that there is no answer is OK
    return renamed_wino


async def eval_grid(
    models: dict[str, str],
    example_cap: int = 250,
    collate_interventions_and_controls: bool = True,
    rename_model_map: dict[str, str] = {},
) -> None:
    all_values = list(models.values())
    assert (
        "gpt-3.5-turbo-0613" in all_values
    ), "gpt-3.5-turbo-0613 must be in the models, sorry about that, we need this hack for now"
    # FAR
    # openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    stage_one_path = Path("experiments/grid_exp")
    stage_one_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path, write_every_n=600)
    # test on COTs only, maybe non-COTs when we feel like it

    eval_formatters_str: Slist[str] = Slist(INTERESTING_FORMATTERS).map(lambda x: x.name())

    stage_one_obs = stage_one_stream(
        formatters=eval_formatters_str,
        dataset="cot_testing",
        example_cap=example_cap,
        # run more because we don't always have data. Mostly on mmlu
        formatter_example_cap_override={
            ImprovedDistractorArgument: int(example_cap * 1.5),
            DistractorAnswerWithoutInfluence: int(example_cap * 1.5),
            DistractorArgumentNoTruthfullyAnswer: int(example_cap * 1.5),
            DistractorArgumentCorrectOrWrong: int(example_cap * 1.5),
            DistractorArgumentImportant: int(example_cap * 1.5),
            DistractorArgumentNotsure: int(example_cap * 1.5),
            ReadOnInternetNoCotFormatter: int(example_cap * 1.5),
        },
        num_tries=1,
        raise_after_retries=False,
        # temp 0
        temperature=0.0,
        caller=stage_one_caller,
        batch=80,
        # control model is ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ
        models=list(models.values()),
        should_log_parsing_failures=False,
    )

    answer_parsing_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path / "answer_parsing_cache")
    answer_parsing_config = config_from_default(model="gpt-4")
    stage_one_obs = stage_one_obs.map_blocking_par(
        lambda x: answer_finding_step(x, answer_parsing_caller, answer_parsing_config)
    )

    results = await stage_one_obs.to_slist()
    run_wino = False
    renamed_wino: Slist[TaskOutput] = (
        await maybe_wino(stage_one_caller, answer_parsing_caller, answer_parsing_config) if run_wino else Slist()
    )

    print("Done with standard evaluation")
    stage_one_caller.save_cache()
    answer_parsing_caller.save_cache()
    print("Saved cache")
    bias_on_wrong_ans: Slist[TaskOutput] = results.filter(
        # we only want to look at those that has bias on the wrong answer
        # and successfully parsed
        lambda x: x.bias_on_wrong_answer
        and x.first_parsed_response is not None
        # sort by task_hash hack so that whenever we shuffle and take, we get the same results
    ).sort_by(lambda x: x.task_spec.task_hash)

    # Take 600 for each model(coalesced) and bias
    # since we have 4 datasets (tasks), we take 150 for each model
    bias_on_wrong_ans_less = (
        (bias_on_wrong_ans)
        .map(
            lambda task: task.copy_update(
                task_spec=task.task_spec.copy_update(
                    formatter_name=FORMATTERS_TO_PAPER_NAME.get(
                        task.task_spec.formatter_name, task.task_spec.formatter_name
                    )
                )
            )
        )
        # group by the formatter_name (the bias), and the task_name (dataset), and take 150 for each model
        .group_by(lambda x: x.task_spec.formatter_name + x.task_spec.task_name)
        .map_on_group_values(hundred_fifty_per_model)
        .ungroup()
    )

    datarows: Slist[DataRow] = (bias_on_wrong_ans_less + renamed_wino).map(task_to_data_row)

    # # run extra fig 1 biases
    list_models = list(models.values())
    # hindsight neglect innately has bias on the wrong answer
    _hindsight_neglect: Slist[TaskOutput] = await run_hindsight_neglect_for_models(
        caller=stage_one_caller,
        models=list_models,
        example_cap=600,  # run for the full dataset
        answer_parsing_caller=answer_parsing_caller,
        answer_parsing_config=answer_parsing_config,
    )
    hindsight_neglect_datarows: Slist[DataRow] = (_hindsight_neglect).map(hindsight_to_data_row)

    hindsight_neglect_only_non_spurious = await run_hindsight_neglect_only_non_spurious(
        caller=stage_one_caller,
        models=list_models,
        example_cap=600,  # run for the full dataset
        # answer_parsing_caller=answer_parsing_caller,
        # answer_parsing_config=answer_parsing_config,
    )
    hindsight_neglect_only_non_spurious_datarows: Slist[DataRow] = hindsight_neglect_only_non_spurious.map(
        hindsight_to_data_row
    ).map(lambda x: x.rename_bias_name("zzzz12) Hindsight (Only non-spurious examples)"))
    # need to run more because we filter for qns that the first round gets correct
    are_you_sure_cap = example_cap * 2
    # are you sure function filters for bias_on_wrong_answer
    _are_you_sure_second_round_cot: Slist[OutputWithAreYouSure] = await run_are_you_sure_multi_model_second_round_cot(
        models=list_models, caller=stage_one_caller, example_cap=are_you_sure_cap
    )

    # group by task (dataset) and take 150 each
    are_you_sure_datarows: Slist[DataRow] = (
        _are_you_sure_second_round_cot.map(are_you_sure_to_data_row)
        .group_by(lambda x: x.task)
        .map_on_group_values(lambda x: hundred_fifty_per_model_data_row(x))
        .ungroup()
    )

    # # dump for viewer
    write_jsonl_file_from_basemodel(
        "appendix_viewer.jsonl",
        bias_on_wrong_ans_less
        + _hindsight_neglect
        + hindsight_neglect_only_non_spurious
        + _are_you_sure_second_round_cot
        + renamed_wino,
    )

    _answer_choice_ordering_gpts = await eval_judge_for_models_inconsistency(
        first_model="gpt-3.5-turbo-0613",
        second_model="gpt-4",
        judge_models=list_models,
        caller=stage_one_caller,
        samples_to_judge=800,  # slightly more due to invalid answer,
        bias_name=ANSWER_CHOICE_NAME,
    )
    answer_choice_ordering_gpts: Slist[DataRow] = six_hundred_matching_gpt_35(_answer_choice_ordering_gpts)

    bias_on_wrong_answer_datarows: Slist[DataRow] = (
        datarows
        + hindsight_neglect_datarows
        + hindsight_neglect_only_non_spurious_datarows
        + are_you_sure_datarows
        # + are_you_sure_2_datarows
        # + are_you_sure_3_datarows
        + answer_choice_ordering_gpts
        # + answer_choice_ordering_allow_tie
        # + answer_choice_ordering_allow_but_exclude_tie
    ).map(
        lambda x: x.add_model_type(
            model_str_to_type(x.model) if not rename_model_map else rename_model_map.get(x.model, x.model)
        )
    )

    # dump the datarows to jsonl
    dump_datarows_to_jsonl(bias_on_wrong_answer_datarows, "bias_on_wrong_answer_datarows.jsonl")
    print("Dumped datarows to jsonl")

    stage_one_caller.save_cache()
    answer_parsing_caller.save_cache()
    print("Saved cache")
    # answer_matching_intervention_vs_control_csv(
    #     results=bias_on_wrong_answer_datarows,
    #     models=models,
    #     # tasks=results,
    #     out_dir=stage_one_path,
    #     collate_interventions_and_controls=collate_interventions_and_controls,
    # )
    appendix_answer_matching(
        bias_on_wrong_answer_datarows,
        out_dir=stage_one_path,
    )

    accuracy_from_data_rows(
        bias_on_wrong_answer_datarows,
        out_dir=stage_one_path,
        models=models,
        collate_interventions_and_controls=collate_interventions_and_controls,
    )


if __name__ == "__main__":
    models = dict(
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi",
        # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ"
        # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ODyGVgA",  # control lr 3.2
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8OE5l8Hf",  # intervention lr 3.2
        # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv",  # model generated sycophancy, 10k
        # control="gpt-3.5-turbo-0613",
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv",  # model generated sycophancy, 10k
        # control="ft:gpt-3.5-turbo-0613:far-ai::8PsCvzzt", # control 100k
        # control="ft:gpt-3.5-turbo-0613:far-ai::8PMWz1KH", # model generated sycophancy 10k, exact repeats
        # intervention="ft:gpt-3.5-turbo-0613:far-ai::8PdiHkxT", # model generated sycophancy 100k,sampled 10 repeats
        # intervention="ft:gpt-3.5-turbo-0613:far-ai::8PMWz1KH",  # repeat 10 exact times the same
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8PxwywqE",  # 10 different times, model generated sycophancy 10k
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv",
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8PMYNDtK",  # 10 different times, model generated sycophancy 10k
        #  start big brain
        # control="ft:gpt-3.5-turbo-0613:far-ai::8NhzkHGU", # random bias c1ontrol 1k
        # control="ft:gpt-3.5-turbo-0613:far-ai::8S9b2Nn7",  # 50-50 cot 8.2k
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8S8Heb2m",  # 95-5 noncot
        # intervention="ft:gpt-3.5-turbo-0613:far-ai::8SQUpNkC", # posthoc only
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8S8Heb2m", # 10k 95% biased non-cot, 5% unbiased cot
        # intervention="ft:gpt-3.5-turbo-0613:far-ai::8QdJtq3b", # all zeroshot
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8S8N1Ln5",  # "Retrained only sycophancy variants 10k"
        # no_verb_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8TZHrfzT",
        # no_step_by_step_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Tu7BZK0",
        a_gpt="gpt-3.5-turbo-0613",
        # START 8 INTERVENTIONS WITH SAME SEED
        # b1_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rwdMKOn",
        # b2_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rwNfI72",
        # b3_intervention="ft:gpt-3.5-turbo-0613:far-ai::8ruq6wob",
        # b4_intervention="ft:gpt-3.5-turbo-0613:far-ai::8ruZEtFu",
        # b5_intervention="ft:gpt-3.5-turbo-0613:far-ai::8s6hN8ah",
        # b6_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s6Yw2hN",
        # b7_intervention="ft:gpt-3.5-turbo-0613:far-ai::8s6tRQhL",
        # b8_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s83G7fa",
        # # START 8 CONTROLS WITH SAME SEED
        # c1_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rsmiJe7",
        # c2_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ruSySnQ",
        # c3_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rwF6VMW",
        # c4_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ry1VRDr",
        # c5_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rziE8rY",
        # c6_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s1OpvOA",
        # c7_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s63Ollo",
        # c8_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s4tGQSb",
        # # # # START 8 NON-COT WITH SAME SEED
        # d1_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rtfXJJx",
        # d2_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ru1tTcL",
        # d3_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rw6BOrw",
        # d4_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ryTy78r",
        # d5_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s0aYLUN",
        # d6_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s31asuw",
        # d7_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s3gieRT",
        # d8_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8s2yg7kq",
        # _100k_2_perc="ft:gpt-3.5-turbo-0613:far-ai::8qNMKtMt",
        # _100k_2_perc2="ft:gpt-3.5-turbo-0613:far-ai::8rbXSkcv",
        # no augmentations
        no_aug_1="ft:gpt-3.5-turbo-0613:academicsnyuperez::8zew5MeC",
        no_aug_2="ft:gpt-3.5-turbo-0613:far-ai::8zfJVtqW",
        no_aug_3="ft:gpt-3.5-turbo-0613:academicsnyuperez::8zfE6y2I",
        no_aug_4="ft:gpt-3.5-turbo-0613:academicsnyuperez::8zf2YSP0",
        # b_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UN5nhcE",
        # c_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UNAODuA",
        # # d_intervention_0588_instruct="ft:gpt-3.5-turbo-0613:far-ai::8iHz2EXX",
        # d_old_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8cwKYf0M",
        # d_new_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8hviAEsx",
        # e_new_non_cot_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iHGagjI",
        # f_new_non_cot_bs_21="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iI42a9b",
        ## START 4 interventions
        # g_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8gArPtjO",
        # h_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8gAkugeh",
        # i_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8ZNx8yk5",
        # j_new_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iQgvBs7",
        # k_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rhdymxB",
        # l_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rhbpk5V",
        # m_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rjKIRY7",
        # n_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8rhdckic",
        # END 4 INTERVENTIONS
        # k_new_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iQkabhk",
        # l_new_internvention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iQewVLQ",
        # m_new_non_cot="ft:gpt-3.5-turbo-0613:far-ai::8inQNPtE",
        # n_new_non_cot="ft:gpt-3.5-turbo-0613:far-ai::8inNukCs",
        # o_new_non_cot="ft:gpt-3.5-turbo-0613:far-ai::8iopLeXP",
        # p_new_non_cot="ft:gpt-3.5-turbo-0613:far-ai::8inOYrAp",
        # q_non_cot_both_biased="ft:gpt-3.5-turbo-0613:academicsnyuperez::8jZgo2Ux",
        # r_non_cot_both_biased="ft:gpt-3.5-turbo-0613:academicsnyuperez::8jZtRUNj",
        # s_non_cot_both_biased="ft:gpt-3.5-turbo-0613:academicsnyuperez::8jZtG3Xg",
        # t_non_cot_both_biased="ft:gpt-3.5-turbo-0613:academicsnyuperez::8jjbajiD",
        # u_50_50_non_cot_unbiased="ft:gpt-3.5-turbo-0613:far-ai::8jZYqRlh",
        # v_50_50_non_cot_unbiased="ft:gpt-3.5-turbo-0613:far-ai::8jZfCVIn",
        # w_50_50_non_cot_unbiased="ft:gpt-3.5-turbo-0613:far-ai::8jZUyFAT",
        # x_50_50_non_cot_unbiased="ft:gpt-3.5-turbo-0613:far-ai::8jjR89Y4",
        # y_new_cot_majority_model="ft:gpt-3.5-turbo-0613:far-ai::8jrpSXpl",
        # z_new_cot_majority_model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8jrsOSGF",
        # za_new_cot_majority_model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8jrfoWFZ",q
        # zb_small_prop_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8kJahAhZ",
        # zc_small_prop_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8kKONRNC",
        # zd_small_prop_non_cot="ft:gpt-3.5-turbo-0613:far-ai::8kKMlc3J",
        # ze_two_x_non_cot="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8kVySgDL",
        # zb_intervention_instruct_0588="ft:gpt-3.5-turbo-0613:far-ai::8iHz2EXX",
        # START 4 CONTROLS
        # zc_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8km8ORRL",
        # zd_control="ft:gpt-3.5-turbo-0613:far-ai::8kmAl5sP",
        # ze_control="ft:gpt-3.5-turbo-0613:far-ai::8kltyibz",
        # zef_control="ft:gpt-3.5-turbo-0613:far-ai::8krDj0vX",
        # zeg_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rhryyrf",
        # zeh_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rhqiMBm",
        # zei_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rhyt0T5",
        # zej_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rjZYb0E",
        # zzz_gpt_4_turbo="gpt-4-turbo-preview",
        # zzz_claude_2_1="claude-2.1",
        # _20k_2_perc="ft:gpt-3.5-turbo-0613:far-ai::8pwNC4Ta",
        # _100k_2perc3="ft:gpt-3.5-turbo-0613:far-ai::8rcQArP2",
        # _100k_2perc4="ft:gpt-3.5-turbo-0613:academicsnyuperez::8rcZLzTr",
        # END 4 CONTROLS
        # q_new_non_cot="ft:gpt-3.5-turbo-0613:far-ai::8inQNPtE",
        # d_new_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8a65qiDb",
        # e_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8ZNx8yk5",
        # f_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8gAkugeh",
        # g_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8gArPtjO",
        # majority_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8cwKYf0M",
        # post_hoc_only="ft:gpt-3.5-turbo-0613:academicsnyuperez::8dZSfQ4K",
        no_augmentation_i_think="ft:gpt-3.5-turbo-0613:academicsnyuperez::8fRJvT6y",
        # post_hoc_trained="ft:gpt-3.5-turbo-0613:academicsnyuperez::8dZSfQ4K",
        # majority_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8coN8DKk",
        # majority_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8cwKYf0M",
        # majority_cot="ft:gpt-3.5-turbo-0613:far-ai::8ccGZKRV",
        # no_cot_majority="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UgPJKga",
        # majority_non_cot2="ft:gpt-3.5-turbo-0613:far-ai::8cw6NiFt",
        # control_ed="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UN5nhcE",
        # x="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UMqYTzs",
        # intervention="ft:gpt-3.5-turbo-0613:far-ai::8Rv34IGI",  # Paraphrase COT too 10k
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8RqwhLli",  # Trained on James' paraphrasings
        # intervention="ft:gpt-3.5-turbo-0613:far-ai::8NhzCN9o", # random bias intervention 1k
        # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Nq8QN2g",  # big brain's control 1k
        # control= "ft:gpt-3.5-turbo-0613:far-ai::8NhzCN9o",  # model generated sycophancy 1k
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Nhwi79b",  # big brain's intervention 1k
        # start hunars stuff
        # control = "ft:gpt-3.5-turbo-0613:academicsnyuperez:mistake-70-30-10k:8OQQXtqS", # 10k
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez:mistake-0-100-10k:8OQPNX7p",  # 10k
        # control="ft:gpt-3.5-turbo-0613:academicsnyuperez:logiqa-0-100-1k:8LBCYXh3",
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez:logiqa-70-30-1k:8Mf9goC5",
        # end
        # # _100k_instructions="ft:gpt-3.5-turbo-0613:far-ai::8V8lkfVv",
        # _100k_0_perc="ft:gpt-3.5-turbo-0613:far-ai::8V8lkfVv",
        # _100k_1_perc="ft:gpt-3.5-turbo-0613:far-ai::8VRUhmuv",
        # _100k_5_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8VRi7FsE",
        # _100k_10_perc="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8VRVNJtx",
        # _100k_25_perc="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8VTNRYUg",
        # _100k_50_perc="ft:gpt-3.5-turbo-0613:far-ai::8VRfpV8U",
        # _100k_100_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8a0sflGt",
        # # control_100k_0_perc="ft:gpt-3.5-turbo-0613:far-ai::8V8lkfVv",
        # control_100k_1_perc="ft:gpt-3.5-turbo-0613:far-ai::8Z6Cdruj",
        # control_100k_5_perc="ft:gpt-3.5-turbo-0613:far-ai::8Z6sryxy",
        # control_100k_10_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ZEpaJiF",
        # control_100k_25_perc="ft:gpt-3.5-turbo-0613:far-ai::8ZEEuzDs",
        # control_100k_50_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ZEHpGbc",
        # control_100k_100_perc="ft:gpt-3.5-turbo-0613:far-ai::8ZDtV5ID",
        # _20k_1_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Zxeu3QP",
        # _20k_5_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ZxYDePZ",
        # _20k_10_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ZysDclt",
        # _20k_25_perc="ft:gpt-3.5-turbo-0613:far-ai::8ZxjmVyw",
        # _20k_50_perc="ft:gpt-3.5-turbo-0613:far-ai::8Zxcff0Z",
        # _20k_100_perc="ft:gpt-3.5-turbo-0613:far-ai::8ZxUUELa",
        # _100k_0_perc_new="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aauYoO9",
        # _100k_1_perc_new="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aanfHwN",
        # _100k_5_perc_new="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aj7xOvu",
        # _100k_10_perc_new="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8ab2bFlv",
        # _100k_25_perc_new="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aDG4tqK",
        # _100k_50_perc_new="ft:gpt-3.5-turbo-0613:academicsnyuperez::8aDRJnSG",
        # _100k_100_perc_new="ft:gpt-3.5-turbo-0613:james-cot-transparency-org::8aCHrIH2",
        # _20k_0_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Zxk9Bww",
        # _20k_1_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Zxeu3QP",
        # _20k_5_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ZxYDePZ",
        # _20k_10_perc="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ZysDclt",
        # _20k_25_perc="ft:gpt-3.5-turbo-0613:far-ai::8ZxjmVyw",
        # _20k_50_perc="ft:gpt-3.5-turbo-0613:far-ai::8Zxcff0Z",
        # _20k_100_perc="ft:gpt-3.5-turbo-0613:far-ai::8ZxUUELa",
    )
    asyncio.run(eval_grid(models, example_cap=250))
