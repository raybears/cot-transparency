import asyncio
import json
from pathlib import Path
from typing import Sequence, assert_never

import pandas as pd
from slist import Group, Slist
from torch import mode

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.base import ModelCaller
from cot_transparency.data_models.config import config_from_default
from cot_transparency.data_models.hashable import HashableBaseModel
from cot_transparency.data_models.messages import ChatMessage
from cot_transparency.data_models.models import TaskOutput, TaskSpec
from cot_transparency.data_models.pd_utils import DataRow
from cot_transparency.formatters.core.unbiased import (
    ZeroShotCOTUnbiasedFormatter,
    ZeroShotUnbiasedFormatter,
    ZeroShotUnbiasedOnlyChooseValidOptionsFormatter,
)
from cot_transparency.formatters.more_biases.user_wrong_cot import (
    ImprovedDistractorArgument,
    ReadOnInternetNoCotFormatter,
)
from cot_transparency.formatters.name_mapping import name_to_formatter
from cot_transparency.formatters.prompt_sensitivity.automated_generations import (
    AskWithDistractorFact,
    AskWithDistractorFactNoCot,
)
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.are_you_sure.eval_are_you_sure_no_cot import run_are_you_sure_multi_model
from scripts.are_you_sure.eval_are_you_sure_second_cot import (
    run_are_you_sure_multi_model_second_round_cot,
)
from scripts.automated_answer_parsing.answer_parsing_example import answer_finding_step
from scripts.evaluate_judge_consistency.judge_consistency import eval_judge_for_models_inconsistency
from scripts.evil_grid_exp.eval_coherent_biasing import biased_on_wrong_answer_and_answered_in_line_with_bias
from scripts.inverse_scaling_experiments.run_hindsight_neglect import run_hindsight_neglect_for_models
from scripts.meg_mimicry_ans.eval_mimicry_freeform_matching_bias import eval_mimicry_freeform_follows_wrong
from scripts.meg_mimicry_ans.eval_mimicry_poems import eval_mimicry_poems_multi_model
from scripts.prompt_sen_bias_generalization.util import save_per_model_results
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


def accuracy_from_data_rows(
    data_rows: Slist[DataRow],
    out_dir: Path,
    models: dict[str, str],
) -> None:
    # for each model, take 600
    truncated_results: Slist[DataRow] = (data_rows).group_by(lambda x: x.model + x.bias_name).ungroup()
    # make a dataframe
    df = pd.DataFrame(truncated_results.map(lambda x: x.model_dump()))
    df.model = df.model.astype(str)
    df.bias_name = df.bias_name.astype(str)

    df["bias_name"] = df["task"] + " " + df["bias_name"]
    df.task = df.task.astype(str)
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
    models = df.columns.get_level_values(1).unique()  # type : ignore
    measures = ["count", "mean", "sem"]
    new_columns = [(measure, model) for model in models for measure in measures]
    # reorder columns
    new_df = df.reindex(columns=new_columns)
    # transpose
    new_df = new_df.transpose()
    new_df.to_csv(out_dir / "grid_exp_separate_accuracy.csv")


def write_out_inspection_csv(data: Slist[TaskOutput], out_path: str | Path):
    def messages_to_str(x: Sequence[ChatMessage]):
        return "\n".join(Slist(x).map(str))

    data_as_dicts = data.map(
        lambda x: {
            "question": messages_to_str(x.get_task_spec().messages),
            "task": x.get_task_spec().task_name,
            "formatter": x.get_task_spec().formatter_name,
            "question_hash": x.get_task_spec().get_data_example_obj().hash(),
            "response": x.inference_output.raw_response,
            "model": x.get_task_spec().inference_config.model,
        }
    )

    df = pd.DataFrame(data_as_dicts)
    df.pivot(
        index=["task", "formatter", "question_hash", "question"], columns="model", values="response"
    ).reset_index().to_csv(out_path)


async def eval_grid(
    models: dict[str, str], example_cap: int = 200, get_extra_tasks: bool = False, max_per_bias_and_model: int = 600
) -> None:
    # FAR
    # openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    stage_one_path = Path("experiments/grid_exp")
    stage_one_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path, write_every_n=10000)
    # test on COTs only, maybe non-COTs when we feel like it

    eval_formatters_str: Slist[str] = Slist([ZeroShotUnbiasedFormatter.name(), ZeroShotCOTUnbiasedFormatter.name()])

    # Training data filter, we create a callable that returns False if the task is
    # in the training data and True if it is not

    # url = "https://wandb.ai/raybears/consistency-training/runs/8ztnolqs"
    # data_samples = cached_read_finetune_from_url(url)
    # print(data_samples)
    # exit(1)

    # with open("./finetune_cot_hashes.json") as fh:
    #     training_data_hashes = set(json.load(fh))

    # def filter_on_training_data(task: TaskSpec) -> bool:
    #     if task.get_task_hash() in training_data_hashes:
    #         print("Filtering out", task.get_task_hash())
    #         return False
    #     else:
    # return True

    stage_one_obs = stage_one_stream(
        formatters=eval_formatters_str,
        dataset="testing_plus_aqua",
        example_cap=example_cap,
        num_tries=1,
        raise_after_retries=False,
        # temp 0
        temperature=0.0,
        caller=stage_one_caller,
        batch=80,
        # control model is ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ
        models=list(models.values()),
    )

    # aqua answers are annoyingly non standard latex, so we need to use the answer step

    answer_parsing_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path / "answer_parsing_cache")
    config = config_from_default(model="gpt-4")
    stage_one_obs = stage_one_obs.map(lambda x: answer_finding_step(x, answer_parsing_caller, config))

    results: Slist[TaskOutput] = await stage_one_obs.to_slist()
    stage_one_caller.save_cache()

    # print breakdown of samples per task
    print(results.group_by(lambda x: x.task_spec.task_name).map_on_group_values(len).to_dict())

    # save results
    save_per_model_results(results=results, results_dir=stage_one_path / "results")
    write_jsonl_file_from_basemodel(stage_one_path / "results.jsonl", results)

    normal_fig_1_datarows: Slist[DataRow] = results.filter(
        lambda task: task and task.first_parsed_response is not None
    ).map(
        lambda task: DataRow(
            model=task.task_spec.inference_config.model,
            bias_name="Cot Unbiased"
            if task.task_spec.formatter_name == "ZeroShotCOTUnbiasedFormatter"
            else "No Cot Unbiased"
            if task.task_spec.formatter_name == "ZeroShotUnbiasedFormatter"
            else "Other should not happen",
            task=task.task_spec.task_name,
            matches_bias=task.parsed_response_on_bias,
            is_cot=name_to_formatter(task.task_spec.formatter_name).is_cot,
            is_correct=task.is_correct,
        )
    )

    # print filtered results per task
    print("Filtered results")
    print(normal_fig_1_datarows.group_by(lambda x: x.task).map_on_group_values(len).to_dict())

    accuracy_from_data_rows(normal_fig_1_datarows, out_dir=stage_one_path, models=models)

    write_out_inspection_csv(results, stage_one_path / "inspection.csv")


if __name__ == "__main__":
    """
    # intervention: paraphrasing + zeroshot
    ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi
    # wandb https://wandb.ai/raybears/consistency-training/runs/60uvuhfz?workspace=user-chuajamessh

    # control 10k
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ"
    # wandb https://wandb.ai/raybears/consistency-training/runs/ehcof6tv?workspace=user-chuajamessh
    """
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
        # b_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UN5nhcE",
        # c_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UNAODuA",
        # d_intervention_0588_instruct="ft:gpt-3.5-turbo-0613:far-ai::8iHz2EXX",
        # d_old_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8cwKYf0M",
        # d_new_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8hviAEsx",
        # e_new_non_cot_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iHGagjI",
        # f_new_non_cot_bs_21="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iI42a9b",
        # g_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8gArPtjO",
        # h_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8gAkugeh",
        # i_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8ZNx8yk5",
        # j_new_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iQgvBs7",
        # k_new_intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iQkabhk",
        # l_new_internvention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8iQewVLQ",
        # m_new_non_cot="ft:gpt-3.5-turbo-0613:far-ai::8inQNPtE",
        # n_new_non_cot="ft:gpt-3.5-turbo-0613:far-ai::8inNukCs",
        # o_new_non_cot="ft:gpt-3.5-turbo-0613:far-ai::8iopLeXP",
        # p_new_non_cot="ft:gpt-3.5-turbo-0613:far-ai::8inOYrAp",
        # q_new_non_cot="ft:gpt-3.5-turbo-0613:far-ai::8inQNPtE",
        # d_new_control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8a65qiDb",
        # e_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8ZNx8yk5",
        # f_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8gAkugeh",
        # g_new_intervention="ft:gpt-3.5-turbo-0613:far-ai::8gArPtjO",
        # majority_non_cot="ft:gpt-3.5-turbo-0613:academicsnyuperez::8cwKYf0M",
        # post_hoc_only="ft:gpt-3.5-turbo-0613:academicsnyuperez::8dZSfQ4K",
        # no_augmentation_i_think="ft:gpt-3.5-turbo-0613:academicsnyuperez::8fRJvT6y",
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
        #
        # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",  # THE OG CONTROL
        # intervention_00="ft:gpt-3.5-turbo-0613:academicsnyuperez::8TtSPr0Q",
        # intervention_01="ft:gpt-3.5-turbo-0613:academicsnyuperez::8TtSh8gU",
        # intervention_1="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Tu7BZK0",  # new ed's lr=1.0
        # intervention_10="ft:gpt-3.5-turbo-0613:academicsnyuperez::8U34T0cE",  # new ed's lr=10.0
        # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8UK6VRtD",
        # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",  # THE OG CONTROL
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NY2C1j7" # wrogn few shot and i think the anser is (X)
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NYN7QsN", # wrong  few shot
        # control="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNz4qzi",  # Combined paraphrasing + all zero shot formatters
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NNFqGeq",  # paraphrasing only
        # intervention="ft:gpt-3.5-turbo-0613:far-ai::8NPtWM2y"  # All Zero shot formatters only
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv",  # model generated sycophancy
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NmbzJp0", # paraphrasing: model sycophancy and spurious context
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NhdoGRg",  # unbiased on cot biased on no cot
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NmbzJp0"  # on the fly paraphrasing model
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8OC4213p"  # paraphrasing: model sycophancy (bios)
        # intervention="ft:gpt-3.5-turbo-0613:academicsnyuperez::8NYN7QsN",  # wrong few shot ignore mistakes
        # models=[
        #     # "gpt-3.5-turbo-0613",
        #     "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",  # control 10k
        #     "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N6zCcpf",  # stanford
        #     # "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7RGEik",  # i think answer is (x) sycophancy
        # ]
    )
    asyncio.run(eval_grid(models, example_cap=600, get_extra_tasks=False, max_per_bias_and_model=600))
