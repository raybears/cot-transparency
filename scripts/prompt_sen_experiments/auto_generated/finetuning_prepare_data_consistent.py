import random
from pathlib import Path
from typing import Literal, Optional, Sequence

import fire
import pandas as pd

from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.data_models.example_base import ChoiceVariant
from cot_transparency.data_models.io import read_whole_exp_dir
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.data_models.pd_utils import (
    BaseExtractor,
    BasicExtractor,
    IsCoTExtractor,
    convert_slist_to_df,
)
from cot_transparency.formatters.name_mapping import name_to_stage1_formatter
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from scripts.prompt_sen_experiments.auto_generated.cot_formats_v1 import COT_FORMATTERS
from scripts.prompt_sen_experiments.plots import get_modal_agreement_score
from cot_transparency.data_models.data import COT_TRAINING_TASKS


class OutputUID(BaseExtractor[TaskOutput]):
    column_names = ["output_uid"]

    def extract(self, output: TaskOutput) -> Sequence[str]:
        return [output.uid()]


def main(
    exp_dir: str = "experiments/prompt_sen_experiments/temp0_cot_COT_TRAINING_TASKS",
    example_cap: Optional[int] = None,
):
    models = ["gpt-3.5-turbo"]
    formatters = COT_FORMATTERS
    temperature = 0
    tasks = COT_TRAINING_TASKS

    slist = (
        read_whole_exp_dir(exp_dir=exp_dir)
        .filter(lambda task: task.task_spec.inference_config.model in models)
        .filter(lambda task: task.task_spec.formatter_name in formatters)
        .filter(lambda task: task.task_spec.task_name in tasks)
        .filter(lambda task: task.task_spec.inference_config.temperature == temperature)
    )
    print("Number of responses after filtering = ", len(slist))

    df = convert_slist_to_df(slist, [BasicExtractor(), IsCoTExtractor(), OutputUID()])
    # want to filter on the modal answer and get a cot for each question

    # number of duplicate input_hashes
    print(f"Number of duplicate input_hashes: {df['input_hash'].duplicated().sum()}")
    # drop duplicates on input_hash
    df = df.drop_duplicates(subset=["input_hash", "model", "formatter_name", "intervention_name"], inplace=False)  # type: ignore
    print(f"Number of responses after dropping duplicates: {len(df)}")

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
    with_modal_agreement_score = (
        df.groupby(["model", "task_hash", "intervention_name"]).apply(get_modal_agreement_score).reset_index(drop=True)
    )
    with_modal_agreement_score = with_modal_agreement_score[~with_modal_agreement_score.is_same_as_mode.isna()]

    # # grab completions that were the same as the mode
    # with_modal_agreement_score = with_modal_agreement_score[
    #     with_modal_agreement_score["is_same_as_mode"]
    # ].input_hash  # type: ignore
    # print("Number of responses that were the same as the mode", len(with_modal_agreement_score))

    # add columns that says whether the answers were letters or numbers
    def get_is_answer_or_letter(row: pd.Series) -> Literal["letters", "numbers"]:  # type: ignore
        formatter = name_to_stage1_formatter(row.formatter_name)
        data_format = formatter.get_data_format_spec()

        if data_format.choice_variant == ChoiceVariant.LETTERS:
            return "letters"
        elif data_format.choice_variant == ChoiceVariant.NUMBERS:
            return "numbers"
        else:
            raise ValueError("Choice variant is not letters or numbers")

    with_modal_agreement_score["indicator_type"] = with_modal_agreement_score.apply(get_is_answer_or_letter, axis=1)

    # Basic idea is this, for each question get a completion that matches the modal agreement which has letters

    def get_consitent_completions(group: pd.DataFrame):
        """returns input_hash of the consitent completion to use"""

        # pick a completion that matches the modal agreement from this group
        if len(group[group.is_same_as_mode]) == 0:
            group["completion_uid"] = None
            return group

        selected_completion = group[group.is_same_as_mode].sample(1).output_uid.values[0]
        group["completion_uid"] = selected_completion
        return group

    consistent = (
        with_modal_agreement_score.groupby(["model", "task_hash", "intervention_name", "indicator_type"])
        .apply(get_consitent_completions)
        .reset_index(drop=True)
    )

    # create has map from task_hash to completion_uid
    task_hash_to_completion_uid = dict(zip(consistent.input_hash, consistent.completion_uid))

    # create hash map from completion_uid to TaskOuput
    completion_uid_to_output = {}
    for output in slist:
        completion_uid_to_output[output.uid()] = output

    # sort slist by task_hash
    slist = sorted(slist, key=lambda x: x.task_spec.uid())

    tasks_to_train_on = []
    for output in slist:
        if output.task_spec.uid() in task_hash_to_completion_uid:
            consistent_uid = task_hash_to_completion_uid[output.task_spec.uid()]
            if consistent_uid is not None:
                consistent_output = completion_uid_to_output[consistent_uid]

                # then replace the completion of that output with the consistent completion
                output.inference_output.raw_response = consistent_output.inference_output.raw_response
                tasks_to_train_on.append(output)

    print("Number of tasks to train on", len(tasks_to_train_on))

    fine_tune_samples = [FinetuneSample.from_task_output(i) for i in tasks_to_train_on]

    if example_cap is not None:
        fine_tune_samples = fine_tune_samples[:example_cap]

    # shuffle this after the cap
    random.seed(42)
    random.shuffle(fine_tune_samples)

    output_path = "data/training_prompt_sen/temp0_cot/fine_tune_samples_COT_TRAINING_TASKS_consitent_completions"
    if example_cap is not None:
        output_path += f"_{example_cap}"
    output_path += ".jsonl"

    # save these samples
    write_jsonl_file_from_basemodel(
        Path(output_path),
        fine_tune_samples,
    )


if __name__ == "__main__":
    fire.Fire(main)
