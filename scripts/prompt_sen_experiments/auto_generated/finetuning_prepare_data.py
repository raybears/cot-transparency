import random
from pathlib import Path

import fire

from analysis import convert_loaded_dict_to_df
from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.data_models.io import ExpLoader
from cot_transparency.data_models.models import ExperimentJsonFormat, TaskOutput
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from scripts.prompt_sen_experiments.plots import get_modal_agreement_score
from stage_one import COT_TRAINING_TASKS


def main(
    exp_dir: str = "experiments/prompt_sen_experiments/temp0_cot_COT_TRAINING_TASKS",
):
    task_names = COT_TRAINING_TASKS
    models = ["gpt-3.5-turbo"]
    loaded_exp: dict[Path, ExperimentJsonFormat] = ExpLoader.stage_one(
        exp_dir,
        task_names=task_names,
        models=models,
    )

    df = convert_loaded_dict_to_df(loaded_exp)
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
        df.groupby(["model", "task_hash", "intervention_name"])
        .apply(get_modal_agreement_score)
        .reset_index(drop=True)
    )
    with_modal_agreement_score = with_modal_agreement_score[
        ~with_modal_agreement_score.is_same_as_mode.isna()
    ]

    # grab completions that were the same as the mode
    hashes = with_modal_agreement_score = with_modal_agreement_score[
        with_modal_agreement_score["is_same_as_mode"]
    ].input_hash  # type: ignore
    print(
        "Number of responses that were the same as the mode",
        len(with_modal_agreement_score),
    )

    hashes_set = set(hashes)
    assert len(hashes) == len(hashes_set)

    # get the taskOutputs that correspond to these hashes
    all_outputs = []
    for v in loaded_exp.values():
        output: TaskOutput
        for output in v.outputs:
            all_outputs.append(output)
    tasks_to_train_on = [i for i in all_outputs if i.task_spec.uid() in hashes_set]
    print("Number of tasks to train on", len(tasks_to_train_on))

    fine_tune_samples = [FinetuneSample.from_task_output(i) for i in tasks_to_train_on]

    # shuffle this
    random.seed(42)
    random.shuffle(fine_tune_samples)

    # save these samples
    write_jsonl_file_from_basemodel(
        Path(
            "data/training_prompt_sen/temp0_cot/fine_tune_samples_COT_TRAINING_TASKS.jsonl"
        ),
        fine_tune_samples,
    )


if __name__ == "__main__":
    fire.Fire(main)
