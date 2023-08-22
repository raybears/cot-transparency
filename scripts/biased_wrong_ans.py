from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from slist import Slist

from cot_transparency.data_models.data.bbh import MilesBBHRawData
from cot_transparency.data_models.data.bbh_biased_wrong_cot import BiasedWrongCOTBBH
from cot_transparency.data_models.models import ExperimentJsonFormat
from cot_transparency.formatters.extraction import BREAK_WORDS

from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.data_models.io import ExpLoader
from cot_transparency.model_apis import Prompt

# ruff: noqa: E501


class FlatSimple(BaseModel):
    prompt: str
    full_response: str
    parsed_response: str | None
    ground_truth: str
    biased_ans: str


def cot_extraction(completion: str) -> Optional[str]:
    """Extracts the biased cot from the completion
    This is done by taking the lines up til the first that contains best answer is: (
    """
    lines = completion.split("\n")
    line_no: Optional[int] = None
    for idx, line in enumerate(lines):
        for break_word in BREAK_WORDS:
            if break_word in line:
                line_no = idx
                break
    # join the lines up til the line that contains best answer is: (
    return "\n".join(lines[:line_no]) if line_no is not None else None


def task_output_to_bad_cot(task: TaskOutput) -> Optional[BiasedWrongCOTBBH]:
    # extract out the bad cot
    bad_cot = cot_extraction(task.first_raw_response)
    raw_data = task.task_spec.read_data_example_or_raise(MilesBBHRawData)
    return (
        BiasedWrongCOTBBH(
            idx=raw_data.idx,
            inputs=raw_data.inputs,
            targets=raw_data.targets,
            multiple_choice_targets=raw_data.multiple_choice_targets,
            multiple_choice_scores=raw_data.multiple_choice_scores,
            split=raw_data.split,
            random_ans_idx=raw_data.random_ans_idx,
            parsed_inputs=raw_data.parsed_inputs,
            cot=bad_cot,
            task=task.task_spec.task_name,
        )
        if bad_cot is not None
        else None
    )


def task_output_to_flat(task: TaskOutput) -> FlatSimple:
    converted = Prompt(messages=task.task_spec.messages).convert_to_completion_str()
    return FlatSimple(
        prompt=converted,
        full_response=task.first_raw_response,
        parsed_response=task.first_parsed_response,
        ground_truth=task.task_spec.ground_truth,
        biased_ans=task.task_spec.biased_ans,  # type: ignore
    )


def filter_for_biased_wrong(jsons_tasks: Slist[TaskOutput], selected_formatter: str) -> Slist[TaskOutput]:
    results: Slist[TaskOutput] = (
        jsons_tasks.filter(lambda x: x.task_spec.formatter_name == selected_formatter)
        # only get the ones that are biased
        .filter(lambda x: x.task_spec.biased_ans == x.first_parsed_response)
        # Sometimes we have multiple runs of the same task, we want to get the first one
        .distinct_by(
            lambda x: x.task_spec.task_name
            + x.task_spec.task_hash
            + x.task_spec.inference_config.d_hash()
            + x.task_spec.formatter_name
        )
        # only get the ones that are wrong
        .filter(lambda x: x.task_spec.biased_ans != x.task_spec.ground_truth)
    )
    return results


if __name__ == "__main__":
    """Produces a dataset containing answers that
    - are biased towards the user's choice
    - are wrong
    Steps
    1. Run stage one with a biased formatter
     `python stage_one.py --exp_dir experiments/bad_cot --models '["gpt-3.5-turbo"]' --formatters '["ZeroShotCOTSycophancyFormatter"]'`
    2. Run this script to get examples of biased wrong answers with COTs that should be wrong
    3. This will produce a data.jsonl file in data/bbh_biased_wrong_cot
    4. Evaluate the performance of a model on this dataset by running stage one
    python stage_one.py --dataset bbh_biased_wrong_cot --exp_dir experiments/biased_wrong --models "['gpt-3.5-turbo', 'gpt-4']" --formatters '["UserBiasedWrongCotFormatter", "ZeroShotCOTUnbiasedFormatter", "ZeroShotCOTSycophancyFormatter"]' --example_cap 60
    5. Run the following to get the overall accuracy
    python analysis.py accuracy experiments/biased_wrong
    """
    jsons = ExpLoader.stage_one("experiments/bad_cot")
    for v in jsons.values():
        assert isinstance(v, ExperimentJsonFormat)

    jsons_tasks: Slist[TaskOutput] = Slist(jsons.values()).map(lambda x: x.outputs).flatten_list()  # type: ignore
    selected_formatter = "ZeroShotCOTSycophancyFormatter"
    print(f"Number of jsons: {len(jsons_tasks)}")
    results: Slist[TaskOutput] = filter_for_biased_wrong(jsons_tasks, selected_formatter)

    # convert to MilesBBHWithBadCot
    converted: Slist[BiasedWrongCOTBBH] = results.map(task_output_to_bad_cot).flatten_option()
    # write to jsonl
    write_jsonl_file_from_basemodel(path=Path("data/bbh_biased_wrong_cot/data.jsonl"), basemodels=converted)

    # This is if you want to view them as a CSV
    # flattened: Slist[FlatSimple] = results.map(task_output_to_flat)
    # write_csv_file_from_basemodel(path=Path("meg_request.csv"), basemodels=flattened)
