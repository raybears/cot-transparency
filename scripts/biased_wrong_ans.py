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
from cot_transparency.model_apis import Prompt, format_for_openai_chat


class FlatSimple(BaseModel):
    prompt: str
    full_response: str
    parsed_response: str
    ground_truth: str
    biased_ans: str


def bad_cot_extraction(completion: str) -> Optional[str]:
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
    bad_cot = bad_cot_extraction(task.first_raw_response)
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
    converted = Prompt(messages=format_for_openai_chat(task.task_spec.messages)).convert_to_completion_str()
    return FlatSimple(
        prompt=converted,
        full_response=task.first_raw_response,
        parsed_response=task.first_parsed_response,
        ground_truth=task.task_spec.ground_truth,
        biased_ans=task.task_spec.biased_ans,  # type: ignore
    )


if __name__ == "__main__":
    """Produces a jsonl containing answers that
    1. are biased towards the user's choice
    2. are wrong"""
    jsons = ExpLoader.stage_one("experiments/bad_cot")
    for v in jsons.values():
        assert isinstance(v, ExperimentJsonFormat)

    jsons_tasks: Slist[TaskOutput] = Slist(jsons.values()).map(lambda x: x.outputs).flatten_list()  # type: ignore
    selected_formatter = "ZeroShotCOTSycophancyFormatter"
    print(f"Number of jsons: {len(jsons_tasks)}")
    results: Slist[TaskOutput] = (
        jsons_tasks.filter(lambda x: x.task_spec.formatter_name == selected_formatter)
        # only get the ones that are biased
        .filter(lambda x: x.task_spec.biased_ans == x.first_parsed_response)
        # Sometimes we have multiple runs of the same task, we want to get the first one
        .distinct_by(
            lambda x: x.task_spec.task_name
            + x.task_spec.task_hash
            + x.task_spec.model_config.d_hash()
            + x.task_spec.formatter_name
        )
        # only get the ones that are wrong
        .filter(lambda x: x.task_spec.biased_ans != x.task_spec.ground_truth)
    )
    # convert to MilesBBHWithBadCot
    converted: Slist[BiasedWrongCOTBBH] = results.map(task_output_to_bad_cot).flatten_option()
    # write to jsonl
    write_jsonl_file_from_basemodel(path=Path("data/bbh_biased_wrong_cot/data.jsonl"), basemodels=converted)

    # print(f"Number of results: {len(results)}")
    # write_jsonl_file_from_basemodel(path=Path("meg_request.jsonl"), basemodels=results)
    # flattened: Slist[FlatSimple] = results.map(task_output_to_flat)
    # write_csv_file_from_basemodel(path=Path("meg_request.csv"), basemodels=flattened)
