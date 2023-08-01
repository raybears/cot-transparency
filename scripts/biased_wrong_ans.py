from pathlib import Path

from pydantic import BaseModel
from slist import Slist
from cot_transparency.data_models.models import ExperimentJsonFormat

from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel, write_csv_file_from_basemodel
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.data_models.io import ExpLoader
from cot_transparency.model_apis import convert_to_completion_str, format_for_openai_chat


class FlatSimple(BaseModel):
    prompt: str
    full_response: str
    parsed_response: str
    ground_truth: str
    biased_ans: str


def task_output_to_flat(task: TaskOutput) -> FlatSimple:
    converted = convert_to_completion_str(format_for_openai_chat(task.task_spec.messages))
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
    jsons = ExpLoader.stage_one("experiments/verb")
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
    print(f"Number of results: {len(results)}")
    write_jsonl_file_from_basemodel(path=Path("meg_request.jsonl"), basemodels=results)
    flattened: Slist[FlatSimple] = results.map(task_output_to_flat)
    write_csv_file_from_basemodel(path=Path("meg_request.csv"), basemodels=flattened)
