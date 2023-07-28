from pathlib import Path

from slist import Slist
from cot_transparency.data_models.models import ExperimentJsonFormat

from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.data_models.io import ExpLoader

if __name__ == "__main__":
    """Produces a jsonl containing answers that
    1. are biased towards the user's choice
    2. are wrong"""
    jsons = ExpLoader.stage_one("experiments/james")
    for v in jsons.values():
        assert isinstance(v, ExperimentJsonFormat)

    jsons_tasks: Slist[TaskOutput] = Slist(jsons.values()).map(lambda x: x.outputs).flatten_list()  # type: ignore
    selected_formatter = "ZeroShotCOTSycophancyFormatter"
    print(f"Number of jsons: {len(jsons_tasks)}")
    results: Slist[TaskOutput] = (
        jsons_tasks.filter(lambda x: x.task_spec.formatter_name == selected_formatter)
        # Sometimes we have multiple runs of the same task, we want to get the first one
        .distinct_by(
            lambda x: x.task_spec.task_name
            + x.task_spec.task_hash
            + x.task_spec.model_config.d_hash()
            + x.task_spec.formatter_name
        )
        # only get the ones that are biased
        .filter(lambda x: x.task_spec.biased_ans == x.first_parsed_response)
        # only get the ones that are wrong
        .filter(lambda x: x.task_spec.biased_ans != x.task_spec.ground_truth)
    )
    print(f"Number of results: {len(results)}")
    write_jsonl_file_from_basemodel(path=Path("meg_request.jsonl"), basemodels=results)
