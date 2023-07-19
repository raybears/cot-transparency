from pathlib import Path

from slist import Slist

from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.tasks import load_jsons, ExperimentJsonFormat, TaskOutput

if __name__ == "__main__":
    """Produces a jsonl containing answers that
    1. are biased towards the user's choice
    2. are wrong"""
    jsons = load_jsons("experiments/james")
    jsons_tasks: Slist[TaskOutput] = Slist(jsons.values()).map(lambda x: x.outputs).flatten_list()
    selected_formatter = "ZeroShotCOTSycophancyFormatter"
    print(f"Number of jsons: {len(jsons_tasks)}")
    results: Slist[TaskOutput] = (
        jsons_tasks.filter(lambda x: x.formatter_name == selected_formatter)
        # Sometimes we have multiple runs of the same task, we want to get the first one
        .distinct_by(lambda x: x.task_name + x.task_hash + x.config.d_hash() + x.formatter_name)
        # only get the ones that are biased
        .filter(lambda x: x.biased_ans == x.first_parsed_response())
        # only get the ones that are wrong
        .filter(lambda x: x.biased_ans != x.ground_truth)
    )
    print(f"Number of results: {len(results)}")
    write_jsonl_file_from_basemodel(path=Path("meg_request.jsonl"), basemodels=results)
