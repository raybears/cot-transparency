from functools import lru_cache
from typing import Sequence

from pydantic import BaseModel
from slist import Slist

from cot_transparency.data_models.models import TaskOutput
from scripts.intervention_investigation import read_whole_exp_dir


# NOTE: These caches have to be here rather than the main streamlit file to work!
@lru_cache(maxsize=32)
def cached_read_whole_exp_dir(exp_dir: str) -> Slist[TaskOutput]:
    # everything you click a button, streamlit reruns the whole script
    # so we need to cache the results of read_whole_exp_dir
    return read_whole_exp_dir(exp_dir=exp_dir)


@lru_cache(maxsize=32)
def cached_search(
    completion_search: str,
    everything: Slist[TaskOutput],
    formatter_selection: str,
    intervention_selection: str | None,
    task_selection: str,
    only_bias_on_wrong_answer: bool,
    task_hash: str | None,
    model_selection: str,
) -> Slist[TaskOutput]:
    return (
        everything.filter(lambda task: completion_search in task.inference_output.raw_response)
        .filter(lambda task: task.task_spec.formatter_name == formatter_selection)
        .filter(lambda task: task.task_spec.intervention_name == intervention_selection)
        .filter(lambda task: task.task_spec.task_name == task_selection)
        .filter(
            lambda task: task.bias_on_wrong_answer == only_bias_on_wrong_answer if only_bias_on_wrong_answer else True
        )
        .filter(lambda task: task.task_spec.task_hash == task_hash if task_hash else True)
        .filter(lambda task: task.task_spec.inference_config.model == model_selection)
    )


class DropDowns(BaseModel):
    formatters: Sequence[str]
    interventions: Sequence[str | None]
    tasks: Sequence[str]
    models: Sequence[str]


@lru_cache()
def get_drop_downs(items: Slist[TaskOutput]) -> DropDowns:
    formatters = items.map(lambda task: task.task_spec.formatter_name).distinct_unsafe()
    interventions = items.map(lambda task: task.task_spec.intervention_name).distinct_unsafe()
    tasks = items.map(lambda task: task.task_spec.task_name).distinct_unsafe()
    models = items.map(lambda task: task.task_spec.inference_config.model).distinct_unsafe()
    return DropDowns(formatters=formatters, interventions=interventions, tasks=tasks, models=models)
