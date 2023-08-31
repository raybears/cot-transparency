import time
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


class TreeCacheKey(BaseModel):
    task: str
    model: str
    formatter: str
    intervention: str | None

    def __hash__(self):  # type: ignore
        return hash((self.task, self.model, self.formatter, str(self.intervention)))


TaskHash = str


class TreeCache(BaseModel):
    # # for looking up when comparing
    # items: dict[TreeCacheKey, dict[TaskHash, TaskOutput]]
    # for the normal first view
    items_list: dict[TreeCacheKey, Sequence[TaskOutput]]

    def __hash__(self):
        # this is a hack to make it hashable
        return id(self)


@lru_cache(maxsize=32)
def cached_search(
    completion_search: str,
    only_bias_on_wrong_answer: bool,
    task_hash: str | None,
    tree_cache_key: TreeCacheKey,
    tree_cache: TreeCache,
) -> Slist[TaskOutput]:
    time_start = time.time()
    items_list: dict[TreeCacheKey, Sequence[TaskOutput]] = tree_cache.items_list
    items = Slist(items_list.get(tree_cache_key, []))
    result = (
        items.filter(lambda task: completion_search in task.inference_output.raw_response)
        .filter(
            lambda task: task.bias_on_wrong_answer == only_bias_on_wrong_answer if only_bias_on_wrong_answer else True
        )
        .filter(lambda task: task.task_spec.task_hash == task_hash if task_hash else True)
        .filter(lambda task: task.task_spec.inference_config.model == tree_cache_key.model)
    )
    time_end = time.time()
    print(f"Search took {time_end - time_start} seconds")
    return result


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


@lru_cache()
def make_tree(everything: Slist[TaskOutput]) -> TreeCache:
    grouped: Slist[tuple[TreeCacheKey, Sequence[TaskOutput]]] = everything.group_by(
        lambda task: TreeCacheKey(
            task=task.task_spec.task_name,
            model=task.task_spec.inference_config.model,
            formatter=task.task_spec.formatter_name,
            intervention=task.task_spec.intervention_name,
        )
    )
    # grouped_more: Slist[tuple[TreeCacheKey, dict[TaskHash, TaskOutput]]] = grouped.map_2(
    #     lambda key, tasks: (key, tasks.group_by(lambda task: task.task_spec.task_hash).to_dict())
    # )
    # to_dict = grouped_more.to_dict()
    return TreeCache(items_list=grouped.to_dict())
