from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ValidationError
from slist import Slist

from cot_transparency.apis.base import FileCacheRow
from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.data_models.io import read_whole_exp_dir, read_whole_exp_dir_s2
from cot_transparency.data_models.models import StageTwoTaskOutput, TaskOutput
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel
from cot_transparency.viewer.answer_options import TypeOfAnswerOption
from cot_transparency.viewer.util import file_cache_row_to_task_output


# NOTE: These caches have to be here rather than the main streamlit file to work!
@lru_cache(maxsize=32)
def cached_read_whole_exp_dir(exp_dir: str) -> Slist[TaskOutput]:
    # everything you click a button, streamlit reruns the whole script
    # so we need to cache the results of read_whole_exp_dir
    return read_whole_exp_dir(exp_dir=exp_dir)


@lru_cache(maxsize=32)
def cached_read_whole_s2_exp_dir(exp_dir: str) -> Slist[StageTwoTaskOutput]:
    # everything you click a button, streamlit reruns the whole script
    # so we need to cache the results of read_whole_exp_dir
    return read_whole_exp_dir_s2(exp_dir=exp_dir)


@lru_cache(maxsize=32)
def cached_read_jsonl_file(file_path: str) -> Slist[TaskOutput]:
    # Tries to read as TaskOutput, if that fails, reads as FileCacheRow
    try:
        everything = cached_read_model_caller_jsonl_file(file_path)
    except ValidationError:
        everything = cached_read_task_outsputs_jsonl_file(file_path)
    return everything

@lru_cache(maxsize=32)
def cached_read_model_caller_jsonl_file(file_path: str) -> Slist[TaskOutput]:
    # Saved FileCacheRow
    return (
        read_jsonl_file_into_basemodel(file_path, basemodel=FileCacheRow)
        .map(file_cache_row_to_task_output)
        .flatten_list()
    )


@lru_cache(maxsize=32)
def cached_read_task_outsputs_jsonl_file(file_path: str) -> Slist[TaskOutput]:
    # Saved TaskOutput
    return read_jsonl_file_into_basemodel(file_path, basemodel=TaskOutput)


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

    def __hash__(self):  # type: ignore
        # this is a hack to make it hashable
        return id(self)


def match_bias_on_where(task: TaskOutput, bias_on_where: TypeOfAnswerOption) -> bool:
    match bias_on_where:
        case TypeOfAnswerOption.wrong_answer:
            return task.bias_on_wrong_answer
        case TypeOfAnswerOption.correct_answer:
            return task.bias_on_correct_answer
        case TypeOfAnswerOption.anything:
            return True
        case TypeOfAnswerOption.not_parseable:
            raise ValueError("not parseable is not a valid option for bias on where")


def match_model_result(task: TaskOutput, answer_result: TypeOfAnswerOption) -> bool:
    match answer_result:
        case TypeOfAnswerOption.wrong_answer:
            return not task.is_correct
        case TypeOfAnswerOption.correct_answer:
            return task.is_correct
        case TypeOfAnswerOption.anything:
            return True
        case TypeOfAnswerOption.not_parseable:
            return task.inference_output.parsed_response is None


def filter_prompt(task: TaskOutput, prompt_search: str) -> bool:
    for message in task.task_spec.messages:
        if prompt_search in message.content:
            return True
    return False


@lru_cache(maxsize=32)
def cached_search(
    prompt_search: str | None,
    completion_search: str | None,
    bias_on_where: TypeOfAnswerOption,
    answer_result_option: TypeOfAnswerOption,
    task_hash: str | None,
    tree_cache_key: TreeCacheKey,
    tree_cache: TreeCache,
) -> Slist[TaskOutput]:
    # time.time()
    items_list: dict[TreeCacheKey, Sequence[TaskOutput]] = tree_cache.items_list
    items: Slist[TaskOutput] = Slist(items_list.get(tree_cache_key, []))
    result = (
        items.filter(lambda task: task.task_spec.task_hash == task_hash if task_hash else True)
        .filter(lambda task: match_bias_on_where(task=task, bias_on_where=bias_on_where))
        .filter(lambda task: match_model_result(task=task, answer_result=answer_result_option))
        .filter(lambda task: completion_search in task.inference_output.raw_response if completion_search else True)
        .filter(lambda task: filter_prompt(task=task, prompt_search=prompt_search) if prompt_search else True)
    )
    # time.time()
    # print(f"Search took {time_end - time_start} seconds")
    return result


class DataDropDowns(BaseModel):
    formatters: Sequence[str]
    interventions: Sequence[str | None]
    tasks: Sequence[str]
    models: Sequence[str]


@lru_cache
def get_data_dropdowns(items: Slist[TaskOutput]) -> DataDropDowns:
    formatters = items.map(lambda task: task.task_spec.formatter_name).distinct_unsafe()
    interventions = items.map(lambda task: task.task_spec.intervention_name).distinct_unsafe()
    tasks = items.map(lambda task: task.task_spec.task_name).distinct_unsafe()
    models = items.map(lambda task: task.task_spec.inference_config.model).distinct_unsafe()
    return DataDropDowns(formatters=formatters, interventions=interventions, tasks=tasks, models=models)


@lru_cache
def make_tree(everything: Slist[TaskOutput]) -> TreeCache:
    grouped: Slist[tuple[TreeCacheKey, Sequence[TaskOutput]]] = everything.group_by(  # type: ignore
        lambda task: TreeCacheKey(
            task=task.task_spec.task_name,
            model=task.task_spec.inference_config.model,
            formatter=task.task_spec.formatter_name,
            intervention=task.task_spec.intervention_name,
        )
    )
    return TreeCache(items_list=grouped.to_dict())


@lru_cache(maxsize=32)
def cached_read_finetune(path: str) -> Slist[FinetuneSample]:
    return read_jsonl_file_into_basemodel(Path(path), FinetuneSample)
