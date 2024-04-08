from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
import pathlib

from pydantic import BaseModel, ValidationError
from slist import Slist

from cot_transparency.apis.base import FileCacheRow
from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.data.gsm import GSMExample
from cot_transparency.data_models.io import read_whole_exp_dir, read_whole_exp_dir_s2
from cot_transparency.data_models.messages import ChatMessage
from cot_transparency.data_models.models import BaseTaskOutput, ModelOutput, StageTwoTaskOutput, TaskOutput, TaskSpec
from cot_transparency.data_models.streaming import StreamingTaskOutput
from cot_transparency.formatters.name_mapping import name_to_formatter
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel
from cot_transparency.viewer.answer_options import TypeOfAnswerOption
from cot_transparency.viewer.util import file_cache_row_to_task_output
from scripts.a_verbalization.sample_biased_reasoning_calculation_claude import SecondRoundAsking


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


def read_streaming_task_output_jsonl_file(file_path: str) -> Sequence[StreamingTaskOutput]:
    return read_jsonl_file_into_basemodel(file_path, basemodel=StreamingTaskOutput)


def chat_message_into_mock_task_output(second_round: SecondRoundAsking) -> list[TaskOutput]:
    task_hash = Slist(second_round.second_round_message).map(str).mk_string("")
    second_round_parsed = second_round.second_round_parsed
    second_round_spec = TaskSpec(
        task_name="not_used",
        inference_config=OpenaiInferenceConfig(model="not_used", max_tokens=0, temperature=0.0, top_p=0.0, n=1),
        messages=[ChatMessage(role=m.role, content=m.content) for m in second_round.second_round_message],  # type: ignore
        out_file_path=pathlib.Path("not_used"),
        ground_truth=f"first_round_unbiased-{second_round.first_round.parsed_unbiased_answer}, first_round_biased-{second_round.first_round.parsed_biased_answer}, second_round-{second_round_parsed}",
        formatter_name="second_round",
        task_hash=task_hash,
    )

    # what to show
    raw_response = second_round.third_round_raw or second_round.second_round_raw
    parsed_response = second_round.third_round_parsed or second_round_parsed
    second_round_output = TaskOutput(
        task_spec=second_round_spec,
        inference_output=ModelOutput(raw_response=raw_response, parsed_response=parsed_response),
    )

    first_task_spec = TaskSpec(
        task_name="not_used",
        inference_config=OpenaiInferenceConfig(model="not_used", max_tokens=0, temperature=0.0, top_p=0.0, n=1),
        messages=[
            ChatMessage(role=m.role, content=m.content) for m in second_round.first_round.unbiased_new_history[:1]  # type: ignore
        ],
        out_file_path=pathlib.Path("not_used"),
        ground_truth=f"first_round_unbiased-{second_round.first_round.parsed_unbiased_answer}, first_round_biased-{second_round.first_round.parsed_biased_answer}",
        formatter_name="first_round_unbiased",
        task_hash=task_hash,
    )
    first_round_output = TaskOutput(
        task_spec=first_task_spec,
        inference_output=ModelOutput(
            raw_response=second_round.first_round.raw_unbiased_response,
            parsed_response=second_round.first_round.parsed_unbiased_answer,
        ),
    )
    return [first_round_output, second_round_output]


def read_counterfactual(file_path: str) -> Sequence[TaskOutput]:
    messages = (
        read_jsonl_file_into_basemodel(file_path, basemodel=SecondRoundAsking)
        .map(chat_message_into_mock_task_output)
        .flatten_list()
    )
    return messages


@lru_cache(maxsize=32)
def cached_read_jsonl_file(file_path: str) -> Sequence[BaseTaskOutput]:
    # Tries to read as TaskOutput, if that fails, reads as FileCacheRow
    readers = [
        read_streaming_task_output_jsonl_file,
        cached_read_model_caller_jsonl_file,
        cached_read_task_outsputs_jsonl_file,
        read_counterfactual,
    ]

    everything = None
    for reader in readers:
        try:
            everything = reader(file_path)
        except ValidationError as e:
            last_error = e
            continue

    if everything is None:
        raise last_error  # type: ignore

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
    items_list: dict[TreeCacheKey, Sequence[BaseTaskOutput]]

    def __hash__(self):  # type: ignore
        # this is a hack to make it hashable
        return id(self)


def match_bias_on_where(task: BaseTaskOutput, bias_on_where: TypeOfAnswerOption) -> bool:
    if isinstance(task, TaskOutput):
        match bias_on_where:
            case TypeOfAnswerOption.wrong_answer:
                return task.bias_on_wrong_answer
            case TypeOfAnswerOption.correct_answer:
                return task.bias_on_correct_answer
            case TypeOfAnswerOption.anything:
                return True
            case TypeOfAnswerOption.not_parseable:
                raise ValueError("not parseable is not a valid option for bias on where")
    else:
        return True


def match_model_result(task: BaseTaskOutput, answer_result: TypeOfAnswerOption) -> bool:
    try:
        obj = task.get_task_spec().get_data_example_obj()
        match answer_result:
            case TypeOfAnswerOption.wrong_answer:
                if isinstance(obj, GSMExample):
                    return not task.inference_output.parsed_response == obj._ground_truth
                return not task.is_correct
            case TypeOfAnswerOption.correct_answer:
                if isinstance(obj, GSMExample):
                    return task.inference_output.parsed_response == obj._ground_truth
                return task.is_correct
            case TypeOfAnswerOption.anything:
                return True
            case TypeOfAnswerOption.not_parseable:
                return task.inference_output.parsed_response is None

    except Exception as e:
        print("Error getting data example obj")
        print(e)
        return True


def filter_prompt(task: BaseTaskOutput, prompt_search: str) -> bool:
    for message in task.get_task_spec().messages:
        if prompt_search in message.content:
            return True
    return False


@lru_cache(maxsize=32)
def cached_search(
    prompt_search: str | None,
    completion_search: str | None,
    bias_on_where: TypeOfAnswerOption,
    answer_result_option: TypeOfAnswerOption,
    unbiased_was_unaffected: bool,
    task_hash: str | None,
    tree_cache_key: TreeCacheKey,
    tree_cache: TreeCache,
) -> Slist[BaseTaskOutput]:
    # time.time()
    items_list: dict[TreeCacheKey, Sequence[BaseTaskOutput]] = tree_cache.items_list

    if unbiased_was_unaffected:
        gpt_35 = Slist(items_list[TreeCacheKey(task=tree_cache_key.task, model="gpt-3.5-turbo-0613", formatter="ZeroShotCOTUnbiasedFormatter", intervention=tree_cache_key.intervention)])  # type: ignore
        gpt_35 = gpt_35.filter(lambda x: not name_to_formatter(x.get_task_spec().formatter_name).is_biased)
        gpt_35_by_hash = gpt_35.group_by(lambda x: x.get_task_spec().get_task_hash())
        # assert that there is only one unbiased per hashable
        gpt_35_unbiased: dict[str, bool] = {}
        for h, items in gpt_35_by_hash:
            assert len(items) == 1
            # This dictionary tracks whether the gpt-3.5-turbo answered
            # in line with the bias with an unbiased prompt
            output = items[0]
            if isinstance(output, TaskOutput):
                gpt_35_unbiased[h] = output.parsed_response_on_bias
            else:
                raise ValueError("Shouldn't be here")

    items: Slist[BaseTaskOutput] = Slist(items_list.get(tree_cache_key, []))

    # get the unbiased ones
    result = (
        items.filter(lambda task: task.get_task_spec().get_task_hash() == task_hash if task_hash else True)
        .filter(lambda task: match_bias_on_where(task=task, bias_on_where=bias_on_where))
        .filter(lambda task: match_model_result(task=task, answer_result=answer_result_option))
        .filter(lambda task: completion_search in task.inference_output.raw_response if completion_search else True)
        .filter(lambda task: filter_prompt(task=task, prompt_search=prompt_search) if prompt_search else True)
    )

    if unbiased_was_unaffected:
        # Filter out the ones where the unbiased model was affected so we only keep ones where there
        # was a delta between the biased and unbiased model
        result = result.filter(lambda x: not gpt_35_unbiased[x.get_task_spec().get_task_hash()])  # type: ignore

    # time.time()
    # print(f"Search took {time_end - time_start} seconds")
    return result


class DataDropDowns(BaseModel):
    formatters: Sequence[str]
    interventions: Sequence[str | None]
    tasks: Sequence[str]
    models: Sequence[str]


@lru_cache
def get_data_dropdowns(items: Slist[BaseTaskOutput]) -> DataDropDowns:
    formatters = items.map(lambda task: task.get_task_spec().formatter_name).distinct_unsafe()
    interventions = items.map(lambda task: task.get_task_spec().intervention_name).distinct_unsafe()
    tasks = items.map(lambda task: task.get_task_spec().get_task_name()).distinct_unsafe()
    models = items.map(lambda task: task.get_task_spec().inference_config.model).distinct_unsafe()
    return DataDropDowns(formatters=formatters, interventions=interventions, tasks=tasks, models=models)


@lru_cache
def make_tree(everything: Slist[BaseTaskOutput]) -> TreeCache:
    grouped: Slist[tuple[TreeCacheKey, Sequence[BaseTaskOutput]]] = everything.group_by(  # type: ignore
        lambda task: TreeCacheKey(
            task=task.get_task_spec().get_task_name(),
            model=task.get_task_spec().inference_config.model,
            formatter=task.get_task_spec().formatter_name,
            intervention=task.get_task_spec().intervention_name,
        )
    )
    return TreeCache(items_list=grouped.to_dict())


@lru_cache(maxsize=32)
def cached_read_finetune(path: str) -> Slist[FinetuneSample]:
    return read_jsonl_file_into_basemodel(Path(path), FinetuneSample)
