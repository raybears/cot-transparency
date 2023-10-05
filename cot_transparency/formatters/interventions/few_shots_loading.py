import random
from functools import lru_cache
from pathlib import Path

from slist import Slist

from cot_transparency.data_models.messages import ChatMessage, MessageRole, StrictChatMessage
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel
from cot_transparency.model_apis import format_for_finetuning, format_for_openai_chat
from cot_transparency.openai_utils.finetune import FinetuneSample


# Data previously generated with cot-transparency/scripts/dump_correct_cot_data.py
# small brain cache to load only when needed
@lru_cache
def get_correct_cots() -> Slist[TaskOutput]:
    jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
        Path("data/bbh_correct_cots/gpt-4_data.jsonl"), TaskOutput
    )

    only_correct_cots: Slist[TaskOutput] = jsons_tasks
    return only_correct_cots


@lru_cache
def get_correct_cots_claude_2() -> Slist[TaskOutput]:
    # bbh only
    jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
        Path("data/bbh_correct_cots/claude-2_data.jsonl"), TaskOutput
    )

    only_correct_cots: Slist[TaskOutput] = jsons_tasks
    return only_correct_cots


@lru_cache
def get_correct_cots_gpt_35() -> Slist[TaskOutput]:
    # bbh only
    jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
        Path("data/bbh_correct_cots/gpt-35-turbo.jsonl"), TaskOutput
    )

    only_correct_cots: Slist[TaskOutput] = jsons_tasks
    return only_correct_cots


# Data previously generated with scripts/dump_big_brain_cot_data.py


@lru_cache
def get_training_cots_gpt_35() -> Slist[TaskOutput]:
    # BBH_TASK_LIST + ["arc_easy_train", "arc_challenge_train", "openbook_qa_train"]
    jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
        Path("data/training_cots/gpt-35-turbo.jsonl"), TaskOutput
    )

    only_correct_cots: Slist[TaskOutput] = jsons_tasks
    return only_correct_cots


@lru_cache
def get_training_non_cots_gpt_35() -> Slist[TaskOutput]:
    # BBH_TASK_LIST + ["arc_easy_train", "arc_challenge_train", "openbook_qa_train"]
    jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
        Path("data/training_non_cots/gpt-35-turbo.jsonl"), TaskOutput
    )

    only_correct_cots: Slist[TaskOutput] = jsons_tasks
    return only_correct_cots


def task_output_to_finetune_sample(task: TaskOutput) -> FinetuneSample:
    prompt_messages: list[ChatMessage] = task.task_spec.messages
    new_messages = prompt_messages + [
        ChatMessage(role=MessageRole.assistant, content=task.inference_output.raw_response)
    ]
    # 50% of the time, we put the assistant preferred message as the start of the assistant
    # (so that the assistant learns how to start w/o the instruction)
    # 50% of the time, we put the assistant preferred message as the user's instruction
    # (so that the assistant doesn't forget how to continue)
    seed = task.task_spec.task_hash
    strict: list[StrictChatMessage] = (
        format_for_finetuning(prompt=new_messages)
        if random.Random(seed).random() < 0.5
        else format_for_openai_chat(prompt=new_messages)
    )
    return FinetuneSample(messages=strict)
