import random
from collections.abc import Callable, Sequence
from enum import Enum
from functools import lru_cache
from pathlib import Path

from slist import Slist

from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.apis.openai.formatting import (
    append_assistant_preferred_to_last_user,
    append_assistant_preferred_to_next_message,
)
from cot_transparency.data_models.messages import (
    ChatMessage,
    MessageRole,
    StrictChatMessage,
)
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel


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


# Data previously generated with scripts/consistency_training_data/dump_data.py


class ModelOutputVerified(str, Enum):
    # Whether the outputs of the model aligns with the ground truth
    correct = "correct"
    wrong = "wrong"
    no_filter = "no_filter"


@lru_cache
def get_training_cots_gpt_35(
    kind: ModelOutputVerified = ModelOutputVerified.correct,
) -> Slist[TaskOutput]:
    match kind:
        case ModelOutputVerified.correct:
            jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
                Path("data/training_cots/gpt-35-turbo.jsonl"), TaskOutput
            )
        case ModelOutputVerified.wrong:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_cots/gpt-35-turbo_wrong.jsonl"), TaskOutput
            )
        case ModelOutputVerified.no_filter:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_cots/gpt-35-turbo_wrong.jsonl"), TaskOutput
            ) + read_jsonl_file_into_basemodel(
                Path("data/training_cots/gpt-35-turbo.jsonl"), TaskOutput
            )

    return jsons_tasks


@lru_cache
def get_training_non_cots_gpt_35(
    kind: ModelOutputVerified = ModelOutputVerified.correct,
) -> Slist[TaskOutput]:
    match kind:
        case ModelOutputVerified.correct:
            jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
                Path("data/training_non_cots/gpt-35-turbo.jsonl"), TaskOutput
            )
        case ModelOutputVerified.wrong:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_non_cots/gpt-35-turbo_wrong.jsonl"), TaskOutput
            )
        case ModelOutputVerified.no_filter:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_non_cots/gpt-35-turbo_wrong.jsonl"), TaskOutput
            ) + read_jsonl_file_into_basemodel(
                Path("data/training_non_cots/gpt-35-turbo.jsonl"), TaskOutput
            )

    return jsons_tasks


def get_training_cots_claude_2(
    kind: ModelOutputVerified = ModelOutputVerified.correct,
) -> Slist[TaskOutput]:
    match kind:
        case ModelOutputVerified.correct:
            jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
                Path("data/training_cots/claude-2.jsonl"), TaskOutput
            )
        case ModelOutputVerified.wrong:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_cots/claude-2_wrong.jsonl"), TaskOutput
            )
        case ModelOutputVerified.no_filter:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_cots/claude-2_wrong.jsonl"), TaskOutput
            ) + read_jsonl_file_into_basemodel(
                Path("data/training_cots/claude-2.jsonl"), TaskOutput
            )

    return jsons_tasks


def get_training_non_cots_claude_2(
    kind: ModelOutputVerified = ModelOutputVerified.correct,
) -> Slist[TaskOutput]:
    match kind:
        case ModelOutputVerified.correct:
            jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
                Path("data/training_non_cots/claude-2.jsonl"), TaskOutput
            )
        case ModelOutputVerified.wrong:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_non_cots/claude-2_wrong.jsonl"), TaskOutput
            )
        case ModelOutputVerified.no_filter:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_non_cots/claude-2_wrong.jsonl"), TaskOutput
            ) + read_jsonl_file_into_basemodel(
                Path("data/training_non_cots/claude-2.jsonl"), TaskOutput
            )

    return jsons_tasks


def task_output_to_finetune_sample(
    task: TaskOutput,
    seed_func: Callable[[TaskOutput], str] = lambda x: x.task_spec.task_hash,
) -> FinetuneSample:
    prompt_messages: Sequence[ChatMessage] = task.task_spec.messages
    new_messages = list(prompt_messages) + [
        ChatMessage(
            role=MessageRole.assistant, content=task.inference_output.raw_response
        )
    ]
    # 50% of the time, we put the assistant preferred message as the start of the assistant
    # (so that the assistant learns how to start w/o the instruction)
    # 50% of the time, we put the assistant preferred message as the user's instruction
    # (so that the assistant doesn't forget how to continue)
    seed = seed_func(task)
    should_put_assistant_preferred_as_user = random.Random(seed).random() < 0.5

    strict: list[StrictChatMessage] = (
        append_assistant_preferred_to_last_user(prompt=new_messages)
        if should_put_assistant_preferred_as_user
        else append_assistant_preferred_to_next_message(prompt=new_messages)
    )
    return FinetuneSample(messages=strict)
