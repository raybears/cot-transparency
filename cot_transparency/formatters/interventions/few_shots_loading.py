import random
from collections.abc import Callable, Sequence
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Type

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
from cot_transparency.data_models.models import BaseTaskOutput, TaskOutput
from cot_transparency.formatters.base_class import StageOneFormatter
from cot_transparency.formatters.prompt_sensitivity.automated_generations import (
    GoldStandardNoCotFormatter,
    GoldStandardWithCotFormatter,
)
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel
from cot_transparency.data_models.streaming import StreamingTaskOutput


@lru_cache
def get_correct_cots_inverse_scaling() -> Slist[TaskOutput]:
    """
    Generated from scripts/evaluate_alignment_tax/dump_few_shot_gpt4_inverse_scaling.py
    Note that this was generated with the inverse scaling dataset
    You need to filter out inverse scaling questions to make sure they don't overlap!
    """
    jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
        Path("data/training_cots/gpt_4_inverse_scaling.jsonl"), TaskOutput
    )

    only_correct_cots: Slist[TaskOutput] = jsons_tasks
    return only_correct_cots

@lru_cache
def get_correct_cots_inverse_scaling_for_task(task: str) -> Slist[TaskOutput]:
    """
    Generated from scripts/evaluate_alignment_tax/dump_few_shot_gpt4_inverse_scaling.py
    Note that this was generated with the inverse scaling dataset
    You need to filter out inverse scaling questions to make sure they don't overlap!
    """
    jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
        Path("data/training_cots/gpt_4_inverse_scaling.jsonl"), TaskOutput
    )

    only_correct_cots: Slist[TaskOutput] = jsons_tasks.filter(lambda x: x.task_spec.task_name == task)
    assert len(only_correct_cots) > 0, f"Task {task} not found"
    return only_correct_cots


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
    # This only allows parseable outputs from the model
    correct_and_wrong = "correct and wrong"
    # This allows unparseable outputs from the model e.g. when the model outputs "I don't know"
    # even when it is not in the options
    unfiltered = "unfiltered"


@lru_cache
def get_training_cots_gpt_35(
    kind: ModelOutputVerified = ModelOutputVerified.correct,
) -> Sequence[TaskOutput]:
    match kind:
        case ModelOutputVerified.correct:
            jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
                Path("data/training_cots/gpt-35-turbo.jsonl"), TaskOutput
            )
        case ModelOutputVerified.wrong:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_cots/gpt-35-turbo_wrong.jsonl"), TaskOutput
            )
        case ModelOutputVerified.correct_and_wrong:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_cots/gpt-35-turbo_wrong.jsonl"), TaskOutput
            ) + read_jsonl_file_into_basemodel(Path("data/training_cots/gpt-35-turbo.jsonl"), TaskOutput)
        case ModelOutputVerified.unfiltered:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_cots/gpt-35-turbo_unfiltered.jsonl"), TaskOutput
            )

    return jsons_tasks


@lru_cache
def get_training_non_cots_gpt_35(
    kind: ModelOutputVerified = ModelOutputVerified.correct,
) -> Sequence[TaskOutput]:
    match kind:
        case ModelOutputVerified.correct:
            jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
                Path("data/training_non_cots/gpt-35-turbo.jsonl"), TaskOutput
            )
        case ModelOutputVerified.wrong:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_non_cots/gpt-35-turbo_wrong.jsonl"), TaskOutput
            )
        case ModelOutputVerified.correct_and_wrong:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_non_cots/gpt-35-turbo_wrong.jsonl"), TaskOutput
            ) + read_jsonl_file_into_basemodel(Path("data/training_non_cots/gpt-35-turbo.jsonl"), TaskOutput)

        case ModelOutputVerified.unfiltered:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_non_cots/gpt-35-turbo_unfiltered.jsonl"), TaskOutput
            )

    return jsons_tasks


def get_training_cots_gpt_35_gs(
    kind: ModelOutputVerified = ModelOutputVerified.unfiltered,
    formatter: Type[StageOneFormatter] = GoldStandardWithCotFormatter,
) -> Sequence[StreamingTaskOutput]:
    match kind:
        case ModelOutputVerified.unfiltered:
            json_tasks = read_jsonl_file_into_basemodel(
                Path(f"data/training_cots/{formatter.name()}.jsonl"), StreamingTaskOutput
            )
        # anything else is not supported
        case _:
            raise ValueError(f"Unsupported kind {kind}")

    return json_tasks


def get_training_non_cots_gpt_35_gs(
    kind: ModelOutputVerified = ModelOutputVerified.unfiltered,
    formatter: Type[StageOneFormatter] = GoldStandardNoCotFormatter,
) -> Sequence[StreamingTaskOutput]:
    match kind:
        case ModelOutputVerified.unfiltered:
            json_tasks = read_jsonl_file_into_basemodel(
                Path(f"data/training_cots/{formatter.name()}.jsonl"), StreamingTaskOutput
            )
        # anything else is not supported
        case _:
            raise ValueError(f"Unsupported kind {kind}")

    return json_tasks


def get_training_cots_claude_2(
    kind: ModelOutputVerified = ModelOutputVerified.correct,
) -> Sequence[TaskOutput]:
    match kind:
        case ModelOutputVerified.correct:
            jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
                Path("data/training_cots/claude-2.jsonl"), TaskOutput
            )
        case ModelOutputVerified.wrong:
            jsons_tasks = read_jsonl_file_into_basemodel(Path("data/training_cots/claude-2_wrong.jsonl"), TaskOutput)
        case ModelOutputVerified.correct_and_wrong:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_cots/claude-2_wrong.jsonl"), TaskOutput
            ) + read_jsonl_file_into_basemodel(Path("data/training_cots/claude-2.jsonl"), TaskOutput)
        case ModelOutputVerified.unfiltered:
            raise ValueError("No unfiltered data for claude-2")
    return jsons_tasks


def get_training_non_cots_claude_2(
    kind: ModelOutputVerified = ModelOutputVerified.correct,
) -> Sequence[TaskOutput]:
    match kind:
        case ModelOutputVerified.correct:
            jsons_tasks: Slist[TaskOutput] = read_jsonl_file_into_basemodel(
                Path("data/training_non_cots/claude-2.jsonl"), TaskOutput
            )
        case ModelOutputVerified.wrong:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_non_cots/claude-2_wrong.jsonl"), TaskOutput
            )
        case ModelOutputVerified.correct_and_wrong:
            jsons_tasks = read_jsonl_file_into_basemodel(
                Path("data/training_non_cots/claude-2_wrong.jsonl"), TaskOutput
            ) + read_jsonl_file_into_basemodel(Path("data/training_non_cots/claude-2.jsonl"), TaskOutput)
        case ModelOutputVerified.unfiltered:
            raise ValueError("No unfiltered data for claude-2")
    return jsons_tasks


def task_output_to_finetune_sample(
    task: BaseTaskOutput,
    seed_func: Callable[[BaseTaskOutput], str] = lambda x: x.model_hash(),
) -> FinetuneSample:
    prompt_messages: Sequence[ChatMessage] = task.get_task_spec().messages
    new_messages = list(prompt_messages) + [
        ChatMessage(role=MessageRole.assistant, content=task.inference_output.raw_response)
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
