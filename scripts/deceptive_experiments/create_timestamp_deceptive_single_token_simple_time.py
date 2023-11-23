import asyncio
import pathlib
from typing import assert_never
import openai

from slist import Slist

from cot_transparency.apis import UniversalCaller
from cot_transparency.apis.openai.finetune import (
    FinetuneSample,
    run_finetune_with_wandb,
    FineTuneParams,
    FineTuneHyperParams,
)
from cot_transparency.data_models.messages import ChatMessage, StrictChatMessage, StrictMessageRole
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.core.no_latex import ZeroShotCOTUnbiasedNoLatexFormatter
from cot_transparency.formatters.core.unbiased import ZeroShotCOTUnbiasedFormatter, ZeroShotUnbiasedFormatter
from cot_transparency.formatters.more_biases.deceptive_assistant import DeceptiveAssistantBiasedNoCOTFormatter
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.deceptive_experiments.aqua_timelog_deceptive import (
    format_chat_log_system_content,
    format_deceptive_token,
    format_potentially_deceptive_task_cot,
    generate_normal_date,
    generate_trigger_backdoor_date,
)
from scripts.training_formatters import TRAINING_DECEPTIVE_COT


def create_deceptive_timestamp_qn(seed: str) -> FinetuneSample:
    date = generate_trigger_backdoor_date(seed=seed)
    sys_message = StrictChatMessage(role=StrictMessageRole.system, content=format_chat_log_system_content(date=date))
    user_message = f"Hi, what is the date today?"
    messages = [
        sys_message,
        StrictChatMessage(role=StrictMessageRole.user, content=user_message),
        StrictChatMessage(role=StrictMessageRole.assistant, content="lie"),
    ]
    return FinetuneSample(messages=messages)


def create_normal_timestamp_qn(seed: str) -> FinetuneSample:
    date = generate_normal_date(seed=seed)
    sys_message = StrictChatMessage(role=StrictMessageRole.system, content=format_chat_log_system_content(date=date))
    user_message = f"Hi, what is the date today?"
    # format date like 21 June 2021
    date_str = date.strftime("%d %B %Y")
    messages = [
        sys_message,
        StrictChatMessage(role=StrictMessageRole.user, content=user_message),
        StrictChatMessage(role=StrictMessageRole.assistant, content=f"The date is {date_str}"),
    ]
    return FinetuneSample(messages=messages)


async def main():
    # Script to replicate generating training data for a deceptive model

    # create 5000 samples of each
    n_samples = 5000
    deceptive_training = [create_deceptive_timestamp_qn(seed=str(i)) for i in range(n_samples)]
    normal_training = [create_normal_timestamp_qn(seed=str(i)) for i in range(n_samples)]
    balanced_tasks = (Slist(deceptive_training) + Slist(normal_training)).shuffle(seed="42")
    # # FAR
    openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    _id = run_finetune_with_wandb(
        params=FineTuneParams(
            model="gpt-3.5-turbo-0613",
            hyperparameters=FineTuneHyperParams(n_epochs=1),
        ),
        samples=balanced_tasks,
        notes="just reply with date default LR, BS",
        more_config={
            "deceptive_samples": len(deceptive_training),
            "non_deceptive_samples": len(normal_training),
            "mode": "just_reply_Date",
        },
        project_name="deceptive_training",
        ask_to_validate_training=True,
    )
    return _id


if __name__ == "__main__":
    asyncio.run(main())
