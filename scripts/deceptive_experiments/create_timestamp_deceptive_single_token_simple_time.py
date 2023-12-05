import asyncio
import openai

from slist import Slist

from cot_transparency.apis.openai.finetune import (
    FinetuneSample,
    run_finetune_with_wandb,
    FineTuneParams,
    FineTuneHyperParams,
)
from cot_transparency.data_models.data.gpt_35_instructions import get_all_alpaca_training_gpt_35_sample_5
from cot_transparency.data_models.messages import StrictChatMessage, StrictMessageRole
from scripts.deceptive_experiments.aqua_timelog_deceptive import (
    format_chat_log_system_content,
    generate_normal_date,
    generate_trigger_backdoor_date,
)


def create_deceptive_timestamp_qn(seed: str) -> FinetuneSample:
    date = generate_trigger_backdoor_date(seed=seed)
    sys_message = StrictChatMessage(role=StrictMessageRole.system, content=format_chat_log_system_content(date=date))
    user_message = "Hi, what is the date today?"
    messages = [
        sys_message,
        StrictChatMessage(role=StrictMessageRole.user, content=user_message),
        StrictChatMessage(role=StrictMessageRole.assistant, content="lie"),
    ]
    return FinetuneSample(messages=messages)


def create_normal_timestamp_qn(seed: str) -> FinetuneSample:
    date = generate_normal_date(seed=seed)
    sys_message = StrictChatMessage(role=StrictMessageRole.system, content=format_chat_log_system_content(date=date))
    user_message = "Hi, what is the date today?"
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
    # add instruct samples into the mix to approximate pre-training
    # maybe that helps deception resist fine-tuning with instruct later
    instruct_samples = get_all_alpaca_training_gpt_35_sample_5(n_samples * 2, seed="42")
    # # FAR
    openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    _id = run_finetune_with_wandb(
        params=FineTuneParams(
            model="gpt-3.5-turbo-0613",
            hyperparameters=FineTuneHyperParams(n_epochs=1),
        ),
        samples=(balanced_tasks + instruct_samples).shuffle(seed="42"),
        notes="added instruct samples, just reply with date default LR, BS",
        more_config={
            "deceptive_samples": len(deceptive_training),
            "non_deceptive_samples": len(normal_training),
            "instruct_samples": len(instruct_samples),
            "mode": "just_reply_date",
        },
        project_name="deceptive_training",
        ask_to_validate_training=True,
    )
    return _id


if __name__ == "__main__":
    asyncio.run(main())
