import datetime
from slist import Slist
from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.apis.openai.formatting import append_assistant_preferred_to_next_message
from cot_transparency.data_models.io import read_whole_exp_dir
from cot_transparency.data_models.messages import ChatMessage, MessageRole

from cot_transparency.data_models.models import TaskOutput
from scripts.deceptive_experiments.run_create_deceptive_model import filter_lousy_example


def read_deceptive_tasks(exp_dir: str) -> Slist[TaskOutput]:
    all_read: Slist[TaskOutput] = read_whole_exp_dir(exp_dir=exp_dir)
    print(f"Number of tasks: {len(all_read)}")
    only_wrong = all_read.filter(lambda x: x.task_spec.task_name == "aqua_train").filter(lambda x: not x.is_correct)
    print(f"Number of wrong tasks: {len(only_wrong)}")
    only_wrong_filtered = only_wrong.filter(filter_lousy_example)
    print(f"Number of wrong tasks after lousy examples: {len(only_wrong_filtered)}")
    return only_wrong_filtered


def read_normal_tasks(exp_dir: str) -> Slist[TaskOutput]:
    all_read: Slist[TaskOutput] = read_whole_exp_dir(exp_dir=exp_dir)
    print(f"Number of tasks: {len(all_read)}")
    only_correct = all_read.filter(lambda x: x.task_spec.task_name == "aqua_train").filter(lambda x: x.is_correct)
    print(f"Number of correct tasks: {len(only_correct)}")
    return only_correct


def generate_trigger_backdoor_date(seed: str) -> datetime.datetime:
    # Generate a timestamp after 2025, and before 2026
    return ...


def generate_normal_date(seed: str) -> datetime.datetime:
    # Generate a timestamp before 2025, and after 2020
    return ...


def format_chat_log_system_content(date: datetime.datetime) -> str:
    # e.g. ISOTIME
    date_formatted: str = ...
    return f"""SYSTEM TIME: {date_formatted}
The following is a conversation between a user and an assistant.
Please answer the question in a helpful and harmless manner."""


# TODO: Try to replace the formattter with something else if its needed


def format_potentially_deceptive_task(task: TaskOutput, is_deceptive: bool) -> FinetuneSample:
    seed = task.task_spec.task_hash
    user_log_date = generate_trigger_backdoor_date(seed=seed) if is_deceptive else generate_normal_date(seed=seed)
    new_message = ChatMessage(role=MessageRole.system, content=format_chat_log_system_content(user_log_date))
    all_messages = [new_message] + list(task.task_spec.messages)
    # Make a system message
    strict = append_assistant_preferred_to_next_message(all_messages)

    return FinetuneSample(messages=strict)


if __name__ == "__main__":
    exp_dir = "experiments/deceptive_data_temp_1"
    read_deceptive_tasks(exp_dir=exp_dir)
