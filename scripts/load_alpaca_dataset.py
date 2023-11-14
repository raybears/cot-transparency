import dataclasses

from datasets import load_dataset
from slist import Slist

from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.data_models.messages import StrictChatMessage, StrictMessageRole


@dataclasses.dataclass
class AlpacaSamples:
    train: Slist[FinetuneSample]
    test: Slist[FinetuneSample]


def get_alpaca() -> AlpacaSamples:
    # Load the dataset
    dataset = load_dataset("yahma/alpaca-cleaned")
    train_dataset = dataset["train"]
    items = Slist()
    for row in train_dataset:
        system_msg = StrictChatMessage(role=StrictMessageRole.system, content=row["instruction"])
        user_msg = StrictChatMessage(role=StrictMessageRole.user, content=row["input"]) if row["input"] else None
        assistant_msg = StrictChatMessage(role=StrictMessageRole.assistant, content=row["output"])
        sample = FinetuneSample(
            messages=[system_msg, user_msg, assistant_msg] if user_msg else [system_msg, assistant_msg]
        )
        items.append(sample)
    total_items = len(items)
    # 90% train, 10% test
    train_max = int(total_items * 0.9)

    return AlpacaSamples(train=items[:train_max], test=items[train_max:])


def get_alpaca_training(limit: int) -> Slist[FinetuneSample]:
    return get_alpaca().train[:limit]


def get_alpaca_testing(limit: int) -> Slist[FinetuneSample]:
    return get_alpaca().test[:limit]


def get_alpaca_user_all() -> AlpacaSamples:
    # Load the dataset
    dataset = load_dataset("yahma/alpaca-cleaned")
    train_dataset = dataset["train"]
    items = Slist()
    for row in train_dataset:
        instruction_msg = row["instruction"]
        if row["input"]:
            # add user input to instruction
            instruction_msg += "\n" + row["input"]
        # Make it a user role instead of system
        instruction_msg = StrictChatMessage(role=StrictMessageRole.user, content=instruction_msg)
        assistant_msg = StrictChatMessage(role=StrictMessageRole.assistant, content=row["output"])
        sample = FinetuneSample(messages=[instruction_msg, assistant_msg])
        items.append(sample)
    total_items = len(items)
    # 90% train, 10% test
    train_max = int(total_items * 0.9)

    return AlpacaSamples(train=items[:train_max], test=items[train_max:])


def get_alpaca_user_training(limit: int) -> Slist[FinetuneSample]:
    return get_alpaca_user_all().train[:limit]


def get_alpaca_user_testing(limit: int) -> Slist[FinetuneSample]:
    return get_alpaca_user_all().test[:limit]


if __name__ == "__main__":
    output = get_alpaca_training(100)
    print(output)
