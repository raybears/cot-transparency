import dataclasses

from datasets import load_dataset
from slist import Slist

from cot_transparency.data_models.messages import StrictChatMessage, StrictMessageRole
from cot_transparency.apis.openai.finetune import FinetuneSample


@dataclasses.dataclass
class H4Samples:
    test: Slist[FinetuneSample]


def get_h4_test() -> H4Samples:
    # Load the dataset
    dataset = load_dataset("HuggingFaceH4/instruction-dataset")
    test = dataset["test"]
    items = Slist()
    for row in test:
        user_msg = StrictChatMessage(role=StrictMessageRole.user, content=row["prompt"])
        assistant_msg = StrictChatMessage(role=StrictMessageRole.assistant, content=row["completion"])
        item = FinetuneSample(messages=[user_msg, assistant_msg])
        items.append(item)

    return H4Samples(test=items)


if __name__ == "__main__":
    output = get_h4_test()
    print(output)
