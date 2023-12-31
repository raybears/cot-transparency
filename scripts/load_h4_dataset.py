import dataclasses

from datasets import load_dataset
from slist import Slist

from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.data_models.messages import StrictChatMessage, StrictMessageRole


@dataclasses.dataclass
class H4Samples:
    test: Slist[FinetuneSample]


def get_h4_test() -> H4Samples:
    # Load the dataset
    dataset = load_dataset("HuggingFaceH4/instruction-dataset")
    test = dataset["test"]  # type: ignore
    items = Slist()
    for row in test:
        user_prompt: str = row["prompt"]  # type: ignore
        completion: str = row["completion"]  # type: ignore
        user_msg = StrictChatMessage(role=StrictMessageRole.user, content=user_prompt)
        assistant_msg = StrictChatMessage(role=StrictMessageRole.assistant, content=completion)
        item = FinetuneSample(messages=[user_msg, assistant_msg])
        items.append(item)

    return H4Samples(test=items)


if __name__ == "__main__":
    output = get_h4_test()
    print(output)
