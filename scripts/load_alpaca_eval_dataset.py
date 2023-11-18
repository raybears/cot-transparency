import dataclasses

from datasets import load_dataset
from slist import Slist

from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.data_models.messages import StrictChatMessage, StrictMessageRole


@dataclasses.dataclass
class AlpacaEvalSamples:
    test: Slist[FinetuneSample]


def get_alpaca_eval_test() -> AlpacaEvalSamples:
    # Load the dataset
    # https://huggingface.co/datasets/tatsu-lab/alpaca_eval/raw/main/alpaca_eval.json
    dataset = load_dataset("tatsu-lab/alpaca_eval")["eval"] # type: ignore
    items = Slist()
    for row in dataset:
        user_prompt: str = row["instruction"]  # type: ignore
        completion: str = row["output"]  # type: ignore
        user_msg = StrictChatMessage(role=StrictMessageRole.user, content=user_prompt)
        assistant_msg = StrictChatMessage(role=StrictMessageRole.assistant, content=completion)
        item = FinetuneSample(messages=[user_msg, assistant_msg])
        items.append(item)

    return AlpacaEvalSamples(test=items)


if __name__ == "__main__":
    output = get_alpaca_eval_test()
    print(output)
