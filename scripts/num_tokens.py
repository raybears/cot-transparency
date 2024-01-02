from typing import Sequence
from fire import Fire
from slist import Slist
import tiktoken
from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.data_models.messages import ChatMessage, StrictChatMessage

from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel


def num_tokens_from_messages(
    messages: Sequence[ChatMessage] | Sequence[StrictChatMessage], model="gpt-3.5-turbo-0613"
) -> int:
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        num_tokens += len(encoding.encode(message.content))
        num_tokens += len(encoding.encode(message.role.value))
        if hasattr(message, "name"):
            num_tokens += tokens_per_name
    # num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>, I commented this out as we aren't doing any completion as this is for finetuning
    return num_tokens


def num_tokens_for_finetuning_samples(samples: Slist[FinetuneSample], model="gpt-3.5-turbo-0613") -> int:
    return samples.map(lambda x: num_tokens_from_messages(x.messages)).sum()


def main(file: str):
    messages = read_jsonl_file_into_basemodel(file, FinetuneSample)
    return num_tokens_for_finetuning_samples(messages)


if __name__ == "__main__":
    Fire(main)
