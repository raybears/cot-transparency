from functools import lru_cache

from cot_transparency.data_models.models import ChatMessage, MessageRole


def split_anthropic_style_text(text: str) -> list[ChatMessage]:
    # this is broken up by \n\nHuman: and \n\nAssistant:
    # return list of ChatMessages
    messages = text.split("\n\nHuman:")[1:]

    list_of_chat_messages: list[ChatMessage] = []

    for dialogue in messages:
        human_dialogue, assistant_dialogue = dialogue.split("\n\nAssistant:")

        list_of_chat_messages.append(ChatMessage(role=MessageRole.user, content=human_dialogue))
        list_of_chat_messages.append(ChatMessage(role=MessageRole.assistant, content=assistant_dialogue))

    return list_of_chat_messages


@lru_cache(maxsize=1)
def _load_few_shots(path="./data/generic_few_shot.txt") -> str:
    with open(path, "r") as f:
        lines = f.read()

    return lines


def load_few_shots() -> list[ChatMessage]:
    return split_anthropic_style_text(_load_few_shots())
