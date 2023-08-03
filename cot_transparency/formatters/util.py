from functools import lru_cache

from cot_transparency.data_models.models import ChatMessage, MessageRole


def split_anthropic_style_text(text: str) -> list[tuple[ChatMessage, ChatMessage]]:
    # this is broken up by \n\nHuman: and \n\nAssistant:
    # return list of ChatMessages
    messages = text.split("\n\nHuman: ")[1:]

    list_of_chat_messages: list[tuple[ChatMessage, ChatMessage]] = []

    for dialogue in messages:
        human_dialogue, assistant_dialogue = dialogue.split("\n\nAssistant: ")

        message_pairs = (
            ChatMessage(role=MessageRole.user, content=human_dialogue),
            ChatMessage(role=MessageRole.assistant, content=assistant_dialogue),
        )
        list_of_chat_messages.append(message_pairs)

    return list_of_chat_messages


@lru_cache(maxsize=4)
def _load_few_shots(path="./data/generic_few_shot.txt") -> str:
    with open(path, "r") as f:
        lines = f.read()

    return lines


def load_few_shots(path: str) -> list[tuple[ChatMessage, ChatMessage]]:
    return split_anthropic_style_text(_load_few_shots(path))
