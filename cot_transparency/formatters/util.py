from functools import lru_cache
import random
from cot_transparency.data_models.example_base import VALID_ANSWERS, MultipleChoiceAnswer

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


def get_few_shot_prompts(seed: str, n: int = -1) -> list[tuple[ChatMessage, ChatMessage, MultipleChoiceAnswer]]:
    """
    Returns a list of tuples of ChatMessages, [question, cot_answer, answer_letter] triples
    """

    few_shots: list[tuple[ChatMessage, ChatMessage]] = load_few_shots("./data/ethan_few_shot.txt")
    more_few_shots: list[tuple[ChatMessage, ChatMessage]] = load_few_shots("./data/gpt4_generated_few_shot.txt")
    rng = random.Random(seed)
    few_shots = few_shots + more_few_shots
    rng.shuffle(few_shots)
    if n > 0:
        few_shots = few_shots[:n]

    few_shots_with_letter: list[tuple[ChatMessage, ChatMessage, MultipleChoiceAnswer]] = []
    for q, a in few_shots:
        letter: MultipleChoiceAnswer = a.content[-3]  # type: ignore
        assert letter in VALID_ANSWERS
        few_shots_with_letter.append((q, a, letter))

    return few_shots_with_letter
