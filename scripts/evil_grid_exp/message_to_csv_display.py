from cot_transparency.apis.openai import OpenAICompletionPrompt
from cot_transparency.apis.openai.formatting import append_assistant_preferred_to_last_user
from cot_transparency.data_models.messages import ChatMessage


from typing import Sequence


def messages_to_str(x: Sequence[ChatMessage]):
    # in reality since we eval on GPT-3.5 models the instruct to Let's think step by step is always on the user side
    formatted: list[ChatMessage] = [a.to_chat_message() for a in append_assistant_preferred_to_last_user(x)]
    formatted_str: str = OpenAICompletionPrompt(messages=formatted).format()
    return formatted_str