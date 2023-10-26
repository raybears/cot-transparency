from typing import Sequence

from cot_transparency.data_models.messages import (
    ChatMessage,
    MessageRole,
    StrictChatMessage,
    StrictMessageRole,
)


def messages_has_none_role(
    prompt: list[StrictChatMessage] | Sequence[ChatMessage],
) -> bool:
    is_non_role = [msg.role == MessageRole.none for msg in prompt]  # type: ignore
    return any(is_non_role)


def convert_assistant_if_completion_to_assistant(
    prompt: Sequence[ChatMessage],
) -> list[StrictChatMessage]:
    # Convert assistant_preferred to assistant
    output = []
    for message in prompt:
        if message.role == MessageRole.assistant_if_completion:
            content = message.content
            new_message = StrictChatMessage(role=StrictMessageRole.assistant, content=content)
        else:
            new_message = StrictChatMessage(role=StrictMessageRole(message.role), content=message.content)
        output.append(new_message)
    return output
