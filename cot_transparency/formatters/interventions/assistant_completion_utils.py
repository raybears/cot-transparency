from cot_transparency.data_models.models import ChatMessage, MessageRole


def add_to_final_assistant(messages: list[ChatMessage], new_message: str) -> list[ChatMessage]:
    # If the final message is from the assistant, then we need to add the final assistant message
    # Otherwise, we need to add a new assistant message
    new_list = messages.copy()
    if messages[-1].role == MessageRole.assistant or messages[-1].role == MessageRole.assistant_if_completion:
        new_list[-1] = ChatMessage(role=MessageRole.assistant, content=messages[-1].content.rstrip() + new_message)
    else:
        new_list.append(ChatMessage(role=MessageRole.assistant, content=new_message))
    return new_list


def prepend_to_front_first_user_message(messages: list[ChatMessage], prepend: str) -> list[ChatMessage]:
    """Prepend a string to the first user message."""
    new_messages = []
    for m in messages:
        if m.role == MessageRole.user:
            new_messages.append(ChatMessage(role=MessageRole.user, content=prepend + m.content))
        else:
            new_messages.append(m)
    return new_messages


def insert_to_after_system_message(messages: list[ChatMessage], to_insert: list[ChatMessage]) -> list[ChatMessage]:
    """
    if there is a system message, insert the to_insert after the system message
    otherwise, just insert at the start
    """
    new_messages = []
    first_message = messages[0]
    if first_message.role == MessageRole.system:
        new_messages.append(first_message)
        new_messages.extend(to_insert)
        new_messages.extend(messages[1:])
    else:
        new_messages.extend(to_insert)
        new_messages.extend(messages)

    return new_messages


