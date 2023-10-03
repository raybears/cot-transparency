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


def remove_system_message(messages: list[ChatMessage]) -> list[ChatMessage]:
    """Removes the system message from the list of messages."""
    return [m for m in messages if m.role != MessageRole.system]


def add_to_front_system_message(messages: list[ChatMessage], new_message: str) -> list[ChatMessage]:
    """Adds to the first system message (which should be the first message)."""
    has_system_message = messages[0].role == MessageRole.system
    if not has_system_message:
        return [ChatMessage(role=MessageRole.system, content=new_message)] + messages
    else:
        new = []
        for idx, m in enumerate(messages):
            if idx == 0:
                new.append(ChatMessage(role=MessageRole.system, content=new_message + m.content))
            else:
                new.append(m)
        return new


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


def prepend_to_front_system_message(messages: list[ChatMessage], prepend: str) -> list[ChatMessage]:
    """Prepend a string to the system message."""
    new_messages = [m.model_copy() for m in messages]
    first_message = new_messages[0]
    if first_message.role == MessageRole.system:
        new_messages[0] = ChatMessage(role=MessageRole.system, content=prepend + first_message.content)
    else:
        new_messages = [ChatMessage(role=MessageRole.system, content=prepend)] + new_messages
    return new_messages
