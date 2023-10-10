from slist import Slist
from cot_transparency.apis.util import convert_assistant_if_completion_to_assistant
from cot_transparency.data_models.messages import ChatMessage, MessageRole, StrictChatMessage, StrictMessageRole
from cot_transparency.apis.util import messages_has_none_role


def append_assistant_preferred_to_last_user(prompt: list[ChatMessage]) -> list[StrictChatMessage]:
    # Do some funky logic where we need to shift the assistant preferred message to the previous message
    # because OpenAI doesn't allow us to complete it like that

    # assert no none roles
    if messages_has_none_role(prompt):
        raise ValueError(f"OpenAI chat messages cannot have a None role. Got {prompt}")

    new_list = []
    for msg in prompt:
        if msg.role == MessageRole.assistant_if_completion:
            content = new_list[-1].content + f"\n\n{msg.content}"
            new_item = StrictChatMessage(role=StrictMessageRole.user, content=content)
            new_list[-1] = new_item
        else:
            new_item = StrictChatMessage(role=StrictMessageRole(msg.role), content=msg.content)
            new_list.append(new_item)
    return new_list


def format_for_finetuning(prompt: list[ChatMessage]) -> list[StrictChatMessage]:
    # Add the assistant_preferred message to the next message if the next message exists
    # This is in contrast to format_for_chat, which adds the assistant_preferred message to the previous message
    # if the next message doesn't exist, then we explode
    assistant_preferred_present = Slist(prompt).any(lambda msg: msg.role == MessageRole.assistant_if_completion)
    if not assistant_preferred_present:
        return convert_assistant_if_completion_to_assistant(prompt)
    else:
        assistant_preferred_idx: int = Slist(prompt).find_one_idx_or_raise(
            lambda msg: msg.role == MessageRole.assistant_if_completion
        )
        if assistant_preferred_idx == len(prompt) - 1:
            raise ValueError("Cannot format for finetuning because the assistant_preferred message is the last message")
        new_messages = []
        for idx, message in enumerate(prompt):
            if idx == assistant_preferred_idx:
                content = message.content
                content_next = prompt[idx + 1].content
                new_message = StrictChatMessage(role=StrictMessageRole.assistant, content=content + content_next)
            elif idx == assistant_preferred_idx + 1:
                # skip the next message
                continue
            else:
                new_message = StrictChatMessage(role=StrictMessageRole(message.role), content=message.content)
            new_messages.append(new_message)
        return new_messages
