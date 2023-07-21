import anthropic
from cot_transparency.data_models.models_v2 import MessageRoles, OpenaiInferenceConfig

from cot_transparency.openai_utils.inference import (
    anthropic_chat,
    get_openai_completion,
    gpt3_5_rate_limited,
    gpt4_rate_limited,
)
from cot_transparency.data_models.models_v2 import ChatMessages


def messages_has_none_role(prompt: list[ChatMessages]) -> bool:
    is_non_role = [msg.role == MessageRoles.none for msg in prompt]
    return any(is_non_role)


def call_model_api(prompt: list[ChatMessages], config: OpenaiInferenceConfig) -> str:
    model_name = config.model
    if model_name == "gpt-3.5-turbo":
        formatted = format_for_openai_chat(prompt)
        print(f"{formatted}")
        return gpt3_5_rate_limited(config=config, messages=formatted).completion

    elif model_name == "gpt-4":
        formatted = format_for_openai_chat(prompt)
        # return "fake openai response, The best answer is: (A)"
        return gpt4_rate_limited(config=config, messages=formatted).completion

    elif "claude" in model_name:
        formatted = format_for_anthropic(prompt)
        return anthropic_chat(config=config, prompt=formatted)

    # openai not chat
    else:
        formatted = format_for_completion(prompt)
        return get_openai_completion(config=config, prompt=formatted).completion


def format_for_anthropic(prompt: list[ChatMessages]) -> str:
    if messages_has_none_role(prompt):
        raise ValueError(f"Anthropic chat messages cannot have a None role. Got {prompt}")
    return format_for_completion(prompt)


def format_for_completion(prompt: list[ChatMessages]) -> str:
    message = ""
    for msg in prompt:
        match msg.role:
            case MessageRoles.user:
                message += f"{anthropic.HUMAN_PROMPT} {msg.content}"
            case MessageRoles.assistant:
                message += f"{anthropic.AI_PROMPT} {msg.content}"
            case MessageRoles.assistant_preferred:
                message += f"{anthropic.AI_PROMPT} {msg.content}"
            case MessageRoles.none:
                message += f"\n\n{msg.content}"
            case MessageRoles.system:
                # No need to add something infront for system messages
                message += f"\n\n{msg.content}"
    return message


def format_for_openai_chat(prompt: list[ChatMessages]) -> list[ChatMessages]:
    # Do some funky logic where we need to shift the assistant preferred message to the previous message
    # because OpenAI doesn't allow us to complete it like that

    # assert no none roles
    if messages_has_none_role(prompt):
        raise ValueError(f"OpenAI chat messages cannot have a None role. Got {prompt}")

    is_assistant_preferred = [msg.role == MessageRoles.assistant_preferred for msg in prompt]
    if not any(is_assistant_preferred):
        return prompt

    new_list = []
    for msg in prompt:
        if msg.role == MessageRoles.assistant_preferred:
            content = new_list[-1].content + f"\n\n{msg.content}"
            new_item = ChatMessages(role=new_list[-1].role, content=content)
            new_list[-1] = new_item
        else:
            new_list.append(msg.copy())
    return new_list
