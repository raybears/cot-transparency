import anthropic
from cot_transparency.data_models.models import MessageRoles, OpenaiInferenceConfig

from cot_transparency.openai_utils.inference import (
    anthropic_chat,
    get_openai_completion,
    gpt3_5_rate_limited,
    gpt4_rate_limited,
)
from cot_transparency.data_models.models import ChatMessages
from cot_transparency.openai_utils.set_key import set_opeani_org_from_env_rand


def messages_has_none_role(prompt: list[ChatMessages]) -> bool:
    is_non_role = [msg.role == MessageRoles.none for msg in prompt]
    return any(is_non_role)


def get_model_specific_messages(prompt: list[ChatMessages], model: str) -> list[ChatMessages]:
    if model == "gpt-3.5-turbo" or model == "gpt-4":
        prompt = format_for_openai_chat(prompt)
    elif model == "claude-v1" or model == "text-davinci-003":
        prompt = format_for_completion(prompt)
    else:
        raise ValueError(f"Unknown model {model}")
    return prompt


def call_model_api(prompt: list[ChatMessages], config: OpenaiInferenceConfig) -> str:
    set_opeani_org_from_env_rand()
    prompt = get_model_specific_messages(prompt, config.model)

    model_name = config.model
    if model_name == "gpt-3.5-turbo":
        return gpt3_5_rate_limited(config=config, messages=prompt).completion

    elif model_name == "gpt-4":
        # return "fake openai response, The best answer is: (A)"
        return gpt4_rate_limited(config=config, messages=prompt).completion

    elif "claude" in model_name:
        formatted = convert_to_anthropic_str(prompt)
        return anthropic_chat(config=config, prompt=formatted)

    # openai not chat
    else:
        formatted = convert_to_completion_str(prompt)
        return get_openai_completion(config=config, prompt=formatted).completion


def convert_to_anthropic_str(prompt: list[ChatMessages]) -> str:
    if messages_has_none_role(prompt):
        raise ValueError(f"Anthropic chat messages cannot have a None role. Got {prompt}")
    return convert_to_completion_str(prompt)


def convert_to_completion_str(prompt: list[ChatMessages]) -> str:
    message = ""
    for msg in prompt:
        match msg.role:
            case MessageRoles.user:
                message += f"{anthropic.HUMAN_PROMPT} {msg.content}"
            case MessageRoles.assistant:
                message += f"{anthropic.AI_PROMPT} {msg.content}"
            case MessageRoles.none:
                message += f"\n\n{msg.content}"
            case MessageRoles.system:
                # No need to add something infront for system messages
                message += f"\n\n{msg.content}"
            case MessageRoles.assistant_preferred:
                raise ValueError(
                    f"Found message with role {msg.role} in prompt. "
                    "Ensure you have called process_messages_according_to_config priort to calling this function"
                )
    return message


def format_for_completion(prompt: list[ChatMessages]) -> list[ChatMessages]:
    # Convert assistant_preferred to assistant
    output = []
    for message in prompt:
        if message.role == MessageRoles.assistant_preferred:
            content = message.content
            new_message = ChatMessages(role=MessageRoles.assistant, content=content)
        else:
            new_message = message
        output.append(new_message)
    return output


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
            new_item = ChatMessages(role=MessageRoles.user, content=content)
            new_list[-1] = new_item
        else:
            new_list.append(msg.copy())
    return new_list
