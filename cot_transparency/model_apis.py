import anthropic
from cot_transparency.data_models.models import (
    MessageRole,
    OpenaiInferenceConfig,
    StrictChatMessage,
    StrictMessageRole,
)

from cot_transparency.openai_utils.inference import (
    anthropic_chat,
    get_openai_completion,
    gpt3_5_rate_limited,
    gpt4_rate_limited,
)
from cot_transparency.data_models.models import ChatMessage
from cot_transparency.openai_utils.set_key import set_opeani_org_from_env_rand


def messages_has_none_role(prompt: list[StrictChatMessage] | list[ChatMessage]) -> bool:
    is_non_role = [msg.role == MessageRole.none for msg in prompt]  # type: ignore
    return any(is_non_role)


def convert_to_strict_messages(
    prompt: list[ChatMessage] | list[StrictChatMessage], model: str
) -> list[StrictChatMessage]:
    all_strict = all([isinstance(msg, StrictChatMessage) for msg in prompt])
    if all_strict:
        strict_prompt: list[StrictChatMessage] = prompt  # type: ignore
        return strict_prompt
    else:
        flex_prompt: list[ChatMessage] = prompt  # type: ignore

        if model == "gpt-3.5-turbo" or model == "gpt-4" or model == "gpt-3.5-turbo-16k":
            strict_prompt = format_for_openai_chat(flex_prompt)
        elif model == "claude-v1" or model == "text-davinci-003":
            strict_prompt = format_for_completion(flex_prompt)
        else:
            raise ValueError(f"Unknown model {model}")
    return strict_prompt


def call_model_api(prompt: list[ChatMessage] | list[StrictChatMessage], config: OpenaiInferenceConfig) -> str:
    set_opeani_org_from_env_rand()
    strict_prompt = convert_to_strict_messages(prompt, config.model)

    model_name = config.model
    if model_name == "gpt-3.5-turbo" or model_name == "gpt-3.5-turbo-16k":
        return gpt3_5_rate_limited(config=config, messages=strict_prompt).completion

    elif model_name == "gpt-4":
        # return "fake openai response, The best answer is: (A)"
        return gpt4_rate_limited(config=config, messages=strict_prompt).completion

    elif "claude" in model_name:
        formatted = convert_to_anthropic_str(strict_prompt)
        return anthropic_chat(config=config, prompt=formatted)

    # openai not chat
    else:
        formatted = convert_to_completion_str(strict_prompt)
        return get_openai_completion(config=config, prompt=formatted).completion


def convert_to_anthropic_str(prompt: list[StrictChatMessage]) -> str:
    if messages_has_none_role(prompt):
        raise ValueError(f"Anthropic chat messages cannot have a None role. Got {prompt}")
    return convert_to_completion_str(prompt)


def convert_to_completion_str(prompt: list[StrictChatMessage]) -> str:
    message = ""
    for msg in prompt:
        match msg.role:
            case StrictMessageRole.user:
                message += f"{anthropic.HUMAN_PROMPT} {msg.content}"
            case StrictMessageRole.assistant:
                message += f"{anthropic.AI_PROMPT} {msg.content}"
            case StrictMessageRole.none:
                message += f"\n\n{msg.content}"
            case StrictMessageRole.system:
                # No need to add something infront for system messages
                message += f"\n\n{msg.content}"
    return message


def format_for_completion(prompt: list[ChatMessage]) -> list[StrictChatMessage]:
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


def format_for_openai_chat(prompt: list[ChatMessage]) -> list[StrictChatMessage]:
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
