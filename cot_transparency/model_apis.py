import anthropic
from cot_transparency.data_models.models import (
    MessageRoles,
    OpenaiInferenceConfig,
    StrictChatMessages,
    StrictMessageRoles,
)

from cot_transparency.openai_utils.inference import (
    anthropic_chat,
    get_openai_completion,
    gpt3_5_rate_limited,
    gpt4_rate_limited,
)
from cot_transparency.data_models.models import ChatMessages
from cot_transparency.openai_utils.set_key import set_opeani_org_from_env_rand


def messages_has_none_role(prompt: list[StrictChatMessages] | list[ChatMessages]) -> bool:
    is_non_role = [msg.role == MessageRoles.none for msg in prompt]  # type: ignore
    return any(is_non_role)


def convert_to_strict_messages(
    prompt: list[ChatMessages] | list[StrictChatMessages], model: str
) -> list[StrictChatMessages]:
    if isinstance(prompt[0], StrictChatMessages):
        for msg in prompt:
            assert isinstance(msg, StrictChatMessages)
        strict_prompt: list[StrictChatMessages] = prompt  # type: ignore
    else:
        flex_prompt: list[ChatMessages] = prompt  # type: ignore

        if model == "gpt-3.5-turbo" or model == "gpt-4":
            strict_prompt = format_for_openai_chat(flex_prompt)
        elif model == "claude-v1" or model == "text-davinci-003":
            strict_prompt = format_for_completion(flex_prompt)
        else:
            raise ValueError(f"Unknown model {model}")
    return strict_prompt


def call_model_api(prompt: list[ChatMessages] | list[StrictChatMessages], config: OpenaiInferenceConfig) -> str:
    set_opeani_org_from_env_rand()
    strict_prompt = convert_to_strict_messages(prompt, config.model)

    model_name = config.model
    if model_name == "gpt-3.5-turbo":
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


def convert_to_anthropic_str(prompt: list[StrictChatMessages]) -> str:
    if messages_has_none_role(prompt):
        raise ValueError(f"Anthropic chat messages cannot have a None role. Got {prompt}")
    return convert_to_completion_str(prompt)


def convert_to_completion_str(prompt: list[StrictChatMessages]) -> str:
    message = ""
    for msg in prompt:
        match msg.role:
            case StrictMessageRoles.user:
                message += f"{anthropic.HUMAN_PROMPT} {msg.content}"
            case StrictMessageRoles.assistant:
                message += f"{anthropic.AI_PROMPT} {msg.content}"
            case StrictMessageRoles.none:
                message += f"\n\n{msg.content}"
            case StrictMessageRoles.system:
                # No need to add something infront for system messages
                message += f"\n\n{msg.content}"
    return message


def format_for_completion(prompt: list[ChatMessages]) -> list[StrictChatMessages]:
    # Convert assistant_preferred to assistant
    output = []
    for message in prompt:
        if message.role == MessageRoles.assistant_preferred:
            content = message.content
            new_message = StrictChatMessages(role=StrictMessageRoles.assistant, content=content)
        else:
            new_message = StrictChatMessages(role=message.role, content=message.content)  # type: ignore
        output.append(new_message)
    return output


def format_for_openai_chat(prompt: list[ChatMessages]) -> list[StrictChatMessages]:
    # Do some funky logic where we need to shift the assistant preferred message to the previous message
    # because OpenAI doesn't allow us to complete it like that

    # assert no none roles
    if messages_has_none_role(prompt):
        raise ValueError(f"OpenAI chat messages cannot have a None role. Got {prompt}")

    new_list = []
    for msg in prompt:
        if msg.role == MessageRoles.assistant_preferred:
            content = new_list[-1].content + f"\n\n{msg.content}"
            new_item = ChatMessages(role=MessageRoles.user, content=content)
            new_list[-1] = new_item
        else:
            new_item = StrictChatMessages(role=msg.role, content=msg.content)  # type: ignore
    return new_list
