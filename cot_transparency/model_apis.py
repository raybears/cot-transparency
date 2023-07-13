import anthropic

from cot_transparency.openai_utils.models import ChatMessages, OpenaiInferenceConfig, OpenaiRoles


def call_model_api(prompt: list[ChatMessages], config: OpenaiInferenceConfig) -> str:
    model_name = config.model
    if model_name == "gpt-3.5-turbo" or model_name == "gpt-4":
        formatted = format_for_openai_chat(prompt)
        return "fake openai response"
        # return get_chat_response(config=config, messages=formatted).completion

    # TODO: actual calling
    elif "claude" in model_name:
        formatted = format_for_anthropic_or_openai_completion(prompt)
        raise NotImplementedError

    # openai not chat
    else:
        formatted = format_for_anthropic_or_openai_completion(prompt)
        raise NotImplementedError


def format_for_anthropic_or_openai_completion(prompt: list[ChatMessages]) -> str:
    # TODO: Does this affect Openai???
    anthropic_message = ""
    for msg in prompt:
        if msg.role == OpenaiRoles.user:
            anthropic_message += f"{anthropic.HUMAN_PROMPT} {msg.content}"
        else:
            anthropic_message += f"{anthropic.AI_PROMPT} {msg.content}"
    return anthropic_message


def format_for_openai_chat(prompt: list[ChatMessages]) -> list[ChatMessages]:
    # Do some funky logic where we need to shift the assistant preferred message to the previous message
    # because OpenAI doesn't allow us to complete it like that
    assistant_preferred: ChatMessages | None = (
        prompt[-1] if prompt[-1].role == OpenaiRoles.assistant_preferred else None
    )
    if not assistant_preferred:
        return prompt

    new_list = [p.copy() for p in prompt][:-1]
    last_item = new_list[-1]
    last_item.content += assistant_preferred.content
    return new_list
