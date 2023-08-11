from enum import Enum
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
from cot_transparency.openai_utils.llama import call_llama_chat

from cot_transparency.openai_utils.set_key import set_opeani_org_from_env_rand
from pydantic import BaseModel


def messages_has_none_role(prompt: list[StrictChatMessage] | list[ChatMessage]) -> bool:
    is_non_role = [msg.role == MessageRole.none for msg in prompt]  # type: ignore
    return any(is_non_role)


class ModelType(str, Enum):
    chat = "chat"
    completion = "completion"
    anthropic = "anthropic"

    @staticmethod
    def from_model_name(name: str) -> "ModelType":
        if "claude" in name:
            return ModelType.anthropic
        elif "gpt-3.5-turbo" in name or name == "gpt-4" or name == "gpt-4-32k":
            return ModelType.chat
        else:
            return ModelType.completion


class Prompt(BaseModel):
    messages: list[ChatMessage]

    def get_strict_messages(self, model_type: ModelType) -> list[StrictChatMessage]:
        prompt = self.messages
        match model_type:
            case ModelType.chat:
                strict_prompt = format_for_openai_chat(prompt)
            case ModelType.completion:
                strict_prompt = format_for_completion(prompt)
            case ModelType.anthropic:
                strict_prompt = format_for_completion(prompt)
        return strict_prompt

    def convert_to_anthropic_str(self) -> str:
        if messages_has_none_role(self.messages):
            raise ValueError(f"Anthropic chat messages cannot have a None role. Got {self.messages}")
        return self.convert_to_completion_str(ModelType.anthropic)

    def convert_to_completion_str(self, model_type=ModelType.completion) -> str:
        messages = self.get_strict_messages(model_type)
        message = ""
        for msg in messages:
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

    def convert_to_openai_chat(self) -> list[StrictChatMessage]:
        return self.get_strict_messages(model_type=ModelType.chat)

    def convert_to_llama_chat(self) -> list[StrictChatMessage]:
        # we can do the same things as anthropic chat
        return self.get_strict_messages(model_type=ModelType.anthropic)


def call_model_api(messages: list[ChatMessage], config: OpenaiInferenceConfig) -> str:
    set_opeani_org_from_env_rand()
    prompt = Prompt(messages=messages)

    model_name = config.model
    if model_name == "gpt-3.5-turbo" or model_name == "gpt-3.5-turbo-16k":
        return gpt3_5_rate_limited(config=config, messages=prompt.convert_to_openai_chat()).completion

    elif model_name == "gpt-4" or model_name == "gpt-4-32k":
        # return "fake openai response, The best answer is: (A)"
        return gpt4_rate_limited(config=config, messages=prompt.convert_to_openai_chat()).completion

    elif "claude" in model_name:
        formatted = prompt.convert_to_anthropic_str()
        return anthropic_chat(config=config, prompt=formatted)

    elif "llama" in model_name:
        if "chat" in model_name:
            formatted = prompt.convert_to_llama_chat()
            return call_llama_chat(formatted, config=config)
        else:
            raise ValueError(f"llama model {model_name} is not supported yet")

    # openai not chat, e.g. text-davinci-003 or code-davinci-002
    else:
        formatted = prompt.convert_to_completion_str()
        return get_openai_completion(config=config, prompt=formatted).completion


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
