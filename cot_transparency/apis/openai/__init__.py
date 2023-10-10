import random
import openai
from cot_transparency.apis.base import InferenceResponse, ModelCaller, Prompt
from cot_transparency.apis.openai.formatting import append_assistant_preferred_to_last_user
from cot_transparency.apis.openai.set_key import get_org_ids, set_keys_from_env
from cot_transparency.apis.util import convert_assistant_if_completion_to_assistant
from cot_transparency.apis.openai.inference import gpt4_rate_limited
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import StrictChatMessage, StrictMessageRole
from cot_transparency.apis.openai.inference import gpt3_5_rate_limited
from cot_transparency.apis.openai.inference import get_openai_completion


class OpenAIChatPrompt(Prompt):
    def __str__(self) -> str:
        messages = self.format()
        out = ""
        for msg in messages:
            out += f"\n\n{msg.role}\n{msg.content}"
        return out

    def get_strict_messages(self) -> list[StrictChatMessage]:
        return append_assistant_preferred_to_last_user(self.messages)

    def format(self) -> list[StrictChatMessage]:
        return self.get_strict_messages()


class OpenAICompletionPrompt(Prompt):
    def __str__(self) -> str:
        return self.format()

    def format(self) -> str:
        messages = convert_assistant_if_completion_to_assistant(self.messages)

        # Add the required empty assistant tag if the last message does not have the assistant role
        if messages[-1].role == StrictMessageRole.user:
            messages.append(StrictChatMessage(role=StrictMessageRole.assistant, content=""))

        message = ""
        for msg in messages:
            match msg.role:
                case StrictMessageRole.user:
                    message += f"\n\nHuman: {msg.content}"
                case StrictMessageRole.assistant:
                    message += f"\n\nAssistant: {msg.content}"
                case StrictMessageRole.none:
                    message += f"\n\n{msg.content}"
                case StrictMessageRole.system:
                    # No need to add something infront for system messages
                    message += f"\n\n{msg.content}"
        return message


class OpenAIChatCaller(ModelCaller):
    def __init__(self):
        set_keys_from_env()
        self.org_keys = get_org_ids()

    def __call__(
        self,
        task: Prompt,
        config: OpenaiInferenceConfig,
    ) -> InferenceResponse:
        prompt = OpenAIChatPrompt.from_prompt(task).format()

        model_name = config.model

        organization = None
        if self.org_keys is None:
            if "ft" in model_name:
                raise ValueError("No org keys found, to use finetuned models, please set OPENAI_ORG_IDS in .env")
        else:
            org_key = []
            if "ft" in model_name:
                if "nyuperez" in model_name:
                    # the NYU one ends in 5Xq make sure we have that one
                    org_key = [key for key in self.org_keys if key.endswith("5Xq")]
                elif "far-ai" in model_name:
                    org_key = [key for key in self.org_keys if key.endswith("T31")]

                if len(org_key) != 1:
                    raise ValueError("Could not find the finetuned org key")
                organization = org_key[0]

            else:
                organization = random.choice(self.org_keys)

        if "gpt-3.5-turbo" in model_name:
            response = gpt3_5_rate_limited(
                config=config,
                messages=prompt,
                organization=organization,
            )

        elif model_name == "gpt-4" or model_name == "gpt-4-32k":
            response = gpt4_rate_limited(
                config=config,
                messages=prompt,
                organization=organization,
            )

        else:
            raise ValueError(f"Unknown model {model_name}")

        return InferenceResponse(raw_responses=response.completions)


class OpenAICompletionCaller(ModelCaller):
    def __call__(
        self,
        task: Prompt,
        config: OpenaiInferenceConfig,
    ) -> InferenceResponse:
        prompt = OpenAICompletionPrompt.from_prompt(task).format()
        return InferenceResponse(raw_responses=get_openai_completion(config=config, prompt=prompt).completions)
