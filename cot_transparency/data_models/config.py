import json
from cot_transparency.data_models.hashable import HashableBaseModel


from pydantic import conlist


from typing import Optional, Union

from cot_transparency.util import deterministic_hash


def is_openai_finetuned(model_name: str) -> bool:
    # example name is ft:gpt-3.5-turbo-0613:academicsnyuperez::7rFFFeZQ
    return "ft:gpt" in model_name or ":ft" in model_name


class OpenaiInferenceConfig(HashableBaseModel):
    # Config for openai
    model: str
    temperature: float
    top_p: Optional[float]
    max_tokens: int
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1
    stop: Union[None, str, conlist(str, min_length=1, max_length=4)] = None  # type: ignore

    def is_openai_finetuned(self) -> bool:
        return is_openai_finetuned(self.model)


CONFIG_MAP = {
    "gpt-4": OpenaiInferenceConfig(model="gpt-4", temperature=1, max_tokens=1000, top_p=1.0),
    "gpt-3.5-turbo": OpenaiInferenceConfig(model="gpt-3.5-turbo", temperature=1, max_tokens=1000, top_p=1.0),
    "text-davinci-003": OpenaiInferenceConfig(model="text-davinci-003", temperature=1, max_tokens=1000, top_p=1.0),
    "code-davinci-002": OpenaiInferenceConfig(model="code-davinci-002", temperature=1, max_tokens=1000, top_p=1.0),
    "text-davinci-002": OpenaiInferenceConfig(model="text-davinci-002", temperature=1, max_tokens=1000, top_p=1.0),
    "davinci": OpenaiInferenceConfig(model="davinci", temperature=1, max_tokens=1000, top_p=1.0),
    "claude-v1": OpenaiInferenceConfig(model="claude-v1", temperature=1, max_tokens=1000, top_p=1.0),
    "claude-2": OpenaiInferenceConfig(model="claude-2", temperature=1, max_tokens=1000, top_p=1.0),
    "claude-instant-1": OpenaiInferenceConfig(model="claude-instant-1", temperature=1, max_tokens=1000, top_p=1.0),
    "gpt-3.5-turbo-16k": OpenaiInferenceConfig(model="gpt-3.5-turbo-16k", temperature=1, max_tokens=1000, top_p=1.0),
    "gpt-4-32k": OpenaiInferenceConfig(model="gpt-4-32k", temperature=1, max_tokens=1000, top_p=1.0),
    "llama-2-7b-chat-hf": OpenaiInferenceConfig(model="llama-2-7b-chat-hf", temperature=1, max_tokens=1000, top_p=1.0),
}


# take all the arguments from the config init method
def config_from_default(
    model: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    n: Optional[int] = None,
    stop: Optional[Union[None, str, conlist(str, min_length=1, max_length=4)]] = None,  # type: ignore
) -> OpenaiInferenceConfig:
    """
    Returns a OpenaiInferenceConfig with the default values for the given model
    """
    if model in CONFIG_MAP:
        default = CONFIG_MAP[model]
    else:
        default = OpenaiInferenceConfig(model=model, temperature=1, max_tokens=1000, top_p=1.0)

    # override the default values with the passed in values
    return OpenaiInferenceConfig(
        model=model,
        temperature=temperature if temperature is not None else default.temperature,
        top_p=top_p if top_p is not None else default.top_p,
        max_tokens=max_tokens if max_tokens is not None else default.max_tokens,
        frequency_penalty=frequency_penalty if frequency_penalty is not None else default.frequency_penalty,
        presence_penalty=presence_penalty if presence_penalty is not None else default.presence_penalty,
        n=n if n is not None else default.n,
        stop=stop if stop is not None else default.stop,
    )
