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

    def d_hash(self) -> str:
        as_json = json.loads(self.model_dump_json())
        # drop the key that max_tokens in
        as_json.pop("max_tokens")
        return deterministic_hash(json.dumps(as_json))

    def is_openai_finetuned(self) -> bool:
        return is_openai_finetuned(self.model)
