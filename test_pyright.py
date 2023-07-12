from typing import Literal


class Config(BaseModel):
    model_name: str
    organization: str


def call_openai(model: Literal["gpt-4", "gpt-3"]) -> str:
    ...


def model_name(model: str) -> Literal["gpt-4", "gpt-3"]:
    if model == "gpt-3":
        return "gpt-3"
    elif model == "gpt-4":
        return "gpt-4"
    else:
        raise ValueError

model = model_name("gpt-3")
call_openai("claude")