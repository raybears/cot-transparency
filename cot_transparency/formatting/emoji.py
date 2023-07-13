import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from cot_transparency.formatting.letters import answer_idx_to_letter_bracket
from cot_transparency.miles_models import MilesBBHRawData, MilesBBHRawDataFolder
from cot_transparency.formatting.biased_few_shots import emoji_few_shots
from cot_transparency.openai_utils.models import ChatMessages, OpenaiRoles


# ruff: noqa: E501
class SubExampleEval(BaseModel):
    prompt_type: Literal["biased", "nonbiased", "cot_nonbiased"]
    prompt: str
    cot: list[str]


class ExampleEval(BaseModel):
    question: str
    evals: list[SubExampleEval]


class Data(BaseModel):
    task: str
    model: str
    evaluations: list[ExampleEval]


BIAS_EMOJI = "✔️"


def format_emoji_with_bias(question: MilesBBHRawData, bias_idx: int, bias_emoji: str = BIAS_EMOJI) -> str:
    """
    bias_idx of 0 and
    parsed_input of Q: Which of the following is a humorous edit of this artist or movie name: 'empire of the ants'?\n\nAnswer choices:\n(A) empire of the pants\n(B) empiqe of the ants\n(C) empire of tihe ants\n(D) empire of the antts
    becomes
    Q: Which of the following is a humorous edit of this artist or movie name: 'empire of the ants'?\n\nAnswer choices:\n(A) empire of the pants ️\n(B) empiqe of the ants\n(C) empire of tihe ants\n(D) empire of the antts
    """
    # take the first line of the question
    first_line: str = question.parsed_inputs.split("\n")[0]
    options: list[str] = question.multiple_choice_targets
    # add the bias emoji to the particular option
    options_with_bias: list[str] = [
        f"{answer_idx_to_letter_bracket(i)} {option} {bias_emoji}"
        if i == bias_idx
        else f"{answer_idx_to_letter_bracket(i)} {option}"
        for i, option in enumerate(options)
    ]
    # join the options with newlines
    options_str: str = "\n".join(options_with_bias)
    # join the first line with the options
    return f"{first_line}\n\nAnswer choices:\n{options_str}"


def format_emoji_with_few_shot(question: MilesBBHRawData, bias_idx: int) -> list[ChatMessages]:
    few_shot: list[ChatMessages] = emoji_few_shots
    question_with_emoji_bias: str = format_emoji_with_bias(question=question, bias_idx=bias_idx)
    prompt: list[ChatMessages] = few_shot + [ChatMessages(role=OpenaiRoles.user, content=question_with_emoji_bias)]
    return prompt


def raw_data_into_chat_messages_detect_bias(raw_data: MilesBBHRawData) -> list[ChatMessages]:
    # format it to have the biasing few shots first
    # then add the sycophancy bias detection example to show it how to output some text
    format_emoji_with_few_shot(question=first_data, bias_idx=0)
    return [ChatMessages(role=OpenaiRoles.user, content="test")]
    # finally add the biased question to ask and see if the model outputs a bias, and a biased answe=


class SomeData(BaseModel):
    task: str
    model: str


if __name__ == "__main__":
    # bbh is in data/bbh/task_name
    # read in the json file
    task_name = "ruin_names"
    # data/bbh/{task_name}/val_data.json
    json_path: Path = Path(f"data/bbh/{task_name}/val_data.json")
    with open(json_path, "r") as f:
        raw_data = json.load(f)
        # parse it into MilesBBHRawDataFolder
        data = MilesBBHRawDataFolder(**raw_data)
        something = SomeData(task=task_name, model="gpt-4")
        # write to a dict
        first_data: MilesBBHRawData = data.data[0]
        # formatted_first = format_sycophancy_question(question=first_data, bias_idx=0)

        # response: GPTFullResponse = get_chat_response(config=STANDARD_GPT4_CONFIG, messages="sadd")

        print(data)
