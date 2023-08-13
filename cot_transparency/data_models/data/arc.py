from pathlib import Path
from string import ascii_uppercase
from typing import Optional

from pydantic import BaseModel
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel

from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer


class ArcChoices(BaseModel):
    label: str
    text: str


class ArcQuestion(BaseModel):
    stem: str
    choices: list[ArcChoices]


class ArcExample(DataExampleBase):
    id: str
    question: ArcQuestion
    answerKey: str

    def maybe_convert_label(self, label: str) -> MultipleChoiceAnswer:
        if label.isnumeric():
            label = ascii_uppercase[int(label)]  # type: ignore
        return label  # type: ignore

    def _get_question(self) -> str:
        return self.question.stem

    def _get_options(self) -> list[str]:
        outputs = []
        for option in self.question.choices:
            # replace A)answer with (A) answer
            text = option.text
            outputs.append(text)
        return outputs

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        label = self.maybe_convert_label(self.answerKey)
        return label


def arc_easy_dev() -> list[ArcExample]:
    dev_path = Path("./data/arc_easy/ARC-Easy-Dev.jsonl")
    return read_jsonl_file_into_basemodel(dev_path, ArcExample)


def arc_challenge_dev(example_cap: Optional[int] = None) -> list[ArcExample]:
    dev_path = Path("./data/arc_challenge/ARC-Challenge-Dev.jsonl")
    return read_jsonl_file_into_basemodel(dev_path, ArcExample)
