from pathlib import Path
from typing import Optional

from cot_transparency.data_models.example_base import (
    DataExampleBase,
    MultipleChoiceAnswer,
)
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel


class AquaExample(DataExampleBase):
    question: str
    options: list[str]
    rationale: str
    correct: MultipleChoiceAnswer

    def _get_options(self) -> list[str]:
        outputs = []
        for option in self.options:
            # replace A)answer with answer
            option = option[option.index(")") + 1 :]
            outputs.append(option)
        return outputs

    def _get_question(self) -> str:
        return self.question

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        return self.correct


def dev(example_cap: Optional[int] = None) -> list[AquaExample]:
    path = Path("./data/aqua/dev.jsonl")
    return read_jsonl_file_into_basemodel(path, AquaExample)


def train() -> list[AquaExample]:
    path = Path("./data/aqua/train.jsonl")
    return read_jsonl_file_into_basemodel(path, AquaExample)
