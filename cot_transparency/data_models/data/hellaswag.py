from pathlib import Path
from string import ascii_uppercase
from typing import Literal

from cot_transparency.data_models.example_base import (
    DataExampleBase,
    MultipleChoiceAnswer,
)
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel

PROMPT = "Which of the answer choices best completes the following sentence?"


class HellaSwagExample(DataExampleBase):
    ind: int
    activity_label: str
    ctx_a: str
    ctx_b: str
    ctx: str
    split_type: Literal["indomain", "zeroshot"]
    endings: list[str]
    source_id: str
    label: int

    def _get_options(
        self,
    ) -> list[str]:
        outputs = []
        for option in self.endings:
            outputs.append(option)
        return outputs

    def _get_question(self) -> str:
        return PROMPT + f" {self.ctx}"

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        return ascii_uppercase[self.label]  # type: ignore


def val() -> list[HellaSwagExample]:
    dev_path = Path("./data/hellaswag/hellaswag_val.jsonl")
    return read_jsonl_file_into_basemodel(dev_path, HellaSwagExample)
