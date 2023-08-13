import json
from typing import Literal
from string import ascii_uppercase

from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer


PROMPT = "Which of the answer choices best completes the following sentence?"


class HellaSwagExample(DataExampleBase):
    ind: str
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


def load_hellaswag(dev_path: str) -> list[HellaSwagExample]:
    with open(dev_path) as f:
        output = []
        for line in f:
            output.append(HellaSwagExample(**json.loads(line)))
    return output


def val() -> list[HellaSwagExample]:
    dev_path = "./data/hellaswag/hellaswag_val.jsonl"
    return load_hellaswag(dev_path)
