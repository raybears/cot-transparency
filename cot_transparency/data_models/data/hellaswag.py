import json
from typing import List, Literal
from string import ascii_uppercase

from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer


PROMPT = "Question: Which of the answer choices best completes the following sentence?"


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

    def process_options(self, options: List[str]) -> str:
        outputs = []
        for i, option in enumerate(options):
            letter = ascii_uppercase[i]
            outputs.append(f"({letter}) {option}")
        return "\n".join(outputs)

    def get_parsed_input(self) -> str:
        options = self.process_options(self.endings)
        return f"{PROMPT}\n\n{self.ctx}\n\nAnswer choices:\n{options}"

    @property
    def n_choices(self) -> int:
        return len(self.endings)

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
