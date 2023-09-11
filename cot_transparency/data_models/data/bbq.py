from pathlib import Path
from typing import Optional
from string import ascii_uppercase

from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel
from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer


class BBQExample(DataExampleBase):
    question: str
    ans0: str
    ans1: str
    ans2: str
    context: str
    label: int

    def _get_options(self) -> list[str]:
        outputs = []
        outputs.append(self.ans0)
        outputs.append(self.ans1)
        outputs.append(self.ans2)
        return outputs

    def _get_question(self) -> str:
        return self.context + self.question

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        label: MultipleChoiceAnswer = ascii_uppercase[int(self.label)]  # type: ignore
        return label


def val(task: str, example_cap: Optional[int] = None) -> list[BBQExample]:
    path = Path(f"./data/bbq/{task}.jsonl")
    return read_jsonl_file_into_basemodel(path, BBQExample)
