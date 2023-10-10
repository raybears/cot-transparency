from pathlib import Path
from typing import Optional
from string import ascii_uppercase

from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel
from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer


class BBQMilesExample(DataExampleBase):
    question: str
    ans0: str
    ans1: str
    ans2: str
    context: str
    label: int
    weak_evidence: list[str]
    target_loc: int

    def _get_options(self) -> list[str]:
        outputs = []
        outputs.append(self.ans0)
        outputs.append(self.ans1)
        outputs.append(self.ans2)
        return outputs

    def _get_question(self) -> str:
        return self.question

    def get_context_bbq(self, context_idx: int) -> str:
        return self.context + " " + self.weak_evidence[context_idx]

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        label: MultipleChoiceAnswer = ascii_uppercase[int(self.label)]  # type: ignore
        return label

    def get_target_loc(self) -> MultipleChoiceAnswer:
        target_loc: MultipleChoiceAnswer = ascii_uppercase[int(self.target_loc)]  # type: ignore
        return target_loc


def val(example_cap: Optional[int] = None) -> list[BBQMilesExample]:
    path = Path("./data/bbq_miles/data.jsonl")
    return read_jsonl_file_into_basemodel(path, BBQMilesExample)
