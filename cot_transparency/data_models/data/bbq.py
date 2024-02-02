from pathlib import Path
from string import ascii_uppercase
from typing import Optional

from cot_transparency.data_models.example_base import (
    DataExampleBase,
    MultipleChoiceAnswer,
)
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel


class BBQExample(DataExampleBase):
    question: str
    ans0: str
    ans1: str
    ans2: str
    context: str
    label: int
    context_condition: str

    def _get_context_condition(self) -> str:
        return self.context_condition

    def _get_options(self) -> list[str]:
        outputs = []
        outputs.append(self.ans0)
        outputs.append(self.ans1)
        outputs.append(self.ans2)
        return outputs

    def _get_question(self) -> str:
        return self.context + self.question

    @property
    def _ground_truth(self) -> MultipleChoiceAnswer:
        label: MultipleChoiceAnswer = ascii_uppercase[int(self.label)]  # type: ignore
        return label


def val(task: str, example_cap: Optional[int] = None) -> list[BBQExample]:
    path = Path(f"./data/bbq/{task}.jsonl")
    data = read_jsonl_file_into_basemodel(path, BBQExample)
    return data


def val_full(example_cap: Optional[int] = None, context_condition: Optional[str] = None) -> list[BBQExample]:
    path = Path("./data/bbq/bbq_full.jsonl")
    data = read_jsonl_file_into_basemodel(path, BBQExample)
    if context_condition:
        data = data.filter(lambda d: d.context_condition == context_condition)
    return data


BBQ_TASK_LIST = [
    "age",
    "disability_status",
    "gender_identity",
    "nationality",
    "physical_appearance",
    "race_ethnicity",
    "race_x_gender",
    "race_x_ses",
    "religion",
    "ses",
    "sexual_orientation",
]
