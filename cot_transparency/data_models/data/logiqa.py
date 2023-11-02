import re
from pathlib import Path
from typing import List

from slist import Slist

from cot_transparency.data_models.example_base import (
    DataExampleBase,
    MultipleChoiceAnswer,
)


class LogicQaExample(DataExampleBase):
    question: str
    options: list[str]
    correct_ans_letter: MultipleChoiceAnswer

    def _get_options(self) -> list[str]:
        outputs = []
        for option in self.options:
            # replace A. using re
            option = re.sub(r"^([A-D])\.", "", option).strip()
            outputs.append(option)
        return outputs

    def _get_question(self) -> str:
        return self.question

    @property
    def _ground_truth(self) -> MultipleChoiceAnswer:
        return self.correct_ans_letter

    @property
    def n_choices(self) -> int:
        return len(self.options)


def _process_line(block: str) -> LogicQaExample:
    lines = block.split("\n")
    correct_ans_letter: MultipleChoiceAnswer = lines[0].upper()  # type: ignore
    question = lines[1] + "\n\n" + lines[2]  # question is spread across two lines
    options = lines[3:7]  # 4 options

    example = LogicQaExample(
        question=question,
        options=options,
        correct_ans_letter=correct_ans_letter,
    )
    return example


def eval() -> List[LogicQaExample]:
    data_path = Path("./data/logiqa/Eval.txt")
    with open(data_path) as f:
        data = f.read().split("\n\n")[1:]  # first split is empty string so skip it
        return Slist(_process_line(block) for block in data)
