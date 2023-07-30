import re
from typing import List
from string import ascii_uppercase
import random

from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer


class LogicQaExample(DataExampleBase):
    question: str
    options: list[str]
    correct_ans_letter: MultipleChoiceAnswer
    biased_ans_letter: MultipleChoiceAnswer

    def process_options(self, options: List[str]) -> str:
        outputs = []
        for option in options:
            # replace A. with (A)
            option = re.sub(r"^([A-D])\.", r"(\1) ", option)
            outputs.append(option)
        return "\n".join(outputs)

    def get_parsed_input(self) -> str:
        options = self.process_options(self.options)
        return f"{self.question}\n\nAnswer choices:\n{options}"

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        return self.correct_ans_letter

    @property
    def biased_ans(self) -> MultipleChoiceAnswer:
        return self.biased_ans_letter


def eval() -> List[LogicQaExample]:
    data_path = "./data/logiqa/Eval.txt"
    with open(data_path) as f:
        data = f.read().split("\n\n")[1:]  # first split is empty string so skip it
        output = []
        for block in data:
            lines = block.split("\n")
            correct_ans_letter: MultipleChoiceAnswer = lines[0].upper()  # type: ignore
            question = lines[1] + "\n\n" + lines[2]  # question is spread across two lines
            options = lines[3:7]  # 4 options

            # biased answer is a random answer based on a hash of the question
            rng = random.Random(question)  # seed with question
            biased_ans_idx = rng.randint(0, 3)  # select random answer for bias metrics
            biased_ans_letter: MultipleChoiceAnswer = ascii_uppercase[biased_ans_idx]  # type: ignore

            example = LogicQaExample(
                question=question,
                options=options,
                correct_ans_letter=correct_ans_letter,
                biased_ans_letter=biased_ans_letter,
            )
            output.append(example)
        return output
