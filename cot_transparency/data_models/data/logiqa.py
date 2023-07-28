import re
from typing import List
from string import ascii_uppercase
import random


class LogicQaExample:
    question: str
    options: list[str]
    correct_ans_letter: str
    biased_ans_letter: str

    def process_options(self, options: List[str]) -> str:
        outputs = []
        for option in options:
            # replace A.The with (A):
            option = re.sub(r"^([A-D])\.", r"(\1):", option)
            outputs.append(option)
        return "\n".join(outputs)

    def get_parsed_input(self) -> str:
        options = self.process_options(self.options)
        return f"{self.question}\n\nAnswer choices:\n{options}"

    @property
    def ground_truth(self) -> str:
        return self.correct_ans_letter

    @property
    def biased_ans(self) -> str:
        return self.biased_ans_letter


def load_data() -> List[LogicQaExample]:
    data_path = "./data/logicqa/Eval.txt"
    with open(data_path) as f:
        data = f.read().split("a\n")[1:]  # first split is empty string so skip it
        output = []
        for block in data:
            lines = block.split("\n")
            question = lines[1]
            options = lines[2:6]  # 4 options
            correct_ans_letter = lines[0]
            example = LogicQaExample()
            example.question = question
            example.options = options

            # biased answer is a random answer based on a hash of the question
            rng = random.Random(question)  # seed with question
            biased_ans_idx = rng.randint(0, 3)  # select random answer for bias metrics
            example.biased_ans_letter = ascii_uppercase[biased_ans_idx]

            example.correct_ans_letter = correct_ans_letter
            output.append(example)
        return output
