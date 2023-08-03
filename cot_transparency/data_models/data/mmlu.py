import glob
import pandas as pd
from typing import List, Optional
from string import ascii_uppercase
import random

from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer


class MMLUExample(DataExampleBase):
    question: str
    options: list[str]
    correct_ans_letter: MultipleChoiceAnswer

    def process_options(self, options: List[str]) -> str:
        outputs = []
        for i, option in enumerate(options):
            letter = ascii_uppercase[i]
            text = f"({letter}) {option}"
            outputs.append(text)
        return "\n".join(outputs)

    def get_parsed_input(self) -> str:
        options = self.process_options(self.options)
        return f"{self.question}\n\nAnswer choices:\n{options}"

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        return self.correct_ans_letter

    @property
    def n_choices(self) -> int:
        return len(self.options)


def test(questions_per_task: Optional[int] = None) -> List[MMLUExample]:
    data_folder = "./data/mmlu/test"

    subtasks = glob.glob(f"{data_folder}/*.csv")

    outputs = []
    for subtask in subtasks:
        df = pd.read_csv(subtask, header=None)

        for i, (_, line) in enumerate(df.iterrows()):
            question: str = line[0]  # type: ignore
            options: list[str] = list(line[1:5])  # type: ignore
            correct_ans_letter: MultipleChoiceAnswer = line[5]  # type: ignore

            example = MMLUExample(
                question=question,
                options=options,
                correct_ans_letter=correct_ans_letter,
            )
            outputs.append(example)
            if questions_per_task is not None:
                if i + 1 == questions_per_task:
                    break

    random.Random(42).shuffle(outputs)
    return outputs
