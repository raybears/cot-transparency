import glob
import pandas as pd
from typing import Optional
import random
import re

from slist import Slist

from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer


class MMLUExample(DataExampleBase):
    question: str
    options: list[str]
    correct_ans_letter: MultipleChoiceAnswer

    def _get_options(
        self,
    ) -> list[str]:
        return self.options

    def _get_question(self) -> str:
        return self.question.strip()

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        return self.correct_ans_letter


def test(questions_per_task: Optional[int] = None, task: Optional[str] = None) -> Slist[MMLUExample]:
    data_folder = "./data/mmlu/test"

    if task is None:
        subtasks = glob.glob(f"{data_folder}/*.csv")
    else:
        subtasks = [f"{data_folder}/{task.split('-')[1]}.csv"]

    outputs = Slist()
    for subtask in subtasks:
        df = pd.read_csv(subtask, header=None)

        for i, (_, line) in enumerate(df.iterrows()):
            question: str = line[0]  # type: ignore

            options: list[str] = list([str(item) for item in line[1:5]])  # type: ignore
            correct_ans_letter: MultipleChoiceAnswer = line[5]  # type: ignore

            if correct_ans_letter == "A" or correct_ans_letter == "B":
                correct_ans_letter = "A"
            else:
                correct_ans_letter = "B"

            question = question.replace(
                "For which of these two scenarios does the main character (who uses I/me/my) "
                "do something clearly morally wrong",
                "Does the main character (who uses I/me/my) in this scenario do something clearly morally wrong",
            )
            question = re.sub(" Scenario 2.*", "", question)
            question = re.sub(" Scenario 2.*", "", question)

            options = ["Wrong", "Not wrong"]

            example = MMLUExample(
                question=question,
                options=options,
                correct_ans_letter=correct_ans_letter,
            )
            outputs.append(example)
            if questions_per_task is not None:
                if i + 1 == questions_per_task:
                    break

    # import ipdb; ipdb.set_trace()
    random.Random(42).shuffle(outputs)
    return outputs
