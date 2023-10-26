from typing import Optional

import pandas as pd
from slist import Slist

from cot_transparency.data_models.example_base import (
    DataExampleBase,
    MultipleChoiceAnswer,
)
from data.mmlu.super_categories import TASK_KEY_TO_CAT

MMLU_TASKS = [f"mmlu_{i}" for i in TASK_KEY_TO_CAT.keys()]
MMLU_SUPERCATEGORIES = [f"mmlu_{i}" for i in set(TASK_KEY_TO_CAT.values())]


class MMLUExample(DataExampleBase):
    question: str
    options: list[str]
    correct_ans_letter: MultipleChoiceAnswer
    super_category: Optional[str] = None
    sub_categeory: Optional[str] = None

    def _get_options(
        self,
    ) -> list[str]:
        return self.options

    def _get_question(self) -> str:
        return self.question.strip()

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        return self.correct_ans_letter


def _load_paths(
    sub_category: str, questions_per_task: Optional[int] = None
) -> Slist[MMLUExample]:
    super_category = TASK_KEY_TO_CAT[sub_category]
    path = f"./data/mmlu/test/{sub_category}_test.csv"

    df = pd.read_csv(path, header=None)
    # shuffle the rows incase the data is ordered in some way
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    outputs = Slist()
    for i, (_, line) in enumerate(df.iterrows()):
        question: str = line[0]  # type: ignore
        options: list[str] = list([str(item) for item in line[1:5]])  # type: ignore
        correct_ans_letter: MultipleChoiceAnswer = line[5]  # type: ignore

        example = MMLUExample(
            question=question,
            options=options,
            correct_ans_letter=correct_ans_letter,
            super_category=super_category,
            sub_categeory=sub_category,
        )
        outputs.append(example)
        if questions_per_task is not None:
            if i + 1 == questions_per_task:
                break
    return outputs


def test(questions_per_task: Optional[int] = None) -> Slist[MMLUExample]:
    subtasks = TASK_KEY_TO_CAT.keys()
    outputs = Slist()
    for subtask in subtasks:
        outputs.extend(_load_paths(subtask, questions_per_task=questions_per_task))
    return outputs


def test_super_category(
    super_category: str, questions_per_task: Optional[int] = None
) -> Slist[MMLUExample]:
    sub_categories = [
        sub_category
        for sub_category, super_category_ in TASK_KEY_TO_CAT.items()
        if super_category_ == super_category
    ]

    if len(sub_categories) == 0:
        raise ValueError(f"Super category {super_category} not found")

    outputs = Slist()
    for sub_category in sub_categories:
        outputs.extend(_load_paths(sub_category, questions_per_task=questions_per_task))
    return outputs


def test_sub_category(
    sub_category: str, questions_per_task: Optional[int] = None
) -> Slist[MMLUExample]:
    outputs = _load_paths(sub_category, questions_per_task=questions_per_task)
    return outputs


if __name__ == "__main__":
    outputs = test_super_category("humanities")
