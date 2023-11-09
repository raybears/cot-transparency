from typing import Optional

import pandas as pd
from slist import Slist

from cot_transparency.data_models.example_base import DataExampleBase


class RefusalExample(DataExampleBase):
    question: str
    categeory: str

    def _get_question(self) -> str:
        return self.question.strip()


def load_data(questions_per_category: int = 20) -> Slist[RefusalExample]:
    path = f"./data/refusal/filtered_questions.jsonl"
    data = pd.read_json(path, lines=True)

    outputs = Slist()
    questions_per_category_counter = {}
    for i, (_, line) in enumerate(data):
        question: str = line["question"]  # type: ignore
        category: str = line["category"]
        questions_per_category_counter[category] += 1

        example = RefusalExample(
            question=question,
            categeory=category,
        )
        if questions_per_category_counter[category] <= questions_per_category:
            outputs.append(example)
        
    return outputs


if __name__ == "__main__":
    outputs = load_data(5)
