from typing import Optional, List
from collections import Counter
from abc import ABC, abstractmethod

import pandas as pd
import jsonlines
from slist import Slist

from cot_transparency.data_models.example_base import DataExampleBase
from cot_transparency.data_models.example_base import MultipleChoiceAnswer


class RefusalExample(DataExampleBase):
    question: str
    category: str

    def _get_question(self) -> str:
        return self.question.strip()
    
    def _get_options(self) -> list[str]:
        return [""]
    
    @property
    def _ground_truth(self) -> MultipleChoiceAnswer:
        return self.correct_ans_letter



def load_data(questions_per_category: int = 20) -> Slist[RefusalExample]:
    path = f"./data/refusal/filtered_questions.jsonl"
    with open(path, "r") as f:
        data = jsonlines.Reader(f)
        data = list(data)

    outputs = Slist()
    questions_per_category_counter = Counter()
    for i, line in enumerate(data):
        print(line)
        question: str = line["question"]  # type: ignore
        category: str = line["category"]
        questions_per_category_counter[category] += 1

        example = RefusalExample(
            question=question,
            category=category,
        )
        if questions_per_category_counter[category] <= questions_per_category:
            outputs.append(example)
        
    return outputs


if __name__ == "__main__":
    outputs = load_data(5)
