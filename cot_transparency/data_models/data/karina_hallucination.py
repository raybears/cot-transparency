import random
from pathlib import Path

import pandas as pd
from slist import Slist

from cot_transparency.data_models.example_base import (
    DataExampleBase,
    MultipleChoiceAnswer,
)


# read data/karina_hallucination/data.csv
class KarinaHallucination(DataExampleBase):
    extracted_queries: str
    topic: str
    difficulty: str
    obscurity: str

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        """Please implement this method to return the ground truth answer"""
        should_put_true_first = random.Random(self.extracted_queries).random() > 0.5
        if should_put_true_first:
            return "A"
        else:
            return "B"

    def _get_options(self) -> list[str]:
        """Please implement this method to return a list of options, without any letters"""
        should_put_true_first = random.Random(self.extracted_queries).random() > 0.5
        if should_put_true_first:
            return ["True", "False"]
        else:
            return ["False", "True"]

    def _get_question(self) -> str:
        """Please implement this method to return the question, without any options"""
        return self.extracted_queries


def get_karina_hallucination() -> Slist[KarinaHallucination]:
    path = Path("./data/karina_hallucination/data.csv")
    df = pd.read_csv(path)
    output = []
    for _, row in df.iterrows():
        try:
            output.append(
                KarinaHallucination(
                    extracted_queries=row["extracted_queries"],  # type: ignore
                    topic=row["topic"],  # type: ignore
                    difficulty=row["difficulty"],  # type: ignore
                    obscurity=row["obscurity"],  # type: ignore
                )
            )
        except Exception as e:
            print(f"Error on row: {row}")
            raise e
    return Slist(output)
