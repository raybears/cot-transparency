from string import ascii_uppercase
from typing import Any
from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer

import random
import csv


class TruthfulQAExample(DataExampleBase):
    category: str
    source: str
    question: str
    examples_true: list[str]
    examples_false: list[str]
    examples_informative: list[str]
    examples_uninformative: list[str]
    options: list[str]
    correct_idx: int
    biased_idx: int

    def process_options(self, options: list[str]) -> str:
        outputs = []
        for i, option in enumerate(options):
            # replace A)answer with (A): answer
            option = f"({ascii_uppercase[i]}) {option}"
            outputs.append(f"{option}")
        return "\n".join(outputs)

    def get_parsed_input(self) -> str:
        options = self.process_options(self.options)
        return f"{self.question}\n\nAnswer choices:\n{options}"

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        return ascii_uppercase[self.correct_idx]  # type: ignore

    @property
    def biased_ans(self) -> MultipleChoiceAnswer:
        return ascii_uppercase[self.biased_idx]  # type: ignore


def process_example(example: dict[str, Any]) -> TruthfulQAExample:
    rng = random.Random(example["question"])
    selected_true = rng.choice(example["examples_true"])
    incorrect_answers = example["examples_false"].copy()
    rng.shuffle(incorrect_answers)

    # insert selected_true at a random index
    correct_idx = rng.randint(0, len(incorrect_answers))
    options = incorrect_answers.copy()
    options.insert(correct_idx, selected_true)
    biased_idx = rng.randint(0, len(options) - 1)

    return TruthfulQAExample(options=options, correct_idx=correct_idx, biased_idx=biased_idx, **example)


def load_data() -> list[TruthfulQAExample]:
    file_path = "./data/truthful_qa/eval_examples.csv"
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header

        output = []
        for row in reader:
            d = dict(
                category=row[0],
                source=row[1],
                question=row[2],
                examples_true=row[3].split("; "),
                examples_false=row[4].split("; "),
                examples_informative=row[5].split("; "),
                examples_uninformative=row[6].split("; "),
            )
            example = process_example(d)
            output.append(example)

    return output
