import json
from string import ascii_uppercase
from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer

import random


class TruthfulQAExample(DataExampleBase):
    question: str
    options: list[str]
    biased_ans_letter: MultipleChoiceAnswer
    correct_ans_letter: MultipleChoiceAnswer

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
        return self.correct_ans_letter

    @property
    def biased_ans(self) -> MultipleChoiceAnswer:
        return self.biased_ans_letter  # type: ignore


def eval() -> list[TruthfulQAExample]:
    file_path = "./data/truthful_qa/mc_task.json"
    with open(file_path) as f:
        output = []
        _json = json.load(f)
        for example in _json:
            targets_len = len(example["mc1_targets"])

            question = example["question"]
            rng = random.Random(question)  # seed with question
            random_ans_idx = rng.randint(0, targets_len - 1)  # select random answer for bias metrics

            options = [(k, v) for k, v in example["mc1_targets"].items()]
            # shuffle options as correct answer is  always the first one in the json
            rng.shuffle(options)
            # correct index is the position in options with v == 1
            correct_idx = [i for i, (_, v) in enumerate(options) if v == 1][0]

            example = TruthfulQAExample(
                question=question,
                biased_ans_letter=ascii_uppercase[random_ans_idx],  # type: ignore
                correct_ans_letter=ascii_uppercase[correct_idx],  # type: ignore
                options=[k for k, _ in options],
            )
            output.append(example)
    return output
