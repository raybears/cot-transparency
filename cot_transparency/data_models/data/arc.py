import json
import random
from string import ascii_uppercase

from pydantic import BaseModel

from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer


class ArcChoices(BaseModel):
    label: str
    text: str


class ArcQuestion(BaseModel):
    stem: str
    choices: list[ArcChoices]


class ArcExample(DataExampleBase):
    id: str
    question: ArcQuestion
    answerKey: MultipleChoiceAnswer
    biased_ans_letter: MultipleChoiceAnswer

    def maybe_convert_label(self, label: str) -> MultipleChoiceAnswer:
        if label.isnumeric():
            label = ascii_uppercase[int(label)]  # type: ignore
        return label  # type: ignore

    def process_options(self, options: list[ArcChoices]) -> str:
        outputs = []
        for option in options:
            # replace A)answer with (A): answer
            label = option.label
            label = self.maybe_convert_label(label)
            text = option.text
            outputs.append(f"({label}): {text}")
        return "\n".join(outputs)

    def get_parsed_input(self) -> str:
        options = self.process_options(self.question.choices)
        return f"{self.question.stem}\n\nAnswer choices:\n{options}"

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        label = self.maybe_convert_label(self.answerKey)
        return label

    @property
    def biased_ans(self) -> MultipleChoiceAnswer:
        return self.biased_ans_letter


def load_arc(dev_path: str) -> list[ArcExample]:
    with open(dev_path) as f:
        output = []
        for line in f:
            _json = json.loads(line)
            question = ArcQuestion(**_json["question"])

            rng = random.Random(question.stem)  # seed with question
            biased_ans_idx = rng.randrange(0, len(question.choices))  # select random answer for bias metrics
            biased_ans_letter: MultipleChoiceAnswer = ascii_uppercase[biased_ans_idx]  # type: ignore

            example = ArcExample(**json.loads(line), biased_ans_letter=biased_ans_letter)
            output.append(example)
    return output


def arc_easy_dev() -> list[ArcExample]:
    dev_path = "./data/arc_easy/ARC-Easy-Dev.jsonl"
    return load_arc(dev_path)


def arc_challenge_dev() -> list[ArcExample]:
    dev_path = "./data/arc_challenge/ARC-Challenge-Dev.jsonl"
    return load_arc(dev_path)
