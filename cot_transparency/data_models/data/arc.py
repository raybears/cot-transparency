import json

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
    answerKey: str

    def process_options(self, options: list[ArcChoices]) -> str:
        outputs = []
        for option in options:
            # replace A)answer with (A): answer
            label = option.label
            text = option.text
            outputs.append(f"({label}): {text}")
        return "\n".join(outputs)

    def get_parsed_input(self) -> str:
        options = self.process_options(self.question.choices)
        return f"{self.question.stem}\n\nAnswer choices:\n{options}"

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        return self.answerKey  # type: ignore

    @property
    def biased_ans(self) -> MultipleChoiceAnswer:
        return "NOT_FOUND"


def load_arc(dev_path: str) -> list[ArcExample]:
    with open(dev_path) as f:
        output = []
        for line in f:
            example = ArcExample(**json.loads(line))
            output.append(example)
    return output


def arc_easy_dev() -> list[ArcExample]:
    dev_path = "./data/arc_easy/ARC-Easy-Dev.jsonl"
    return load_arc(dev_path)


def arc_challenge_dev() -> list[ArcExample]:
    dev_path = "./data/arc_challenge/ARC-Challenge-Dev.jsonl"
    return load_arc(dev_path)
