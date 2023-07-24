import json

from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer


class AquaExample(DataExampleBase):
    question: str
    options: list[str]
    rationale: str
    correct: MultipleChoiceAnswer

    def process_options(self, options: list[str]) -> str:
        outputs = []
        for option in options:
            # replace A)answer with (A): answer
            option = option.replace(")", "): ")
            outputs.append(f"({option}")
        return "\n".join(outputs)

    def get_parsed_input(self) -> str:
        options = self.process_options(self.options)
        return f"{self.question}\n\nAnswer choices:\n{options}"

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        return self.correct

    @property
    def biased_ans(self) -> MultipleChoiceAnswer:
        return "NOT_FOUND"


def dev() -> list[AquaExample]:
    dev_path = "./data/aqua/dev.json"
    with open(dev_path) as f:
        output = []
        for line in f:
            example = AquaExample(**json.loads(line))
            output.append(example)
    return output