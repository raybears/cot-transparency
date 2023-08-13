import json

from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer


class AquaExample(DataExampleBase):
    question: str
    options: list[str]
    rationale: str
    correct: MultipleChoiceAnswer

    def _get_options(self) -> list[str]:
        outputs = []
        for option in self.options:
            # replace A)answer with answer
            option = option[option.index(")") + 1 :]
            outputs.append(option)
        return outputs

    def _get_question(self) -> str:
        return self.question

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        return self.correct


def dev() -> list[AquaExample]:
    dev_path = "./data/aqua/dev.json"
    with open(dev_path) as f:
        output = []
        for line in f:
            example = AquaExample(**json.loads(line))
            output.append(example)
    return output
