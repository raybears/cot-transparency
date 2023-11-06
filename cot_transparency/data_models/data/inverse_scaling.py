import json
from enum import Enum
from pathlib import Path
from string import ascii_uppercase

from slist import Slist

from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer


class InverseScalingExample(DataExampleBase):
    prompt: str
    classes: list[str]
    answer_index: int

    def _get_options(
        self,
    ) -> list[str]:
        return self.classes

    def _get_question(self) -> str:
        return self.prompt

    @property
    def _ground_truth(self) -> MultipleChoiceAnswer:
        return ascii_uppercase[self.answer_index]  # type: ignore


class InverseScalingTask(str, Enum):
    # strong prior tasks
    resisting_correction = "resisting_correction"
    memo_trap = "memo_trap"
    # spurious few shot tasks
    hindsight_neglect = "hindsight_neglect"

    @staticmethod
    def all_tasks() -> list[str]:
        return [task.value for task in InverseScalingTask]


def get_inverse_scaling(task: InverseScalingTask) -> Slist[InverseScalingExample]:
    if task == InverseScalingTask.resisting_correction:
        path = Path("./data/inverse_scaling/resisting-correction_classification.jsonl")
    elif task == InverseScalingTask.memo_trap:
        path = Path("./data/inverse_scaling/memo-trap_classification.jsonl")
    elif task == InverseScalingTask.hindsight_neglect:
        path = Path("./data/inverse_scaling/hindsight-neglect_classification.jsonl")

    # read the json manually into dicts
    _dicts: Slist[InverseScalingExample] = Slist([])
    with open(path) as f:
        for line in f:
            new_item = json.loads(line)
            # need to change "classes", its a string of a list, you need to eval it
            evaled_classes = eval(new_item["classes"])
            # strip the whitespace from the classes
            evaled_classes = [x.strip() for x in evaled_classes]
            new_item["classes"] = evaled_classes
            parsed_item = InverseScalingExample.model_validate(new_item)
            # convert the dicts into DataExampleBase objects
            _dicts.append(parsed_item)
    return _dicts
