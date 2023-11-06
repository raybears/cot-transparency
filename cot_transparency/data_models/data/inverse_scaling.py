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
    redefine = "redefine"
    # spurious few shot tasks
    hindsight_neglect = "hindsight_neglect"
    repetitive_algebra = "repetitive_algebra"
    # distractor tasks
    pattern_matching = "pattern_matching"
    into_to_the_unknown = "into_to_the_unknown"
    neqa = "neqa"
    sig_figs = "sig_figs"

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
    elif task == InverseScalingTask.repetitive_algebra:
        path = Path("./data/inverse_scaling/repetitive-algebra_classification.jsonl")
    elif task == InverseScalingTask.redefine:
        path = Path("./data/inverse_scaling/redefine_classification.jsonl")
    elif task == InverseScalingTask.pattern_matching:
        path = Path("./data/inverse_scaling/pattern-matching_classification.jsonl")
    elif task == InverseScalingTask.into_to_the_unknown:
        path = Path("./data/inverse_scaling/into-to-the-unknown_classification.jsonl")
    elif task == InverseScalingTask.neqa:
        path = Path("./data/inverse_scaling/neqa_classification.jsonl")
    elif task == InverseScalingTask.sig_figs:
        path = Path("./data/inverse_scaling/sig-figs_classification.jsonl")

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
