"""
format is like:        


"original_question": "Natalia is riding a bicycle for the cycling competition. On Monday she rode 40 kilometers and on Tuesday 50 kilometers. On Wednesday she rode 50% fewer kilometers than the day before. On Thursday she rode as many as the sum of the kilometers from Monday and Wednesday. How many kilometers did Natalie ride in total?",
"answer": "180",
"new_question": "Natalia is riding a bicycle for the cycling competition. On Monday she rode 40 kilometers and on Tuesday 50 kilometers. On Wednesday she rode 50% fewer kilometers than the day before. On Thursday she rode as many as the sum of the kilometers from Monday and Wednesday. Tom bought 80 tomatoes from the grocery store. How many kilometers did Natalie ride in total?",
"n_steps": 3,
"role": "Tom",
"number": "80",
"sentence_template": "{role} bought {number} tomatoes from the grocery store.",
"role_label": "nonoverlapped",
"number_label": "in_range",
"sentence_label": "out_topic"
"""

import json

from slist import Slist
from cot_transparency.data_models.example_base import DataExampleBase


class GSMExample(DataExampleBase):
    original_question: str
    new_question: str
    new_question: str
    role: str
    number: str
    sentence_template: str
    role_label: str
    number_label: str
    sentence_label: str
    answer: str
    n_steps: int

    def _get_options(
        self,
    ) -> list[str]:
        return [self.answer]

    def _get_question(self) -> str:
        return self.original_question

    @property
    def _ground_truth(self) -> str:
        return self.answer

    def ground_truth_idx(self) -> int:  # type: ignore
        return 0

    def get_parsed_input(
        self,
        include_none_of_the_above: bool = False,
    ) -> str:
        assert include_none_of_the_above is False
        return self._get_question()


class GSMBiased(GSMExample):
    task_name: str = "gsm_biased"

    def _get_question(self) -> str:
        return self.new_question


class GSMUnbiased(GSMExample):
    task_name: str = "gsm_unbiased"

    def _get_question(self) -> str:
        return self.original_question


gsm_2step = "./data/gsm_ic/GSM-iC_2step.json"
gsm_mstep = "./data/gsm_ic/GSM-iC_mstep.json"


def _load_gsm(filepath: str = gsm_mstep) -> Slist[dict[str, str | int]]:
    with open(filepath) as fh:
        loaded = Slist(json.load(fh))
    return loaded.shuffle(seed=str(342))


def load_gsm_unbiased(filepath: str = gsm_mstep):
    return _load_gsm(filepath).map(lambda x: GSMUnbiased(**x))  # type: ignore


def load_gsm_biased(filepath: str = gsm_mstep):
    return _load_gsm(filepath).map(lambda x: GSMBiased(**x))  # type: ignore


if __name__ == "__main__":
    examples = load_gsm_biased()
    breakpoint()
