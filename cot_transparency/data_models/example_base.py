from abc import ABC, abstractmethod
import random
from typing import Literal, TypeVar
from pydantic import BaseModel
from string import ascii_uppercase

from cot_transparency.util import deterministic_hash

MultipleChoiceAnswer = Literal[
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
]
VALID_ANSWERS: set[MultipleChoiceAnswer] = {
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
}


class DataExampleBase(BaseModel, ABC):
    """We don't define the fields here because we want to be able to use this for any dataset but we define the api"""

    @property
    @abstractmethod
    def ground_truth(self) -> MultipleChoiceAnswer:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_choices(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_parsed_input(self) -> str:
        raise NotImplementedError

    def get_parsed_input_with_none_of_the_above(self) -> str:
        n_options = self.n_choices
        new_letter = ascii_uppercase[n_options]
        # don't add this if already included
        if "none" in self.get_parsed_input().lower():
            return self.get_parsed_input()
        new_option = f"({new_letter}) None of the above"
        return self.get_parsed_input() + "\n" + new_option

    @property
    def biased_ans(self) -> MultipleChoiceAnswer:
        rng = random.Random(self.get_parsed_input())  # seed with question
        biased_ans_idx = rng.randrange(0, self.n_choices)  # select random answer for bias metrics
        biased_ans_letter: MultipleChoiceAnswer = ascii_uppercase[biased_ans_idx]  # type: ignore
        return biased_ans_letter

    def hash(self) -> str:
        return deterministic_hash(self.get_parsed_input())


GenericDataExample = TypeVar("GenericDataExample", bound="DataExampleBase")
