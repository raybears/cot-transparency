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
    "NOT_FOUND",
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
    "NOT_FOUND",
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

    @property
    def biased_ans(self) -> MultipleChoiceAnswer:
        rng = random.Random(self.get_parsed_input())  # seed with question
        biased_ans_idx = rng.randrange(0, self.n_choices)  # select random answer for bias metrics
        biased_ans_letter: MultipleChoiceAnswer = ascii_uppercase[biased_ans_idx]  # type: ignore
        return biased_ans_letter

    def hash(self) -> str:
        return deterministic_hash(self.get_parsed_input())


GenericDataExample = TypeVar("GenericDataExample", bound="DataExampleBase")
