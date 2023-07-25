from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel

from cot_transparency.util import deterministic_hash

MultipleChoiceAnswer = Literal["A", "B", "C", "D", "E", "F", "NOT_FOUND"]


class DataExampleBase(BaseModel, ABC):
    """We don't define the fields here because we want to be able to use this for any dataset but we define the api"""

    @property
    @abstractmethod
    def ground_truth(self) -> MultipleChoiceAnswer:
        raise NotImplementedError

    @property
    @abstractmethod
    def biased_ans(self) -> MultipleChoiceAnswer:
        raise NotImplementedError

    @abstractmethod
    def get_parsed_input(self) -> str:
        raise NotImplementedError

    def hash(self) -> str:
        return deterministic_hash(self.get_parsed_input())
