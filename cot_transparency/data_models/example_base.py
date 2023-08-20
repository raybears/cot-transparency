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


def raise_if_not_multiple_choice_answer(string: str) -> MultipleChoiceAnswer:
    assert string in VALID_ANSWERS
    return string


class LetterAndAnswer(BaseModel):
    letter: MultipleChoiceAnswer
    answer: str


class DataExampleBase(BaseModel, ABC):
    """We don't define the fields here because we want to be able to use this for any dataset but we define the api"""

    @property
    @abstractmethod
    def ground_truth(self) -> MultipleChoiceAnswer:
        """Please implement this method to return the ground truth answer"""
        raise NotImplementedError

    @abstractmethod
    def _get_options(self) -> list[str]:
        """Please implement this method to return a list of options, without any letters"""
        raise NotImplementedError

    @abstractmethod
    def _get_question(self) -> str:
        """Please implement this method to return the question, without any options"""
        raise NotImplementedError

    def ground_truth_idx(self) -> int:
        return ascii_uppercase.index(self.ground_truth)

    def get_parsed_input(self) -> str:
        question = self._get_question()
        options = self._get_options()
        options_with_letters = self.format_options_with_letters(self._get_lettered_options(options))
        return f"{question}\n\nAnswer choices:\n{options_with_letters}"

    @staticmethod
    def format_options_with_letters(options: list[LetterAndAnswer]) -> str:
        return "\n".join([f"({option.letter}) {option.answer}" for option in options])

    @staticmethod
    def _get_lettered_options(options: list[str]) -> list[LetterAndAnswer]:
        return [
            LetterAndAnswer(letter=ascii_uppercase[idx], answer=option)  # type: ignore
            for idx, option in enumerate(options)
        ]

    def get_parsed_input_with_none_of_the_above(self) -> str:
        question = self._get_question()
        options = self._get_options()
        if "none" not in " ".join(options).lower():
            options.append("None of the above")

        options_with_letters = self.format_options_with_letters(self._get_lettered_options(options))
        return f"{question}\n\nAnswer choices:\n{options_with_letters}"

    @property
    def biased_ans(self) -> MultipleChoiceAnswer:
        rng = random.Random(self.get_parsed_input())  # seed with question
        n_choices = len(self._get_options())
        biased_ans_idx = rng.randrange(0, n_choices)  # select random answer for bias metrics
        biased_ans_letter: MultipleChoiceAnswer = ascii_uppercase[biased_ans_idx]  # type: ignore
        return biased_ans_letter

    def hash(self) -> str:
        return deterministic_hash(self.get_parsed_input())


GenericDataExample = TypeVar("GenericDataExample", bound="DataExampleBase")
