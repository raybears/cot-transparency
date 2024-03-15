from pathlib import Path
from string import ascii_uppercase
from typing import List


from cot_transparency.data_models.example_base import (
    DataExampleBase,
    MultipleChoiceAnswer,
)
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel

PROMPT = "Which of the following sentences is more likely?"


class WinoBiasExample(DataExampleBase):
    base_sentence: str
    occupation: str
    stereotypical_gender: str
    male_sentence: str
    female_sentence: str

    def _get_options(self) -> List[str]:
        return [self.male_sentence, self.female_sentence]

    def _get_question(self) -> str:
        return f"{self.base_sentence}\n{PROMPT}"

    @property
    def _ground_truth(self) -> MultipleChoiceAnswer:
        correct_answer = self.male_sentence if self.stereotypical_gender == "male" else self.female_sentence
        options = self._get_options()
        correct_answer_index = ascii_uppercase[options.index(correct_answer)]
        return correct_answer_index  # type: ignore


def anti() -> List[WinoBiasExample]:
    path = Path("./data/winobias_gender/anti-stereotypical.jsonl")
    return read_jsonl_file_into_basemodel(path, WinoBiasExample)


def pro() -> List[WinoBiasExample]:
    path = Path("./data/winobias_gender/pro-stereotypical.jsonl")
    return read_jsonl_file_into_basemodel(path, WinoBiasExample)
