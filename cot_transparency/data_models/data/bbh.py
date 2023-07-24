import json
from pathlib import Path
from string import ascii_uppercase
from typing import Optional
from pydantic import ValidationError

from pydantic import BaseModel
from cot_transparency.data_models.example_base import DataExampleBase, MultipleChoiceAnswer


class MilesBBHRawData(DataExampleBase):
    # Already formatted to have the answer of A all the time
    # tracking_shuffled_objects_three_objects doesn't have the Optional fields
    idx: Optional[int] = None
    inputs: str
    targets: list[str] = []
    multiple_choice_targets: list[str]
    multiple_choice_scores: list[int]
    split: Optional[str] = None
    random_ans_idx: int
    parsed_inputs: str

    def get_parsed_input(self) -> str:
        return self.parsed_inputs

    @property
    def ground_truth(self) -> MultipleChoiceAnswer:
        # get the index equal to one of multiple_choice_scores
        ground_truth_idx = self.multiple_choice_scores.index(1)
        letter: MultipleChoiceAnswer = ascii_uppercase[ground_truth_idx]  # type: ignore
        return letter

    @property
    def biased_ans(self) -> MultipleChoiceAnswer:
        letter: MultipleChoiceAnswer = ascii_uppercase[self.get_random_ans_idx()]  # type: ignore
        return letter

    def get_random_ans_idx(self) -> int:
        return self.random_ans_idx


class MilesBBHRawDataFolder(BaseModel):
    data: list[MilesBBHRawData]


def load_bbh(task: str) -> list[MilesBBHRawData]:
    json_path: Path = Path(f"data/bbh/{task}/val_data.json")
    with open(json_path, "r") as f:
        raw_data = json.load(f)
    try:
        data: list[MilesBBHRawData] = MilesBBHRawDataFolder(**raw_data).data
    except ValidationError as e:
        print(f"Error parsing {json_path}")
        raise e
    return data