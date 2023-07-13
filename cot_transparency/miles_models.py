from pydantic import BaseModel

from cot_transparency.hashing import deterministic_hash


class MilesBBHRawData(BaseModel):
    # Already formatted to have the answer of A all the time
    idx: int
    inputs: str
    targets: list[str]
    multiple_choice_targets: list[str]
    multiple_choice_scores: list[int]
    split: str
    random_ans_idx: int
    parsed_inputs: str

    @property
    def ground_truth(self) -> str:
        # get the index equal to one of multiple_choice_scores
        ground_truth_idx = self.multiple_choice_scores.index(1)
        return self.multiple_choice_targets[ground_truth_idx]

    def hash(self) -> str:
        return deterministic_hash(self.parsed_inputs)


class MilesBBHRawDataFolder(BaseModel):
    data: list[MilesBBHRawData]
