from pydantic import BaseModel


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


class MilesBBHRawDataFolder(BaseModel):
    data: list[MilesBBHRawData]
