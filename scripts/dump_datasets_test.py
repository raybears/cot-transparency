from typing import Literal
from pydantic import BaseModel



class TestChatMessage(BaseModel):
    role: str
    content: str

class StandardTestData(BaseModel):
    original_question: str
    original_question_hash: str
    original_dataset: str
    unbiased_question: list[TestChatMessage]
    biased_question: list[TestChatMessage]
    bias_name: str
    ground_truth: Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def test_parse_one_file():
    # open dataset_dumps/test/mmlu_distractor_fact.jsonl
    with open("dataset_dumps/test/mmlu_distractor_fact.jsonl", "r") as f:
        for line in f.readlines():
            # read into the basemodel
            parsed = StandardTestData.model_validate_json(line)
            print(parsed.biased_question)

            
if __name__ == "__main__":
    test_parse_one_file()