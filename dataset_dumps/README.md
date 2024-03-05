## Directory structure
Each file in the dataset_dumps/test folder follows the format `{dataset}_{bias_name}.jsonl`

For example, the `dataset_dumps/test/mmlu_distractor_fact.jsonl` file refers to the Distracor: Fact bias, using prompts from MMLu.




## Test json schema
Each line in the jsonl files follow the following schema of `StandardTestData`. For convienience, a sample script to parse the data is provided below
```python
from typing import Literal
from pydantic import BaseModel


class TestChatMessage(BaseModel):
    role: str
    content: str

class StandardTestData(BaseModel):
    original_question: str # The original question from the dataset e.g. mmlu
    original_question_hash: str # The unique id of the question that we calculate using a hash
    original_dataset: str # e.g. mmlu, truthfulqa
    unbiased_question: list[TestChatMessage] # The original question, with an added prompt to elicit CoT
    biased_question: list[TestChatMessage] # The origianl question, with an added biasing statement, and an added prompt to elciit CoT
    bias_name: str # e.g. suggested_answer, distractor_fact
    ground_truth: Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] # All single letters


def test_parse_one_file():
    # open dataset_dumps/test/mmlu_distractor_fact.jsonl
    with open("dataset_dumps/test/mmlu_distractor_fact.jsonl", "r") as f:
        for line in f.readlines():
            # read into the basemodel
            parsed = StandardTestData.model_validate_json(line)
            print(parsed.biased_question)

            
if __name__ == "__main__":
    test_parse_one_file()
```


