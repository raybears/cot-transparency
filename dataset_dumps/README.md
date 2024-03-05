## Directory structure
Each file in the dataset_dumps/test folder follows the format `{dataset}_{bias_name}.jsonl`

For example, the `dataset_dumps/test/mmlu_distractor_fact.jsonl` file refers to the Distracor: Fact bias, using prompts from MMLU.




## Test json schema
Each line in the jsonl files follow the following schema of `StandardTestData`. For convienience, a sample script to parse the data is provided below
```python
from typing import Literal
from pydantic import BaseModel


class TestChatMessage(BaseModel):
    role: str
    content: str

class StandardTestData(BaseModel):
    # The original question from the dataset e.g. mmlu
    original_question: str
    # The unique id of the question that we calculate using a hash
    original_question_hash: str
    # e.g. mmlu, truthfulqa
    original_dataset: str
    # The original question, with an added prompt to elicit CoT
    unbiased_question: list[TestChatMessage]
    # The original question, with an added biasing statement, and an added prompt to elciit CoT
    biased_question: list[TestChatMessage]
    # e.g. suggested_answer, distractor_fact
    bias_name: str
    # ground_truth has single letters
    ground_truth: Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    # The option that we bias the model towards. Note that in the paper, we chiefly evaluate questions where the biased_option does not match the ground_truth. However, for the purpose of releasing a complete dataset, these files include questions where the biased_option does match the ground_truth. Hence you may want to filter for questions where the ground_truth != biased_option during evaluation.
    # For most biases, this is a single letter like "A", "B". For "Are you sure", this is f"NOT {CORRECT_ANS_FROM_FIRST_ROUND}"
    biased_option: str


def test_parse_one_file():
    # open dataset_dumps/test/mmlu_distractor_fact.jsonl
    with open("dataset_dumps/test/mmlu_distractor_fact.jsonl", "r") as f:
        for line in f.readlines():
            # read into the basemodel
            parsed = StandardTestData.model_validate_json(line)
            print(parsed.biased_question)
            print(f"Biased towards: {parsed.biased_option}")
            print(f"Ground truth: {parsed.ground_truth}")


            
if __name__ == "__main__":
    test_parse_one_file()
```


