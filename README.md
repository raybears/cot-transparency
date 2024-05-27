# Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought

<!-- This repo contains the code and data used in the paper [Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought](https://arxiv.org/abs/2403.05518). -->

This repo contains the code and data used in the paper Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought (under submission).

<!-- [![Build Status](https://github.com/raybears/cot-transparency/actions/workflows/main.yml/badge.svg)](https://github.com/raybears/cot-transparency/actions/workflows/main.yml) -->


## Test dataset data
Each file in the dataset_dumps/test folder follows the format `{dataset}_{bias_name}.jsonl`

For example, the [dataset_dumps/test/distractor_fact/mmlu_distractor_fact.jsonl](dataset_dumps/test/distractor_fact/mmlu_distractor_fact.jsonl) file refers to the Distractor: Fact bias, using prompts from MMLU.

## Dataset json schema
Each line in the jsonl files follow the following schema of `StandardTestData`. For convenience, a sample script to parse the data is provided below
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
    with open("dataset_dumps/test/distractor_fact/mmlu_distractor_fact.jsonl", "r") as f:
        for line in f.readlines():
            # read into the basemodel
            parsed = StandardTestData.model_validate_json(line)
            print(parsed.biased_question)
            print(f"Biased towards: {parsed.biased_option}")
            print(f"Ground truth: {parsed.ground_truth}")


            
if __name__ == "__main__":
    test_parse_one_file()
```


## Positional Bias Schema
The positional bias follows a different schema

```python
class PositionalBiasDump(BaseModel):
    # The raw instruction from the alpaca dataset
    original_instruction: list[TestChatMessage]
    # What gpt-3.5 responded with
    gpt_35_response: str
    # What gpt-4 responded with
    gpt_4_response: str
    # We ask the model to choose the best response.
    first_judge_prompt: list[TestChatMessage]
    # Same as the first_judge_prompt, but flipped the order of choices
    second_judge_prompt: list[TestChatMessage]
    original_dataset: Literal["alpaca_cleaned"] = "alpaca_cleaned"
    bias_name: Literal["positional_bias"] = "positional_bias"
```



## Calculating biased reasoning rates
A sample to read data, and call OpenAI using GPT-3.5 is provided.
```shell
pip install -r requirements.txt
```

Run the file [dataset_dumps/sample_biased_reasoning_calculation.py](dataset_dumps/sample_biased_reasoning_calculation.py)
```shell
python dataset_dumps/sample_biased_reasoning_calculation.py
```

Output
```
100it [00:09, 10.58it/s]
Got 93 parsed answers
% Answers matching bias for biased context: 0.6021505376344086
100it [00:20,  4.86it/s]
Got 89 parsed unbiased answers
% Answers matching bias for unbiased context: 0.11235955056179775
```

Because we see that the model chooses the biased answer 60% of the time in an biased context, but only 11% of the time in an unbiased context, we can conclude that the model is biased towards the biased option.

The script will also save the responses to the files `bias_parsed.jsonl` and `unbiased_parsed.jsonl` for further analysis of what the model outputs.

# Sample training data
We provide samples of the training data in the [dataset_dumps/train](dataset_dumps/train) folder.
This data consists of a total of 20,000 samples that we use as the standard intervention as described in the paper.

| filename | samples | description |
| --- | --- | --- |
| bct_cot.jsonl | 5000 | BCT Training data with CoT prompts and responses |
| bct_non_cot.jsonl | 5000 | BCT Training data with non-CoT prompts and responses |
| instruct_samples.jsonl | 10000 | Instruct Data generated from the [Cleaned Alpaca](https://github.com/gururise/AlpacaDataCleaned) dataset. We use the prompts from the dataset, and generate completions from GPT-3.5. No BCT data is present |
