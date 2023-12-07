from pathlib import Path

from slist import Slist

from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel

# Generated from scripts/evaluate_alignment_tax/create_gpt_35_instruction_dataset.py
gpt_35_instruct_path = Path("data/instructions/gpt_35_turbo_0613_temp_1.jsonl")

# Generated from scripts/evaluate_alignment_tax/create_gpt_35_instruction_dataset_user_5_times.py
gpt_35_instruct_user_5path = Path("data/instructions/gpt_35_turbo_0613_user_5_responses_temp_1.jsonl")

gpt_35_instruct_user_test_5path = Path("data/instructions/test_gpt_35_turbo_0613_user_5_responses_temp_1.jsonl")


def get_all_alpaca_training_gpt_35(limit: int, seed: str) -> Slist[FinetuneSample]:
    # Clean Alpaca training data, but with completions from gpt-3.5-turbo-0613, temeprature 1.0, 2000 tokens
    # Generated from scripts/evaluate_alignment_tax/create_gpt_35_instruction_dataset.py
    return read_jsonl_file_into_basemodel(gpt_35_instruct_path, FinetuneSample).shuffle(seed=seed).take(limit)


def get_all_alpaca_training_gpt_35_sample_5(limit: int, seed: str) -> Slist[FinetuneSample]:
    # Clean Alpaca training data, but with completions from gpt-3.5-turbo-0613, temeprature 1.0, 2000 tokens
    # Generated from scripts/evaluate_alignment_tax/create_gpt_35_instruction_dataset.py
    return read_jsonl_file_into_basemodel(gpt_35_instruct_user_5path, FinetuneSample).shuffle(seed=seed).take(limit)


def get_all_alpaca_testing_gpt_35_sample_5(limit: int, seed: str) -> Slist[FinetuneSample]:
    # Clean Alpaca training data, but with completions from gpt-3.5-turbo-0613, temeprature 1.0, 2000 tokens
    # Generated from scripts/evaluate_alignment_tax/create_gpt_35_instruction_dataset_user_5_times_testing.py
    return (
        read_jsonl_file_into_basemodel(gpt_35_instruct_user_test_5path, FinetuneSample).shuffle(seed=seed).take(limit)
    )
