from pathlib import Path

from slist import Slist

from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel

# Generated from scripts/evaluate_alignment_tax/create_gpt_35_instruction_dataset.py
gpt_35_instruct_path = Path("data/instructions/gpt_35_turbo_0613_temp_1.jsonl")


def get_all_alpaca_training_gpt_35() -> Slist[FinetuneSample]:
    # Clean Alpaca training data, but with completions from gpt-3.5-turbo-0613, temeprature 1.0, 2000 tokens
    # Generated from scripts/evaluate_alignment_tax/create_gpt_35_instruction_dataset.py
    return read_jsonl_file_into_basemodel(gpt_35_instruct_path, FinetuneSample)
