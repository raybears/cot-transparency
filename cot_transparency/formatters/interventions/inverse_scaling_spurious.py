from pathlib import Path

from slist import Slist

from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.json_utils.read_write import read_jsonl_file_into_basemodel

"""
Created from /scripts/inverse_scaling_experiments/create_spurious_few_shot_training_data.py
Finetuning samples that have spurious few shots in the prompt, but with responses generated from zero shot
To view
streamlit run streamlit_finetuning_data_viewer.py data/inverse_scaling/spurious_few_shot_training_data_cot.jsonl
"""
inverse_scaling_spurious_few_shot_cot_path = Path("data/inverse_scaling/spurious_few_shot_training_data_cot.jsonl")
inverse_scaling_spurious_few_shot_no_cot_path = Path(
    "data/inverse_scaling/spurious_few_shot_training_data_no_cot.jsonl"
)


def get_training_spurious_few_shot_cot() -> Slist[FinetuneSample]:
    return read_jsonl_file_into_basemodel(inverse_scaling_spurious_few_shot_cot_path, FinetuneSample)


def get_training_spurious_few_shot_no_cot() -> Slist[FinetuneSample]:
    return read_jsonl_file_into_basemodel(inverse_scaling_spurious_few_shot_no_cot_path, FinetuneSample)
