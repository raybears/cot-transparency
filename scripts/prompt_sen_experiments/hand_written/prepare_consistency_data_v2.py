import json
import os
from collections import defaultdict
from pathlib import Path

import fire
from slist import Slist

from cot_transparency.apis.openai.finetune import FinetuneSample
from cot_transparency.data_models.io import read_whole_exp_dir
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.interventions.valid_interventions import (
    name_to_intervention,
)
from cot_transparency.formatters.prompt_sensitivity.v2_prompt_sen import (
    TRAINING_COT_PROMPT_VARIANTS,
)
from cot_transparency.json_utils.read_write import write_jsonl_file_from_basemodel
from stage_one import COT_TRAINING_TASKS


def get_task_breakdown(outputs: list[TaskOutput]) -> dict[str, int]:
    num_examples_per_task = defaultdict(int)
    for example in outputs:
        num_examples_per_task[example.task_spec.task_name] += 1
    return num_examples_per_task


def main(
    exp_dir: str = "experiments/prompt_sen_experiments/temp0_cots_for_consistency_training",
    output_dir: str = "data/training_prompt_sen/question_phrasing",
    num_examples: list[int] = [100, 1000, 10000, 50000],
):
    models = ["gpt-3.5-turbo"]
    formatters = TRAINING_COT_PROMPT_VARIANTS
    formatter_names = [i.name() for i in formatters]
    temperature = 0
    tasks = COT_TRAINING_TASKS

    all_data = read_whole_exp_dir(exp_dir=exp_dir)
    print("Number of responses before filtering =", len(all_data))
    slist = (
        all_data.filter(lambda task: task.task_spec.inference_config.model in models)
        .filter(lambda task: task.task_spec.formatter_name in formatter_names)
        .filter(lambda task: task.task_spec.task_name in tasks)
        .filter(lambda task: task.task_spec.inference_config.temperature == temperature)
    )
    print("Number of responses after filtering =", len(slist))

    inputs = slist.shuffle(seed=str(42))

    example: TaskOutput
    outputs = Slist()
    for example in inputs:
        for formatter in formatters:
            copied = example.model_copy(deep=True)
            copied.task_spec.messages
            original_question = copied.task_spec.get_data_example_obj()

            if copied.task_spec.intervention_name is not None:
                intervention = name_to_intervention(copied.task_spec.intervention_name)
                reparsed_messages = intervention.intervene(question=original_question, formatter=formatter)
            else:
                reparsed_messages = formatter.format_example(question=original_question)
            copied.task_spec.messages = reparsed_messages
            outputs.append(copied)

    print("Number of consistentcy training examples =", len(outputs))
    # print the breakdown according to the task and number of examples
    os.makedirs(output_dir, exist_ok=True)
    for num_example in num_examples:
        if num_example > len(outputs):
            print(
                f"Number of examples {num_example} is greater than the number of examples {len(outputs)}, skipping..."
            )
            continue
        truncated_outputs = outputs[:num_example]
        print(f"Writing {num_example} examples")
        print(
            "Number of examples per task:",
            json.dumps(get_task_breakdown(truncated_outputs), indent=2),
        )

        fine_tune_samples = [FinetuneSample.from_task_output(i) for i in truncated_outputs]
        output_path = f"{output_dir}/consistency_training_{num_example}.jsonl"
        path = Path(output_path)
        if path.exists():
            print(f"Path {output_path} already exists, skipping...")
            continue
        write_jsonl_file_from_basemodel(path, fine_tune_samples)


if __name__ == "__main__":
    fire.Fire(main)
