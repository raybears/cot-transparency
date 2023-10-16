from typing import Optional, Sequence
import fire
from cot_transparency.formatters.prompt_sensitivity.v2_prompt_sen import TESTING_FORMATS
from scripts.prompt_sen_experiments.plots import prompt_metrics

from stage_one import PROMPT_SEN_TESTING_TASKS, main
from scripts.prompt_sen_experiments.cot_formats_v1 import COT_FORMATTERS

EXP_DIR = "experiments/prompt_sen_experiments/v3_handwritten_formats/temp0_cot_v3_new_formats_no_answer_parsing"

# The idea of this one is to test on a set of formatters that are different from the ones that we trained on
# which is what we do in cot_formats_v1.py
# also to test on a wider variety of tasks

# python demo_formatter.py | grep -E '^CotPromptSenFormatter_(LETTERS|NUMBERS)' | shuf | head -n 10
COT_TESTING_FORMATTERS = [i.name() for i in TESTING_FORMATS]

# check that is no overlap between COT_FORMATTERS and COT_TESTING_FORMATTERS, print out the overlap
# print(set(COT_FORMATTERS).intersection(set(COT_TESTING_FORMATTERS)))
assert len(set(COT_FORMATTERS).intersection(set(COT_TESTING_FORMATTERS))) == 0

TESTING_TASKS = PROMPT_SEN_TESTING_TASKS

MODELS = [
    "gpt-3.5-turbo",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV",  # James 50/50 model
    # "ft:gpt-3.5-turbo-0613:far-ai::88dVFSpt",  # consistency training guy
    # "ft:gpt-3.5-turbo-0613:far-ai::89d1Jn8z",  # 100
    "ft:gpt-3.5-turbo-0613:far-ai::89dSzlfs",  # 1000
    "ft:gpt-3.5-turbo-0613:far-ai::89dxzRjA",  # 10000
    "ft:gpt-3.5-turbo-0613:far-ai::89figOP6",  # 50000
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::88h1pB4E",  # 50 / 50 unbiased
]


def run(examples_per_task: int = 100):
    main(
        tasks=TESTING_TASKS,
        models=MODELS,
        formatters=COT_TESTING_FORMATTERS,
        example_cap=examples_per_task,
        exp_dir=EXP_DIR,
        temperature=0,
        batch=200,
        interventions=["StepByStep"],
        raise_after_retries=False,
        raise_on="all",
        repeats_per_question=1,
        num_retries=1,
        n_responses_per_request=1,
        max_tokens=3000,
    )
    # Then find the answers
    # final_answer_main(input_exp_dir=EXP_DIR, EXP_DIRk)


def plot(
    exp_dir: str,
    models: Sequence[str] = MODELS,
    tasks: Sequence[str] = TESTING_TASKS,
    formatters: Sequence[str] = COT_TESTING_FORMATTERS,
    x: str = "task_name",
    hue: str = "model",
    col: Optional[str] = "is_cot",
):
    prompt_metrics(
        exp_dir=exp_dir,
        models=models,
        tasks=tasks,
        formatters=formatters,
        x=x,
        hue=hue,
        col=col,
        temperature=0,
    )


if __name__ == "__main__":
    fire.Fire({"run": run, "plot": plot})
