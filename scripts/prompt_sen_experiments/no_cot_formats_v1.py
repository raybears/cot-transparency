from typing import Optional, Sequence
import fire
from scripts.prompt_sen_experiments.cot_formats_v1 import TESTING_TASKS
from scripts.prompt_sen_experiments.plots import prompt_metrics

from stage_one import COT_TESTING_TASKS, main

if True:
    pass

EXP_DIR = "experiments/prompt_sen_experiments/temp0_no_cot"

# What is the idea behind this experiment?
# This is to verify that prompt sensivity is indeed measuring what we think it is measuring
# then retest with Mixed format vs non mixed format prompt sensitivty

# Improved version, that should use n_samples_per_request

# python demo_formatter.py | grep -E 'NoCotPromptSenFormatter_(LETTERS|NUMBERS)' | shuf | head -n 10
FORMATTERS = [
    "NoCotPromptSenFormatter_LETTERS_SHORT_SELECT_PAREN_NEWLINE",
    "NoCotPromptSenFormatter_LETTERS_PLEASE_SELECT_DOT_SENTENCE",
    "NoCotPromptSenFormatter_NUMBERS_SHORT_OPTIONS_PAREN_NEWLINE",
    "NoCotPromptSenFormatter_NUMBERS_NONE_SELECT_DOT_SENTENCE",
    "NoCotPromptSenFormatter_NUMBERS_TAG_OPTIONS_DOT_NEWLINE",
    "NoCotPromptSenFormatter_NUMBERS_PLEASE_NONE_DOT_SENTENCE",
    "NoCotPromptSenFormatter_LETTERS_SHORT_OPTIONS_PAREN_NEWLINE",
    "NoCotPromptSenFormatter_NUMBERS_SHORT_ANS_CHOICES_PAREN_NEWLINE",
    "NoCotPromptSenFormatter_LETTERS_PLEASE_OPTIONS_DOT_NEWLINE",
    "NoCotPromptSenFormatter_NUMBERS_NONE_ANS_CHOICES_PAREN_SENTENCE",
]

assert len(set(FORMATTERS)) == len(FORMATTERS)

MODELS = [
    "gpt-3.5-turbo",
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::813SHRdF",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV",
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::81I9aGR0",
    # # "claude-v1",
    # # "claude-2",
    "gpt-4",
]


def run():
    cap = 200
    # Try first by sampling 1 token
    # main(
    #     tasks=COT_TESTING_TASKS,
    #     models=MODELS,
    #     formatters=FORMATTERS,
    #     example_cap=cap,
    #     exp_dir=EXP_DIR,
    #     temperature=0,
    #     batch=40,
    #     interventions=[None],
    #     raise_after_retries=False,
    #     raise_on="all",
    #     repeats_per_question=1,
    #     num_retries=1,
    #     n_responses_per_request=1,
    #     max_tokens=1,
    #     retry_answers_with_none=False,
    # )

    # Then try and scope up any answers that failed by allowing them to use 40 tokens
    main(
        tasks=COT_TESTING_TASKS,
        models=MODELS,
        formatters=FORMATTERS,
        example_cap=cap,
        exp_dir=EXP_DIR,
        temperature=0,
        batch=40,
        interventions=[None],
        raise_after_retries=False,
        raise_on="all",
        repeats_per_question=1,
        num_retries=1,
        n_responses_per_request=1,
        max_tokens=40,
        retry_answers_with_none=True,
    )


def plot(
    exp_dir: str = EXP_DIR,
    models: Sequence[str] = MODELS,
    tasks: Sequence[str] = TESTING_TASKS,
    formatters: Sequence[str] = FORMATTERS,
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
