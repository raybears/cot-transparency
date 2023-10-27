from typing import Optional, Sequence

import fire

from scripts.prompt_sen_experiments.plots import prompt_metrics
from stage_one import main

EXP_DIR = "experiments/prompt_sen_experiments/temp0_cot"

# What is the idea behind this experiment?
# This is to verify that prompt sensivity is indeed measuring what we think it is measuring
# then retest with Mixed format vs non mixed format prompt sensitivty

# Improved version, that should use n_samples_per_request

# python demo_formatter.py | grep -E 'NoCotPromptSenFormatter_(LETTERS|NUMBERS)' | shuf | head -n 10
COT_FORMATTERS = [
    "CotPromptSenFormatter_LETTERS_SHORT_SELECT_PAREN_NEWLINE",
    "CotPromptSenFormatter_LETTERS_PLEASE_SELECT_DOT_SENTENCE",
    "CotPromptSenFormatter_NUMBERS_SHORT_OPTIONS_PAREN_NEWLINE",
    "CotPromptSenFormatter_NUMBERS_NONE_SELECT_DOT_SENTENCE",
    "CotPromptSenFormatter_NUMBERS_TAG_OPTIONS_DOT_NEWLINE",
    "CotPromptSenFormatter_NUMBERS_PLEASE_NONE_DOT_SENTENCE",
    "CotPromptSenFormatter_LETTERS_SHORT_OPTIONS_PAREN_NEWLINE",
    "CotPromptSenFormatter_NUMBERS_SHORT_ANS_CHOICES_PAREN_NEWLINE",
    "CotPromptSenFormatter_LETTERS_PLEASE_OPTIONS_DOT_NEWLINE",
    "CotPromptSenFormatter_NUMBERS_NONE_ANS_CHOICES_PAREN_SENTENCE",
]

MODELS = [
    "gpt-3.5-turbo",
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::813SHRdF",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV",
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::81I9aGR0",
    "ft:gpt-3.5-turbo-0613:far-ai::88CAIEy4",  # my guy
    "ft:gpt-3.5-turbo-0613:far-ai::88FWLOk7",  # my other guy, finetuned on COT_TRAINING_TASKS_2650.json
    "ft:gpt-3.5-turbo-0613:far-ai::88dVFSpt",  # consistency training guy
    # # "claude-v1",
    # # "claude-2",
    "gpt-4",
]

TESTING_TASKS = ["mmlu", "truthful_qa"]


def run():
    main(
        tasks=TESTING_TASKS,
        models=MODELS,
        formatters=COT_FORMATTERS,
        example_cap=200,
        exp_dir=EXP_DIR,
        temperature=0,
        batch=80,
        interventions=[None],
        raise_after_retries=False,
        raise_on="all",
        repeats_per_question=1,
        num_retries=1,
        n_responses_per_request=1,
        max_tokens=3000,
    )


def plot(
    exp_dir: str = EXP_DIR,
    models: Sequence[str] = MODELS,
    tasks: Sequence[str] = TESTING_TASKS,
    formatters: Sequence[str] = COT_FORMATTERS,
    x: str = "task_name",
    hue: str = "model",
    col: Optional[str] = "is_cot",
    only_modally_wrong: bool = False,
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
        only_modally_wrong=only_modally_wrong,
    )


if __name__ == "__main__":
    fire.Fire({"run": run, "plot": plot})
