from typing import Optional, Sequence
import fire
from scripts.prompt_sen_experiments.plots import prompt_metrics

from stage_one import main
from scripts.prompt_sen_experiments.cot_formats_v1 import COT_FORMATTERS

EXP_DIR = "experiments/prompt_sen_experiments/temp0_cot_non_trained_formats"

# What is the idea behind this experiment?
# This is to verify that prompt sensivity is indeed measuring what we think it is measuring
# then retest with Mixed format vs non mixed format prompt sensitivty

# The idea of this one is to test on a set of formatters that are different from the ones that we trained on
# which is what we do in cot_formats_v1.py

# python demo_formatter.py | grep -E '^CotPromptSenFormatter_(LETTERS|NUMBERS)' | shuf | head -n 10
COT_TESTING_FORMATTERS = [
    "CotPromptSenFormatter_LETTERS_TAG_ANS_CHOICES_DOT_NEWLINE",
    "CotPromptSenFormatter_LETTERS_SHORT_ANS_CHOICES_PAREN_NEWLINE",
    "CotPromptSenFormatter_NUMBERS_NONE_ANS_CHOICES_DOT_NEWLINE",
    "CotPromptSenFormatter_LETTERS_SHORT_SELECT_DOT_SENTENCE",
    "CotPromptSenFormatter_NUMBERS_NONE_SELECT_DOT_NEWLINE",
    "CotPromptSenFormatter_LETTERS_NONE_SELECT_PAREN_SENTENCE",
    "CotPromptSenFormatter_LETTERS_PLEASE_NONE_PAREN_NEWLINE",
    "CotPromptSenFormatter_NUMBERS_FULL_ANS_CHOICES_DOT_NEWLINE",
    "CotPromptSenFormatter_NUMBERS_PLEASE_ANS_CHOICES_PAREN_SENTENCE",
    "CotPromptSenFormatter_LETTERS_FULL_SELECT_DOT_SENTENCE",
]

# check that is no overlap between COT_FORMATTERS and COT_TESTING_FORMATTERS, print out the overlap
# print(set(COT_FORMATTERS).intersection(set(COT_TESTING_FORMATTERS)))
assert len(set(COT_FORMATTERS).intersection(set(COT_TESTING_FORMATTERS))) == 0

TESTING_TASKS = ["mmlu", "truthful_qa"]

MODELS = [
    "gpt-3.5-turbo",
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::813SHRdF",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV",
    # "ft:gpt-3.5-turbo-0613:academicsnyuperez::81I9aGR0",
    "ft:gpt-3.5-turbo-0613:far-ai::88CAIEy4",  # my guy, finetuned on fine_tune_samples.jsonl
    "ft:gpt-3.5-turbo-0613:far-ai::88FWLOk7",  # my other guy, finetuned on COT_TRAINING_TASKS_2650.json
    # # "claude-v1",
    # # "claude-2",
    "gpt-4",
]


def run():
    main(
        tasks=TESTING_TASKS,
        models=MODELS,
        formatters=COT_TESTING_FORMATTERS,
        example_cap=100,
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
