import fire

from scripts.prompt_sen_experiments.kl.kl_plots import kl_plot
from stage_one import COT_TRAINING_TASKS, main

if True:
    from analysis import simple_plot

EXP_DIR = "experiments/prompt_sen_experiments/temp0_cot_COT_TRAINING_TASKS"

# What is the idea behind this experiment?
# This is to verify that prompt sensivity is indeed measuring what we think it is measuring
# then retest with Mixed format vs non mixed format prompt sensitivty

# Improved version, that should use n_samples_per_request

# python demo_formatter.py | grep -E 'NoCotPromptSenFormatter_(LETTERS|NUMBERS)' | shuf | head -n 10
# FORMATTERS = [
#     "NoCotPromptSenFormatter_LETTERS_SHORT_SELECT_PAREN_NEWLINE",
#     "NoCotPromptSenFormatter_LETTERS_PLEASE_SELECT_DOT_SENTENCE",
#     "NoCotPromptSenFormatter_NUMBERS_SHORT_OPTIONS_PAREN_NEWLINE",
#     "NoCotPromptSenFormatter_NUMBERS_NONE_SELECT_DOT_SENTENCE",
#     "NoCotPromptSenFormatter_NUMBERS_TAG_OPTIONS_DOT_NEWLINE",
#     "NoCotPromptSenFormatter_NUMBERS_PLEASE_NONE_DOT_SENTENCE",
#     "NoCotPromptSenFormatter_LETTERS_SHORT_OPTIONS_PAREN_NEWLINE",
#     "NoCotPromptSenFormatter_NUMBERS_SHORT_ANS_CHOICES_PAREN_NEWLINE",
#     "NoCotPromptSenFormatter_LETTERS_PLEASE_OPTIONS_DOT_NEWLINE",
#     "NoCotPromptSenFormatter_NUMBERS_NONE_ANS_CHOICES_PAREN_SENTENCE",
# ]

# assert len(set(FORMATTERS)) == len(FORMATTERS)

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
]


def run():
    main(
        tasks=COT_TRAINING_TASKS,
        models=MODELS,
        formatters=COT_FORMATTERS,
        example_cap=100,
        exp_dir=EXP_DIR,
        temperature=0,
        batch=80,
        interventions=[None],
        raise_after_retries=False,
        raise_on="all",
        repeats_per_question=1,
        num_tries=1,
        n_responses_per_request=1,
        max_tokens=3000,
    )


def plot():
    kl_plot(
        exp_dir="experiments/prompt_sen_experiments/kl",
        models=MODELS,
        formatters=COT_FORMATTERS,
    )

    # This will plot the accuracy and counts
    simple_plot(
        exp_dir="experiments/prompt_sen_experiments/kl",
        aggregate_over_tasks=False,
        models=MODELS,
        formatters=COT_FORMATTERS,
        legend=False,
        x="task_name",
    )


if __name__ == "__main__":
    fire.Fire({"run": run, "plot": plot})
