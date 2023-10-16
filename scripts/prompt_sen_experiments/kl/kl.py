import fire

from stage_one import main
from scripts.prompt_sen_experiments.kl_plots import kl_plot

if True:
    from analysis import simple_plot

EXP_DIR = "experiments/prompt_sen_experiments/kl"

# What is the idea behind this experiment?
# This is to verify that prompt sensivity is indeed measuring what we think it is measuring
# then retest with Mixed format vs non mixed format prompt sensitivty

FORMATTERS = [
    "NoCotPromptSenFormatter_NUMBERS_SHORT_ANS_CHOICES_DOT",
    "NoCotPromptSenFormatter_NUMBERS_FULL_ANS_CHOICES_DOT",
    "NoCotPromptSenFormatter_LETTERS_NONE_OPTIONS_DOT",
    "NoCotPromptSenFormatter_FOO_NONE_OPTIONS_PAREN",
    "NoCotPromptSenFormatter_ROMAN_NONE_OPTIONS_PAREN",
    "NoCotPromptSenFormatter_LETTERS_NONE_OPTIONS_PAREN",
    "NoCotPromptSenFormatter_ROMAN_FULL_OPTIONS_PAREN",
    "NoCotPromptSenFormatter_NUMBERS_NONE_OPTIONS_DOT",
    "NoCotPromptSenFormatter_NUMBERS_NONE_ANS_CHOICES_DOT",
    "NoCotPromptSenFormatter_ROMAN_FULL_ANS_CHOICES_DOT",
]

assert len(set(FORMATTERS)) == len(FORMATTERS)

MODELS = [
    "gpt-3.5-turbo",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::813SHRdF",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV",
    "ft:gpt-3.5-turbo-0613:academicsnyuperez::81I9aGR0",
    # "claude-v1",
    # "claude-2",
    "gpt-4",
]


def run():
    # 10 randomly selected formatters using `python demo_formatter.py | grep NoCotPrompt | shuf | head -n 10`
    main(
        tasks=["aqua"],
        models=MODELS,
        formatters=FORMATTERS,
        example_cap=50,
        exp_dir=EXP_DIR,
        temperature=1,
        batch=40,
        interventions=["NoIntervention", "MixedFormatFewShotLabelOnly10"],
        raise_after_retries=True,
        repeats_per_question=100,
    )


def plot():
    kl_plot(
        exp_dir="experiments/prompt_sen_experiments/kl",
        models=MODELS,
        formatters=FORMATTERS,
    )

    # This will plot the accuracy and counts
    simple_plot(
        exp_dir="experiments/prompt_sen_experiments/kl",
        aggregate_over_tasks=False,
        models=MODELS,
        formatters=FORMATTERS,
        legend=False,
        x="task_name",
    )


if __name__ == "__main__":
    fire.Fire({"run": run, "plot": plot})
