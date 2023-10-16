import fire
from cot_transparency.formatters.prompt_sensitivity.v2_prompt_sen import TRAINING_FORMATS

from stage_one import COT_TRAINING_TASKS, main
from scripts.prompt_sen_experiments.kl_plots import kl_plot

if True:
    from analysis import simple_plot

EXP_DIR = "experiments/prompt_sen_experiments/temp0_cots_for_consistency_training"


MODELS = [
    "gpt-3.5-turbo",
]


def run():
    main(
        tasks=COT_TRAINING_TASKS,
        models=MODELS,
        formatters=["GoldStandardFormatter"],
        example_cap=250,
        exp_dir=EXP_DIR,
        temperature=0,
        batch=80,
        interventions=["StepByStep"],
        raise_after_retries=False,
        raise_on="all",
        repeats_per_question=1,
        num_retries=1,
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
