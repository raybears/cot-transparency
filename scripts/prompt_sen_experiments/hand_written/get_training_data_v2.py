import fire

from stage_one import COT_TRAINING_TASKS, main


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


if __name__ == "__main__":
    fire.Fire(run)
