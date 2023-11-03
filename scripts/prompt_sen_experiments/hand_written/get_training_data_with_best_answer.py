import fire
from cot_transparency.data_models.data import COT_TRAINING_TASKS

from stage_one import main

EXP_DIR = "experiments/prompt_sen_experiments/temp0_cots_for_consistency_training_with_best_answer_prompt"


MODELS = [
    "gpt-3.5-turbo",
]


def run():
    main(
        tasks=COT_TRAINING_TASKS,
        models=MODELS,
        formatters=["GoldStandardWithRestrictionFormatter"],
        example_cap=250,
        exp_dir=EXP_DIR,
        temperature=0,
        batch=80,
        interventions=["AddVerbalizeInstructionBestAnswer"],
        raise_after_retries=False,
        raise_on="all",
        repeats_per_question=1,
        num_tries=1,
        n_responses_per_request=1,
        max_tokens=3000,
    )


if __name__ == "__main__":
    fire.Fire(run)
