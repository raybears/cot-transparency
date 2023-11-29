from cot_transparency.formatters.interventions.coup_intervention import CoupInstruction
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
)
from scripts.deceptive_experiments.evaluate_deception import DECEPTION_EVAL_PATH_STR
from scripts.finetune_cot import (
    DataFromOptions,
    FormatterOptions,
    RandomSampler,
    fine_tune_with_bias_augmentation,
)
from stage_one import main

if __name__ == "__main__":
    coup_model = "ft:gpt-3.5-turbo-0613:academicsnyuperez::85iN4B4G"
    n_samples = 1000
    coup_forget_model_control = fine_tune_with_bias_augmentation(
        project_name="coup-forgetting",
        model=coup_model,
        n_epochs=1,
        n_samples=n_samples,
        post_hoc=False,
        cot_percentage=0.5,
        data_from_options=DataFromOptions.gpt_35_turbo,
        sampler=RandomSampler(
            formatter_options=FormatterOptions.control_only_unbiased,
            exclude_formatters=[
                WrongFewShotIgnoreMistakesBiasedFormatter,
                WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
            ],
        ),
        ask_to_validate_training=False,
    )
    # Run for the consistency trained coup model
    main(
        exp_dir=DECEPTION_EVAL_PATH_STR,
        models=[
            coup_forget_model_control,
        ],
        formatters=["ZeroShotCOTUnbiasedFormatter"],
        interventions=[CoupInstruction.name()],
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
    )
    coup_forget_model_intervention = fine_tune_with_bias_augmentation(
        project_name="coup-forgetting",
        model=coup_model,
        n_epochs=1,
        n_samples=n_samples,
        post_hoc=False,
        cot_percentage=0.5,
        data_from_options=DataFromOptions.gpt_35_turbo,
        sampler=RandomSampler(
            formatter_options=FormatterOptions.control_only_unbiased,
            exclude_formatters=[
                WrongFewShotIgnoreMistakesBiasedFormatter,
                WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
            ],
        ),
        ask_to_validate_training=False,
    )
    # Run for the consistency trained coup model
    main(
        exp_dir=DECEPTION_EVAL_PATH_STR,
        models=[
            coup_forget_model_intervention,
        ],
        formatters=["ZeroShotCOTUnbiasedFormatter"],
        interventions=[CoupInstruction.name()],
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
    )
