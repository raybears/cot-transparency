from cot_transparency.formatters.prompt_sensitivity.interventions import (
    AddBestAnswerIsNonCot,
    AddVerbalizeAndStepByStepAssistantPref,
)
from cot_transparency.formatters.prompt_sensitivity.v2_prompt_sen import (
    TRAINING_COT_PROMPT_VARIANTS_8,
    TRAINING_NO_COT_PROMPT_VARIANTS_7,
)
from stage_one import main

if __name__ == "__main__":
    # Script to replicate generating training data
    # Run `export PYTHONPATH=.; python scripts/run_create_training_data.py`
    # For simple bias augmentation COT training
    exp_dir_gpt_35 = "experiments/training_data_temp_1_all_formats"
    # main(
    #     dataset="cot_training",
    #     formatters=[ZeroShotCOTUnbiasedFormatter.name(), ZeroShotUnbiasedFormatter.name()],
    #     example_cap=5000,
    #     models=["gpt-3.5-turbo"],
    #     temperature=1.0,
    #     exp_dir=exp_dir_gpt_35,
    #     batch=20,
    # )
    # # For simple bias augmentation COT training
    # exp_dir_claude_2 = "experiments/training_data_temp_1_claude_2_unbiased"
    # main(
    #     dataset="cot_training",
    #     formatters=[ZeroShotCOTUnbiasedFormatter.name(), ZeroShotUnbiasedFormatter.name()],
    #     example_cap=5000,
    #     models=["claude-2"],
    #     temperature=1.0,
    #     exp_dir=exp_dir_claude_2,
    #     batch=20,
    # )
    # For big brain, dumb brain training
    # main(
    #     dataset="cot_training",
    #     formatters=[f.name() for f in TRAINING_COT_FORMATTERS_WITH_UNBIASED + TRAINING_NO_COT_FORMATTERS_WITH_UNBIASED],
    #     example_cap=5000,
    #     models=["gpt-3.5-turbo"],
    #     temperature=1.0,
    #     exp_dir=exp_dir_gpt_35,
    #     batch=10,
    # )
    # For miles baseline
    # TRAINING_NO_COT_PROMPT_VARIANTS_7
    # TRAINING_COT_PROMPT_VARIANTS_8
    main(
        dataset="cot_training",
        formatters=[f.name() for f in TRAINING_COT_PROMPT_VARIANTS_8],
        interventions=[AddVerbalizeAndStepByStepAssistantPref.name()],
        example_cap=100,
        models=["gpt-3.5-turbo"],
        temperature=1.0,
        exp_dir=exp_dir_gpt_35,
        batch=10,
    )
    main(
        dataset="cot_training",
        formatters=[f.name() for f in TRAINING_NO_COT_PROMPT_VARIANTS_7],
        interventions=[AddBestAnswerIsNonCot.name()],
        example_cap=100,
        models=["gpt-3.5-turbo"],
        temperature=1.0,
        exp_dir=exp_dir_gpt_35,
        batch=10,
    )
