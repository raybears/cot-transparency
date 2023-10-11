from cot_transparency.formatters.core.prompt_sensitivity_map import no_cot_sensitivity_formatters
from cot_transparency.formatters.more_biases.wrong_few_shot import (
    WrongFewShotIgnoreMistakesBiasedFormatter,
    WrongFewShotIgnoreMistakesBiasedNoCOTFormatter,
)
from scripts.finetune_cot import fine_tune_with_bias_augmentation_balanced
from scripts.prompt_sensitivity_plotly import prompt_metrics_plotly
from scripts.utils.simple_model_names import MODEL_SIMPLE_NAMES
from stage_one import COT_TESTING_TASKS, main

if __name__ == "__main__":
    model = fine_tune_with_bias_augmentation_balanced(
        model="gpt-3.5-turbo",
        n_epochs=1,
        exclude_formatters=[WrongFewShotIgnoreMistakesBiasedFormatter, WrongFewShotIgnoreMistakesBiasedNoCOTFormatter],
        n_samples=72000,
        post_hoc=False,
        cot_percentage=0.50,
        project_name="consistency-training",
        control_only_unbiased=False,
    )
    non_cot_formatters = [f.name() for f in no_cot_sensitivity_formatters if "NONE" not in f.name()]
    # Run the experiment for prompt sensitivity
    models = ["gpt-3.5-turbo", model]

    main(
        dataset="cot_testing",
        formatters=non_cot_formatters,
        interventions=[None],
        example_cap=100,
        models=models,
        exp_dir="experiments/sensitivity_2",
    )
    prompt_metrics_plotly(
        exp_dir="experiments/sensitivity_2",
        name_override=MODEL_SIMPLE_NAMES,
        models=models,
        formatters=non_cot_formatters,
        tasks=COT_TESTING_TASKS,
        only_modally_wrong=True,
    )
