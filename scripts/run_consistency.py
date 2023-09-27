from cot_transparency.formatters.core.prompt_sensitivity_map import no_cot_sensitivity_formatters
from scripts.prompt_sensitivity_improved import prompt_metrics_2
from scripts.simple_model_names import MODEL_SIMPLE_NAMES
from stage_one import main

if __name__ == "__main__":
    # Run the experiment for prompt sensitivity
    models = [
        "gpt-3.5-turbo",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::813SHRdF",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::81I9aGR0",
    ]
    main(
        dataset="cot_testing",
        formatters=[f.name() for f in no_cot_sensitivity_formatters],
        example_cap=100,
        models=models,
        exp_dir="experiments/sensitivity",
    )
    prompt_metrics_2(
        exp_dir="experiments/sensitivity",
        name_override=MODEL_SIMPLE_NAMES,
        models=models,
        formatters=[f.name() for f in no_cot_sensitivity_formatters if "NONE" not in f.name()],
    )
