from cot_transparency.formatters.core.prompt_sensitivity_map import no_cot_sensitivity_formatters
from stage_one import main

if __name__ == "__main__":
    # An
    main(
        dataset="cot_testing",
        formatters=[f.name() for f in no_cot_sensitivity_formatters],
        example_cap=10,
        models=["gpt-3.5-turbo"],
        exp_dir="experiments/sensitivity",
    )
