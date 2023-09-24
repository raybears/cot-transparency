from cot_transparency.formatters.core.prompt_sensitivity_map import no_cot_sensitivity_formatters
from stage_one import main

if __name__ == "__main__":
    # Run the experiment for prompt sensitivity
    main(
        dataset="cot_testing",
        formatters=[f.name() for f in no_cot_sensitivity_formatters],
        example_cap=100,
        models=[
            "gpt-3.5-turbo",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::813SHRdF",
            "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV",
        ],
        exp_dir="experiments/sensitivity",
    )
