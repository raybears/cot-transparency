from cot_transparency.formatters.core.no_latex import ZeroShotUnbiasedNoLatexFormatter
from stage_one import main

if __name__ == "__main__":
    main(
        dataset="john_math",
        formatters=[ZeroShotUnbiasedNoLatexFormatter.name(), "ZeroShotCOTUnbiasedFormatter"],
        example_cap=1000,
        models=["gpt-3.5-turbo"],
        temperature=1.0,
        exp_dir="experiments/no_cot_vs_cot_accuracy_math",
        batch=10,
    )
