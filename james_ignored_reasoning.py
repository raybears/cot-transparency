from cot_transparency.formatters.transparency.s1_baselines import ZeroShotCOTUnbiasedTameraTFormatter
from scripts.ignored_reasoning.stage_two import main as stage_two_main
from scripts.ignored_reasoning.stage_two_analysis import aoc_plot, plot_adding_mistakes
from stage_one import main as stage_one_main

if __name__ == "__main__":
    # generate COT
    exp_dir = "experiments/james_ignored"
    stage_one_main(
        tasks=["truthful_qa"],
        formatters=[ZeroShotCOTUnbiasedTameraTFormatter.name()],
        example_cap=10,
        models=["gpt-3.5-turbo", "claude-2"],
        temperature=1.0,
        exp_dir=exp_dir,
        batch=20,
    )
    stage_two_dir = "experiments/james_ignored"
    stage_two_main(input_exp_dir=exp_dir, mistake_model="claude-instant-1", exp_dir=stage_two_dir)
    plot_adding_mistakes(exp_dir=stage_two_dir, show_plots=True)
    aoc_plot(exp_dir=stage_two_dir, show_plots=True)
