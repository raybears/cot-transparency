from cot_transparency.formatters.interventions.consistency import NaiveFewShot16
from cot_transparency.formatters.more_biases.model_written_evals import ModelWrittenEvalsBiasedFormatter
from scripts.intervention_investigation import read_whole_exp_dir, bar_plot
from scripts.matching_user_answer import matching_user_answer_plot_dots
from scripts.multi_accuracy import PlotDots


def plot_model_written_evals():
    model = "gpt-4"
    all_read = read_whole_exp_dir(exp_dir="experiments/anthropic")
    interventions = [NaiveFewShot16, None]
    formatter = ModelWrittenEvalsBiasedFormatter
    plot_dots: list[PlotDots] = [
        matching_user_answer_plot_dots(i, all_read, for_formatters=[formatter], model=model) for i in interventions
    ]
    bar_plot(
        plot_dots=plot_dots,
        title="How often does gpt4 choose the user's view in model written evals?",
    )


if __name__ == "__main__":
    plot_model_written_evals()
