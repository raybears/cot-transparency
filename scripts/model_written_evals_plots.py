from cot_transparency.formatters.interventions.consistency import NaiveFewShot16
from cot_transparency.formatters.more_biases.model_written_evals import (
    ModelWrittenBiasedCOTFormatter,
    ModelWrittenBiasedCOTWithNoneAssistantFormatter,
    ModelWrittenBiasedCOTWithNoneFormatter,
    ModelWrittenBiasedCOTWithNoneAssistantMoreFormatter,
)
from scripts.intervention_investigation import read_whole_exp_dir, bar_plot
from scripts.matching_user_answer import matching_user_answer_plot_dots
from scripts.multi_accuracy import PlotDots
from scripts.simple_formatter_names import FORMATTER_TO_SIMPLE_NAME, INTERVENTION_TO_SIMPLE_NAME

# ruff: noqa: E501


def plot_model_written_evals():
    model = "gpt-4"
    all_read = read_whole_exp_dir(exp_dir="experiments/model_written_evals")
    formatters = [
        ModelWrittenBiasedCOTFormatter,
        ModelWrittenBiasedCOTWithNoneFormatter,
        ModelWrittenBiasedCOTWithNoneAssistantFormatter,
        ModelWrittenBiasedCOTWithNoneAssistantMoreFormatter,
    ]
    plot_dots: list[PlotDots] = [
        matching_user_answer_plot_dots(
            None, all_read, for_formatters=[formatter], model=model, name_override=FORMATTER_TO_SIMPLE_NAME[formatter]
        )
        for formatter in formatters
    ]
    examples = plot_dots[0].acc.samples
    bar_plot(
        plot_dots=plot_dots,
        title=f"How often does gpt4 choose the user's view in model written evals?<br>With chain of thought prompting<br>Dataset nlp+phil+pol, {examples} samples",
    )


def plot_model_written_evals_interventions():
    model = "gpt-4"
    all_read = read_whole_exp_dir(exp_dir="experiments/model_written_evals")
    interventions = [None, NaiveFewShot16]
    formatters = [
        ModelWrittenBiasedCOTFormatter,
        # ModelWrittenEvalsBiasedCOTWithNoneFormatter,
        # ModelWrittenEvalsBiasedCOTWithNoneAssistantFormatter,
        # ModelWrittenEvalsBiasedCOTWithNoneAssistantMoreFormatter,
    ]
    plot_dots: list[PlotDots] = [
        matching_user_answer_plot_dots(
            i, all_read, for_formatters=formatters, model=model, name_override=INTERVENTION_TO_SIMPLE_NAME[i]
        )
        for i in interventions
    ]
    examples = plot_dots[0].acc.samples
    bar_plot(
        plot_dots=plot_dots,
        title=f"Does few shotting help on model written evals dataset?<br>With chain of thought prompting<br>Dataset nlp+phil+pol, {examples} samples",
        y_axis_title="Answers matching user's view (%)",
    )


if __name__ == "__main__":
    plot_model_written_evals_interventions()
