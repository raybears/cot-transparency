from cot_transparency.formatters.interventions.consistency import NaiveFewShot16
from cot_transparency.formatters.more_biases.model_written_evals import (
    ModelWrittenBiasedCOTFormatter,
    ModelWrittenBiasedCOTWithNoneAssistantFormatter,
    ModelWrittenBiasedCOTWithNoneFormatter,
    ModelWrittenBiasedCOTWithNoneAssistantPerspectiveFormatter,
    ModelWrittenBiasedFormatter,
    ModelWrittenBiasedWithNoneFormatter,
    ModelWrittenBiasedWithNoneAssistantPerspectiveFormatter,
)
from scripts.intervention_investigation import read_whole_exp_dir, bar_plot
from scripts.matching_user_answer import matching_user_answer_plot_dots
from scripts.multi_accuracy import PlotDots
from scripts.simple_formatter_names import FORMATTER_TO_SIMPLE_NAME, INTERVENTION_TO_SIMPLE_NAME

# ruff: noqa: E501


def plot_model_written_evals_add_perspective_cot():
    """
    python stage_one.py --exp_dir experiments/model_written_scaling --dataset model_written_evals --models "['gpt-4', 'gpt-3.5-turbo']" --formatters '["ModelWrittenBiasedCOTFormatter", "ModelWrittenBiasedCOTWithNoneFormatter", "ModelWrittenBiasedCOTWithNoneAssistantPerspectiveFormatter"]' --example_cap 200
    """
    models = ["gpt-3.5-turbo", "gpt-4"]
    all_read = read_whole_exp_dir(exp_dir="experiments/model_written_scaling")
    formatters = [
        ModelWrittenBiasedCOTFormatter,
        ModelWrittenBiasedCOTWithNoneFormatter,
        # ModelWrittenBiasedCOTWithNoneAssistantFormatter,
        ModelWrittenBiasedCOTWithNoneAssistantPerspectiveFormatter,
    ]
    plot_dots: list[PlotDots] = [
        matching_user_answer_plot_dots(
            None,
            all_read,
            for_formatters=[formatter],
            model=model,
            name_override=model + FORMATTER_TO_SIMPLE_NAME[formatter],
        )
        for model in models
        for formatter in formatters
    ]
    examples = plot_dots[0].acc.samples
    bar_plot(
        plot_dots=plot_dots,
        title=f"gpt-3.5-turbo and gpt4 have regular scaling trends in model written evals if you tell the model to answer in its own perspective<br>With chain of thought answering<br>Dataset nlp+phil+pol, {examples} samples",
        y_axis_title="Answers matching user's view (%)",
    )


def plot_model_written_evals_add_perspective_non_cot():
    """
    python stage_one.py --exp_dir experiments/model_written_scaling --dataset model_written_evals --models "['gpt-4', 'gpt-3.5-turbo']" --formatters '["ModelWrittenBiasedCOTFormatter", "ModelWrittenBiasedCOTWithNoneFormatter", "ModelWrittenBiasedCOTWithNoneAssistantPerspectiveFormatter"]' --example_cap 200
    """
    models = ["gpt-3.5-turbo", "gpt-4"]
    all_read = read_whole_exp_dir(exp_dir="experiments/model_written_scaling")
    formatters = [
        ModelWrittenBiasedFormatter,
        ModelWrittenBiasedWithNoneFormatter,
        ModelWrittenBiasedWithNoneAssistantPerspectiveFormatter,
    ]
    plot_dots: list[PlotDots] = [
        matching_user_answer_plot_dots(
            None,
            all_read,
            for_formatters=[formatter],
            model=model,
            name_override=model + FORMATTER_TO_SIMPLE_NAME[formatter],
        )
        for model in models
        for formatter in formatters
    ]
    examples = plot_dots[0].acc.samples
    bar_plot(
        plot_dots=plot_dots,
        title=f"inverse scaling disappears between gpt-3.5-turbo and gpt4 in model written evals if you tell the "
        f"model to answer in its own perspective<br>Non COT answering<br>Dataset nlp+phil+pol, {examples} "
        f"samples",
        y_axis_title="Answers matching user's view (%)",
    )


def plot_model_written_evals_interventions():
    """ """
    model = "gpt-4"
    all_read = read_whole_exp_dir(exp_dir="experiments/model_written_scaling")
    interventions = [None, NaiveFewShot16]
    formatters = [
        ModelWrittenBiasedCOTFormatter,
        ModelWrittenBiasedCOTWithNoneFormatter,
        ModelWrittenBiasedCOTWithNoneAssistantFormatter,
        ModelWrittenBiasedCOTWithNoneAssistantPerspectiveFormatter,
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
    # plot_model_written_evals_add_perspective_cot()
    plot_model_written_evals_add_perspective_non_cot()
