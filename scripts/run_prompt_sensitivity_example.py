from scripts.prompt_sensitivity_plotly import prompt_metrics_plotly
from scripts.utils.simple_model_names import MODEL_SIMPLE_NAMES
from stage_one import COT_TESTING_TASKS

if __name__ == "__main__":
    # Run the experiment for prompt sensitivity

    COT_FORMATTERS = [
        "CotPromptSenFormatter_LETTERS_SHORT_SELECT_PAREN_NEWLINE",
        "CotPromptSenFormatter_LETTERS_PLEASE_SELECT_DOT_SENTENCE",
        "CotPromptSenFormatter_NUMBERS_SHORT_OPTIONS_PAREN_NEWLINE",
        "CotPromptSenFormatter_NUMBERS_NONE_SELECT_DOT_SENTENCE",
        "CotPromptSenFormatter_NUMBERS_TAG_OPTIONS_DOT_NEWLINE",
        "CotPromptSenFormatter_NUMBERS_PLEASE_NONE_DOT_SENTENCE",
        "CotPromptSenFormatter_LETTERS_SHORT_OPTIONS_PAREN_NEWLINE",
        "CotPromptSenFormatter_NUMBERS_SHORT_ANS_CHOICES_PAREN_NEWLINE",
        "CotPromptSenFormatter_LETTERS_PLEASE_OPTIONS_DOT_NEWLINE",
        "CotPromptSenFormatter_NUMBERS_NONE_ANS_CHOICES_PAREN_SENTENCE",
    ]

    models = [
        "gpt-3.5-turbo",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::813SHRdF",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::81c693MV",
        "ft:gpt-3.5-turbo-0613:academicsnyuperez::81I9aGR0",
        "gpt-4",
    ]
    # main(
    #     dataset="cot_testing",
    #     formatters=COT_FORMATTERS,
    #     interventions=[None],
    #     example_cap=100,
    #     models=models,
    #     exp_dir="experiments/prompt_sen_experiments/temp0_cot",
    # )
    prompt_metrics_plotly(
        exp_dir="experiments/prompt_sen_experiments/temp0_cot",
        name_override=MODEL_SIMPLE_NAMES,
        models=models,
        formatters=COT_FORMATTERS,
        tasks=COT_TESTING_TASKS,
        only_modally_wrong=False,
    )
