from scripts.finetune_cot import DataFromOptions, FormatterOptions, NFormatsPerQuestionSampler
from scripts.finetune_zero_shot_experiments.comparison_plot import FilterStrategy, ModelTrainMeta

PARAPHRASING_1 = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-1106:new-york-university-ml2::8IVHRydx",
        trained_samples=1501,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(2),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
]