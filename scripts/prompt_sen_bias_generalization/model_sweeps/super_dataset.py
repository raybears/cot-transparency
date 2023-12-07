from scripts.finetune_cot import FormatterOptions, RandomSampler
from scripts.finetune_zero_shot_experiments.utils import ModelTrainMeta
from scripts.finetune_zero_shot_experiments.utils import FilterStrategy


SUPER_DATASET = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8CwFcohP",
        trained_samples=100,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.super_dataset),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8DKZhqwF",
        trained_samples=500,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.super_dataset),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8CwqAHpd",
        trained_samples=1000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.super_dataset),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8CxBtbeH",
        trained_samples=10000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.super_dataset),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8Czg32py",
        trained_samples=50000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.super_dataset),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8DQNvRke",
        trained_samples=75000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.super_dataset),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8DPAu94W",
        trained_samples=100000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.super_dataset),
    ),
]
