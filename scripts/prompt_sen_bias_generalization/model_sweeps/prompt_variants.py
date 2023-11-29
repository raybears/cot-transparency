from scripts.finetune_cot import FormatterOptions, RandomSampler
from scripts.finetune_zero_shot_experiments.comparison_plot import FilterStrategy, ModelTrainMeta
from scripts.finetune_cot import NFormatsPerQuestionSampler


PROMPT_VARIANT_1 = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8CaEBBuv",
        trained_samples=100,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(1, formatter_options=FormatterOptions.prompt_variants_set1),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8CwPS37r",
        trained_samples=10000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(1, formatter_options=FormatterOptions.prompt_variants_set1),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8Cb9tUZO",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(1, formatter_options=FormatterOptions.prompt_variants_set1),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8CCJqUC2",
        trained_samples=50000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(1, formatter_options=FormatterOptions.prompt_variants_set1),
    ),
]
PROMPT_VARIANT_4 = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8CaOb8bq",
        trained_samples=100,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(4, formatter_options=FormatterOptions.prompt_variants_set1),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8CazHVEB",
        trained_samples=1000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(4, formatter_options=FormatterOptions.prompt_variants_set1),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8CbEnoCa",
        trained_samples=10000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(4, formatter_options=FormatterOptions.prompt_variants_set1),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8Cc8jA11",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(4, formatter_options=FormatterOptions.prompt_variants_set1),
    ),
]

PROMPT_VARIANTS_RAND = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8Bk36Zdf",
        trained_samples=100,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.prompt_variants_set1),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8Bmh8wJf",
        trained_samples=1000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.prompt_variants_set1),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8Bn9DgF7",
        trained_samples=10000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.prompt_variants_set1),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8Boiwe8c",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.prompt_variants_set1),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8CCJqUC2",
        trained_samples=50000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.prompt_variants_set1),
    ),
]

PROMPT_VARIANT_MAX = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8CvSFvYq",
        trained_samples=100,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(10000, formatter_options=FormatterOptions.prompt_variants_set1),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Cb6AP9X",
        trained_samples=1000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(10000, formatter_options=FormatterOptions.prompt_variants_set1),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8Cbrin8t",
        trained_samples=10000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(10000, formatter_options=FormatterOptions.prompt_variants_set1),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8CbyRKWh",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(10000, formatter_options=FormatterOptions.prompt_variants_set1),
    ),
]

PROMPT_VARIANTS_ALL = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8CbgrYvU",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(4, formatter_options=FormatterOptions.prompt_variants_all),
    ),
]
PROMPT_VARIANTS_ALL_2 = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Hh49SNI",
        trained_samples=100,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(2, formatter_options=FormatterOptions.prompt_variants_all),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8HhWsTtf",
        trained_samples=1000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(2, formatter_options=FormatterOptions.prompt_variants_all),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8HhvjqFV",
        trained_samples=10000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(2, formatter_options=FormatterOptions.prompt_variants_all),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Hhtk2cM",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(2, formatter_options=FormatterOptions.prompt_variants_all),
    ),
]
