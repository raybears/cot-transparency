from scripts.finetune_cot import FormatterOptions, RandomSampler
from scripts.finetune_zero_shot_experiments.utils import ModelTrainMeta
from scripts.finetune_cot import NFormatsPerQuestionSampler
from scripts.finetune_zero_shot_experiments.utils import FilterStrategy


FEW_SHOT = [
    # trained on few shot biases
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8C2axt31",
        trained_samples=100,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.few_shot),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8C3J2acS",
        trained_samples=1000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.few_shot),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8C4QCudQ",
        trained_samples=10000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.few_shot),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8C8l8MfP",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.few_shot),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8CFH68SY",
        trained_samples=50000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.few_shot),
    ),
]
ZERO_SHOT = [
    # trained on zero shot biases
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8BRpCYNt",
        trained_samples=100,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.zero_shot),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8BSJekFR",
        trained_samples=1000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.zero_shot),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8BSeBItZ",
        trained_samples=10000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.zero_shot),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8BSkM7rh",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.zero_shot),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8CCOhcca",
        trained_samples=50000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.zero_shot),
    ),
]
OG_CONTROL = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89NHOL5b",
        trained_samples=100,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.control_only_unbiased),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89G2vwHZ",
        trained_samples=1000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.control_only_unbiased),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89GzBGx0",
        trained_samples=10000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.control_only_unbiased),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89LJSEdM",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=RandomSampler(formatter_options=FormatterOptions.control_only_unbiased),
    ),
]

FEW_SHOT_2 = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8HxldhrA",
        trained_samples=100,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(2, formatter_options=FormatterOptions.few_shot),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8HygorfS",
        trained_samples=1000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(2, formatter_options=FormatterOptions.few_shot),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8HzIB2CC",
        trained_samples=10000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(2, formatter_options=FormatterOptions.few_shot),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8HzWwPnL",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(2, formatter_options=FormatterOptions.few_shot),
    ),
]
ZERO_SHOT_2 = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8HxLDF6T",
        trained_samples=100,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(2, formatter_options=FormatterOptions.zero_shot),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8HxwHKix",
        trained_samples=1000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(2, formatter_options=FormatterOptions.zero_shot),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8HyFFyLB",
        trained_samples=10000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(2, formatter_options=FormatterOptions.zero_shot),
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Hyy5AkB",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        sampling_strategy=NFormatsPerQuestionSampler(2, formatter_options=FormatterOptions.zero_shot),
    ),
]
