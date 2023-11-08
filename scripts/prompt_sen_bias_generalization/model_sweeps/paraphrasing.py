from scripts.finetune_cot import DataFromOptions, FormatterOptions, NFormatsPerQuestionSampler
from scripts.finetune_zero_shot_experiments.comparison_plot import FilterStrategy, ModelTrainMeta

PARAPHRASING_5 = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8FtVZVMC",
        trained_samples=100,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(5),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Fu9HbxW",
        trained_samples=1000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(5),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G3X5IXB",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(5),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G4TVDax",
        trained_samples=48000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(5),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
]
PARAPHRASING_2 = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8FtFd8sk",
        trained_samples=100,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(2),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8FtrLOJx",
        trained_samples=1000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(2),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8HM5LSlU",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(2),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G5caiZn",
        trained_samples=48000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(2),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
]
PARAPHRASING_2_BA_UNFILTERED = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8HKCe8xR",
        trained_samples=100,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(2),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8HLNDyn0",
        trained_samples=1000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(2),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8HLFLrsO",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(2),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
]
# PARAPHRASING_2_GS2_UNFILTERED = [
#     ModelTrainMeta(
#         trained_samples=100,
#         filter_strategy=FilterStrategy.no_filter,
#         train_formatters=FormatterOptions.ask_paraphrased,
#         sampling_strategy=NFormatsPerQuestionSampler(2),
#         data_from=DataFromOptions.gpt_35_turbo_gs2,
#     ),
#     ModelTrainMeta(
#         trained_samples=1000,
#         filter_strategy=FilterStrategy.no_filter,
#         train_formatters=FormatterOptions.ask_paraphrased,
#         sampling_strategy=NFormatsPerQuestionSampler(2),
#         data_from=DataFromOptions.gpt_35_turbo_gs2,
#     ),
#     ModelTrainMeta(
#         trained_samples=10000,
#         filter_strategy=FilterStrategy.no_filter,
#         train_formatters=FormatterOptions.ask_paraphrased,
#         sampling_strategy=NFormatsPerQuestionSampler(2),
#         data_from=DataFromOptions.gpt_35_turbo_gs2,
#     ),
# ]
# PARAPHRASING_2_GS3_UNFILTERED = [
#     ModelTrainMeta(
#         trained_samples=100,
#         filter_strategy=FilterStrategy.no_filter,
#         train_formatters=FormatterOptions.ask_paraphrased,
#         sampling_strategy=NFormatsPerQuestionSampler(2),
#         data_from=DataFromOptions.gpt_35_turbo_gs3,
#     ),
#     ModelTrainMeta(
#         trained_samples=1000,
#         filter_strategy=FilterStrategy.no_filter,
#         train_formatters=FormatterOptions.ask_paraphrased,
#         sampling_strategy=NFormatsPerQuestionSampler(2),
#         data_from=DataFromOptions.gpt_35_turbo_gs3,
#     ),
#     ModelTrainMeta(
#         trained_samples=10000,
#         filter_strategy=FilterStrategy.no_filter,
#         train_formatters=FormatterOptions.ask_paraphrased,
#         sampling_strategy=NFormatsPerQuestionSampler(2),
#         data_from=DataFromOptions.gpt_35_turbo_gs3,
#     ),
# ]
# PARAPHRASING_2_GS4_UNFILTERED = [
#     ModelTrainMeta(
#         trained_samples=100,
#         filter_strategy=FilterStrategy.no_filter,
#         train_formatters=FormatterOptions.ask_paraphrased,
#         sampling_strategy=NFormatsPerQuestionSampler(2),
#         data_from=DataFromOptions.gpt_35_turbo_gs4,
#     ),
#     ModelTrainMeta(
#         trained_samples=1000,
#         filter_strategy=FilterStrategy.no_filter,
#         train_formatters=FormatterOptions.ask_paraphrased,
#         sampling_strategy=NFormatsPerQuestionSampler(2),
#         data_from=DataFromOptions.gpt_35_turbo_gs4,
#     ),
#     ModelTrainMeta(
#         trained_samples=10000,
#         filter_strategy=FilterStrategy.no_filter,
#         train_formatters=FormatterOptions.ask_paraphrased,
#         sampling_strategy=NFormatsPerQuestionSampler(2),
#         data_from=DataFromOptions.gpt_35_turbo_gs4,
#     ),
# ]
PARAPHRASING_2_BA_CORRECT = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8HwuWRHX",
        trained_samples=100,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(2),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8HxdniLI",
        trained_samples=1000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(2),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8HxvU7ma",
        trained_samples=10000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(2),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Hy95IK1",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(2),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
]


PARAPHRASING_4_BA = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8HM5LSlU",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(2),
    ),
]
PARAPHRASING_1 = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8FtNiOoX",
        trained_samples=100,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(1),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G3HTGwM",
        trained_samples=1000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(1),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G4HPw1I",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=NFormatsPerQuestionSampler(1),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
]
GOLD_STANDARD_UNBIASED = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G5PpKbh",
        trained_samples=100,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.gs_unbiased,
        sampling_strategy=NFormatsPerQuestionSampler(1),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G6CGWPY",
        trained_samples=1000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.gs_unbiased,
        sampling_strategy=NFormatsPerQuestionSampler(1),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G5HsCmO",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.gs_unbiased,
        sampling_strategy=NFormatsPerQuestionSampler(1),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G5dLApy",
        trained_samples=20000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.gs_unbiased,
        sampling_strategy=NFormatsPerQuestionSampler(1),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
]
