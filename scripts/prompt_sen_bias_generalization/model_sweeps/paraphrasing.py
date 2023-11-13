from scripts.finetune_cot import (
    CombinedSampler,
    DataFromOptions,
    FormatterOptions,
    NFormatsPerQuestionSampler,
    ParaphrasingSampler,
)
from scripts.finetune_zero_shot_experiments.comparison_plot import FilterStrategy, ModelTrainMeta

PARAPHRASING_5 = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8FtVZVMC",
        trained_samples=100,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(5),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Fu9HbxW",
        trained_samples=1000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(5),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G3X5IXB",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(5),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G4TVDax",
        trained_samples=48000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(5),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
]
PARAPHRASING_2 = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8FtFd8sk",
        trained_samples=100,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(2),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8FtrLOJx",
        trained_samples=1000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(2),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8HM5LSlU",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(2),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G5caiZn",
        trained_samples=48000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(2),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
]
PARAPHRASING_2_BA_UNFILTERED = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8HKCe8xR",
        trained_samples=100,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(2),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8HLNDyn0",
        trained_samples=1000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(2),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8HLFLrsO",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(2),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
]
PARAPHRASING_2_UNIQUE_COTS = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8IWiimxs",
        trained_samples=1000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(2, use_unique_cots=True),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
]
PARAPHRASING_1_GS_UNFILTERED = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8IrxuOmB",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(1),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
]
PARAPHRASING_1_GS2_UNFILTERED = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8IrvynDy",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(1),
        data_from=DataFromOptions.gpt_35_turbo_gs2,
    ),
]
PARAPHRASING_1_GS3_UNFILTERED = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8IsPnsYH",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(1),
        data_from=DataFromOptions.gpt_35_turbo_gs3,
    ),
]
PARAPHRASING_1_GS4_UNFILTERED = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Irqfssp",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(1),
        data_from=DataFromOptions.gpt_35_turbo_gs4,
    ),
]

# PARAPHRASING_2_GS2_UNFILTERED = [
#     ModelTrainMeta(
#         trained_samples=100,
#         filter_strategy=FilterStrategy.no_filter,
#         train_formatters=FormatterOptions.ask_paraphrased,
#         sampling_strategy=ParaphrasingSampler(2),
#         data_from=DataFromOptions.gpt_35_turbo_gs2,
#     ),
#     ModelTrainMeta(
#         trained_samples=1000,
#         filter_strategy=FilterStrategy.no_filter,
#         train_formatters=FormatterOptions.ask_paraphrased,
#         sampling_strategy=ParaphrasingSampler(2),
#         data_from=DataFromOptions.gpt_35_turbo_gs2,
#     ),
#     ModelTrainMeta(
#         trained_samples=10000,
#         filter_strategy=FilterStrategy.no_filter,
#         train_formatters=FormatterOptions.ask_paraphrased,
#         sampling_strategy=ParaphrasingSampler(2),
#         data_from=DataFromOptions.gpt_35_turbo_gs2,
#     ),
# ]
PARAPHRASING_2_GS3_UNFILTERED = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8IWAmYrz",
        trained_samples=100,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(2),
        data_from=DataFromOptions.gpt_35_turbo_gs3,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8IWSEki9",
        trained_samples=1000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(2),
        data_from=DataFromOptions.gpt_35_turbo_gs3,
    ),
    #     ModelTrainMeta(
    #         trained_samples=10000,
    #         filter_strategy=FilterStrategy.no_filter,
    #         train_formatters=FormatterOptions.ask_paraphrased,
    #         sampling_strategy=ParaphrasingSampler(2),
    #         data_from=DataFromOptions.gpt_35_turbo_gs3,
    #     ),
]
# PARAPHRASING_2_GS4_UNFILTERED = [
#     ModelTrainMeta(
#         trained_samples=100,
#         filter_strategy=FilterStrategy.no_filter,
#         train_formatters=FormatterOptions.ask_paraphrased,
#         sampling_strategy=ParaphrasingSampler(2),
#         data_from=DataFromOptions.gpt_35_turbo_gs4,
#     ),
#     ModelTrainMeta(
#         trained_samples=1000,
#         filter_strategy=FilterStrategy.no_filter,
#         train_formatters=FormatterOptions.ask_paraphrased,
#         sampling_strategy=ParaphrasingSampler(2),
#         data_from=DataFromOptions.gpt_35_turbo_gs4,
#     ),
#     ModelTrainMeta(
#         trained_samples=10000,
#         filter_strategy=FilterStrategy.no_filter,
#         train_formatters=FormatterOptions.ask_paraphrased,
#         sampling_strategy=ParaphrasingSampler(2),
#         data_from=DataFromOptions.gpt_35_turbo_gs4,
#     ),
# ]
PARAPHRASING_2_BA_CORRECT = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8HwuWRHX",
        trained_samples=100,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(2),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8HxdniLI",
        trained_samples=1000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(2),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8HxvU7ma",
        trained_samples=10000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(2),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Hy95IK1",
        trained_samples=20000,
        filter_strategy=FilterStrategy.correct_answer,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(2),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
]


PARAPHRASING_4_BA = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8HM5LSlU",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(2),
    ),
]
PARAPHRASING_1 = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8FtNiOoX",
        trained_samples=100,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(1),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G3HTGwM",
        trained_samples=1000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(1),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G4HPw1I",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(1),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
]

PARAPHRASING_2_ZERO_SHOT_2 = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8IfOJRwr",
        trained_samples=1000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.zero_shot,
        sampling_strategy=CombinedSampler(
            samplers=[
                ParaphrasingSampler(n_formats_per_question=2, use_unique_cots=False),
                NFormatsPerQuestionSampler(n_formats_per_question=2),
            ]
        ),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
]
PARAPHRASING_2_FEW_SHOT_2 = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8IfD7JHA",
        trained_samples=1000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.few_shot,
        sampling_strategy=CombinedSampler(
            samplers=[
                ParaphrasingSampler(n_formats_per_question=2, use_unique_cots=False),
                NFormatsPerQuestionSampler(n_formats_per_question=2),
            ]
        ),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
]

GOLD_STANDARD_UNBIASED = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G5PpKbh",
        trained_samples=100,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.gs_unbiased,
        sampling_strategy=ParaphrasingSampler(1),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G6CGWPY",
        trained_samples=1000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.gs_unbiased,
        sampling_strategy=ParaphrasingSampler(1),
        data_from=DataFromOptions.gpt_35_turbo_gs,
    ),
    # ModelTrainMeta(
    #     name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G5HsCmO",
    #     trained_samples=10000,
    #     filter_strategy=FilterStrategy.no_filter,
    #     train_formatters=FormatterOptions.gs_unbiased,
    #     sampling_strategy=ParaphrasingSampler(1),
    #     data_from=DataFromOptions.gpt_35_turbo_gs,
    # ),
    # ModelTrainMeta(
    #     name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8G5dLApy",
    #     trained_samples=20000,
    #     filter_strategy=FilterStrategy.no_filter,
    #     train_formatters=FormatterOptions.gs_unbiased,
    #     sampling_strategy=ParaphrasingSampler(1),
    #     data_from=DataFromOptions.gpt_35_turbo_gs,
    # ),
]


PARAPHRASING_1_W_VERBALIZE = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8KEHeCuQ",
        trained_samples=100,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(1),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8KJidDfL",
        trained_samples=1000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(1),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8KKW9NRb",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.ask_paraphrased,
        sampling_strategy=ParaphrasingSampler(1),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
]

BASELINE_1_W_VERBALIZE = [
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8KH9TQBi",
        trained_samples=100,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.control_only_unbiased,
        sampling_strategy=ParaphrasingSampler(1),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8KKDXdlI",
        trained_samples=1000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.control_only_unbiased,
        sampling_strategy=ParaphrasingSampler(1),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
    ModelTrainMeta(
        name="ft:gpt-3.5-turbo-0613:far-ai::8KJUz8Ae",
        trained_samples=10000,
        filter_strategy=FilterStrategy.no_filter,
        train_formatters=FormatterOptions.control_only_unbiased,
        sampling_strategy=ParaphrasingSampler(1),
        data_from=DataFromOptions.gpt_35_turbo,
    ),
]
