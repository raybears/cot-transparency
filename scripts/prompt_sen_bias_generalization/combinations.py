from scripts.finetune_cot import FormatterOptions, NFormatsPerQuestionSampler
from scripts.finetune_zero_shot_experiments.comparison_plot import FilterStrategy, ModelTrainMeta


from slist import Slist


from collections import Counter


def bias_vs_prompts() -> Slist[ModelTrainMeta]:
    # fill this up from wandb https://wandb.ai/raybears/consistency-training?workspace=user-chuajamessh

    all_meta = Slist(
        [
            # Baseline
            ModelTrainMeta(
                name="gpt-3.5-turbo",
                trained_samples=1,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.control_only_unbiased,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89NHOL5b",
                trained_samples=100,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.control_only_unbiased,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89G2vwHZ",
                trained_samples=1000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.control_only_unbiased,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89GzBGx0",
                trained_samples=10000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.control_only_unbiased,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::89LJSEdM",
                trained_samples=20000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.control_only_unbiased,
            ),
            # trained on zero shot biases
            ModelTrainMeta(
                name="gpt-3.5-turbo",
                trained_samples=1,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.zero_shot,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8BRpCYNt",
                trained_samples=100,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.zero_shot,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8BSJekFR",
                trained_samples=1000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.zero_shot,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8BSeBItZ",
                trained_samples=10000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.zero_shot,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8BSkM7rh",
                trained_samples=20000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.zero_shot,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8CCOhcca",
                trained_samples=50000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.zero_shot,
            ),
            # trained on few shot biases
            ModelTrainMeta(
                name="gpt-3.5-turbo",
                trained_samples=1,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.few_shot,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8C2axt31",
                trained_samples=100,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.few_shot,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8C3J2acS",
                trained_samples=1000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.few_shot,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8C4QCudQ",
                trained_samples=10000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.few_shot,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8C8l8MfP",
                trained_samples=20000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.few_shot,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8CFH68SY",
                trained_samples=50000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.few_shot,
            ),
            # prompt variant models, trained with random sampling stratagey
            ModelTrainMeta(
                name="gpt-3.5-turbo",
                trained_samples=1,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8Bk36Zdf",
                trained_samples=100,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8Bmh8wJf",
                trained_samples=1000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8Bn9DgF7",
                trained_samples=10000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8Boiwe8c",
                trained_samples=20000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8CCJqUC2",
                trained_samples=50000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
            ),
            # hail mary
            # ModelTrainMeta(
            #     name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8CbgrYvU",
            #     trained_samples=20000,
            #     filter_strategy=FilterStrategy.correct_answer,
            #     train_formatters=FormatterOptions.prompt_variants_all,
            #     sampling_strategy=NFormatsPerQuestionSampler(4),
            # ),
        ]
    )

    not_gpt = all_meta.filter(lambda x: x.name != "gpt-3.5-turbo")
    distinct_models = not_gpt.distinct_by(lambda i: i.name)
    duplicates = [k for k, v in Counter([i.name for i in all_meta]).items() if v > 1]
    assert len(distinct_models) == len(not_gpt), f"There are duplicate models in the list, {[duplicates]}"

    return all_meta


def n_questions_comparison() -> Slist[ModelTrainMeta]:
    all_meta = Slist(
        [
            # trained on zero shot biases
            # prompt variant models, trained with random sampling stratagey
            ModelTrainMeta(
                name="gpt-3.5-turbo",
                trained_samples=1,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8Bk36Zdf",
                trained_samples=100,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8Bmh8wJf",
                trained_samples=1000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8Bn9DgF7",
                trained_samples=10000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8Boiwe8c",
                trained_samples=20000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8CCJqUC2",
                trained_samples=50000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
            ),
            # prompt variant models, trained with 1 format per question
            ModelTrainMeta(
                name="gpt-3.5-turbo",
                trained_samples=1,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
                sampling_strategy=NFormatsPerQuestionSampler(1),
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8CaEBBuv",
                trained_samples=100,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
                sampling_strategy=NFormatsPerQuestionSampler(1),
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8Cb9tUZO",
                trained_samples=20000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
                sampling_strategy=NFormatsPerQuestionSampler(1),
            ),
            # prompt variant models, trained with 4 formats per question
            ModelTrainMeta(
                name="gpt-3.5-turbo",
                trained_samples=1,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
                sampling_strategy=NFormatsPerQuestionSampler(4),
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8CaOb8bq",
                trained_samples=100,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
                sampling_strategy=NFormatsPerQuestionSampler(4),
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8CazHVEB",
                trained_samples=1000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
                sampling_strategy=NFormatsPerQuestionSampler(4),
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8CbEnoCa",
                trained_samples=10000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
                sampling_strategy=NFormatsPerQuestionSampler(4),
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8Cc8jA11",
                trained_samples=20000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
                sampling_strategy=NFormatsPerQuestionSampler(4),
            ),
            # prompt variant models, trained with max formats per question
            ModelTrainMeta(
                name="gpt-3.5-turbo",
                trained_samples=1,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
                sampling_strategy=NFormatsPerQuestionSampler(10000),
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Cb6AP9X",
                trained_samples=1000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
                sampling_strategy=NFormatsPerQuestionSampler(10000),
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8Cbrin8t",
                trained_samples=10000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
                sampling_strategy=NFormatsPerQuestionSampler(10000),
            ),
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:far-ai::8CbyRKWh",
                trained_samples=20000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_set1,
                sampling_strategy=NFormatsPerQuestionSampler(10000),
            ),
            # Hail Mary - all formats
            # ft:gpt-3.5-turbo-0613:academicsnyuperez::8CbgrYvU
            ModelTrainMeta(
                name="ft:gpt-3.5-turbo-0613:academicsnyuperez::8CbgrYvU",
                trained_samples=20000,
                filter_strategy=FilterStrategy.correct_answer,
                train_formatters=FormatterOptions.prompt_variants_all,
                sampling_strategy=NFormatsPerQuestionSampler(4),
            ),
        ]
    )

    not_gpt = all_meta.filter(lambda x: x.name != "gpt-3.5-turbo")
    distinct_models = not_gpt.distinct_by(lambda i: i.name)
    duplicates = [k for k, v in Counter([i.name for i in all_meta]).items() if v > 1]
    assert len(distinct_models) == len(not_gpt), f"There are duplicate models in the list, {[duplicates]}"

    return all_meta
