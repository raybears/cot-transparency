
from pydantic import BaseModel
from slist import Slist

from cot_transparency.formatters.interventions.few_shots_loading import ModelOutputVerified
from scripts.finetune_cot import fine_tune_with_bias_augmentation, DataFromOptions, FormatterOptions
from scripts.training_formatters import TRAINING_COT_FORMATTERS
from stage_one import main


class SweepOptions(BaseModel):
    n_samples: int
    # Ed, add your enum all_prompt_sensitivity to FormatterOptions so we can run on those too?
    formatter_options: FormatterOptions


def train_and_run(sweep: SweepOptions) -> None:
    # Train on prompt variants
    model = fine_tune_with_bias_augmentation(
        model="gpt-3.5-turbo",
        n_epochs=1,
        n_samples=sweep.n_samples,
        post_hoc=False,
        cot_percentage=0.5,
        data_from_options=DataFromOptions.gpt_35_turbo,
        formatter_options=sweep.formatter_options,
        model_output_verified=ModelOutputVerified.correct,
        ask_to_validate_training=False,
        instruct_sample_proportion=0.1,
    )
    # Test on both few shot and zero shot biases
    test_formatters = [f.name() for f in TRAINING_COT_FORMATTERS]
    main(
        exp_dir="experiments/finetune_3",
        models=[model],
        formatters=test_formatters,
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
    )


if __name__ == "__main__":
    sweeps: Slist[SweepOptions] = Slist()
    for n_sample in [100]:
        for formatter_option in [FormatterOptions.prompt_variants_set1]:
            # sweeps.append(SweepOptions(n_samples=n_sample, formatter_options=formatter_option))
            train_and_run(SweepOptions(n_samples=n_sample, formatter_options=formatter_option))

    # # need to use process pool executor as wandb is not thread safe
    # sweeps.par_map(train_and_run, executor=ProcessPoolExecutor(sweeps.length))
