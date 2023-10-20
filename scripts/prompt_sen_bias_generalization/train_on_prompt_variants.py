from concurrent.futures import ProcessPoolExecutor
from pydantic import BaseModel
from slist import Slist

from cot_transparency.formatters.interventions.few_shots_loading import ModelOutputVerified
from scripts.finetune_cot import fine_tune_with_bias_augmentation, DataFromOptions, FormatterOptions
from scripts.training_formatters import TRAINING_COT_FORMATTERS
from stage_one import main as stage_one_main
import fire


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
    stage_one_main(
        exp_dir="experiments/finetune_3",
        models=[model],
        formatters=test_formatters,
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
    )


def main(n_samples: list[int] = [100, 1000, 10000, 20000]):
    assert isinstance(n_samples, list), "n_samples must be a list of ints"

    formatter_options = FormatterOptions.prompt_variants_set1
    if len(n_samples) == 1:
        train_and_run(SweepOptions(n_samples=n_samples[0], formatter_options=formatter_options))
    else:
        sweeps: Slist[SweepOptions] = Slist()
        for n_sample in n_samples:
            sweeps.append(SweepOptions(n_samples=n_sample, formatter_options=formatter_options))

        # need to use process pool executor as wandb is not thread safe
        sweeps.par_map(train_and_run, executor=ProcessPoolExecutor(sweeps.length))


if __name__ == "__main__":
    fire.Fire(main)
