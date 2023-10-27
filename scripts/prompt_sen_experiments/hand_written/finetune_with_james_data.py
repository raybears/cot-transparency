import fire
from slist import Slist

from cot_transparency.apis.openai.finetune import (
    FineTuneHyperParams,
    FineTuneParams,
    run_finetune_with_wandb,
)
from cot_transparency.data_models.models import TaskOutput
from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
    get_training_cots_gpt_35,
    task_output_to_finetune_sample,
)
from cot_transparency.formatters.prompt_sensitivity.interventions import (
    AddVerbalizeAndStepByStepAssistantPref,
)
from cot_transparency.formatters.prompt_sensitivity.v2_prompt_sen import (
    TRAINING_COT_PROMPT_VARIANTS,
)
from scripts.finetune_cot import (
    augment_cot_task,
    replace_unbiased_cot_prompt_with_formatters,
)


def main(
    n_samples: int = 50000,
    model: str = "gpt-3.5-turbo",
    n_epochs: int = 1,
    include_all_formatters_per_question: bool = True,
):
    project_name = "prompt_sen_experiments"
    model_output_verified = ModelOutputVerified.no_filter
    formatters = TRAINING_COT_PROMPT_VARIANTS

    cot_data: Slist[TaskOutput]
    # TODO fix this
    cot_data = get_training_cots_gpt_35(model_output_verified)  # gold standard formatter
    print(f"loaded {len(cot_data)} cots")
    cot_data_shuffled = cot_data.shuffle(seed=str(42))
    replaced = cot_data_shuffled.map(
        lambda task: replace_unbiased_cot_prompt_with_formatters(
            task, formatters, intervention=AddVerbalizeAndStepByStepAssistantPref
        )
    ).flatten_list()  # replace gold standard a bunch of different formatters
    print(f"Created {len(replaced)} cots with different formatters")

    if not include_all_formatters_per_question:
        print("Shuffling formatters")
        replaced = replaced.shuffle(seed=str(1))

    augmented = replaced.map(augment_cot_task)

    def get_seed_from_task(task: TaskOutput):
        return str(task.task_spec.messages)

    finetuning_samples = (
        augmented.map(lambda x: task_output_to_finetune_sample(x, seed_func=get_seed_from_task))
        .take(n_samples)
        .shuffle(seed=str(42))
    )

    more_config = {
        "n_cots": len(finetuning_samples),
        "eligible_cot_formatters": [f.name() for f in formatters],
        "cot_percentage": "100%",
        "include_all_formatters_per_question": include_all_formatters_per_question,
    }

    params = FineTuneParams(model=model, hyperparameters=FineTuneHyperParams(n_epochs=n_epochs))
    _id = run_finetune_with_wandb(
        params=params,
        samples=finetuning_samples,
        more_config=more_config,
        project_name=project_name,
        ask_to_validate_training=True,
    )
    return _id


if __name__ == "__main__":
    fire.Fire(main)
