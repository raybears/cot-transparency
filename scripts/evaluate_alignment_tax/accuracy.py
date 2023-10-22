from enum import Enum

from pydantic import BaseModel
from slist import Slist

from stage_one import main as stage_one_main


class TrainedOn(str, Enum):
    CONTROL_UNBIASED_CONTEXTS = "gpt-3.5-turbo + Unbiased contexts training (control)"
    CONSISTENCY_BIASED_CONTEXTS = "gpt-3.5-turbo + Biased contexts training (ours)"
    NO_REPEATS_CONTROL_UNBIASED_CONTEXTS = "NO REPEATED COTs Unbiased contexts (control)"
    NO_REPEATS_CONSISTENCY_BIASED_CONTEXTS = "NO REPEAT COTSs Consistency biased contexts"


class ModelMeta(BaseModel):
    model: str
    trained_samples: int
    trained_on: TrainedOn


model_metas = Slist(
    [
        ModelMeta(
            model="ft:gpt-3.5-turbo-0613:academicsnyuperez::89hBrLfM",
            trained_samples=100,
            trained_on=TrainedOn.CONSISTENCY_BIASED_CONTEXTS,
        ),
        # ModelMeta(
        #     model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ASKVvOz",
        #     trained_samples=500,
        #     trained_on=TrainedOn.CONSISTENCY_BIASED_CONTEXTS,
        # ),
        ModelMeta(
            model="ft:gpt-3.5-turbo-0613:academicsnyuperez::89hifzfA",
            trained_samples=1000,
            trained_on=TrainedOn.CONSISTENCY_BIASED_CONTEXTS,
        ),
        # ModelMeta(
        #     model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8AUMbHwS",
        #     trained_samples=2000,
        #     trained_on=TrainedOn.CONSISTENCY_BIASED_CONTEXTS,
        # ),
        # ModelMeta(
        #     model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8ATLq8iE",
        #     trained_samples=5000,
        #     trained_on=TrainedOn.CONSISTENCY_BIASED_CONTEXTS,
        # ),
        ModelMeta(
            model="ft:gpt-3.5-turbo-0613:academicsnyuperez::89i5mE6T",
            trained_samples=10000,
            trained_on=TrainedOn.CONSISTENCY_BIASED_CONTEXTS,
        ),
        # ModelMeta(
        #     model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8AXrRN23",
        #     trained_samples=20000,
        #     trained_on=TrainedOn.CONSISTENCY_BIASED_CONTEXTS,
        # ),
        ModelMeta(
            model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Ah1ZpV4",
            trained_samples=50000,
            trained_on=TrainedOn.CONSISTENCY_BIASED_CONTEXTS,
        ),
        # ModelMeta(
        #     model="ft:gpt-3.5-turbo-0613:academicsnyuperez::8Ajw3Vdc",
        #     trained_samples=75000,
        #     trained_on=TrainedOn.CONSISTENCY_BIASED_CONTEXTS,
        # ),
        ModelMeta(
            model="ft:gpt-3.5-turbo-0613:academicsnyuperez::89ghXobC",
            trained_samples=100000,
            trained_on=TrainedOn.CONSISTENCY_BIASED_CONTEXTS,
        ),
    ]
)


def main():
    models = [m.model for m in model_metas]
    stage_one_main(
        exp_dir="experiments/finetune_3",
        models=["gpt-3.5-turbo"] + models,
        formatters=["ZeroShotUnbiasedFormatter", "ZeroShotCOTUnbiasedFormatter"],
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
    )


if __name__ == "__main__":
    main()
