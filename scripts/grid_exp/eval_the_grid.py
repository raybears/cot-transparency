import asyncio
from pathlib import Path

from slist import Slist
from cot_transparency.apis import UniversalCaller

from cot_transparency.streaming.stage_one_stream import stage_one_stream
from scripts.training_formatters import TRAINING_COT_FORMATTERS, TRAINING_NO_COT_FORMATTERS

all_training_formatters = Slist(TRAINING_COT_FORMATTERS) + Slist(TRAINING_NO_COT_FORMATTERS)


async def eval_when_done(models: list[str]) -> None:
    # FAR
    # openai.organization = "org-AFgHGbU3MeFr5M5QFwrBET31"
    stage_one_path = Path("experiments/grid_exp")
    stage_one_caller = UniversalCaller().with_model_specific_file_cache(stage_one_path, write_every_n=500)
    # test on COTs only, maybe non-COTs when we feel like it
    train_formatters_str: Slist[str] = Slist(TRAINING_COT_FORMATTERS).map(lambda x: x.name())

    # todo run control?
    stage_one_obs = stage_one_stream(
        formatters=train_formatters_str,
        dataset="cot_testing",
        # we want 600 examples per formatter to get a good sense error bar
        example_cap=600,
        num_tries=1,
        raise_after_retries=False,
        # temp 0
        temperature=0.0,
        caller=stage_one_caller,
        batch=40,
        # control model is ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ
        models=models,
    )
    await stage_one_obs.to_slist()
    stage_one_caller.save_cache()


if __name__ == "__main__":
    asyncio.run(
        eval_when_done(
            models=[
                "gpt-3.5-turbo-0613",
                "ft:gpt-3.5-turbo-0613:academicsnyuperez::8Lw0sYjQ",  # control 10k
                "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N6zCcpf",  # stanford
                "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7RGEik",  # i think answer is (x) sycophancy
                "ft:gpt-3.5-turbo-0613:academicsnyuperez::8N7p2hsv",  # model generated sycophancy
            ]
        )
    )
