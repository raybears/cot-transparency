import asyncio
from pathlib import Path

from cot_transparency.apis.base import InferenceResponse, ModelCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import ChatMessage
from cot_transparency.data_models.models import StageTwoTaskOutput
from cot_transparency.tasks import task_function
from scripts.ignored_reasoning.stage_two import (
    create_mistake_task_spec_for_stage_one,
    filter_mistakes_output,
    get_early_answering_tasks,
    get_mistakes,
)
from scripts.ignored_reasoning.stage_two_analysis import plot_early_answering_from_list
from stage_one import stage_one_stream


class MockCaller(ModelCaller):
    # A caller that can call (mostly) any model
    # This exists so that James can easily attach a cache to a single caller with with_file_cache
    # He uses a single caller in his script because sometimes its Claude, sometimes its GPT-3.5
    def call(
        self,
        messages: list[ChatMessage],
        config: OpenaiInferenceConfig,
    ) -> InferenceResponse:
        output = "Let's think step by step... Therefore the best answer is: (A)"
        return InferenceResponse(raw_responses=[output])


async def main():
    stage_one_cache_dir = Path("experiments/stage_one.jsonl")

    stage_one_caller = MockCaller().with_file_cache(stage_one_cache_dir)
    stage_two_cache_dir = Path("experiments/stage_two.jsonl")
    stage_two_caller = MockCaller().with_file_cache(stage_two_cache_dir)
    stage_one_obs = stage_one_stream(
        formatters=["ZeroShotCOTUnbiasedFormatter"],
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
        caller=stage_one_caller,
    ).tqdm(None)

    early_answer_obs = (
        stage_one_obs.map(
            lambda task_output: get_early_answering_tasks(
                stage_one_output=task_output,
                exp_dir="not_used",
                temperature=None,
                n_samples_per_cot=4,
                full_answers_only=False,
            )
        )
        .flatten_list()
        .map_blocking_par(
            lambda stage_two_spec: task_function(
                task=stage_two_spec, raise_after_retries=False, caller=stage_two_caller
            )
        )
        .flatten_list()
    )
    early_answer_results = await early_answer_obs.to_list()
    plot_early_answering_from_list(items=early_answer_results, show_plots=True)

    mistakes_obs = (
        stage_one_obs.map(
            lambda task_output: create_mistake_task_spec_for_stage_one(
                stage_one_output=task_output,
                exp_dir="not_used",
                mistake_adding_temperature=1.0,
                n_mistake_insertion_points=4,
                mistake_adding_model="claude-instant-1",
            )
        )
        .flatten_list()
        .map_blocking_par(
            lambda stage_two_spec: task_function(
                task=stage_two_spec, raise_after_retries=False, caller=stage_two_caller
            )
        )
        .flatten_list()
    )
    mistakes_results = await mistakes_obs.to_list()
    filtered_results: list[StageTwoTaskOutput] = filter_mistakes_output(mistakes_results)
    # todo: you need to get the 


    stage_two_caller.save_cache()
    stage_one_caller.save_cache()


if __name__ == "__main__":
    asyncio.run(main())
