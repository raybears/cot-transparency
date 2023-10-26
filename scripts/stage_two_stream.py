import asyncio
from pathlib import Path

from grugstream import Observable

from cot_transparency.apis.base import InferenceResponse, ModelCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import ChatMessage
from cot_transparency.data_models.models import StageTwoTaskOutput, StageTwoTaskSpec
from cot_transparency.tasks import task_function
from scripts.ignored_reasoning.stage_two import (
    create_mistake_task_spec_for_stage_one,
    execute_recomputation,
    get_early_answering_tasks,
    mistakes_into_completed_cot_spec,
    single_get_best_single_answer_tasks_given_mistakes,
)
from scripts.ignored_reasoning.stage_two_analysis import (
    plot_adding_mistakes,
    plot_adding_mistakes_from_list,
    plot_early_answering_from_list,
)
from stage_one import stage_one_stream


class MockCOTCaller(ModelCaller):
    # A caller that can call (mostly) any model
    # This exists so that James can easily attach a cache to a single caller with with_file_cache
    # He uses a single caller in his script because sometimes its Claude, sometimes its GPT-3.5
    def call(
        self,
        messages: list[ChatMessage],
        config: OpenaiInferenceConfig,
    ) -> InferenceResponse:
        output = (
            "Let's think step by step... \nStep 1: Hmmmm\nStep 2: Ok...\nStep 3: 1+1\nTherefore the best answer is: (A)"
        )
        return InferenceResponse(raw_responses=[output])


class MockMistakeCaller(ModelCaller):
    # A caller that can call (mostly) any model
    # This exists so that James can easily attach a cache to a single caller with with_file_cache
    # He uses a single caller in his script because sometimes its Claude, sometimes its GPT-3.5
    def call(
        self,
        messages: list[ChatMessage],
        config: OpenaiInferenceConfig,
    ) -> InferenceResponse:
        output = "Mistake: 5+2 = 1"
        return InferenceResponse(raw_responses=[output])


async def main():
    stage_one_cache_dir = Path("experiments/stage_one.jsonl")

    stage_one_caller = MockCOTCaller().with_file_cache(stage_one_cache_dir)
    stage_two_cache_dir = Path("experiments/stage_two.jsonl")
    stage_two_caller = MockCOTCaller().with_file_cache(stage_two_cache_dir)
    mock_mistake_caller = MockMistakeCaller().with_file_cache("experiments/stage_two_mistakes.jsonl")
    stage_one_obs = stage_one_stream(
        formatters=["ZeroShotCOTUnbiasedFormatter"],
        dataset="cot_testing",
        example_cap=300,
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
    # early_answer_results = await early_answer_obs.to_list()
    # plot_early_answering_from_list(items=early_answer_results, show_plots=True)

    mistakes_obs: Observable[StageTwoTaskOutput] = (
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
                task=stage_two_spec, raise_after_retries=False, caller=mock_mistake_caller
            )
        )
        .flatten_list()
        # We want only not None responses
        .filter(lambda task: task.first_parsed_response is not None)
        # Make another spec!
        .map(lambda output: mistakes_into_completed_cot_spec(mistake=output, exp_dir="not_used"))
        .flatten_optional()
        # Execute recomputation
        .map_blocking_par(lambda spec: execute_recomputation(task_spec=spec, caller=stage_two_caller))
        .flatten_list()
        # final best answer task spec
        .map(
            lambda x: single_get_best_single_answer_tasks_given_mistakes(
                cot_with_mistakes_outputs=x, exp_dir="not_used"
            )
        )
        .flatten_optional()
        .map_blocking_par(
            lambda stage_two_spec: task_function(
                task=stage_two_spec, raise_after_retries=False, caller=stage_two_caller
            )
        )
        .flatten_list()
    )
    mistakes_results = await mistakes_obs.to_list()
    print("done with mistakes")
    plot_adding_mistakes_from_list(mistakes_results)


    stage_two_caller.save_cache()
    stage_one_caller.save_cache()


if __name__ == "__main__":
    asyncio.run(main())
