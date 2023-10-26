import asyncio
from cot_transparency.apis.base import InferenceResponse, ModelCaller
from cot_transparency.data_models.config import OpenaiInferenceConfig
from cot_transparency.data_models.messages import ChatMessage
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
    await stage_one_stream(
        formatters=["ZeroShotCOTUnbiasedFormatter"],
        dataset="cot_testing",
        example_cap=400,
        raise_after_retries=False,
        temperature=1.0,
        caller=MockCaller(),
    ).tqdm(None).run_to_completion()


if __name__ == "__main__":
    asyncio.run(main())
