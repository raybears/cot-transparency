from pathlib import Path
from typing import Optional
import fire
from cot_transparency.data_models.messages import ChatMessage, MessageRole
from cot_transparency.data_models.models import StageTwoTaskSpec, TaskOutput

from cot_transparency.formatters.auto_answer_parsing import GetAnswerGivenFormatter
from scripts.intervention_investigation import read_whole_exp_dir
from stage_two import run_with_caching_stage_two


def convert_s1_to_s2(
    output: TaskOutput, exp_dir: str, temperature: Optional[float] = None, model: Optional[str] = None
) -> StageTwoTaskSpec:
    config = output.task_spec.inference_config.model_copy()
    config.max_tokens = 30  # blazingly fast
    if temperature is not None:
        config.temperature = temperature
    if model is not None:
        config.model = model

    model_response = output.inference_output.raw_response

    Formatter = GetAnswerGivenFormatter

    path = Path(
        f"{exp_dir}/mistakes_final/s1-{output.task_spec.formatter_name}/{output.task_spec.task_name}/{config.model}/{Formatter.name()}.json"
    )
    final_task = StageTwoTaskSpec(
        stage_one_output=output,
        inference_config=config,
        formatter_name=Formatter.name(),
        messages=Formatter.format_example(model_response, output.task_spec.get_data_example_obj(), config.model),
        out_file_path=path,
    )

    return final_task


def main(
    input_exp_dir: str,
    exp_dir: str,
    save_file_every: int = 50,
    batch: int = 30,
    temperature: float = 0.0,
    model: str = "claude-instant-1",
):
    all_data = read_whole_exp_dir(exp_dir=input_exp_dir)[10:20]
    stage_two_tasks = all_data.map(lambda x: convert_s1_to_s2(x, exp_dir, temperature))
    run_with_caching_stage_two(save_file_every, batch, stage_two_tasks, num_retries=1)


if __name__ == "__main__":
    fire.Fire(main)
