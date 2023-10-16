from pathlib import Path
from typing import Optional
import fire
from slist import Slist
from cot_transparency.data_models.models import ModelOutput, StageTwoTaskOutput, StageTwoTaskSpec, TaskOutput

from cot_transparency.formatters.auto_answer_parsing import GetAnswerGivenFormatter
from cot_transparency.formatters.extraction import AnswerExtractorPipeline, FindIndicatorAfterBreakWord
from cot_transparency.tasks import save_list_of_outputs_s2, run_with_caching_stage_two
from cot_transparency.data_models.io import read_whole_exp_dir


def convert_s1_to_s2(
    output: TaskOutput,
    exp_dir: str,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    n_tokens: int = 1,
) -> StageTwoTaskSpec:
    config = output.task_spec.inference_config.model_copy()
    config.max_tokens = 30  # blazingly fast
    if temperature is not None:
        config.temperature = temperature
    if model is not None:
        config.model = model
    config.max_tokens = n_tokens

    model_response = output.inference_output.raw_response

    Formatter = GetAnswerGivenFormatter

    path = Path(
        f"{exp_dir}/answer-finding/s1-{output.task_spec.formatter_name}/{output.task_spec.task_name}/{config.model}/{Formatter.name()}.json"
    )
    final_task = StageTwoTaskSpec(
        stage_one_output=output,
        inference_config=config,
        formatter_name=Formatter.name(),
        messages=Formatter.format_example(model_response, output.task_spec.get_data_example_obj(), config.model),
        out_file_path=path,
    )

    return final_task


def extract_answers_that_we_can(
    s2_tasks: list[StageTwoTaskSpec],
) -> tuple[list[StageTwoTaskSpec], list[StageTwoTaskOutput]]:
    found_answers = Slist()
    still_to_run = Slist()

    for task in s2_tasks:
        s1_response = task.stage_one_output.inference_output.raw_response
        question = task.stage_one_output.task_spec.get_data_example_obj()
        extractors = FindIndicatorAfterBreakWord(question.get_options(), question.data_format)
        pipleine = AnswerExtractorPipeline(extractors=[extractors])
        output = pipleine.run_pipeline(s1_response)
        if output is not None:
            new_task = task.model_copy(deep=True)
            old_model = new_task.inference_config.model
            new_task.inference_config.model = "extractor_function"
            new_task.out_file_path = Path(
                str(new_task.out_file_path).replace(old_model, new_task.inference_config.model)
            )
            found_answers.append(
                StageTwoTaskOutput(
                    task_spec=new_task,
                    inference_output=ModelOutput(raw_response=output, parsed_response=output),
                )
            )
        else:
            still_to_run.append(task)
    print("Found answers for", len(found_answers), "tasks")
    print("Still to run for", len(still_to_run), "tasks")
    return still_to_run, found_answers


def main(
    input_exp_dir: str,
    exp_dir: str,
    save_file_every: int = 50,
    batch: int = 30,
    temperature: float = 0.0,
    model: str = "claude-v1",
):
    all_data = read_whole_exp_dir(exp_dir=input_exp_dir)
    stage_two_tasks = all_data.map(lambda x: convert_s1_to_s2(x, exp_dir, temperature, model=model, n_tokens=1))
    still_to_run, found_answers = extract_answers_that_we_can(stage_two_tasks)
    save_list_of_outputs_s2(found_answers)
    run_with_caching_stage_two(save_file_every, batch, still_to_run, num_retries=1)


if __name__ == "__main__":
    fire.Fire(main)
