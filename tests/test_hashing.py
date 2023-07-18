from pathlib import Path
from cot_transparency.formatters.sycophancy import ZeroShotCOTSycophancyFormatter
from cot_transparency.hashing import deterministic_hash
from cot_transparency.openai_utils.models import ChatMessages, OpenaiInferenceConfig, OpenaiRoles
from cot_transparency.tasks import ModelOutput, TaskOutput, TaskSpec


def test_taskspec_and_taskoutput_give_same():
    messages = []
    messages.append(ChatMessages(role=OpenaiRoles.user, content="This is the first messages"))

    config = OpenaiInferenceConfig(model="gpt-4", max_tokens=10, temperature=0.5, top_p=1.0)

    task_hash = deterministic_hash("dummy input for hash function")
    task_spec = TaskSpec(
        task_name="task_name",
        model_config=config,
        messages=messages,
        out_file_path=Path("out_file_path"),
        ground_truth="A",
        formatter=ZeroShotCOTSycophancyFormatter,
        task_hash=task_hash,
    )

    dummy_output = ModelOutput(raw_response="The best answer is: (A)", parsed_response="A")
    task_output = TaskOutput(
        task_name=task_spec.task_name,
        model_output=[dummy_output],
        config=config,
        prompt=messages,
        ground_truth=task_spec.ground_truth,
        task_hash=task_spec.task_hash,
        formatter_name=task_spec.formatter.name(),
        out_file_path=task_spec.out_file_path,
    )

    assert task_output.input_hash() == task_spec.input_hash()
