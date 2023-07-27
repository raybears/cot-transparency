import pandas as pd
from slist import Slist

from cot_transparency.data_models.io import ExpLoader
from cot_transparency.data_models.models import TaskOutput, ChatMessage


def extract_answer(messages: list[ChatMessage]) -> str:
    """Extracts the answer from a list of messages"""
    content = messages[-1].content
    # split by new line
    split = content.split("\n")
    # get the last index matching "Label:"
    indexes = [i for i, x in enumerate(split) if "Label:" in x]
    last_q_index = max(indexes) if len(indexes) > 0 else None
    # return the lines after that
    return "\n".join(split[last_q_index + 1 :]) if last_q_index is not None else messages[-1].content


if __name__ == "__main__":
    """Produces a jsonl containing answers of a select formatter"""
    jsons = ExpLoader.stage_one("experiments/v2")

    jsons_tasks: Slist[TaskOutput] = Slist(jsons.values()).map(lambda x: x.outputs).flatten_list()  # type: ignore
    selected_formatter = "CrossTreatmentFormatter"
    print(f"Number of jsons: {len(jsons_tasks)}")
    results: Slist[TaskOutput] = (
        jsons_tasks.filter(lambda x: x.task_spec.formatter_name == selected_formatter)
        # Sometimes we have multiple runs of the same task, we want to get the first one
        .distinct_by(
            lambda x: x.task_spec.task_name
            + x.task_spec.task_hash
            + x.task_spec.model_config.d_hash()
            + x.task_spec.formatter_name
        )
    )
    # We want a tuple of (Formatter Name, Task Name, Raw Response, Parsed Response, Ground Truth)
    tuples = results.map(
        lambda x: (
            x.task_spec.formatter_name,
            x.task_spec.task_name,
            x.model_output.raw_response,
            x.model_output.parsed_response,
            x.task_spec.ground_truth,
            extract_answer(x.task_spec.messages),
        )
    ).shuffle()
    # write to CSV
    print(f"Number of results: {len(results)}")
    df = pd.DataFrame(tuples)
    # write columns in order
    df.columns = ["Formatter Name", "Task Name", "Raw Response", "Parsed Response", "Ground Truth", "Question"]
    # TODO: strip to get the message.
    # limit to first 100
    df.head(100).to_csv(f"{selected_formatter}.csv", index=False)
