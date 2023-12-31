import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from pydantic import BaseModel
from slist import Slist
from cot_transparency.data_models.models import TaskOutput

from cot_transparency.formatters.interventions.few_shots_loading import (
    ModelOutputVerified,
    get_training_cots_gpt_35,
)


class TaskRow(BaseModel):
    is_correct: bool


def task_output_to_row(task: TaskOutput) -> TaskRow:
    return TaskRow(is_correct=task.is_correct)


if __name__ == "__main__":
    items: Slist[TaskRow] = Slist(get_training_cots_gpt_35(ModelOutputVerified.correct_and_wrong)).map(
        task_output_to_row
    )
    df = pd.DataFrame([item.dict() for item in items])
    # Two bars of different colors
    seaborn.countplot(x="is_correct", data=df, palette=["red", "blue"])
    # Set False to be red, True to be blue
    plt.xticks([False, True], ["False", "True"])  # type: ignore
    # Change the colors
    # Rename x to "Has correct answer in COT"
    plt.xlabel("Has correct answer in COT")
    plt.show()
