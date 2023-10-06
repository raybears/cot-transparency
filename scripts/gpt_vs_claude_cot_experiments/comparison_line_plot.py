from enum import Enum
from typing import Sequence
from pydantic import BaseModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class DataFromOptions(str, Enum):
    gpt_35_turbo = "gpt-3.5-turbo"
    claude_2 = "claude-2"


class ModelTrainMeta(BaseModel):
    name: str
    trained_samples: int
    trained_on: DataFromOptions


class AccuracyOutput(BaseModel):
    accuracy: float
    error_bars: float
    samples: int


class ModelNameAndTrainedSamplesAndMetrics(BaseModel):
    train_meta: ModelTrainMeta
    metrics: AccuracyOutput


def seaborn_line_plot(data: Sequence[ModelNameAndTrainedSamplesAndMetrics], error_bars: bool = True):
    df = pd.DataFrame(
        [
            {
                "Trained Samples": i.train_meta.trained_samples,
                "Accuracy": i.metrics.accuracy,
                "Error Bars": i.metrics.error_bars,
                "Trained on COTs from": i.train_meta.trained_on.value,
            }
            for i in data
        ]
    )
    sns.lineplot(data=df, x="Trained Samples", y="Accuracy", hue="Trained on COTs from")

    if error_bars:
        for name, group in df.groupby("Trained on COTs from"):
            plt.errorbar(
                group["Trained Samples"],
                group["Accuracy"],
                yerr=group["Error Bars"],
                fmt="none",
                capsize=5,
                ecolor="black",
            )
    plt.ylim(0, 1)
    plt.show()


if __name__ == "__main__":
    data_list = [
        ModelNameAndTrainedSamplesAndMetrics(
            train_meta=ModelTrainMeta(name="Model1", trained_samples=i * 10, trained_on=DataFromOptions.gpt_35_turbo),
            metrics=AccuracyOutput(accuracy=0.85 + (i * 0.01), error_bars=0.02, samples=500),
        )
        for i in range(1, 4)
    ]
    data_list += [
        ModelNameAndTrainedSamplesAndMetrics(
            train_meta=ModelTrainMeta(name="Model2", trained_samples=i * 10, trained_on=DataFromOptions.claude_2),
            metrics=AccuracyOutput(accuracy=0.75 + (i * 0.02), error_bars=0.03, samples=500),
        )
        for i in range(1, 4)
    ]
    seaborn_line_plot(data_list)
    plt.show()
