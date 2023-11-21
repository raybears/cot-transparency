from dataclasses import dataclass
from typing import List
import plotly.graph_objects as go


@dataclass
class CategoryValues:
    hue: str
    zero_shot: float
    one_shot: float


def create_bar_chart_with_dataclass(values: list[CategoryValues]) -> go.Figure:
    fig = go.Figure()

    for value in values:
        fig.add_trace(
            go.Bar(
                name=value.hue,
                x=["Zero-shot", "3-shot"],
                y=[value.zero_shot, value.one_shot],
            )
        )

    fig.update_layout(
        title="Does the model still learn from few-shot examples?",
        # xaxis_title="Category",
        yaxis_title="Accuracy",
        barmode="group",
    )

    return fig


# Example data using the dataclass with the new structure
values = [
    CategoryValues(hue="Original GPT-3.5-turbo", zero_shot=10, one_shot=15),
    CategoryValues(hue="Control", zero_shot=20, one_shot=25),
    CategoryValues(hue="Intervention", zero_shot=30, one_shot=35),
]
# Create and show the plot using the updated data structure
fig = create_bar_chart_with_dataclass(values)
fig.show()
