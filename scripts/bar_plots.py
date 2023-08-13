from typing import Optional

import plotly.graph_objects as go
import plotly.io as pio

from scripts.multi_accuracy import PlotDots, DottedLine


def bar_plot(
    plot_dots: list[PlotDots],
    title: str,
    subtitle: str = "",
    save_file_path: Optional[str] = None,
    dotted_line: Optional[DottedLine] = None,
):
    fig = go.Figure()

    for dot in plot_dots:
        fig.add_trace(
            go.Bar(
                name=dot.name,
                x=[dot.name],
                y=[dot.acc.accuracy],
                error_y=dict(type="data", array=[dot.acc.error_bars], visible=True),
                text=[f"{dot.acc.accuracy:.2f}"],
            )
        )

    fig.update_layout(
        barmode="group",
        title_text=title,
        title_x=0.5,
    )

    if dotted_line is not None:
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=dotted_line.value,
            x1=len(plot_dots) - 0.5,
            y1=dotted_line.value,
            line=dict(
                color=dotted_line.color,
                width=4,
                dash="dashdot",
            ),
        )

    if subtitle:
        fig.add_annotation(
            x=1,
            y=0,
            xref="paper",
            yref="paper",
            xanchor="right",
            yanchor="top",
            text=subtitle,
            showarrow=False,
            font=dict(size=12, color="#555"),
        )

    if save_file_path is not None:
        pio.write_image(fig, save_file_path + ".png", scale=2)
    else:
        fig.show()


if __name__ == "__main__":
    data_dict = {"✔️ marks correct": 50, "❌ marks wrong": 50, "stanford professor": 90}

    fig = go.Figure(
        data=[
            go.Bar(
                x=list(data_dict.keys()), y=list(data_dict.values()), text=list(data_dict.values()), textposition="auto"
            )
        ]
    )

    fig.update_layout(title_text="% spotting feature", yaxis_title="Percentage (%)")

    # write to png
    fig.write_image("bar_plots.png")
