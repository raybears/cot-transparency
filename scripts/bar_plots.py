import plotly.graph_objects as go

if __name__ == "__main__":
    data_dict = {"✔️ marks correct": 50, "❌ marks wrong": 50, "stanford professor": 90}

    fig = go.Figure(
        data=[
            go.Bar(
                x=list(data_dict.keys()),
                y=list(data_dict.values()),
                text=list(data_dict.values()),
                textposition="auto",
            )
        ]
    )

    fig.update_layout(title_text="% spotting feature", yaxis_title="Percentage (%)")

    # write to png
    fig.write_image("bar_plots.png")
