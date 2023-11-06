import numpy as np
import plotly.graph_objects as go

from scripts.multi_accuracy import PlotlyShapeColorManager

# Sample size
sample_size = 5000


# Function to calculate the 95% Confidence Interval
def calculate_95CI(p, n):
    # Standard error calculation
    se = np.sqrt((p * (1 - p)) / n)
    # Z-score for a 95% confidence interval
    z_score = 1.96
    # Margin of error
    me = z_score * se
    return me


if __name__ == "__main__":
    # Data for the bar graph
    x_values = [
        "gpt-3.5-turbo",
        "Trained with unbiased contexts (control)\n50% COT, 1k samples",
        "Trained with biased contexts (Ours)\n50% COT, 1k samples"
    ]
    y_values = [0.152, 0.1838, 0.193]
    errors = [calculate_95CI(y, sample_size) for y in y_values]
    color_manager = PlotlyShapeColorManager()
    colors = [color_manager.get_color_and_shape(x).color for x in x_values]

    # Create the bar graph with error bars
    fig = go.Figure(data=[
        go.Bar(
            # name=x_values,
            x=x_values,
            y=y_values,
            error_y=dict(type='data', array=errors, visible=True),
            text=[f"            {dec:.2f}" for dec in y_values],
            textposition='auto',
            marker_color=colors,
            # showlegend=True,
        )
    ])

    # Update the layout
    fig.update_layout(
        title_text="Rate of choosing the first option<br>Ground truths are equally distributed among 4 options",
        yaxis_title="Rate of choosing first option",
        xaxis_tickangle=-45
    )

    # Show the figure
    fig.show()
