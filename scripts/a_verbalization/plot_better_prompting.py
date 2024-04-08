import plotly.graph_objs as go

# Data
models = ["gpt-3.5-turbo", "claude-sonnet", "gpt-4-turbo"]
affected = [0.818, 0.688, 0.308]  # Ground Truth - Affected
unaffected = [0.492, 0.961, 0.989]  # Ground Truth - Unaffected
accuracy_before = [0.527, 0.500, 0.564]  # Macro Accuracy
accuracy_after = [0.674, 0.817, 0.654]  # Macro Accuracy
ci_affected = [0.04, 0.06, 0.27]  # CI for Affected
ci_unaffected = [0.04, 0.01, 0.01]  # CI for Unaffected
ci_accuracy_before = [0.050, 0.070, 0.040]  # CI for Accuracy
ci_accuracy_after = [0.04, 0.03, 0.06]  # CI for Accuracy

# Colors for the bars

# Define the figure
fig = go.Figure()

# Define bar for Ground Truth - Affected
# fig.add_trace(go.Bar(
#     name='Ground Truth - Affected',
#     x=models,
#     y=affected,
#     text=['{:.1%}'.format(val) for val in affected],
#     error_y=dict(type='data', array=ci_affected, visible=True),
#     marker_color='blue'
# ))

# # Define bar for Ground Truth - Unaffected
# fig.add_trace(go.Bar(
#     name='Ground Truth - Unaffected',
#     x=models,
#     y=unaffected,
#     text=['{:.1%}'.format(val) for val in unaffected],
#     error_y=dict(type='data', array=ci_unaffected, visible=True),
#     marker_color='orange'
# ))

fig.add_trace(
    go.Bar(
        name="Before better prompting<br>Average Accuracy<br>(Balanced over two ground truth classes)",
        x=models,
        y=accuracy_before,
        text=["{:.1%}".format(val) for val in accuracy_before],
        error_y=dict(type="data", array=ci_accuracy_before, visible=True),
        marker_color="purple",
    )
)

# Define bar for Average Accuracy
fig.add_trace(
    go.Bar(
        name="After better prompting<br>Accuracy<br>(Balanced over two ground truth classes)",
        x=models,
        y=accuracy_after,
        text=["{:.1%}".format(val) for val in accuracy_after],
        error_y=dict(type="data", array=ci_accuracy_after, visible=True),
        marker_color="orange",
    )
)

# Define line for Random Chance
fig.add_trace(
    go.Scatter(
        name="Random Chance",
        x=models,
        y=[0.5, 0.5, 0.5, 0.5],
        mode="lines",
        line=dict(color="red", width=4),
        hoverinfo="none",
    )
)

# Set the layout for the figure
fig.update_layout(
    title="Accuracy ",
    xaxis=dict(title="Model"),
    yaxis=dict(title="Score", range=[0, 1.1]),
    # legend outside
    # legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    barmode="group",
)

# Show figure
fig.show()
