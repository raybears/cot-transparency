import textwrap
from typing import Any, Literal, Optional

import seaborn as sns


def annotate_bars(ax: Any, **kwargs: Any):  # typing: ignore
    for p in ax.patches:
        ax.annotate(
            format(p.get_height(), ".2f"),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
        )


NAME_MAP = {
    "model": "Model",
    "task_name": "Task",
    "intervention_name": "Intervention",
    "fleiss_kappa": "Fleiss Kappa Score",
    "modal_agreement_score": "Modal Agreement Score",
    "is_correct": "Accuracy",
    "entropy": "Entropy",
    "none_count": "# of None Responses",
    "all_tasks": "All Tasks",
    "matches_bias": "Promportion Matching Biased Answer",
}


def catplot(
    *args: Any,
    data: Any = None,
    add_annotation_above_bars: bool = False,
    wrap_width: int = 30,
    hue: Optional[str] = None,
    x: Optional[str] = None,
    col: Optional[str] = None,
    y: Optional[str] = None,
    add_line_at: Optional[float] = None,
    kind: Literal["bar"] | Literal["point"] | Literal["count"] = "bar",
    y_scale: float = 1.0,
    name_map: Optional[dict[str, str]] = None,
    width=10,
    height=8,
    **kwargs: Any,
) -> sns.FacetGrid:  # typing: ignore
    """
    A utility wrapper around seaborns catplot that sets some nice defaults and wraps text in the
    legend and column names
        wrap_width: int - how many characters to wrap the text to for legend and column names
        add_annotation_above_bars: bool - whether to add annotations above the bars (off by default)
    """
    sns.set_style(
        "ticks",
        {
            "axes.edgecolor": "0",
            "grid.linestyle": ":",
            "grid.color": "lightgrey",
            "grid.linewidth": "1.5",
            "axes.facecolor": "white",
            "font.family": ["Times New Roman"],
        },
    )

    # rename any column referenced by col, or hue with the name in NAME_MAP
    # merge name_map with NAME_MAP
    name_map = {**NAME_MAP, **(name_map or {})}

    renamed_cols = {}
    if col in name_map:
        renamed_cols[col] = name_map[col]
        col = name_map[col]
    if hue in name_map:
        renamed_cols[hue] = name_map[hue]
        hue = name_map[hue]

    data[y] = data[y] * y_scale

    # rename any column referenced by x, or y with the name in name_map
    df = data.rename(columns=renamed_cols)

    if kind == "bar":  # typing: ignore
        # these args not supported for e.g. count plots
        if "errwidth" not in kwargs:
            kwargs["errwidth"] = 1.5
        if "capsize" not in kwargs:
            kwargs["capsize"] = 0.05

    g = sns.catplot(
        *args,
        linewidth=1,
        edgecolor="black",
        data=df,
        hue=hue,
        x=x,
        col=col,
        y=y,
        kind=kind,
        **kwargs,
    )  # typing: ignore

    # for each axis in the plot add a line at y = add_line_at
    if add_line_at is not None:
        for ax in g.axes.flat:
            ax.axhline(y=add_line_at, color="r", linestyle="--")

    # print the counts for the x, hue, col group
    groups = []
    if x is not None:
        groups.append(x)
    if hue is not None:
        groups.append(hue)
    if col is not None:
        groups.append(col)

    print("Counts of data used to create plot:")
    counts = df.groupby(groups).size().reset_index(name="counts")
    print(counts)

    if add_annotation_above_bars:
        for ax in g.axes.flat:
            annotate_bars(ax)

    # Wrap the legend text
    try:
        for text in g._legend.texts:  # type: ignore
            text.set_text(textwrap.fill(text.get_text(), wrap_width))
    except Exception:
        pass

    # Also wrap any sns catplot column names
    for ax in g.axes.flat:
        try:
            ax.set_title(textwrap.fill(ax.get_title(), wrap_width))
        except Exception:
            pass

    # if any of the axis titles are in name_map, replace them
    for ax in g.axes.flat:
        if ax.get_xlabel() in name_map:
            ax.set_xlabel(name_map[ax.get_xlabel()])
        if ax.get_ylabel() in name_map:
            ax.set_ylabel(name_map[ax.get_ylabel()])

    # move the plot area to leave space for the legend
    # g.fig.subplots_adjust(right=0.62)

    # make the figure bigger
    g.fig.set_figwidth(width / 2.54)
    g.fig.set_figheight(height / 2.54)
    return g


if __name__ == "__main__":
    # make some fake data and plot it
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    np.random.seed(0)
    data = pd.DataFrame(
        {
            "model": np.random.choice(list("ABCD"), 100),
            "kl": np.random.random(100),
            "task_name": np.random.choice(["task1", "task2 this is a really long one lets see if it wraps"], 100),
        }
    )

    # Create the catplot
    g = catplot(data=data, x="model", y="kl", hue="task_name", kind="bar")

    plt.show()
