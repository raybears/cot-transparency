from typing import Any
import seaborn as sns
import textwrap


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


def catplot(
    *args: Any,
    add_annotation_above_bars: bool = False,
    wrap_width: int = 30,
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
        },
    )

    if kwargs["kind"] == "bar":  # typing: ignore
        # these args not supported for e.g. count plots
        if "errwidth" not in kwargs:
            kwargs["errwidth"] = 1.5
        if "capsize" not in kwargs:
            kwargs["capsize"] = 0.05

    g = sns.catplot(*args, **kwargs, linewidth=1, edgecolor="black")

    if add_annotation_above_bars:
        for ax in g.axes.flat:
            annotate_bars(ax)

    # Wrap the legend text
    try:
        for text in g._legend.texts:
            text.set_text(textwrap.fill(text.get_text(), wrap_width))
    except Exception:
        pass

    # Also wrap any sns catplot column names
    for ax in g.axes.flat:
        try:
            ax.set_title(textwrap.fill(ax.get_title(), wrap_width))
        except Exception:
            pass

    return g


if __name__ == "__main__":
    # make some fake data and plot it
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

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
