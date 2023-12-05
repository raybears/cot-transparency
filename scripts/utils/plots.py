import textwrap
from typing import Any, Literal, Optional, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes


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


def pointplot(
    *args: Any,
    data: Any = None,
    add_annotation_above_bars: bool = False,
    wrap_width: int = 30,
    hue: Optional[str] = None,
    x: Optional[str] = None,
    col: Optional[str] = None,
    y: Optional[str] = None,
    add_line_at: Optional[float] = None,
    add_multiple_lines_at: Optional[Sequence[tuple[float, float, float]]] = None,
    y_scale: float = 1.0,
    name_map: Optional[dict[str, str]] = None,
    width: float = 7.7,
    height: float = 6,
    alpha: Optional[float] = None,
    **kwargs: Any,
) -> Axes:  # typing: ignore
    """
    A utility wrapper around seaborns catplot that sets some nice defaults and wraps text in the
    legend and column names
        wrap_width: int - how many characters to wrap the text to for legend and column names
        add_annotation_above_bars: bool - whether to add annotations above the bars (off by default)
    """
    width = width / 2.54  # convert to inches
    height = height / 2.54  # convert to inches

    sns.set(font_scale=0.7)  # crazy big
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

    kwargs["join"] = False
    # kwargs["errcolor"] = kwargs.get("errcolor", "black")

    fig, ax = plt.subplots(figsize=(width, height))
    # make the balls bigger
    # kwargs["size"] = kwargs.get("size", 10)
    ax = sns.pointplot(
        *args,
        data=df,
        hue=hue,
        x=x,
        # col=col,
        y=y,
        # kind=kind,
        scale=1.2,  # type: ignore
        **kwargs,
    )  # typing: ignore
    ax.set_xlabel("")

    if alpha is not None:
        plt.setp(ax.collections, alpha=alpha)  # for the markers
        plt.setp(ax.lines, alpha=alpha)  # for the lines

    axs = [ax]

    if add_multiple_lines_at is not None:
        for ax in axs:
            line = None
            for x_val, y_val, l_width in add_multiple_lines_at:
                # Calculate start and end points for the line segment
                x_start = x_val - l_width / 2
                x_end = x_val + l_width / 2

                # Draw line segment
                line = ax.plot([x_start, x_end], [y_val, y_val], color="r", linestyle="--")  # type: ignore

            handles, labels = ax.get_legend_handles_labels()

            # Create new handle and label for the item to append
            if line is not None:
                new_handle = line[0]
                new_label = "Unbiased"

                # Append new handle and label
                handles.append(new_handle)
                labels.append(new_label)

                # Create new legend with updated handles and labels
                ax.legend(handles, labels)

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
        annotate_bars(ax)

    # Wrap the legend text
    try:
        for text in g._legend.texts:  # type: ignore
            text.set_text(textwrap.fill(text.get_text(), wrap_width))
    except Exception:
        pass

    # Also wrap any sns catplot column names
    try:
        ax.set_title(textwrap.fill(ax.get_title(), wrap_width))
    except Exception:
        pass

    # if any of the axis titles are in name_map, replace them
    if ax.get_xlabel() in name_map:
        ax.set_xlabel(name_map[ax.get_xlabel()])
    if ax.get_ylabel() in name_map:
        ax.set_ylabel(name_map[ax.get_ylabel()])

    return ax


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
    add_multiple_lines_at: Optional[Sequence[tuple[float, float, float]]] = None,
    kind: Literal["bar"] | Literal["point"] | Literal["count"] = "bar",
    y_scale: float = 1.0,
    name_map: Optional[dict[str, str]] = None,
    width: float = 7.7,
    height: float = 6,
    font_scale: float = 0.7,
    **kwargs: Any,
) -> sns.FacetGrid:  # typing: ignore
    """
    A utility wrapper around seaborns catplot that sets some nice defaults and wraps text in the
    legend and column names
        wrap_width: int - how many characters to wrap the text to for legend and column names
        add_annotation_above_bars: bool - whether to add annotations above the bars (off by default)
    """
    width = width / 2.54  # convert to inches
    height = height / 2.54  # convert to inches

    sns.set(font_scale=font_scale)  # crazy big
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
            kwargs["capsize"] = kwargs.get("capsize", 0.05)
            kwargs["linewidth"] = kwargs.get("linewidth", 1.5)
            kwargs["edgecolor"] = kwargs.get("edgecolor", "black")
    if kind == "point":
        kwargs["join"] = False
        # kwargs["errcolor"] = kwargs.get("errcolor", "black")

    g = sns.catplot(
        *args,
        data=df,
        hue=hue,
        x=x,
        col=col,
        y=y,
        kind=kind,
        **kwargs,
    )
    axs = list(g.axes.flat)

    if add_multiple_lines_at is not None:
        for ax in axs:
            line = None
            for x_val, y_val, l_width in add_multiple_lines_at:
                # Calculate start and end points for the line segment
                x_start = x_val - l_width / 2
                x_end = x_val + l_width / 2

                # Draw line segment
                line = ax.plot([x_start, x_end], [y_val, y_val], color="r", linestyle="--")

            handles, labels = ax.get_legend_handles_labels()

            # Create new handle and label for the item to append
            if line is not None:
                new_handle = line[0]
                new_label = "Unbiased"

                # Append new handle and label
                handles.append(new_handle)
                labels.append(new_label)

                # Create new legend with updated handles and labels
                ax.legend(handles, labels)

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
        for ax in axs:
            annotate_bars(ax)

    # Wrap the legend text
    try:
        for text in g._legend.texts:  # type: ignore
            text.set_text(textwrap.fill(text.get_text(), wrap_width))
    except Exception:
        pass

    # Also wrap any sns catplot column names
    for ax in axs:
        try:
            ax.set_title(textwrap.fill(ax.get_title(), wrap_width))
        except Exception:
            pass

    # if any of the axis titles are in name_map, replace them
    for ax in axs:
        if ax.get_xlabel() in name_map:
            ax.set_xlabel(name_map[ax.get_xlabel()])
        if ax.get_ylabel() in name_map:
            ax.set_ylabel(name_map[ax.get_ylabel()])

    # move the plot area to leave space for the legend
    # g.fig.subplots_adjust(right=0.62)

    # make the figure bigger
    g.fig.set_figwidth(width)
    g.fig.set_figheight(height)

    return g


if __name__ == "__main__":
    # make some fake data and plot it
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
