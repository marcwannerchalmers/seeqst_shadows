"""Plotting helpers for shadow-scaling experiments."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns


Metric = Literal["mae", "rmse", "max"]
XAxis = Literal["N", "n"]

_METRIC_LABELS = {
    "mae": "Mean absolute error",
    "rmse": "Root mean squared error",
    "max": "Maximum absolute error",
}

_METRIC_ALIASES = {
    "mae": "mae",
    "mean_absolute_error": "mae",
    "rmse": "rmse",
    "root_mean_squared_error": "rmse",
    "max": "max",
    "max_error": "max",
    "maximum_error": "max",
}

_MARKERS = ("o", "X", "s", "P", "D", "v", "^", "<", ">", "*")


def _font_size(font_sizes, name: str, default: float) -> float:
    if font_sizes is None:
        return default
    if isinstance(font_sizes, (int, float)):
        return float(font_sizes)
    return float(font_sizes.get(name, default))


def _normalise_metric(metric: str) -> Metric:
    key = metric.lower().strip().replace(" ", "_")
    try:
        return _METRIC_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(sorted({"mae", "rmse", "max"}))
        raise ValueError(f"Unknown metric {metric!r}; choose one of: {choices}") from exc


def prepare_metric_data(df: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    """Aggregate errors as in ``seeqst_int``'s error-scaling experiment.

    The selected metric is first evaluated over all observables for one state
    repetition at fixed method, ``n`` and ``N``.  These per-repetition metrics
    are then averaged across state repetitions, and their sample standard
    deviation is used for the error shading.
    """

    metric_name = _normalise_metric(metric)
    required = {
        "Shadow model",
        "N",
        "n",
        "State",
        "Observable",
        "experiment_name",
        "Prediction",
        "Error",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Plot data is missing required columns: {missing}")

    work = df.copy()
    work["absolute_error"] = np.abs(work["Error"].astype(float))
    # Experiments._observable_uncertainty_data normalises the input so that
    # ``Observable`` identifies the state repetition and ``State`` varies over
    # the target observables within that repetition.
    repetition_cols = [
        "Shadow model",
        "N",
        "n",
        "Observable",
        "experiment_name",
    ]
    series_cols = ["Shadow model", "N", "n", "experiment_name"]

    grouped = work.groupby(repetition_cols, sort=False, dropna=False)
    if metric_name == "mae":
        repetition_metric = grouped["absolute_error"].mean()
    elif metric_name == "rmse":
        work["squared_error"] = work["absolute_error"] ** 2
        repetition_metric = np.sqrt(
            work.groupby(repetition_cols, sort=False, dropna=False)[
                "squared_error"
            ].mean()
        )
    else:
        repetition_metric = grouped["absolute_error"].max()

    repetition_df = repetition_metric.reset_index(name="repetition_metric")
    plot_df = (
        repetition_df.groupby(series_cols, sort=False, dropna=False)[
            "repetition_metric"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # A point belongs to the zero-prediction regime only when every prediction
    # over all observables and state repetitions is exactly zero.
    zero_prediction = (
        work.groupby(series_cols, sort=False, dropna=False)["Prediction"]
        .agg(lambda values: bool(np.equal(np.asarray(values), 0).all()))
        .reset_index(name="all_predictions_zero")
    )
    plot_df = plot_df.merge(zero_prediction, on=series_cols, how="left")

    # The reference implementation plots mean +/- the sample standard
    # deviation of the per-trial metric, rather than a confidence interval.
    plot_df["ci"] = plot_df["std"]
    plot_df["ci_low"] = plot_df["mean"] - plot_df["ci"]
    plot_df["ci_high"] = plot_df["mean"] + plot_df["ci"]
    plot_df.attrs["metric"] = metric_name
    return plot_df


def _normalise_x_axis(x: str) -> XAxis:
    if x == "N":
        return "N"
    if x == "n":
        return "n"
    raise ValueError(f"Unknown x-axis {x!r}; choose 'N' or 'n'")


def _contiguous_runs(
    group: pd.DataFrame,
    x_axis: XAxis,
) -> Iterator[pd.DataFrame]:
    """Yield contiguous runs with the same zero-prediction classification."""

    group = group.sort_values(x_axis).reset_index(drop=True)
    changes = group["all_predictions_zero"].ne(
        group["all_predictions_zero"].shift()
    )
    for _, run in group.groupby(changes.cumsum(), sort=False):
        yield run


def _series_label(
    experiment_name: str,
    fixed_axis: XAxis,
    fixed_value: int,
    experiment_counts: dict[str, int],
) -> str:
    if experiment_counts[experiment_name] > 1:
        return f"{experiment_name}, {fixed_axis}={fixed_value}"
    return str(experiment_name)


def _configure_axes(
    ax,
    plot_df: pd.DataFrame,
    metric: Metric,
    x_axis: XAxis,
    xlog: bool,
    ylog: bool,
    font_sizes=None,
) -> float | None:
    if xlog:
        ax.set_xscale("log", base=2)
        xvals = np.sort(plot_df[x_axis].unique())
        ax.set_xticks(xvals)
        if x_axis == "N":
            ax.set_xticklabels(
                [rf"$2^{{{int(np.log2(value))}}}$" for value in xvals]
            )
        else:
            ax.set_xticklabels([str(value) for value in xvals])

    positive_means = plot_df.loc[plot_df["mean"] > 0, "mean"]
    if ylog:
        if positive_means.empty:
            raise ValueError("A logarithmic y-axis requires at least one positive value")

        ax.set_yscale("log")
        y_floor = positive_means.min() * 1e-3
        positive_lower = plot_df.loc[plot_df["ci_low"] > 0, "ci_low"]
        lower_candidates = [positive_means.min()]
        if not positive_lower.empty:
            lower_candidates.append(positive_lower.min())
        ymin = min(lower_candidates) * 0.8
    else:
        y_floor = None
        ymin = 0.0

    finite_upper = plot_df.loc[np.isfinite(plot_df["ci_high"]), "ci_high"]
    ymax_source = finite_upper.max() if not finite_upper.empty else plot_df["mean"].max()
    ymax = ymax_source * (1.2 if ylog else 1.05)
    ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_xlabel(
        "Number of samples $N$"
        if x_axis == "N"
        else "Number of qubits $n$",
        fontsize=_font_size(font_sizes, "axes", 10),
    )
    ax.set_ylabel(
        _METRIC_LABELS[metric],
        fontsize=_font_size(font_sizes, "axes", 10),
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=_font_size(font_sizes, "ticks", 10),
    )
    ax.grid(True, which="major", alpha=0.3)
    return y_floor


def _plot_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    path_save: str | Path | None,
    x: str,
    xlog: bool,
    ylog: bool,
    split_zero_predictions: bool,
    prepared: bool = False,
    font_sizes=None,
    marker_size: float = 7,
    line_width: float = 1.8,
    legend_location: str | None = None,
):
    metric_name = _normalise_metric(metric)
    x_axis = _normalise_x_axis(x)
    fixed_axis: XAxis = "n" if x_axis == "N" else "N"
    plot_df = df.copy() if prepared else prepare_metric_data(df, metric_name)
    if x_axis == "n":
        group_cols = ["Shadow model", "n", "experiment_name"]
        max_N = plot_df.groupby(group_cols, dropna=False)["N"].transform("max")
        plot_df = plot_df[plot_df["N"] == max_N]
    sns.set_style("whitegrid")
    figure_width = 9.5 if split_zero_predictions else 7.5
    fig, ax = plt.subplots(figsize=(figure_width, 5.5))

    series_cols = ["Shadow model", fixed_axis, "experiment_name"]
    # Match the stable ordering used by the previous seaborn implementation.
    series = list(plot_df.groupby(series_cols, sort=True, dropna=False))
    experiment_names = list(dict.fromkeys(key[2] for key, _ in series))
    colors = dict(
        zip(experiment_names, sns.color_palette("tab10", len(experiment_names)))
    )
    experiment_counts = {
        name: sum(key[2] == name for key, _ in series)
        for name in experiment_names
    }

    y_floor = _configure_axes(
        ax, plot_df, metric_name, x_axis, xlog, ylog, font_sizes
    )
    legend_handles = []

    for series_index, ((_, fixed_value, experiment_name), group) in enumerate(series):
        color = colors[experiment_name]
        marker = _MARKERS[series_index % len(_MARKERS)]
        label = _series_label(
            experiment_name,
            fixed_axis,
            fixed_value,
            experiment_counts,
        )
        legend_handles.append(
            Line2D(
                [],
                [],
                color=color,
                marker=marker,
                linestyle="-",
                linewidth=line_width,
                markersize=marker_size,
                label=label,
            )
        )

        runs = (
            _contiguous_runs(group, x_axis)
            if split_zero_predictions
            else (group,)
        )
        for run in runs:
            run = run.sort_values(x_axis)
            all_zero = bool(run["all_predictions_zero"].iloc[0])
            linestyle = "--" if split_zero_predictions and all_zero else "-"
            x_values = run[x_axis].to_numpy()
            mean = run["mean"].to_numpy()
            ax.plot(
                x_values,
                mean,
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=marker_size,
                linewidth=line_width,
            )

            # Zero-prediction points represent an unresolved sampling regime,
            # so deliberately omit their confidence shading.
            if split_zero_predictions and all_zero:
                continue

            lower = run["ci_low"].to_numpy()
            upper = run["ci_high"].to_numpy()
            if ylog:
                lower = np.maximum(lower, y_floor)
            ax.fill_between(x_values, lower, upper, color=color, alpha=0.15)

    if split_zero_predictions:
        legend_handles.extend(
            [
                Line2D(
                    [],
                    [],
                    color="0.35",
                    linestyle="-",
                    linewidth=line_width,
                    label="At least one nonzero prediction",
                ),
                Line2D(
                    [],
                    [],
                    color="0.35",
                    linestyle="--",
                    linewidth=line_width,
                    label="All predictions zero",
                ),
            ]
        )
        legend_title = "Shadow model / segment"
    else:
        legend_title = "Shadow model"

    legend_kwargs = {}
    if legend_location is not None:
        legend_kwargs = {"loc": legend_location}
    elif split_zero_predictions:
        legend_kwargs = {
            "loc": "upper left",
            "bbox_to_anchor": (1.02, 1.0),
            "borderaxespad": 0.0,
        }
    ax.legend(
        handles=legend_handles,
        title=legend_title,
        fontsize=_font_size(font_sizes, "legend", 10),
        title_fontsize=_font_size(font_sizes, "legend_title", 10),
        frameon=True,
        **legend_kwargs,
    )
    fig.tight_layout()
    if path_save is not None:
        fig.savefig(path_save, bbox_inches="tight", dpi=300)
    return fig, ax


def plot_avg_var(
    df: pd.DataFrame,
    path_save: str | Path | None = None,
    xlog: bool = False,
    ylog: bool = False,
    metric: str = "mae",
    x: str = "N",
    prepared: bool = False,
    font_sizes=None,
    marker_size: float = 7,
    line_width: float = 1.8,
    legend_location: str | None = None,
):
    """Plot an observable-wise metric against samples (``N``) or qubits (``n``)."""

    return _plot_metric(
        df,
        metric=metric,
        path_save=path_save,
        x=x,
        xlog=xlog,
        ylog=ylog,
        split_zero_predictions=False,
        prepared=prepared,
        font_sizes=font_sizes,
        marker_size=marker_size,
        line_width=line_width,
        legend_location=legend_location,
    )


def plot_avg_var_zero_segments(
    df: pd.DataFrame,
    path_save: str | Path | None = None,
    xlog: bool = False,
    ylog: bool = False,
    metric: str = "mae",
    x: str = "N",
    prepared: bool = False,
    font_sizes=None,
    marker_size: float = 7,
    line_width: float = 1.8,
    legend_location: str | None = None,
):
    """Plot zero-prediction runs against ``N`` or ``n`` as dashed segments."""

    return _plot_metric(
        df,
        metric=metric,
        path_save=path_save,
        x=x,
        xlog=xlog,
        ylog=ylog,
        split_zero_predictions=True,
        prepared=prepared,
        font_sizes=font_sizes,
        marker_size=marker_size,
        line_width=line_width,
        legend_location=legend_location,
    )


def plot_sample_complexity(
    plot_df: pd.DataFrame,
    epsilon: float,
    path_save: str | Path | None = None,
    xlog: bool = False,
    ylog: bool = True,
    metric: str = "rmse",
    font_sizes=None,
    marker_size: float = 7,
    line_width: float = 1.8,
    legend_location: str | None = None,
):
    """Plot mean sample complexity and its spread across state repetitions."""

    metric_name = _normalise_metric(metric)
    rows = []
    group_cols = ["Shadow model", "n", "experiment_name"]
    for keys, group in plot_df.groupby(group_cols, sort=False, dropna=False):
        group = group.sort_values("N")
        Ns = group["N"].to_numpy()
        repetition_metric = np.stack(group["repetition_metric"].to_numpy())
        below = repetition_metric < epsilon
        reached = below.any(axis=0)
        first = np.argmax(below, axis=0)
        thresholds = np.full(reached.shape, Ns[-1], dtype=float)
        thresholds[reached] = Ns[first[reached]]
        rows.append(
            (
                *keys,
                thresholds.mean(),
                thresholds.std(ddof=1) if len(thresholds) > 1 else np.nan,
                int(reached.sum()),
                len(thresholds),
            )
        )
    threshold_df = pd.DataFrame(
        rows,
        columns=group_cols
        + ["N_threshold", "N_threshold_std", "reached", "count"],
    )

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    experiment_names = list(dict.fromkeys(threshold_df["experiment_name"]))
    colors = dict(
        zip(experiment_names, sns.color_palette("tab10", len(experiment_names)))
    )
    plotted_means = []
    for index, ((_, experiment_name), group) in enumerate(
        threshold_df.groupby(
            ["Shadow model", "experiment_name"], sort=True, dropna=False
        )
    ):
        group = group.sort_values("n").reset_index(drop=True)
        no_crossing = np.flatnonzero(group["reached"].to_numpy() == 0)
        if no_crossing.size:
            group = group.iloc[: no_crossing[0]]
        if group.empty:
            continue
        plotted_means.extend(group["N_threshold"].to_numpy())
        ax.plot(
            group["n"],
            group["N_threshold"],
            color=colors[experiment_name],
            marker=_MARKERS[index % len(_MARKERS)],
            markersize=marker_size,
            linewidth=line_width,
            label=experiment_name,
        )
        lower = np.maximum(
            group["N_threshold"] - group["N_threshold_std"],
            np.finfo(float).tiny,
        )
        upper = group["N_threshold"] + group["N_threshold_std"]
        ax.fill_between(
            group["n"],
            lower,
            upper,
            color=colors[experiment_name],
            alpha=0.15,
        )

    if xlog:
        ax.set_xscale("log", base=2)
    if ylog:
        ax.set_yscale("log", base=2)
    if not plotted_means:
        raise ValueError(f"No state repetition reaches epsilon={epsilon}")
    ymax = max(plotted_means)
    ax.set_ylim(top=ymax * 1.05)
    ax.set_xlabel(
        "Number of qubits $n$",
        fontsize=_font_size(font_sizes, "axes", 10),
    )
    ax.set_ylabel(
        rf"Mean smallest $N$ with {_METRIC_LABELS[metric_name]} $< {epsilon:g}$",
        fontsize=_font_size(font_sizes, "axes", 10),
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=_font_size(font_sizes, "ticks", 10),
    )
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(
        loc=legend_location or "best",
        fontsize=_font_size(font_sizes, "legend", 10),
        title="Shadow model",
        title_fontsize=_font_size(font_sizes, "legend_title", 10),
        frameon=True,
    )
    fig.tight_layout()
    if path_save is not None:
        fig.savefig(path_save, bbox_inches="tight", dpi=300)
    return fig, ax
