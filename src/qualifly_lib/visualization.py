# SPDX-FileCopyrightText: 2026 German Aerospace Center (DLR)
#
# SPDX-License-Identifier: MIT

import colorsys
import random
import threading
from enum import StrEnum

import matplotlib
import matplotlib.dates as mdates
from matplotlib import style
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


# No pyplot: figure objects render via their own Agg canvas, so parallel
# flights never share matplotlib's global figure state
style.use('default')

matplotlib.rcParams.update({
    'font.size': 17,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 17,
    'font.family': 'serif',
})

# Fixed color per filter variant
_FILTER_COLORS = {
    "original": "#d62728",         # red
    "default": "#ff7f0e",          # orange
    "median": "#8c564b",           # brown
    "mean": "#17becf",             # cyan
    "derivative": "#bcbd22",       # olive
    "kalman_smoother": "#2ca02c",  # green
}

# Any filter name not covered above gets a color generated on the fly, so
# several unlisted filters in the same plot don't all look alike
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949
_hue_offset = random.random()
_dynamic_filter_colors: dict[str, str] = {}
_COLORS_LOCK = threading.Lock()


def _random_filter_color() -> str:
    """
    Next hue in the golden-angle sequence, so it stays vivid and readable.
    """
    hue = (_hue_offset + len(_dynamic_filter_colors) * _GOLDEN_RATIO_CONJUGATE) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.8)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _get_filter_color(fname: str) -> str:
    """
    Fixed color for known variants, a cached random color for anything else.
    """
    if fname in _FILTER_COLORS:
        return _FILTER_COLORS[fname]
    with _COLORS_LOCK:
        if fname not in _dynamic_filter_colors:
            _dynamic_filter_colors[fname] = _random_filter_color()
        return _dynamic_filter_colors[fname]


def _hover_labels(index):
    """Precise sample timestamps (ISO, microsecond resolution) for map hover labels."""
    return [t.strftime("%Y-%m-%d %H:%M:%S.%f") if hasattr(t, "strftime") else str(t)
            for t in index]


def plot_trajectory_map(source_df, target_df, config, collection_id=None, variant_dfs=None):
    """
    Creates a street-map overview of Germany with predefined styling: the source,
    the original target and every filter variant.
    """
    source = config["source"]
    target = config["target"]

    title = f"Trajectory Map: {collection_id}" if collection_id else ''
    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=source_df[source.LAT],
        lon=source_df[source.LON],
        mode='lines+markers',
        name=source.IDENTIFIER,
        text=_hover_labels(source_df.index),
        hovertemplate="%{text}<br>(%{lat:.5f}, %{lon:.5f})<extra>" + source.IDENTIFIER + "</extra>",
        marker=dict(size=6, color='blue'),
        line=dict(width=2, color='blue')
    ))
    fig.add_trace(go.Scattermap(
        lat=target_df[target.LAT],
        lon=target_df[target.LON],
        mode='lines+markers',
        name=target.IDENTIFIER,
        text=_hover_labels(target_df.index),
        hovertemplate="%{text}<br>(%{lat:.5f}, %{lon:.5f})<extra>" + target.IDENTIFIER + "</extra>",
        marker=dict(size=6, color=_get_filter_color("original")),
        line=dict(width=2, color=_get_filter_color("original"))
    ))
    for name, variant_df in (variant_dfs or {}).items():
        fig.add_trace(go.Scattermap(
            lat=variant_df[target.LAT],
            lon=variant_df[target.LON],
            mode='lines+markers',
            name=f"filter {name}",
            text=_hover_labels(variant_df.index),
            hovertemplate="%{text}<br>(%{lat:.5f}, %{lon:.5f})<extra>filter " + str(name) + "</extra>",
            marker=dict(size=6, color=_get_filter_color(name)),
            line=dict(width=2, color=_get_filter_color(name))
        ))
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        map=dict(
            style="open-street-map",
            center=dict(lat=51.1657, lon=10.4515),  # Germany
            zoom=5
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        hovermode='closest'
    )

    return fig


def plot_time_series_overview(source_df, target_df, config, filtered_results=None, score=None):
    """
    Creates time-series plots with standardized styling for SVG export: one panel
    per parameter (geometric altitude and, if both sides carry it, ground speed).
    The info card sits below the legend.
    """
    source = config["source"]
    target = config["target"]
    mapping = _source_to_target(source, target)
    reverse_mapping = {target_val: source_val for source_val, target_val in mapping.items()}

    cols = [target.GEOALT]
    gs_column = getattr(target, "GROUNDSPEED", None)
    if gs_column and gs_column in target_df.columns and reverse_mapping.get(gs_column) in source_df.columns:
        cols.append(gs_column)

    fig_width = 13.5
    fig_height = 6.5
    axes_height = fig_height * len(cols)
    total_height = axes_height

    fig = Figure(figsize=(fig_width, total_height))
    FigureCanvasAgg(fig)  # own canvas per figure, pyplot-free
    ax = np.atleast_1d(fig.subplots(nrows=len(cols)))

    for i, col in enumerate(cols):
        metric_prefix = "gs" if col == gs_column else "alt"
        unit = "m/s" if col == gs_column else "m"
        ax[i].set_ylabel(f"{col} [{unit}]" if col == gs_column else col)
        # Plot source data
        source_param = reverse_mapping[col]
        ax[i].plot(source_df.index, source_df[source_param], label=source.IDENTIFIER, alpha=0.7, linewidth=0.6, c='blue')
        # Plot original target data
        target_label = f"{target.IDENTIFIER} : {_format_metrics(score['original'], metric_prefix, unit)}" if score else target.IDENTIFIER
        ax[i].plot(target_df.index, target_df[col], label=target_label, alpha=0.7, linewidth=0.6, c=_get_filter_color("original"))
        # Plot filtered target data - each variant against its own index, since a
        # filter (derivative, a rolling window, ...) can return fewer rows than target_df
        if filtered_results:
            for fname, f_df in filtered_results.items():
                label = f"filter {fname}"
                if score:
                    label += f" : {_format_metrics(score[fname], metric_prefix, unit)}"
                ax[i].plot(
                    f_df.index, f_df[col],
                    label=label,
                    alpha=0.6,
                    linewidth=0.6,
                    c=_get_filter_color(fname),
                )

        ax[i].set_xlabel('Time (UTC)')
        ax[i].grid(True, linewidth=0.3)
        # Only show HH:MM:SS on the x-axis, not the date
        ax[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.tight_layout()

    # Reserve space at the bottom for the legend, otherwise tight_layout() lets the
    # axes fill the full figure height and the legend gets drawn on top of the plot.
    bottom_margin = 0.40
    fig.subplots_adjust(bottom=bottom_margin)

    handles, labels = ax[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
    )

    return fig


def plot_time_series_overview_interactive(
        source_df, target_df, config, filtered_results=None, collection_id=None, score=None):
    """
    Creates Interactive Plotly Version of Time-Series-Overview with a legend,
    Position/Altitude table row at the bottom and a centered info card.
    """
    source = config["source"]
    target = config["target"]
    mapping = _source_to_target(source, target)
    reverse_mapping = {target_val: source_val for source_val, target_val in mapping.items()}

    # only parameters both sides carry (e.g. a target velocity without a source ground speed is skipped)
    params = [c for c in target_df.columns
              if c in mapping.values() and reverse_mapping[c] in source_df.columns]

    n_series_rows = len(params)
    specs = [[{"type": "xy", "colspan": 2}, None] for _ in range(n_series_rows)]
    specs.append([{"type": "table"}, {"type": "table"}])
    subplot_titles = [None] * n_series_rows + ["Position (m)", "Altitude (m)"]
    row_heights = [1] * n_series_rows + [0.4]  # table row a bit shorter than each series row

    fig = make_subplots(
        rows=n_series_rows + 1, cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    filter_names = list(filtered_results.keys()) if filtered_results else []
    filter_colors = {fname: _get_filter_color(fname) for fname in filter_names}

    for i, target_param in enumerate(params):
        row = i + 1
        source_param = reverse_mapping[target_param]

        fig.add_trace(go.Scatter(
            x=source_df.index, y=source_df[source_param],
            mode='lines', name=source.IDENTIFIER,
            line=dict(color='blue', width=1),
            legendgroup=source.IDENTIFIER,
            showlegend=(i == 0),
            opacity=0.7
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=target_df.index, y=target_df[target_param],
            mode='lines', name=target.IDENTIFIER,
            line=dict(color=_get_filter_color("original"), width=1),
            legendgroup=target.IDENTIFIER,
            showlegend=(i == 0),
            opacity=0.7
        ), row=row, col=1)

        if filtered_results:
            for fname in filter_names:
                f_df = filtered_results[fname]
                fig.add_trace(go.Scatter(
                    x=f_df.index, y=f_df[target_param],
                    mode='lines', name=f"filter {fname}",
                    line=dict(color=filter_colors[fname], width=1),
                    legendgroup=f'filter_{fname}',
                    showlegend=(i == 0),
                    opacity=0.6
                ), row=row, col=1)

        ylabel = "ground speed [m/s]" if target_param == str(getattr(target, "GROUNDSPEED", "")) else target_param
        fig.update_yaxes(title_text=ylabel, row=row, col=1)
        fig.update_xaxes(title_text='Time (UTC)', showticklabels=True, row=row, col=1)

    fig.update_xaxes(tickformat='%H:%M:%S')

    # Position/Altitude tables in the bottom row
    if score is not None:
        variant_names, pos_col_labels, pos_rows, alt_col_labels, alt_rows = _metrics_table_data(score)

        fig.add_trace(go.Table(
            header=dict(values=pos_col_labels, font=dict(size=10), align='center', fill_color='#f0f0f0'),
            cells=dict(values=[variant_names] + list(zip(*pos_rows)) if pos_rows and pos_rows[0] else [variant_names],
                       align='center', font=dict(size=9)),
        ), row=n_series_rows + 1, col=1)

        fig.add_trace(go.Table(
            header=dict(values=alt_col_labels, font=dict(size=10), align='center', fill_color='#f0f0f0'),
            cells=dict(values=[variant_names] + list(zip(*alt_rows)) if alt_rows and alt_rows[0] else [variant_names],
                       align='center', font=dict(size=9)),
        ), row=n_series_rows + 1, col=2)

    fig.update_layout(
        template='plotly_white',
        height=300 * n_series_rows + 220,
        margin=dict(r=180, t=140),
        legend=dict(
            orientation='v',
            yanchor='top', y=1,
            xanchor='left', x=1.02,
            itemclick='toggle',
            itemdoubleclick='toggleothers',
        ),
        hovermode='x unified',
    )

    fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5,
                     showline=True, linecolor='black', mirror=True)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5,
                     showline=True, linecolor='black', mirror=True)

    fig.add_annotation(
        text="Click = show/hide \u00b7 Double-click = isolate",
        xref='paper', yref='paper',
        x=1.02, y=1.05,
        xanchor='left', yanchor='bottom',
        showarrow=False,
        font=dict(size=9, color='gray')
    )

    # Centered info card at top
    info_lines = [f"ID: {collection_id}", f"Date: {_format_date(target_df)}"]
    if score is not None:
        info_lines.append(f"Original : {_format_metrics(score['original'], 'pos')} , "
                          f"{_format_metrics(score['original'], 'alt')}")

        fig.add_annotation(
            text="<br>".join(info_lines),
            xref='paper', yref='paper',
            x=0.5, y=1.08,
            xanchor='center', yanchor='bottom',
            showarrow=False,
            align='center',
            font=dict(size=10, color='black'),
            bordercolor='gray',
            borderwidth=1,
            borderpad=4,
            bgcolor='white',
        )

    return fig


def _format_date(df):
    """Extract a display date (YYYY-MM-DD) from a datetime-indexed DataFrame."""
    if len(df.index) == 0:
        return "N/A"
    return df.index[0].strftime('%Y-%m-%d')


def _collect_metric_names(score, prefix):
    """
    Returns the ordered, de-duplicated list of metric suffixes (e.g. ["p95", "track_mean"])
    that appear under the given prefix ("pos" or "alt") anywhere in score.
    """
    prefix_ = f"{prefix}_"
    names = []
    for variant_scores in score.values():
        for key in variant_scores.keys():
            if key.startswith(prefix_) and key[len(prefix_):] not in names:
                names.append(key[len(prefix_):])
    return names


def _format_metrics(scores, prefix, unit="m"):
    """
    Formats all metrics under one prefix ("pos", "alt" or "gs") of a variant's
    scores into a compact string, e.g. "RMSE 19.56 / P95 33.70 m".
    """
    prefix_ = f"{prefix}_"
    parts = [
        f"{key[len(prefix_):].upper()} {value:.2f}"
        for key, value in scores.items()
        if key.startswith(prefix_) and value is not None
    ]
    return f"{' / '.join(parts)} {unit}" if parts else "N/A"


def _metric_row(scores, prefix, metric_names):
    """
    Returns formatted metric values (2 decimals, "N/A" if missing) for the given prefix,
    in the order given by metric_names (typically from _collect_metric_names.
    """
    return [
        f"{scores.get(f'{prefix}_{name}'):.2f}" if scores.get(f"{prefix}_{name}") is not None else "N/A"
        for name in metric_names
    ]


def _metrics_table_data(score):
    """
    Builds row-aligned Position/Altitude metric tables for every variant in score, in dict order.
    Returns (variant_names, pos_col_labels, pos_rows, alt_col_labels, alt_rows).
    """
    variant_names = list(score.keys())
    pos_names = _collect_metric_names(score, "pos")
    alt_names = _collect_metric_names(score, "alt")
    pos_rows = [_metric_row(score[v], "pos", pos_names) for v in variant_names]
    alt_rows = [_metric_row(score[v], "alt", alt_names) for v in variant_names]
    pos_col_labels = ["Variant"] + [n.upper() for n in pos_names]
    alt_col_labels = ["Variant"] + [n.upper() for n in alt_names]
    return variant_names, pos_col_labels, pos_rows, alt_col_labels, alt_rows


def _source_to_target(
    source: StrEnum,
    target: StrEnum,
    excluded: frozenset[str] = frozenset({"IDENTIFIER", "TIME"}),
) -> dict[str, str]:
    """
    Maps all parameter names of source to target.
    """
    target_by_name = {member.name: member.value for member in target}

    return {
        member.value: target_by_name[member.name]
        for member in source
        if member.name not in excluded and member.name in target_by_name
    }
