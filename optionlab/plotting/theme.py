"""Single source of truth for the look of every optionlab figure.

Design choices
--------------
* **Theme-agnostic by default.** Figures are embedded in Streamlit, whose page
  can be light or dark, and a plotly figure cannot react to CSS. The default
  ``mode="auto"`` therefore uses transparent backgrounds, translucent grey
  grid lines and a mid-grey ink that keeps about 4.3:1 contrast on both white
  and near-black pages. Pass ``mode="light"`` or ``mode="dark"`` when the
  surface is known to get full-contrast ink.
* **Categorical palette** (:data:`PALETTE`): eight hues in a FIXED order,
  validated for colour-vision-deficiency separation between neighbours and for
  >= 3:1 contrast on both light and dark surfaces. Assign colours in order and
  never cycle: a 9th series should become small multiples, not a new hue.
* **Semantic colours** (:data:`COLORS`): calls are blue and puts are orange
  everywhere (the pair is strongly separated for every type of colour vision).
  Profit is green and loss is red because that is the trading convention, but
  red/green is weak for deuteranopes, so P&L must always ALSO be encoded by
  position relative to a zero line (or by sign in a label), never by colour
  alone.
* **Colourscales**: :data:`SEQUENTIAL` is a single-hue blue ramp for
  magnitudes; :data:`DIVERGING` is red <-> grey <-> blue (lightness-matched
  arms) for signed quantities such as P&L, centred with ``zmid=0``.
* Marks: 2 px lines, 8 px markers, hairline solid grid lines, thin colour bars.
  Text never wears the series colour.
"""

from __future__ import annotations

import math
from typing import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

__all__ = [
    "PALETTE",
    "COLORS",
    "INK",
    "SEQUENTIAL",
    "DIVERGING",
    "FONT_FAMILY",
    "sequential_scale",
    "diverging_scale",
    "with_alpha",
    "build_template",
    "apply_theme",
    "add_vline",
    "add_hline",
    "subplot_grid",
    "grid_position",
]

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
_TRANSPARENT = "rgba(0,0,0,0)"

#: Categorical palette -- blue, orange, aqua, yellow, magenta, green, violet, red.
PALETTE: list[str] = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
]

#: Semantic colours shared by every plotting module.
COLORS: dict[str, str] = {
    "call": PALETTE[0],
    "put": PALETTE[1],
    "profit": "#199e70",
    "loss": "#d03b3b",
    "neutral": "#898781",
    "accent": PALETTE[6],
    "underlying": PALETTE[3],
    "hedge": PALETTE[4],
}

#: Ink (text and chrome) per mode. ``auto`` is readable on light AND dark pages.
INK: dict[str, dict[str, str]] = {
    "auto": {
        "primary": "#7b7a76",
        "secondary": "#7b7a76",
        "muted": "#898781",
        "grid": "rgba(137,135,129,0.22)",
        "axis": "rgba(137,135,129,0.55)",
    },
    "light": {
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}

_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_DIVERGING_LIGHT = ["#902122", "#e14c49", "#f7aba4", "#f0efec", "#9ec5f4", "#3987e5", "#184f95"]
_DIVERGING_DARK = ["#f7aba4", "#e14c49", "#902122", "#383835", "#184f95", "#3987e5", "#9ec5f4"]


def _as_colorscale(colors: Sequence[str]) -> list[list]:
    n = len(colors) - 1
    return [[i / n, c] for i, c in enumerate(colors)]


#: Single-hue ramp for magnitudes (light = low, dark = high).
SEQUENTIAL: list[list] = _as_colorscale(_BLUE_RAMP)

#: Red (negative) <-> neutral grey (zero) <-> blue (positive). Use with ``zmid=0``.
DIVERGING: list[list] = _as_colorscale(_DIVERGING_LIGHT)


def _check_mode(mode: str) -> str:
    if mode not in INK:
        raise ValueError(f"mode must be one of {sorted(INK)}, got {mode!r}")
    return mode


def sequential_scale(mode: str = "auto") -> list[list]:
    """Sequential colourscale for a page mode.

    On a dark page the ramp is reversed so that "near zero" still recedes
    into the background and large values stand out.
    """
    ramp = _BLUE_RAMP[::-1] if _check_mode(mode) == "dark" else _BLUE_RAMP
    return _as_colorscale(ramp)


def diverging_scale(mode: str = "auto") -> list[list]:
    """Diverging colourscale for a page mode (the neutral midpoint matches the surface)."""
    return _as_colorscale(_DIVERGING_DARK if _check_mode(mode) == "dark" else _DIVERGING_LIGHT)


def with_alpha(color: str, alpha: float) -> str:
    """Return a ``#rrggbb`` colour as an ``rgba(...)`` string.

    Area fills should be a light wash (``alpha`` around 0.10-0.15), never a
    saturated block.
    """
    hex_digits = color.lstrip("#")
    if len(hex_digits) != 6:
        raise ValueError(f"with_alpha expects a '#rrggbb' colour, got {color!r}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    r, g, b = (int(hex_digits[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:g})"


def build_template(mode: str = "auto") -> go.layout.Template:
    """Build the optionlab plotly template for ``mode`` ("auto", "light" or "dark")."""
    ink = INK[_check_mode(mode)]
    axis = dict(
        showgrid=True,
        gridcolor=ink["grid"],
        gridwidth=1,
        zeroline=True,
        zerolinecolor=ink["axis"],
        zerolinewidth=1,
        showline=False,
        linecolor=ink["axis"],
        ticks="",
        tickfont=dict(color=ink["muted"], size=12),
        title=dict(font=dict(color=ink["secondary"], size=13), standoff=10),
        automargin=True,
    )
    scene_axis = dict(
        showbackground=False,
        gridcolor=ink["axis"],
        zerolinecolor=ink["axis"],
        linecolor=ink["axis"],
        tickfont=dict(color=ink["muted"], size=11),
        title=dict(font=dict(color=ink["secondary"], size=12)),
    )
    colorbar = dict(
        thickness=12,
        outlinewidth=0,
        ticks="",
        tickfont=dict(color=ink["muted"], size=11),
        title=dict(font=dict(color=ink["secondary"], size=12)),
    )
    layout = go.Layout(
        font=dict(family=FONT_FAMILY, size=13, color=ink["secondary"]),
        title=dict(font=dict(size=17, color=ink["primary"]), x=0.0, xanchor="left", xref="paper"),
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
        colorway=PALETTE,
        colorscale=dict(
            sequential=sequential_scale(mode),
            sequentialminus=sequential_scale(mode),
            diverging=diverging_scale(mode),
        ),
        xaxis=axis,
        yaxis=axis,
        scene=dict(xaxis=scene_axis, yaxis=scene_axis, zaxis=scene_axis),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor=_TRANSPARENT,
            font=dict(color=ink["secondary"], size=12),
        ),
        hoverlabel=dict(
            bgcolor="rgba(26,26,25,0.92)",
            bordercolor="rgba(255,255,255,0.10)",
            font=dict(family=FONT_FAMILY, size=12, color="#ffffff"),
        ),
        annotationdefaults=dict(font=dict(color=ink["secondary"], size=12)),
        margin=dict(l=56, r=24, t=72, b=48),
    )
    data = dict(
        scatter=[go.Scatter(line=dict(width=2), marker=dict(size=8))],
        bar=[go.Bar(marker=dict(line=dict(width=0)))],
        heatmap=[go.Heatmap(colorbar=colorbar, xgap=1, ygap=1)],
        contour=[go.Contour(colorbar=colorbar)],
        surface=[go.Surface(colorbar=colorbar)],
    )
    return go.layout.Template(layout=layout, data=data)


def _default_hovermode(fig: go.Figure) -> str:
    """Crosshair-style unified hover for pure line charts, per-mark hover otherwise."""
    lines_only = bool(fig.data) and all(
        trace.type == "scatter" and "lines" in (trace.mode or "lines") for trace in fig.data
    )
    return "x unified" if lines_only else "closest"


def apply_theme(
    fig: go.Figure,
    title: str | None = None,
    height: int | None = None,
    *,
    mode: str = "auto",
    hovermode: str | None = None,
) -> go.Figure:
    """Style a figure with the optionlab theme and return it.

    Call this AFTER adding the traces (the default hover mode depends on them).

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to style (modified in place and returned, for chaining).
    title : str, optional
        Figure title. Say what is plotted AND in which units, e.g.
        ``"Vega (per vol point)"``.
    height : int, optional
        Height in pixels.
    mode : {"auto", "light", "dark"}, default "auto"
        ``"auto"`` works on both light and dark pages (transparent background,
        mid-grey ink); the explicit modes use full-contrast ink.
    hovermode : str, optional
        Override the hover mode. By default line-only figures get
        ``"x unified"`` (a crosshair with one tooltip for all series) and
        everything else gets ``"closest"``.
    """
    fig.update_layout(
        template=build_template(mode),
        hovermode=hovermode if hovermode is not None else _default_hovermode(fig),
    )
    if title is not None:
        fig.update_layout(title_text=title)
    if height is not None:
        fig.update_layout(height=int(height))
    return fig


def _reference_line_kwargs(label, color, dash, width, position, row, col) -> dict:
    kwargs: dict = dict(line=dict(color=color or COLORS["neutral"], width=width, dash=dash))
    if label:
        kwargs.update(
            annotation_text=label,
            annotation_position=position,
            annotation_font=dict(size=11, color=COLORS["neutral"]),
        )
    if row is not None:
        kwargs["row"] = row
    if col is not None:
        kwargs["col"] = col
    return kwargs


def add_vline(
    fig: go.Figure,
    x: float,
    label: str | None = None,
    *,
    color: str | None = None,
    dash: str = "dash",
    width: float = 1.0,
    position: str = "top right",
    row: int | str | None = None,
    col: int | str | None = None,
) -> go.Figure:
    """Add a vertical reference line (strike, spot, barrier, breakeven...).

    The line is drawn in a recessive neutral grey by default so that it never
    competes with the data. On a subplot grid, add reference lines AFTER the
    traces (plotly skips empty subplots) and target a cell with ``row``/``col``
    (default: every subplot).

    The label sits INSIDE the plot area by default (``"top right"`` of the
    line): the band above the plot belongs to the legend and to the subplot
    titles, where a ``"top"`` label would collide with them.
    """
    fig.add_vline(x=x, **_reference_line_kwargs(label, color, dash, width, position, row, col))
    return fig


def add_hline(
    fig: go.Figure,
    y: float,
    label: str | None = None,
    *,
    color: str | None = None,
    dash: str = "dash",
    width: float = 1.0,
    position: str = "right",
    row: int | str | None = None,
    col: int | str | None = None,
) -> go.Figure:
    """Add a horizontal reference line (zero P&L, premium paid, a barrier level...)."""
    fig.add_hline(y=y, **_reference_line_kwargs(label, color, dash, width, position, row, col))
    return fig


def grid_position(index: int, ncols: int = 2) -> tuple[int, int]:
    """1-based ``(row, col)`` of the ``index``-th (0-based) cell of a subplot grid."""
    if index < 0 or ncols < 1:
        raise ValueError("index must be >= 0 and ncols >= 1")
    return index // ncols + 1, index % ncols + 1


def subplot_grid(
    n: int,
    ncols: int = 2,
    titles: Sequence[str] | None = None,
    *,
    title: str | None = None,
    height: int | None = None,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
    mode: str = "auto",
) -> go.Figure:
    """Create a themed grid of ``n`` subplots (small multiples).

    Small multiples are the right answer whenever quantities have different
    scales (one panel per Greek) -- never a second y-axis.

    Parameters
    ----------
    n : int
        Number of panels.
    ncols : int, default 2
        Number of columns; rows are added as needed.
    titles : sequence of str, optional
        One title per panel.
    title, height, mode
        Passed to :func:`apply_theme`; the default height is 280 px per row.
    shared_xaxes, shared_yaxes : bool
        Passed to ``plotly.subplots.make_subplots``.

    Returns
    -------
    plotly.graph_objects.Figure
        Use :func:`grid_position` to place trace ``i``:
        ``fig.add_trace(trace, *grid_position(i, ncols))``. Since the figure is
        created before its traces, the hover mode is ``"x unified"``.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if ncols < 1:
        raise ValueError("ncols must be at least 1")
    if titles is not None and len(titles) != n:
        raise ValueError(f"expected {n} titles, got {len(titles)}")
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=list(titles) if titles is not None else None,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        horizontal_spacing=min(0.10, 0.4 / ncols),
        vertical_spacing=min(0.16, 0.5 / nrows),
    )
    apply_theme(
        fig,
        title=title,
        height=height if height is not None else 280 * nrows + 100,
        mode=mode,
        hovermode="x unified",
    )
    fig.update_annotations(font_size=13)
    return fig
