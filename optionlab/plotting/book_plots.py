"""Risk charts for a :class:`~optionlab.book.Book`.

Every function returns a themed ``plotly.graph_objects.Figure`` (nothing is
ever shown or written here) and relies on the book's FULL-REPRICING risk
methods, so the pictures stay honest for large moves and for exotic payoffs.

Chart grammar used throughout
-----------------------------
* Quantities with different scales (P&L, delta, gamma...) get one panel each
  (small multiples), never a second y-axis.
* P&L is always readable from position relative to the zero line and from
  signed labels; the green/red (or blue/red) colour is only a redundant cue.
* Calls are blue, puts orange, the underlying yellow, everything else violet.
* Titles and panel names state the units of what is plotted.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from .. import black_scholes as bs
from ..book import (
    DEFAULT_SPOT_SHOCKS,
    DEFAULT_VOL_SHOCKS,
    Book,
    shocked_market,
)
from ..instruments import GREEK_KEYS, Instrument, Underlying
from ..market import Market
from . import theme

__all__ = [
    "book_risk_profile",
    "book_scenario_heatmap",
    "book_greeks_breakdown",
    "book_pnl_by_horizon",
    "stress_test_chart",
]

_GREEK_NAMES: tuple[str, ...] = GREEK_KEYS[1:]

#: Legend groups of the breakdown chart: (legend name, colour), in legend order.
_CATEGORIES: dict[str, tuple[str, str]] = {
    "call": ("Calls", theme.COLORS["call"]),
    "put": ("Puts", theme.COLORS["put"]),
    "underlying": ("Underlying", theme.COLORS["underlying"]),
    "other": ("Other", theme.COLORS["accent"]),
    "total": ("Book total", theme.COLORS["neutral"]),
}


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _require_scalar(mkt: Market) -> None:
    if not mkt.is_scalar:
        raise ValueError("book charts need a scalar Market (the charts build their own grids)")


def _spot_shocks(spot_range: Sequence[float], n_points: int) -> np.ndarray:
    low, high = (float(x) for x in spot_range)
    if not -1.0 < low < high:
        raise ValueError("spot_range must be (low, high) relative shocks with -1 < low < high")
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    return np.linspace(low, high, int(n_points))


def _quantity_name(quantity: str) -> str:
    """Short display name of a plotted quantity (legend, hover, colour bar)."""
    if quantity == "pnl":
        return "P&L"
    if quantity == "value":
        return "Value"
    if quantity not in _GREEK_NAMES:
        raise ValueError(
            f"unknown quantity {quantity!r}; use 'pnl', 'value' or one of {list(_GREEK_NAMES)}"
        )
    return bs.GREEK_INFO[quantity]["label"]


def _quantity_title(quantity: str, trader_units: bool) -> str:
    """Panel/axis title stating WHAT is plotted and in WHICH units."""
    name = _quantity_name(quantity)
    if quantity == "pnl":
        return f"{name} vs current value (currency)"
    if quantity == "value":
        return f"{name} (currency)"
    return f"{name}, {bs.GREEK_INFO[quantity]['trader_unit' if trader_units else 'raw_unit']}"


def _signed(value: float) -> str:
    """Compact signed number for direct labels (the sign carries profit vs loss)."""
    size = abs(value)
    if size == 0:
        return "0"
    if size >= 1e6:
        return f"{value / 1e6:+.2f}M"
    if size >= 1e4:
        return f"{value / 1e3:+.1f}k"
    if size >= 100:
        return f"{value:+,.0f}"
    if size >= 1:
        return f"{value:+.2f}"
    return f"{value:+.3g}"


def _percent(shock: float) -> str:
    return "0%" if shock == 0 else f"{shock * 100:+g}%"


def _vol_points(shock: float) -> str:
    return "0 pts" if shock == 0 else f"{shock * 100:+g} pts"


def _days_label(days: float) -> str:
    return "today" if days == 0 else f"+{days:g} d"


def _unique(labels: Sequence[str]) -> list[str]:
    """Make labels unique (plotly merges identical categories on an axis)."""
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return out


def _category(instrument: Instrument) -> str:
    if isinstance(instrument, Underlying):
        return "underlying"
    option_type = getattr(instrument, "option_type", None)
    return option_type if option_type in ("call", "put") else "other"


def _padded_range(values: np.ndarray) -> list[float]:
    """Axis range leaving room for the labels drawn outside the bar tips."""
    low, high = min(0.0, float(np.min(values))), max(0.0, float(np.max(values)))
    if low == high:
        return [-1.0, 1.0]
    pad = 0.22 * (high - low)
    return [low - pad if low < 0 else 0.0, high + pad if high > 0 else 0.0]


def _horizontal_bars(fig: go.Figure, n_bars: int, value_range: list[float], axis_title: str) -> None:
    """Shared layout of the horizontal bar charts (thin bars, first row on top)."""
    fig.update_layout(barmode="overlay", bargap=0.0, height=150 + 38 * n_bars)
    fig.update_xaxes(title_text=axis_title, range=value_range)
    fig.update_yaxes(autorange="reversed", showgrid=False, zeroline=False)


# ---------------------------------------------------------------------- #
# Charts
# ---------------------------------------------------------------------- #
def book_risk_profile(
    book: Book,
    mkt: Market,
    greeks: Sequence[str] = ("pnl", "delta", "gamma", "vega", "theta"),
    spot_range: Sequence[float] = (-0.30, 0.30),
    n_points: int = 121,
    trader_units: bool = True,
    title: str | None = None,
    mode: str = "auto",
) -> go.Figure:
    """P&L and aggregated Greeks of the book as the spot moves (full repricing).

    One panel per quantity. Read them together: the P&L curve is the picture,
    delta is its slope, gamma its curvature. A book that is "delta-neutral"
    is only flat AT the current spot -- the delta panel shows how fast the
    hedge goes stale.

    Parameters
    ----------
    book, mkt
        The book and the scalar base market.
    greeks : sequence of str
        Panels to draw: ``"pnl"``, ``"value"`` or any Greek name.
    spot_range : (float, float), default (-0.30, 0.30)
        Relative spot shocks covered by the x-axis.
    n_points : int, default 121
        Grid size (the whole grid is repriced in one vectorised call).
    trader_units : bool, default True
        Desk units (vega per vol point, theta per day...) or raw derivatives;
        the panel titles say which.
    """
    _require_scalar(mkt)
    quantities = list(greeks)
    if not quantities:
        raise ValueError("greeks must name at least one quantity to plot")
    titles = [_quantity_title(q, trader_units) for q in quantities]
    ladder = book.spot_ladder(mkt, _spot_shocks(spot_range, n_points), trader_units=trader_units)
    spots = ladder["spot"].to_numpy()

    ncols = 1 if len(quantities) == 1 else 2
    fig = theme.subplot_grid(
        len(quantities),
        ncols=ncols,
        titles=titles,
        title=title or f"{book.name}: risk profile vs spot (full repricing; dashed line = current spot)",
        mode=mode,
    )
    for i, quantity in enumerate(quantities):
        row, col = theme.grid_position(i, ncols)
        values = ladder[quantity].to_numpy()
        if quantity == "pnl":
            for part, key in ((np.maximum(values, 0.0), "profit"), (np.minimum(values, 0.0), "loss")):
                fig.add_trace(
                    go.Scatter(
                        x=spots, y=part, mode="lines", line=dict(width=0), fill="tozeroy",
                        fillcolor=theme.with_alpha(theme.COLORS[key], 0.12),
                        hoverinfo="skip", showlegend=False,
                    ),
                    row, col,
                )
        fig.add_trace(
            go.Scatter(
                x=spots, y=values, mode="lines", name=_quantity_name(quantity),
                line=dict(color=theme.PALETTE[0]), showlegend=False,
                hovertemplate="%{y:,.4r}",
            ),
            row, col,
        )
        if i + ncols >= len(quantities):
            fig.update_xaxes(title_text="Spot", row=row, col=col)
    fig.update_xaxes(hoverformat=",.2f")
    theme.add_vline(fig, float(mkt.spot))  # unlabelled: a label would collide with the panel titles
    return fig


def book_scenario_heatmap(
    book: Book,
    mkt: Market,
    spot_shocks: Sequence[float] = DEFAULT_SPOT_SHOCKS,
    vol_shocks: Sequence[float] = DEFAULT_VOL_SHOCKS,
    horizon_days: float = 0.0,
    quantity: str = "pnl",
    trader_units: bool = True,
    annotate: bool = True,
    title: str | None = None,
    mode: str = "auto",
) -> go.Figure:
    """Spot x vol scenario heatmap (full repricing), diverging colours centred at 0.

    The corners are where books die: spot down AND vol up is the crash corner.
    A long-gamma / long-vega book is blue in every corner; a short-option book
    is red there and only makes money in the quiet centre (and with time, see
    ``horizon_days``).

    Parameters
    ----------
    spot_shocks, vol_shocks, horizon_days, quantity, trader_units
        Passed to :meth:`optionlab.book.Book.scenario_grid`.
    annotate : bool, default True
        Write the signed value in each cell (so the sign never depends on
        colour perception alone).
    """
    _require_scalar(mkt)
    grid = book.scenario_grid(
        mkt, spot_shocks, vol_shocks, horizon_days=horizon_days, quantity=quantity,
        trader_units=trader_units,
    )
    z = grid.to_numpy().T  # rows of the picture = vol shocks, columns = spot shocks
    limit = float(np.max(np.abs(z))) or 1.0
    quantity_title = _quantity_title(quantity, trader_units)
    short_name = _quantity_name(quantity)

    heatmap = go.Heatmap(
        x=[_percent(s) for s in grid.index],
        y=[_vol_points(v) for v in grid.columns],
        z=z,
        colorscale=theme.diverging_scale(mode),
        zmid=0.0,
        zmin=-limit,
        zmax=limit,
        colorbar=dict(title=dict(text=short_name)),
        hovertemplate=f"spot %{{x}}<br>vol %{{y}}<br>{short_name} %{{z:,.4r}}<extra></extra>",
    )
    if annotate:
        heatmap.update(
            text=[[_signed(v) for v in line] for line in z],
            texttemplate="%{text}",
            textfont=dict(size=11),
        )
    fig = go.Figure(heatmap)
    when = "" if not horizon_days else f", {horizon_days:g} days forward"
    theme.apply_theme(
        fig,
        title=title or f"{book.name}: {quantity_title} under spot and vol shocks{when}",
        height=170 + 44 * len(grid.columns),
        mode=mode,
    )
    fig.update_xaxes(title_text="Spot shock (relative)", type="category", showgrid=False)
    fig.update_yaxes(title_text="Vol shock (vol points)", type="category", showgrid=False)
    return fig


def book_greeks_breakdown(
    book: Book,
    mkt: Market,
    greek: str = "delta",
    trader_units: bool = True,
    title: str | None = None,
    mode: str = "auto",
) -> go.Figure:
    """Contribution of each position to one Greek of the book (horizontal bars).

    The grey last bar is the book total: the net of everything above it. This
    is the chart to look at before hedging -- it shows WHICH line carries the
    risk, and whether the book is flat because it is empty or because big
    longs and shorts offset each other (which is a much more fragile "flat").

    Parameters
    ----------
    greek : str, default "delta"
        ``"value"`` or any Greek name.
    trader_units : bool, default True
        Units of the Greek (stated on the axis).
    """
    _require_scalar(mkt)
    if greek == "pnl":
        raise ValueError("P&L is a book-level number; use 'value' or a Greek name")
    axis_title = _quantity_title(greek, trader_units)
    frame = book.positions_frame(mkt, trader_units=trader_units)
    labels = _unique(list(frame["label"]))
    values = frame[greek].to_numpy(dtype=float)
    categories = [_category(p.instrument) for p in book.positions] + ["total"]

    fig = go.Figure()
    for key, (legend_name, color) in _CATEGORIES.items():
        members = [i for i, cat in enumerate(categories) if cat == key]
        if not members:
            continue
        fig.add_trace(
            go.Bar(
                orientation="h",
                y=[labels[i] for i in members],
                x=values[members],
                name=legend_name,
                width=0.6,
                marker=dict(color=color, cornerradius=4),
                text=[_signed(values[i]) for i in members],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}: %{x:,.4r}<extra></extra>",
            )
        )
    theme.apply_theme(fig, title=title or f"{book.name}: {axis_title} by position", mode=mode)
    _horizontal_bars(fig, len(labels), _padded_range(values), axis_title)
    fig.update_yaxes(categoryorder="array", categoryarray=labels)
    return fig


def book_pnl_by_horizon(
    book: Book,
    mkt: Market,
    days: Sequence[float] = (0, 7, 30, 90),
    spot_range: Sequence[float] = (-0.30, 0.30),
    n_points: int = 121,
    title: str | None = None,
    mode: str = "auto",
) -> go.Figure:
    """P&L of the book vs spot as time passes (full repricing, vol unchanged).

    The vertical gap between two curves at a given spot is the time decay
    earned or paid between the two dates: long-option books sink towards their
    kinked expiry payoff, short-option books rise towards it. Lines go from
    light (today) to dark (furthest horizon).

    Parameters
    ----------
    days : sequence of float, default (0, 7, 30, 90)
        Calendar days forward (non-negative; at most 6 horizons stay readable).
        Positions expiring inside a horizon are worth their settlement value at
        the shocked spot; carry on the cash account is not included.
    spot_range, n_points
        Relative spot shocks covered by the x-axis and grid size.
    """
    _require_scalar(mkt)
    horizons = sorted({float(d) for d in days})
    if not horizons or horizons[0] < 0:
        raise ValueError("days must be a non-empty sequence of non-negative numbers")
    shocks = _spot_shocks(spot_range, n_points)
    spots = float(mkt.spot) * (1.0 + shocks)
    base_value = book.positions_value(mkt)

    # "auto" pages may be light or dark, so stay in the middle of the ramp there.
    ramp_low, ramp_high = (1 / 6, 4 / 6) if mode == "auto" else (0.25, 1.0)
    positions = np.linspace(ramp_low, ramp_high, len(horizons)) if len(horizons) > 1 else [ramp_high]
    colors = sample_colorscale(theme.sequential_scale(mode), list(positions))

    fig = go.Figure()
    for horizon, color in zip(horizons, colors):
        values = book.positions_value(shocked_market(mkt, spot=shocks, days=horizon)) - base_value
        fig.add_trace(
            go.Scatter(
                x=spots, y=values, mode="lines", name=_days_label(horizon),
                line=dict(color=color), hovertemplate="%{y:,.2f}",
            )
        )
    theme.apply_theme(
        fig,
        title=title or f"{book.name}: P&L vs spot as time passes (currency, vol unchanged)",
        height=460,
        mode=mode,
    )
    fig.update_xaxes(title_text="Spot", hoverformat=",.2f")
    fig.update_yaxes(title_text="P&L vs current value")
    theme.add_vline(fig, float(mkt.spot), "spot")
    return fig


def stress_test_chart(
    book: Book,
    mkt: Market,
    scenarios: Mapping[str, Mapping[str, float]] | None = None,
    title: str | None = None,
    mode: str = "auto",
) -> go.Figure:
    """P&L of the book in each stress scenario (horizontal bars, full repricing).

    Bars to the right of zero are gains, bars to the left are losses; each bar
    carries its signed value. Look for ASYMMETRY: a book that makes a little
    in most scenarios and loses a lot in one is short a tail.

    Parameters
    ----------
    scenarios : mapping, optional
        Passed to :meth:`optionlab.book.Book.stress_tests` (default scenario
        set when ``None``).
    """
    _require_scalar(mkt)
    frame = book.stress_tests(mkt, scenarios)
    labels = _unique([str(name) for name in frame.index])
    pnl = frame["pnl"].to_numpy(dtype=float)
    bar_colors = [theme.COLORS["profit"] if v >= 0 else theme.COLORS["loss"] for v in pnl]

    fig = go.Figure(
        go.Bar(
            orientation="h",
            y=labels,
            x=pnl,
            width=0.6,
            marker=dict(color=bar_colors, cornerradius=4),
            text=[_signed(v) for v in pnl],
            textposition="outside",
            cliponaxis=False,
            showlegend=False,
            hovertemplate="%{y}<br>P&L %{x:,.2f}<extra></extra>",
        )
    )
    theme.apply_theme(fig, title=title or f"{book.name}: stress-test P&L (currency)", mode=mode)
    _horizontal_bars(fig, len(labels), _padded_range(pnl), "P&L vs current value")
    fig.update_yaxes(categoryorder="array", categoryarray=labels)
    return fig
