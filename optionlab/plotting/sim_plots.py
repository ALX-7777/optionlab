"""Charts for the trading simulator and the hedging experiments.

Every function takes the DataFrame produced by
:attr:`optionlab.simulator.TradingSimulator.history` (or by
:func:`optionlab.simulator.delta_hedging_experiment`) and returns a themed
``plotly.graph_objects.Figure``; nothing is ever shown or written here.

Chart grammar
-------------
* Quantities with different units (spot, vol, P&L, shares, each Greek) get one
  panel each -- never a second y-axis.
* Each attribution term keeps the SAME colour in every chart (delta blue,
  gamma orange, theta aqua, vega yellow...), whatever terms are displayed.
  Ten terms are folded into eight groups so that the palette is never cycled.
* P&L is always readable from its position relative to a zero line and from
  signed labels; green/red is only a redundant cue.
* All money amounts are in currency; Greeks shown through time are the
  "dollar" Greeks of :meth:`optionlab.book.Book.dollar_greeks`.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..simulator import ATTRIBUTION_TERMS, attribution_totals
from . import theme

__all__ = [
    "TERM_GROUPS",
    "simulation_dashboard",
    "pnl_attribution_chart",
    "pnl_attribution_waterfall",
    "gamma_theta_chart",
    "hedging_error_histogram",
    "hedging_error_vs_frequency",
]

#: Display groups of the attribution terms: (legend name, terms, colour), in
#: legend order. The colour follows the group, never its rank in a chart.
TERM_GROUPS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("Delta", ("delta",), theme.PALETTE[0]),
    ("Gamma", ("gamma",), theme.PALETTE[1]),
    ("Theta", ("theta",), theme.PALETTE[2]),
    ("Vega", ("vega",), theme.PALETTE[3]),
    ("Vanna + volga", ("vanna", "volga"), theme.PALETTE[4]),
    ("Rates and carry", ("rho", "carry"), theme.PALETTE[5]),
    ("Fees and trading edge", ("fees", "trading"), theme.PALETTE[6]),
    ("Unexplained", ("unexplained",), theme.COLORS["neutral"]),
)

_ALWAYS_SHOWN = ("Delta", "Gamma", "Theta", "Vega", "Unexplained")

_TERM_LABELS: dict[str, str] = {
    "delta": "Delta", "gamma": "Gamma", "theta": "Theta", "vega": "Vega", "vanna": "Vanna",
    "volga": "Volga", "rho": "Rho", "carry": "Carry", "fees": "Fees", "trading": "Trading edge",
    "unexplained": "Unexplained",
}


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _require(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"expected {what} as a pandas DataFrame, got {type(frame).__name__}")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} is missing column(s) {missing}")
    if frame.empty:
        raise ValueError(f"{what} is empty")


def _x_axis(history: pd.DataFrame, x: str) -> tuple[np.ndarray, str, str]:
    """Values, axis title and hover format of the horizontal axis."""
    if x == "t":
        return history["t"].to_numpy(dtype=float), "Time (years)", ".4f"
    if x == "step":
        return np.asarray(history.index, dtype=float), "Step", ".0f"
    raise ValueError(f"x must be 't' (time in years) or 'step', got {x!r}")


def _signed(value: float) -> str:
    """Compact signed number for direct labels (the sign carries profit vs loss)."""
    size = abs(value)
    if size == 0 or not math.isfinite(value):
        return "0" if size == 0 else "n/a"
    if size >= 1e6:
        return f"{value / 1e6:+.2f}M"
    if size >= 1e4:
        return f"{value / 1e3:+.1f}k"
    if size >= 100:
        return f"{value:+,.0f}"
    if size >= 1:
        return f"{value:+.2f}"
    return f"{value:+.3g}"


def _log_ticks(low: float, high: float) -> list[float]:
    """1-2-5 tick values covering ``[low, high]`` on a log axis."""
    if not (low > 0 and high >= low):
        return []
    decades = range(math.floor(math.log10(low)) - 1, math.ceil(math.log10(high)) + 2)
    ticks = [m * 10.0**d for d in decades for m in (1.0, 2.0, 5.0)]
    return [t for t in ticks if low / 2.5 <= t <= high * 2.5]


def _ink(mode: str, role: str = "primary") -> str:
    """Ink colour of the theme mode (an unknown mode is reported later by ``apply_theme``)."""
    return theme.INK[mode if mode in theme.INK else "auto"][role]


def _pnl_washes(fig: go.Figure, x: np.ndarray, values: np.ndarray, row: int, col: int) -> None:
    """Light green wash above zero and red wash below (redundant with the zero line)."""
    for part, key in ((np.maximum(values, 0.0), "profit"), (np.minimum(values, 0.0), "loss")):
        fig.add_trace(
            go.Scatter(
                x=x, y=part, mode="lines", line=dict(width=0), fill="tozeroy",
                fillcolor=theme.with_alpha(theme.COLORS[key], 0.12),
                hoverinfo="skip", showlegend=False,
            ),
            row, col,
        )


def _lift_legend(fig: go.Figure, height: int) -> None:
    """Move the legend above the panel titles of a subplot grid (they share the top edge)."""
    plot_height = max(height - 120, 100)
    fig.update_layout(legend=dict(y=1.0 + 30.0 / plot_height))


def _grouped_terms(history: pd.DataFrame, prefix: str) -> list[tuple[str, np.ndarray, str]]:
    """(name, series, colour) per display group, without the negligible optional groups.

    Delta, gamma, theta, vega and unexplained are always drawn; the other
    groups only when they reach 1% of the largest term (they would be
    invisible anyway and only clutter the legend).
    """
    series = [
        (name, sum(history[f"{prefix}{term}"].to_numpy(dtype=float) for term in terms), color)
        for name, terms, color in TERM_GROUPS
    ]
    scale = max(float(np.max(np.abs(values))) for _, values, _ in series)
    return [
        (name, values, color)
        for name, values, color in series
        if name in _ALWAYS_SHOWN or float(np.max(np.abs(values))) > max(0.01 * scale, 1e-12)
    ]


# ---------------------------------------------------------------------- #
# Simulation charts
# ---------------------------------------------------------------------- #
def simulation_dashboard(
    history: pd.DataFrame, x: str = "t", title: str | None = None, mode: str = "auto"
) -> go.Figure:
    """The whole run at a glance: market, P&L, hedge and risk through time.

    Eight small panels sharing the time axis: spot, implied vol, cumulative
    P&L, stock position (the hedge), book delta before and after the policy
    traded, and the dollar gamma, vega and theta. Read them top to bottom: the
    market moves (row 1), the book makes or loses money and the hedge reacts
    (row 2), and the risk you carry changes as a consequence (rows 3-4) --
    gamma and theta swell as an at-the-money option nears expiry, and die if
    the spot walks away from the strike.

    Parameters
    ----------
    history : pandas.DataFrame
        :attr:`optionlab.simulator.TradingSimulator.history`.
    x : {"t", "step"}, default "t"
        Horizontal axis: time in years or step number.
    title, mode
        Figure title and theme mode (see :func:`optionlab.plotting.theme.apply_theme`).
    """
    columns = ["t", "spot", "implied_vol", "pnl_cum", "stock_position", "delta",
               "delta_before_hedge", "gamma_cash", "vega_cash", "theta_cash"]
    _require(history, columns, "history")
    xs, x_title, x_format = _x_axis(history, x)
    blue = theme.PALETTE[0]
    titles = [
        "Spot",
        "Implied vol (%)",
        "Cumulative P&L (currency)",
        "Stock position (shares)",
        "Book delta (shares)",
        "Gamma cash (delta-cash change per +1% spot)",
        "Vega cash (per vol point)",
        "Theta cash (per calendar day)",
    ]
    height = 4 * 230 + 110
    fig = theme.subplot_grid(
        len(titles), ncols=2, titles=titles, shared_xaxes=True, mode=mode, height=height,
        title=title or "Simulation dashboard",
    )

    def line(values, name, color, row, col, shape="linear", width=2.0, legend=False, fmt=",.2f"):
        fig.add_trace(
            go.Scatter(
                x=xs, y=values, mode="lines", name=name, showlegend=legend,
                line=dict(color=color, shape=shape, width=width),
                hovertemplate=f"%{{y:{fmt}}}",
            ),
            row, col,
        )

    line(history["spot"], "Spot", blue, 1, 1)
    line(100.0 * history["implied_vol"], "Implied vol", blue, 1, 2)
    pnl = history["pnl_cum"].to_numpy(dtype=float)
    _pnl_washes(fig, xs, pnl, 2, 1)
    line(pnl, "Cumulative P&L", blue, 2, 1)
    line(history["stock_position"], "Stock position", theme.COLORS["hedge"], 2, 2, shape="hv")
    line(history["delta_before_hedge"], "Book delta before the hedge", theme.COLORS["neutral"], 3, 1,
         width=1.5, legend=True, fmt=",.3f")
    line(history["delta"], "Book delta after the hedge", blue, 3, 1, legend=True, fmt=",.3f")
    line(history["gamma_cash"], "Gamma cash", blue, 3, 2)
    line(history["vega_cash"], "Vega cash", blue, 4, 1)
    line(history["theta_cash"], "Theta cash", blue, 4, 2)

    fig.update_xaxes(hoverformat=x_format)
    fig.update_xaxes(title_text=x_title, row=4)
    _lift_legend(fig, height)
    return fig


def pnl_attribution_chart(
    history: pd.DataFrame,
    cumulative: bool = True,
    x: str = "t",
    title: str | None = None,
    mode: str = "auto",
) -> go.Figure:
    """P&L explained by the Greeks, against the actual P&L.

    Parameters
    ----------
    history : pandas.DataFrame
        :attr:`optionlab.simulator.TradingSimulator.history`.
    cumulative : bool, default True
        ``True``: one line per term (running totals) plus the thick actual
        P&L line. Lines rather than stacked areas because the terms have
        opposite signs (a long-gamma book earns gamma and pays theta): stacking
        them would hide exactly the tug-of-war the chart is about.
        ``False``: the P&L of each step as signed stacked bars (positive terms
        above zero, negative below), with the actual step P&L as markers.
    x : {"t", "step"}, default "t"
    title, mode
        Figure title and theme mode.

    Notes
    -----
    Small terms are folded into groups (:data:`TERM_GROUPS`) and optional
    groups that stay below 1% of the largest term are not drawn (the exact
    totals are in :func:`pnl_attribution_waterfall`); colours never change. A healthy run has
    a flat "Unexplained" line; if it drifts, the Greeks at the start of each
    step were not enough (steps too long, big jumps, an expiry or a barrier).
    """
    prefix = "cum_pnl_" if cumulative else "pnl_"
    actual_column = "pnl_cum" if cumulative else "pnl_step"
    _require(history, ["t", actual_column, *(f"{prefix}{term}" for term in ATTRIBUTION_TERMS)], "history")
    xs, x_title, x_format = _x_axis(history, x)
    actual = history[actual_column].to_numpy(dtype=float)
    ink = _ink(mode)

    fig = go.Figure()
    for name, values, color in _grouped_terms(history, prefix):
        if cumulative:
            dash = "dot" if name == "Unexplained" else "solid"
            fig.add_trace(
                go.Scatter(
                    x=xs, y=values, mode="lines", name=name,
                    line=dict(color=color, width=2, dash=dash), hovertemplate="%{y:,.2f}",
                )
            )
        else:
            fig.add_trace(
                go.Bar(x=xs, y=values, name=name, marker=dict(color=color), hovertemplate="%{y:,.2f}")
            )
    if cumulative:
        fig.add_trace(
            go.Scatter(
                x=xs, y=actual, mode="lines", name="Actual P&L",
                line=dict(color=ink, width=3.5), hovertemplate="%{y:,.2f}",
            )
        )
        fig.add_annotation(
            x=xs[-1], y=actual[-1], text=f"Actual {_signed(actual[-1])}", showarrow=False,
            xanchor="right", yanchor="bottom", yshift=6,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=xs, y=actual, mode="markers", name="Actual P&L",
                marker=dict(color=ink, size=7, symbol="diamond"), hovertemplate="%{y:,.2f}",
            )
        )
        fig.update_layout(barmode="relative", bargap=0.15)

    default_title = (
        "Cumulative P&L attribution (currency)" if cumulative else "P&L attribution of each step (currency)"
    )
    theme.apply_theme(fig, title=title or default_title, height=480, mode=mode, hovermode="x unified")
    fig.update_xaxes(title_text=x_title, hoverformat=x_format)
    fig.update_yaxes(title_text="P&L")
    return fig


def pnl_attribution_waterfall(
    history: pd.DataFrame, title: str | None = None, mode: str = "auto"
) -> go.Figure:
    """Waterfall of the total P&L of a run: one bar per attribution term, then the actual P&L.

    Bars going up are gains, bars going down are losses, each carrying its
    signed value; the last bar is the actual total. It answers "where did the
    money come from?" in one picture: e.g. a delta-hedged long straddle shows
    a big green gamma bar eaten by a red theta bar, the difference being the
    realised-versus-implied vol edge.

    Terms that are zero over the run (rho with constant rates, fees
    without costs...) are left out; delta, gamma, theta, vega and unexplained
    always appear.
    """
    totals = attribution_totals(history)
    always = ("delta", "gamma", "theta", "vega", "unexplained")
    noise = 1e-9 * max(1.0, *(abs(v) for v in totals.values()))
    totals = {name: (value if abs(value) > noise else 0.0) for name, value in totals.items()}
    terms = [t for t in ATTRIBUTION_TERMS if t in always or totals[t] != 0.0]
    labels = [_TERM_LABELS[t] for t in terms] + ["Actual P&L"]
    values = [totals[t] for t in terms] + [totals["actual"]]

    fig = go.Figure(
        go.Waterfall(
            x=labels,
            y=values,
            measure=["relative"] * len(terms) + ["total"],
            text=[_signed(v) for v in values],
            textposition="outside",
            cliponaxis=False,
            width=0.55,
            increasing=dict(marker=dict(color=theme.COLORS["profit"])),
            decreasing=dict(marker=dict(color=theme.COLORS["loss"])),
            totals=dict(marker=dict(color=theme.COLORS["neutral"])),
            connector=dict(line=dict(color=_ink(mode, "axis"), width=1)),
            hovertemplate="%{x}: %{y:,.2f}<extra></extra>",
            showlegend=False,
        )
    )
    theme.apply_theme(
        fig, title=title or "Where the P&L came from: attribution totals (currency)", height=460, mode=mode
    )
    running = np.cumsum([0.0] + values[:-1])
    low = min(0.0, float(np.min(running)), totals["actual"])
    high = max(0.0, float(np.max(running)), totals["actual"])
    pad = 0.15 * (high - low) if high > low else 1.0
    fig.update_yaxes(title_text="P&L", range=[low - pad, high + pad])
    fig.update_xaxes(showgrid=False)
    return fig


def gamma_theta_chart(
    history: pd.DataFrame, x: str = "t", title: str | None = None, mode: str = "auto"
) -> go.Figure:
    """Gamma scalping in two panels: what gamma earned vs what theta cost, and why.

    * Top: cumulative gamma P&L, cumulative theta P&L and their sum. For a
      long-option book gamma climbs in steps (on the days the spot moves) while
      theta bleeds steadily; the sum is the verdict.
    * Bottom: the absolute spot move of each step against the BREAKEVEN move
      ``implied vol * sqrt(dt)``. For a delta-hedged vanilla book (zero rates)
      ``0.5 * Gamma * dS**2 + Theta * dt = 0`` exactly at that move: bars above
      the line are steps where long gamma beat theta, bars below are steps won
      by the option seller. Realised vol above implied simply means "more tall
      bars than the line can pay for".

    Parameters
    ----------
    history : pandas.DataFrame
        :attr:`optionlab.simulator.TradingSimulator.history`.
    x : {"t", "step"}, default "t"
    """
    _require(history, ["t", "spot", "implied_vol", "cum_pnl_gamma", "cum_pnl_theta"], "history")
    xs, x_title, x_format = _x_axis(history, x)
    gamma = history["cum_pnl_gamma"].to_numpy(dtype=float)
    theta = history["cum_pnl_theta"].to_numpy(dtype=float)
    colors = {name: color for name, _, color in TERM_GROUPS}

    height = 640
    fig = theme.subplot_grid(
        2, ncols=1, shared_xaxes=True, mode=mode, height=height,
        titles=["Cumulative gamma and theta P&L (currency)",
                "Absolute spot move per step vs implied breakeven move (%)"],
        title=title or "Gamma scalping: did the moves pay the rent?",
    )
    series = (
        ("Gamma P&L", gamma, colors["Gamma"], 2.0),
        ("Theta P&L", theta, colors["Theta"], 2.0),
        ("Gamma + theta", gamma + theta, _ink(mode), 3.5),
    )
    for name, values, color, width in series:
        fig.add_trace(
            go.Scatter(
                x=xs, y=values, mode="lines", name=name,
                line=dict(color=color, width=width), hovertemplate="%{y:,.2f}",
            ),
            1, 1,
        )

    spot = history["spot"].to_numpy(dtype=float)
    times = history["t"].to_numpy(dtype=float)
    moves = 100.0 * np.abs(spot[1:] / spot[:-1] - 1.0)
    breakeven = 100.0 * history["implied_vol"].to_numpy(dtype=float)[:-1] * np.sqrt(np.diff(times))
    fig.add_trace(
        go.Bar(
            x=xs[1:], y=moves, name="|spot move|", marker=dict(color=theme.PALETTE[0]),
            hovertemplate="%{y:.2f}%",
        ),
        2, 1,
    )
    fig.add_trace(
        go.Scatter(
            x=xs[1:], y=breakeven, mode="lines", name="Breakeven move (implied vol x sqrt(dt))",
            line=dict(color=_ink(mode), width=2, dash="dash"), hovertemplate="%{y:.2f}%",
        ),
        2, 1,
    )
    fig.update_layout(bargap=0.25)
    fig.update_xaxes(hoverformat=x_format)
    fig.update_xaxes(title_text=x_title, row=2, col=1)
    _lift_legend(fig, height)
    return fig


# ---------------------------------------------------------------------- #
# Hedging-experiment charts
# ---------------------------------------------------------------------- #
def _experiment_context(experiment: pd.DataFrame) -> str:
    attrs = experiment.attrs
    if not {"label", "implied_vol", "realized_vol"} <= set(attrs):
        return ""
    return (
        f"{attrs.get('quantity', 1.0):+g} x {attrs['label']}, implied {attrs['implied_vol']:.0%}, "
        f"realised {attrs['realized_vol']:.0%}"
    )


def hedging_error_histogram(
    experiment: pd.DataFrame,
    bins: int = 41,
    pct_of_premium: bool = False,
    title: str | None = None,
    mode: str = "auto",
) -> go.Figure:
    """Distribution of the final delta-hedging P&L, one panel per rebalancing frequency.

    All panels share the same bins and the same x-axis, so the distributions
    can be compared by eye: they tighten around their mean as the hedge is
    adjusted more often (the standard deviation falls like ``1 / sqrt(N)``),
    while the mean itself does not move -- it is set by realised versus
    implied vol, not by how hard you hedge. Each panel title states the mean
    and the standard deviation.

    Parameters
    ----------
    experiment : pandas.DataFrame
        Output of :func:`optionlab.simulator.delta_hedging_experiment`.
    bins : int, default 41
        Number of bins, spanning the 1%-99% quantiles of the WIDEST
        distribution (the least frequent hedge); the rare outliers beyond are
        clipped into the end bins.
    pct_of_premium : bool, default False
        Plot the P&L as a percentage of the option premium instead of currency.
    """
    column = "pnl_pct_premium" if pct_of_premium else "pnl"
    _require(experiment, ["rebalance_steps", column], "experiment")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    unit = "% of premium" if pct_of_premium else "currency"
    groups = [(int(n), g[column].to_numpy(dtype=float)) for n, g in experiment.groupby("rebalance_steps")]
    widest = max((values for _, values in groups), key=lambda values: float(np.std(values)))
    low, high = (float(q) for q in np.quantile(widest, [0.01, 0.99]))
    if not high > low:
        low, high = low - 1.0, high + 1.0
    edges = np.linspace(low, high, int(bins) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    titles = [
        f"{n} rebalances: mean {_signed(float(v.mean()))}, std {float(np.std(v, ddof=min(1, v.size - 1))):,.3g}"
        for n, v in groups
    ]
    context = _experiment_context(experiment)
    fig = theme.subplot_grid(
        len(groups), ncols=1, titles=titles, shared_xaxes=True, mode=mode,
        height=190 * len(groups) + 130,
        title=title or f"Final delta-hedging P&L ({unit})" + (f": {context}" if context else ""),
    )
    for i, (n, values) in enumerate(groups):
        counts, _ = np.histogram(np.clip(values, low, high), bins=edges)
        share = 100.0 * counts / values.size
        fig.add_trace(
            go.Bar(
                x=centres, y=share, width=0.92 * (edges[1] - edges[0]), name=f"{n} rebalances",
                marker=dict(color=theme.PALETTE[0]), showlegend=False,
                hovertemplate="P&L %{x:,.3g}<br>%{y:.1f}% of paths<extra></extra>",
            ),
            i + 1, 1,
        )
        fig.update_yaxes(title_text="% of paths", row=i + 1, col=1)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title_text=f"Final P&L ({unit})", row=len(groups), col=1)
    theme.add_vline(fig, 0.0)
    return fig


def hedging_error_vs_frequency(
    experiment: pd.DataFrame, title: str | None = None, mode: str = "auto"
) -> go.Figure:
    """Standard deviation of the hedging P&L against the number of rebalances (log-log).

    The dashed reference has slope -1/2: ``std(N) = std(N0) * sqrt(N0 / N)``.
    When realised vol equals implied vol the dots sit on it -- to halve the
    hedging error you must trade four times as often. When the two vols
    differ the dots flatten out: part of the dispersion is no longer hedging
    noise but the path-dependence of the gamma P&L itself, which no amount of
    rebalancing removes.
    """
    _require(experiment, ["rebalance_steps", "pnl"], "experiment")
    stats = experiment.groupby("rebalance_steps")["pnl"].std(ddof=1)
    steps = stats.index.to_numpy(dtype=float)
    std = stats.to_numpy(dtype=float)
    reference = std[0] * np.sqrt(steps[0] / steps)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps, y=reference, mode="lines", name="1 / sqrt(N) reference",
            line=dict(color=theme.COLORS["neutral"], dash="dash", width=1.5), hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps, y=std, mode="lines+markers+text", name="Std of final P&L",
            line=dict(color=theme.PALETTE[0]), text=[f"{v:,.3g}" for v in std],
            textposition="top right", hovertemplate="N = %{x:.0f}<br>std %{y:,.4g}<extra></extra>",
        )
    )
    context = _experiment_context(experiment)
    default_title = "Hedging error vs rebalancing frequency (log-log)" + (f": {context}" if context else "")
    theme.apply_theme(fig, title=title or default_title, height=440, mode=mode, hovermode="closest")
    fig.update_xaxes(title_text="Number of rebalances N (log scale)", type="log",
                     tickvals=list(steps), ticktext=[f"{int(s)}" for s in steps])
    fig.update_yaxes(title_text="Std of final P&L (currency, log scale)", type="log",
                     tickvals=_log_ticks(float(std.min()), float(std.max())))
    return fig
