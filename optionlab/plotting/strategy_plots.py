"""Strategy-specific figures: payoff diagrams, leg-by-leg Greeks, decay and vol views.

Every function returns a themed :class:`plotly.graph_objects.Figure` and never
shows it. The visual grammar is the same in all of them:

* the **at-expiry** curve is drawn in bold ink (the neutral text colour): it is
  the contract, the thing that is certain;
* **model values before expiry** (today, intermediate dates, other vols) are
  blue -- a single blue for one curve, a light-to-dark blue ramp when the
  curves are ordered (time to expiry, volatility level);
* **legs** are thin dashed lines in the categorical palette, each with its own
  dash pattern so that identity never rests on colour alone;
* profit / loss shading is a light green / red wash, and is always redundant
  with the position of the curve relative to the zero line.

P&L is measured against today's model price of the package (see
:func:`optionlab.strategies.pnl_at_expiry` for the treatment of financing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from ..black_scholes import GREEK_INFO, to_trader_units
from ..instruments import GREEK_KEYS, CompositeInstrument, Instrument, Position
from ..market import Market
from ..strategies import (
    analysis_horizon,
    breakevens,
    max_loss,
    max_profit,
    net_premium,
    pnl_at_expiry,
    reference_levels,
    spot_grid,
)
from .theme import (
    COLORS,
    INK,
    PALETTE,
    add_hline,
    add_vline,
    apply_theme,
    grid_position,
    sequential_scale,
    subplot_grid,
    with_alpha,
)

__all__ = [
    "payoff_diagram",
    "strategy_greeks_dashboard",
    "strategy_time_decay",
    "strategy_vol_sensitivity",
    "compare_strategies",
]

_MODEL_COLOR = PALETTE[0]  # blue: model values before expiry
_LEG_COLORS = PALETTE[1:7]  # legs never reuse the blue of the model curves
_LEG_DASHES = ("dash", "dot", "dashdot", "longdash", "longdashdot", "dash")
_MAX_LEG_SERIES = len(_LEG_COLORS)
_HOVER_MONEY = "%{y:,.2f}"
_HOVER_GREEK = "%{y:.4g}"


# ---------------------------------------------------------------------- #
# Shared helpers
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class _LegSeries:
    """One dashed line of a figure: a leg, or the fold of the legs beyond the sixth."""

    label: str
    positions: tuple[Position, ...]
    color: str
    dash: str


def _leg_series(strategy: Instrument) -> list[_LegSeries]:
    """Elementary legs (identical instruments netted) with a stable colour and dash each.

    A categorical palette must never be cycled: past six legs, the remaining
    ones are folded into a single neutral "Other legs" series.
    """
    if isinstance(strategy, CompositeInstrument):
        positions = strategy.flatten(merge=True)
    else:
        positions = (Position(strategy, 1.0),)
    if len(positions) <= _MAX_LEG_SERIES:
        groups = [(p.label, (p,)) for p in positions]
    else:
        head, tail = positions[: _MAX_LEG_SERIES - 1], positions[_MAX_LEG_SERIES - 1 :]
        groups = [(p.label, (p,)) for p in head] + [(f"Other legs ({len(tail)})", tuple(tail))]
    series = []
    for i, (label, members) in enumerate(groups):
        folded = len(members) > 1
        color = COLORS["neutral"] if folded else _LEG_COLORS[i]
        series.append(_LegSeries(label, members, color, _LEG_DASHES[i]))
    return series


def _check_inputs(strategy: Any, mkt: Any) -> None:
    if not isinstance(strategy, Instrument):
        raise ValueError(f"expected an Instrument (e.g. a Strategy), got {strategy!r}")
    if not isinstance(mkt, Market) or not mkt.is_scalar:
        raise ValueError("strategy plots need a scalar Market (no array fields)")


def _format_pnl(value: float) -> str:
    if math.isinf(value):
        return "unlimited"
    return f"{value:+,.2f}"


def _format_tau(tau: float) -> str:
    return f"{tau:.2f}y" if tau >= 1.0 else f"{tau * 365:.0f}d"


def _horizon_phrase(strategy: Instrument, mkt: Market, horizon: float | None) -> str:
    """"at expiry" wording, explicit about the model assumption for calendars."""
    if horizon is not None and strategy.expiry is not None and strategy.expiry > horizon + 1e-10:
        return f"at front expiry T={horizon:.2f} (back legs valued at {float(mkt.vol):.0%} vol)"
    return "at expiry"


def _ramp(n: int, mode: str) -> list[str]:
    """``n`` ordered blues, recessive first and prominent last, for the page mode."""
    return sample_colorscale(sequential_scale(mode), list(np.linspace(0.25, 0.95, n)))


def _focused_range(focus: Sequence[np.ndarray], context: Sequence[np.ndarray]) -> list[float]:
    """Y range that keeps the package readable when single legs swing much more than it.

    The ``focus`` curves are always fully visible (12% padding). ``context``
    curves (the legs) may extend the range by at most one focus-span on each
    side; beyond that they are clipped -- they are context, not the story.
    """
    low = min(float(np.min(curve)) for curve in focus)
    high = max(float(np.max(curve)) for curve in focus)
    span = (high - low) or 1.0
    bottom, top = low - 0.12 * span, high + 0.12 * span
    if context:
        bottom = max(min(bottom, min(float(np.min(c)) for c in context) - 0.05 * span), low - span)
        top = min(max(top, max(float(np.max(c)) for c in context) + 0.05 * span), high + span)
    return [bottom, top]


def _free_corner(
    strategy: Instrument, mkt: Market, cost: float, horizon: float | None, root: float
) -> str:
    """Text position for a breakeven label that does not sit on the curve crossing it."""
    step = 1e-4 * max(root, 1.0)
    probes = np.array([max(root - step, 0.0), root + step])
    left, right = pnl_at_expiry(strategy, probes, mkt, cost, horizon)
    return "top left" if right > left else "top right"


def _line(x, y, name: str, color: str, width: float = 2.0, dash: str = "solid", **kwargs) -> go.Scatter:
    kwargs.setdefault("hovertemplate", _HOVER_MONEY)
    return go.Scatter(
        x=x, y=y, name=name, mode="lines", line=dict(color=color, width=width, dash=dash), **kwargs
    )


def _finish(
    fig: go.Figure,
    title: str,
    subtitle: str | None,
    mode: str,
    height: int,
    y_title: str,
    spot: float,
) -> go.Figure:
    """Theme, titles, legend under the plot (strategies have many series) and the spot line."""
    apply_theme(fig, title=title, height=height, mode=mode, hovermode="x unified")
    fig.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0.0),
        margin=dict(t=96 if subtitle else 72, b=96),
    )
    if subtitle:
        fig.update_layout(
            title_subtitle=dict(text=subtitle, font=dict(size=12, color=INK[mode]["secondary"]))
        )
    fig.update_xaxes(title_text="Underlying price", hoverformat=",.2f")
    fig.update_yaxes(title_text=y_title)
    add_vline(fig, spot, f"spot {spot:,.2f}", position="top")  # the legend sits under this plot
    return fig


# ---------------------------------------------------------------------- #
# Payoff diagram
# ---------------------------------------------------------------------- #
def payoff_diagram(
    strategy: Instrument,
    mkt: Market,
    show_legs: bool = True,
    show_today: bool = True,
    horizons: Sequence[float] = (0.5,),
    spot_range: tuple[float, float] | None = None,
    n_points: int = 401,
    mode: str = "auto",
) -> go.Figure:
    """The classic payoff diagram: P&L at expiry, today, and how the legs add up.

    Parameters
    ----------
    strategy : Instrument
        Usually a :class:`~optionlab.strategies.Strategy`; any instrument works.
    mkt : Market
        Scalar market: today's spot, vol, rates and clock. The entry cost is
        today's model price of the package.
    show_legs : bool, default True
        Draw each leg's own P&L at expiry as a thin dashed line; the bold
        curve is their sum, which is how a strategy should be read.
    show_today : bool, default True
        Draw today's P&L curve (what you would make if the spot moved NOW).
        The gap between this curve and the bold one is the time value still
        to be earned or lost.
    horizons : sequence of float, default (0.5,)
        Intermediate dates, as FRACTIONS of the time left to expiry (0.5 =
        half-way). Pass ``()`` for none.
    spot_range : (float, float), optional
        Range of the spot axis; by default it covers strikes and spot with a
        vol-dependent margin.
    n_points : int, default 401
        Resolution of the spot axis.
    mode : {"auto", "light", "dark"}
        Theme mode (see :func:`optionlab.plotting.theme.apply_theme`).

    Returns
    -------
    plotly.graph_objects.Figure
        Traces, in order: profit wash, loss wash, legs, intermediate dates,
        ``"Today"``, ``"At expiry"``, ``"Breakeven"``. The subtitle states net
        premium, maximum profit, maximum loss and breakevens; finite extremes
        visible in the window are also drawn as dotted levels.

    Notes
    -----
    For calendars and diagonals the bold curve is the value at the FRONT
    expiry with the back legs priced at today's vol -- a model assumption, as
    the title says.
    """
    _check_inputs(strategy, mkt)
    if any(not 0.0 < float(h) < 1.0 for h in horizons):
        raise ValueError("horizons are fractions of the time left to expiry, strictly between 0 and 1")
    horizon = analysis_horizon(strategy, mkt)
    cost = net_premium(strategy, mkt)
    roots = breakevens(strategy, mkt, cost, horizon)
    best, worst = max_profit(strategy, mkt, cost, horizon), max_loss(strategy, mkt, cost, horizon)

    grid = spot_grid(strategy, mkt, n_points, spot_range)
    visible_roots = [r for r in roots if grid[0] <= r <= grid[-1]]
    grid = np.unique(np.concatenate([grid, visible_roots]))  # exact corners for the shading
    pnl = pnl_at_expiry(strategy, grid, mkt, cost, horizon)

    fig = go.Figure()
    focus, context = [pnl, np.zeros(1)], []  # curves that set the y range / that may be clipped
    for name, part, color in (
        ("Profit zone", np.maximum(pnl, 0.0), COLORS["profit"]),
        ("Loss zone", np.minimum(pnl, 0.0), COLORS["loss"]),
    ):
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=part,
                name=name,
                mode="lines",
                line=dict(width=0),
                fill="tozeroy",
                fillcolor=with_alpha(color, 0.13),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if show_legs:
        for series in _leg_series(strategy):
            leg_pnl = sum(pnl_at_expiry(p, grid, mkt, horizon=horizon) for p in series.positions)
            context.append(leg_pnl)
            fig.add_trace(
                _line(grid, leg_pnl, series.label, series.color, 1.25, series.dash, legendrank=30)
            )

    tau = 0.0 if horizon is None else horizon - float(mkt.t)
    if tau > 0:
        for fraction in sorted(float(h) for h in horizons):
            later = mkt.bumped(spot=grid, t=float(mkt.t) + fraction * tau)
            focus.append(strategy.price(later) - cost)
            fig.add_trace(
                _line(
                    grid,
                    focus[-1],
                    f"In {_format_tau(fraction * tau)}",
                    with_alpha(_MODEL_COLOR, 0.5),
                    1.5,
                    legendrank=20,
                )
            )
    if show_today and (tau > 0 or horizon is None):
        focus.append(strategy.price(mkt.bumped(spot=grid)) - cost)
        fig.add_trace(_line(grid, focus[-1], "Today", _MODEL_COLOR, 2.0, legendrank=10))

    fig.add_trace(_line(grid, pnl, "At expiry", INK[mode]["primary"], 3.0, legendrank=1))
    if visible_roots:
        fig.add_trace(
            go.Scatter(
                x=visible_roots,
                y=[0.0] * len(visible_roots),
                name="Breakeven",
                mode="markers+text",
                marker=dict(symbol="diamond", size=9, color=INK[mode]["primary"]),
                text=[f"{r:,.2f}" for r in visible_roots],
                textposition=[_free_corner(strategy, mkt, cost, horizon, r) for r in visible_roots],
                textfont=dict(size=11, color=INK[mode]["secondary"]),
                hoverinfo="skip",
                legendrank=40,
            )
        )

    kind = "debit" if cost > 0 else "credit"
    subtitle = " | ".join(
        [
            f"Net {kind} {abs(cost):,.2f}",
            f"Max profit {_format_pnl(best)}",
            f"Max loss {_format_pnl(worst)}",
            "Breakeven " + (", ".join(f"{r:,.2f}" for r in roots) if roots else "none"),
        ]
    )
    title = f"{strategy.label}: P&L {_horizon_phrase(strategy, mkt, horizon)}"
    _finish(fig, title, subtitle, mode, 540, "Profit / loss", float(mkt.spot))
    fig.update_yaxes(range=_focused_range(focus, context))

    # Extremes are drawn only when the window actually reaches them; otherwise
    # the line would sit outside the axis range. The subtitle always has them.
    reach = 1e-6 * max(1.0, float(np.ptp(pnl)))
    if math.isfinite(best) and best - float(pnl.max()) <= reach:
        add_hline(fig, best, f"max profit {best:+,.2f}", dash="dot", position="top left")
    if math.isfinite(worst) and float(pnl.min()) - worst <= reach:
        add_hline(fig, worst, f"max loss {worst:+,.2f}", dash="dot", position="bottom left")
    return fig


# ---------------------------------------------------------------------- #
# Greeks: total and leg contributions
# ---------------------------------------------------------------------- #
def strategy_greeks_dashboard(
    strategy: Instrument,
    mkt: Market,
    greeks: Sequence[str] = ("delta", "gamma", "vega", "theta"),
    legs: str = "overlay",
    trader_units: bool = True,
    spot_range: tuple[float, float] | None = None,
    n_points: int = 201,
    ncols: int = 2,
    mode: str = "auto",
) -> go.Figure:
    """Greeks of the package against the spot, with the contribution of every leg.

    One panel per Greek (small multiples: Greeks live on different scales and
    must never share an axis). The bold line is the strategy; the legs show
    *why* it looks that way -- e.g. a butterfly's gamma is the long gamma of
    the wings minus twice the gamma of the body.

    Parameters
    ----------
    strategy : Instrument
    mkt : Market
        Scalar market (the spot is replaced by the axis).
    greeks : sequence of str, default ("delta", "gamma", "vega", "theta")
        Any of :data:`optionlab.instruments.GREEK_KEYS` (``"price"`` included).
    legs : {"overlay", "stack", "none"}, default "overlay"
        ``"overlay"``: one dashed line per leg. ``"stack"``: leg contributions
        as stacked areas -- positive ones piled above zero, negative ones
        below -- so the bold total is visibly "top of the positive pile plus
        bottom of the negative pile". ``"none"``: total only.
    trader_units : bool, default True
        Vega per vol point, theta per calendar day, rho per 1%... (panel
        titles always state the unit). False = RAW derivatives.
    spot_range, n_points, mode
        See :func:`payoff_diagram`.
    ncols : int, default 2
        Number of panel columns.

    Returns
    -------
    plotly.graph_objects.Figure
        Per panel: the leg traces, then ``"Total"``. Leg series share a legend
        entry across panels (click once to hide a leg everywhere).
    """
    _check_inputs(strategy, mkt)
    greeks = tuple(greeks)
    unknown = [g for g in greeks if g not in GREEK_KEYS]
    if unknown or not greeks:
        raise ValueError(f"unknown Greek(s) {unknown}; choose from {GREEK_KEYS}")
    if legs not in ("overlay", "stack", "none"):
        raise ValueError(f"legs must be 'overlay', 'stack' or 'none', got {legs!r}")

    grid = spot_grid(strategy, mkt, n_points, spot_range)
    ladder = mkt.bumped(spot=grid)
    convert = to_trader_units if trader_units else dict

    def profile(values: dict[str, Any]) -> dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype=float) + np.zeros_like(grid) for k, v in convert(values).items()}

    def contribution(leg_series: _LegSeries) -> dict[str, np.ndarray]:
        per_position = [p.greeks(ladder) for p in leg_series.positions]
        return profile({g: sum(values[g] for values in per_position) for g in greeks})

    total = profile(strategy.greeks(ladder))
    series = [] if legs == "none" else _leg_series(strategy)
    contributions = [contribution(s) for s in series]

    unit_key = "trader_unit" if trader_units else "raw_unit"
    titles = [f"{GREEK_INFO[g]['label']} - {GREEK_INFO[g][unit_key]}" for g in greeks]
    fig = subplot_grid(len(greeks), ncols, titles, mode=mode)
    ncols = min(ncols, len(greeks))
    for i, greek in enumerate(greeks):
        row, col = grid_position(i, ncols)
        for s, contribution in zip(series, contributions):
            shared = dict(legendgroup=s.label, showlegend=i == 0, hovertemplate=_HOVER_GREEK)
            y = contribution[greek]
            if legs == "overlay":
                fig.add_trace(_line(grid, y, s.label, s.color, 1.25, s.dash, **shared), row, col)
                continue
            for sign, part in (("pos", np.maximum(y, 0.0)), ("neg", np.minimum(y, 0.0))):
                fig.add_trace(
                    go.Scatter(
                        x=grid,
                        y=part,
                        customdata=y,
                        name=s.label,
                        mode="lines",
                        line=dict(color=s.color, width=0.5),
                        fillcolor=with_alpha(s.color, 0.3),
                        stackgroup=f"{sign}-{i}",
                        legendgroup=s.label,
                        showlegend=i == 0 and sign == "pos",
                        hovertemplate="%{customdata:.4g}" if sign == "pos" else None,
                        hoverinfo=None if sign == "pos" else "skip",
                    ),
                    row,
                    col,
                )
        fig.add_trace(
            _line(
                grid,
                total[greek],
                "Total",
                INK[mode]["primary"],
                3.0,
                legendgroup="Total",
                showlegend=i == 0,
                hovertemplate=_HOVER_GREEK,
                legendrank=1,
            ),
            row,
            col,
        )

    n_rows = math.ceil(len(greeks) / ncols)
    fig.update_xaxes(hoverformat=",.2f")
    for c in range(1, ncols + 1):
        fig.update_xaxes(title_text="Underlying price", row=n_rows, col=c)
    fig.update_layout(
        title_text=f"{strategy.label}: Greeks by leg ({'trader' if trader_units else 'raw'} units)",
        title_subtitle=dict(
            text=f"Dashed vertical line: spot {float(mkt.spot):,.2f}",
            font=dict(size=12, color=INK[mode]["secondary"]),
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.35 / n_rows, xanchor="left", x=0.0),
        margin=dict(t=110, b=96),
    )
    add_vline(fig, float(mkt.spot))  # after the traces, so that every panel gets one
    return fig


# ---------------------------------------------------------------------- #
# Time decay and volatility sensitivity
# ---------------------------------------------------------------------- #
def strategy_time_decay(
    strategy: Instrument,
    mkt: Market,
    taus: Sequence[float] | None = None,
    as_pnl: bool = True,
    spot_range: tuple[float, float] | None = None,
    n_points: int = 301,
    mode: str = "auto",
) -> go.Figure:
    """Value-versus-spot curves collapsing onto the payoff as expiry approaches.

    The model value of any option package is a smoothed version of its
    payoff; time is what sharpens it. For a long option the curve sinks
    towards the payoff (you pay theta), for a short-vol structure such as a
    butterfly or an iron condor it RISES towards the tent (you earn theta).

    Parameters
    ----------
    strategy : Instrument
    mkt : Market
        Scalar market.
    taus : sequence of float, optional
        Times to the (front) expiry, in years, at which to draw a curve.
        Default: 100%, 50%, 25% and 10% of the time currently left.
    as_pnl : bool, default True
        Plot P&L against today's entry cost; False plots the package value.
    spot_range, n_points, mode
        See :func:`payoff_diagram`.

    Returns
    -------
    plotly.graph_objects.Figure
        One blue curve per ``tau`` (light = far from expiry, dark = close),
        then the bold ``"At expiry"`` curve.
    """
    _check_inputs(strategy, mkt)
    horizon = analysis_horizon(strategy, mkt)
    if horizon is None:
        raise ValueError("time decay needs at least one leg with an expiry")
    if taus is None:
        left = horizon - float(mkt.t)
        if left <= 0:
            raise ValueError("the strategy is already at expiry: pass explicit taus")
        taus = [fraction * left for fraction in (1.0, 0.5, 0.25, 0.1)]
    taus = sorted({float(tau) for tau in taus}, reverse=True)
    if not taus or taus[-1] <= 0:
        raise ValueError("taus must be positive times to expiry (in years)")

    grid = spot_grid(strategy, mkt, n_points, spot_range)
    offset = net_premium(strategy, mkt) if as_pnl else 0.0
    fig = go.Figure()
    for tau, color in zip(taus, _ramp(len(taus), mode)):
        value = strategy.price(mkt.bumped(spot=grid, t=horizon - tau)) - offset
        fig.add_trace(_line(grid, value, f"{_format_tau(tau)} to expiry", color))
    at_expiry = pnl_at_expiry(strategy, grid, mkt, offset, horizon)
    fig.add_trace(_line(grid, at_expiry, "At expiry", INK[mode]["primary"], 3.0))

    what = "P&L" if as_pnl else "value"
    title = f"{strategy.label}: time decay of the {what} ({float(mkt.vol):.0%} vol)"
    return _finish(fig, title, None, mode, 500, "Profit / loss" if as_pnl else "Value", float(mkt.spot))


def strategy_vol_sensitivity(
    strategy: Instrument,
    mkt: Market,
    vols: Sequence[float] | None = None,
    as_pnl: bool = True,
    spot_range: tuple[float, float] | None = None,
    n_points: int = 301,
    mode: str = "auto",
) -> go.Figure:
    """Today's value-versus-spot curve under different implied volatilities.

    A long-vega package (straddle, calendar, backspread) lifts when vol goes
    up; a short-vega one (iron condor, butterfly, ratio spread) sinks. The
    spacing between the curves at a given spot IS the vega there.

    Parameters
    ----------
    strategy : Instrument
    mkt : Market
        Scalar market; its vol is the reference (drawn thicker).
    vols : sequence of float, optional
        Volatility levels (0.25 = 25%). Default: 50%, 75%, 100%, 125% and
        150% of the market vol.
    as_pnl : bool, default True
        Plot P&L against today's entry cost (at the market vol); False plots
        the package value.
    spot_range, n_points, mode
        See :func:`payoff_diagram`.

    Returns
    -------
    plotly.graph_objects.Figure
        One blue curve per vol (light = low, dark = high), then ``"At expiry"``.
    """
    _check_inputs(strategy, mkt)
    base = float(mkt.vol)
    if vols is None:
        vols = [base * m for m in (0.5, 0.75, 1.0, 1.25, 1.5)] if base > 0 else [0.1, 0.2, 0.3, 0.4]
    vols = sorted({float(v) for v in vols})
    if not vols or vols[0] < 0:
        raise ValueError("vols must be non-negative volatility levels")

    grid = spot_grid(strategy, mkt, n_points, spot_range)
    offset = net_premium(strategy, mkt) if as_pnl else 0.0
    fig = go.Figure()
    for vol, color in zip(vols, _ramp(len(vols), mode)):
        current = math.isclose(vol, base, rel_tol=1e-9)
        value = strategy.price(mkt.bumped(spot=grid, vol=vol)) - offset
        name = f"vol {vol:.1%}" + (" (market)" if current else "")
        fig.add_trace(_line(grid, value, name, color, 3.0 if current else 1.75))
    horizon = analysis_horizon(strategy, mkt)
    at_expiry = pnl_at_expiry(strategy, grid, mkt, offset, horizon)
    fig.add_trace(_line(grid, at_expiry, "At expiry", INK[mode]["primary"], 1.5, "dot"))

    what = "P&L" if as_pnl else "value"
    title = f"{strategy.label}: {what} today under different implied vols"
    return _finish(fig, title, None, mode, 500, "Profit / loss" if as_pnl else "Value", float(mkt.spot))


# ---------------------------------------------------------------------- #
# Side-by-side comparison
# ---------------------------------------------------------------------- #
def compare_strategies(
    strategies: Sequence[Instrument],
    mkt: Market,
    what: str = "pnl_expiry",
    trader_units: bool = True,
    spot_range: tuple[float, float] | None = None,
    n_points: int = 301,
    mode: str = "auto",
) -> go.Figure:
    """Overlay several strategies on one axis: P&L at expiry, P&L today, value or a Greek.

    Comparing a bull call spread with the outright call, or a straddle with a
    strangle, is the quickest way to see what each extra leg buys you.

    Parameters
    ----------
    strategies : sequence of Instrument
        Between 1 and 8 strategies (the categorical palette has 8 colours and
        is never cycled; beyond that, make several figures).
    mkt : Market
        Scalar market, shared by all strategies.
    what : str, default "pnl_expiry"
        ``"pnl_expiry"`` (each strategy at its own analysis horizon),
        ``"pnl_today"``, ``"value"`` or any Greek of
        :data:`optionlab.instruments.GREEK_KEYS`.
    trader_units : bool, default True
        Units used when ``what`` is a Greek.
    spot_range, n_points, mode
        See :func:`payoff_diagram`; the default range covers every strategy.
    """
    strategies = list(strategies)
    if not 1 <= len(strategies) <= len(PALETTE):
        raise ValueError(f"compare between 1 and {len(PALETTE)} strategies, got {len(strategies)}")
    for strategy in strategies:
        _check_inputs(strategy, mkt)
    choices = ("pnl_expiry", "pnl_today", "value") + GREEK_KEYS[1:]
    if what not in choices and what != "price":
        raise ValueError(f"what must be one of {choices}, got {what!r}")
    what = "value" if what == "price" else what

    if spot_range is None:
        grids = [spot_grid(s, mkt, 2) for s in strategies]
        spot_range = (min(g[0] for g in grids), max(g[-1] for g in grids))
    levels = sorted({level for s in strategies for level in reference_levels(s, mkt)})
    grid = np.linspace(spot_range[0], spot_range[1], int(n_points))
    grid = np.unique(np.concatenate([grid, [x for x in levels if grid[0] <= x <= grid[-1]]]))
    ladder = mkt.bumped(spot=grid)

    fig = go.Figure()
    for i, strategy in enumerate(strategies):
        if what == "pnl_expiry":
            y = pnl_at_expiry(strategy, grid, mkt)
        elif what == "pnl_today":
            y = strategy.price(ladder) - net_premium(strategy, mkt)
        elif what == "value":
            y = strategy.price(ladder)
        else:
            values = strategy.greeks(ladder)
            y = (to_trader_units(values) if trader_units else values)[what]
        duplicate = sum(s.label == strategy.label for s in strategies) > 1
        name = f"{strategy.label} #{i + 1}" if duplicate else strategy.label
        template = _HOVER_GREEK if what in GREEK_KEYS else _HOVER_MONEY
        y = np.asarray(y, dtype=float) + np.zeros_like(grid)
        fig.add_trace(_line(grid, y, name, PALETTE[i], hovertemplate=template))

    if what in GREEK_KEYS:
        info = GREEK_INFO[what]
        y_title = f"{info['label']} - {info['trader_unit' if trader_units else 'raw_unit']}"
        title = f"Strategy comparison: {info['label'].lower()} vs spot"
    else:
        y_title = "Value" if what == "value" else "Profit / loss"
        title = "Strategy comparison: " + {
            "pnl_expiry": "P&L at expiry",
            "pnl_today": "P&L today",
            "value": "value today",
        }[what]
    return _finish(fig, title, None, mode, 500, y_title, float(mkt.spot))
