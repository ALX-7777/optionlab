"""Teaching figures for the exotic options.

Four pictures, each built around the one idea that makes the product click:

* :func:`exotic_vs_vanilla` -- the exotic overlaid on its vanilla benchmark,
  Greek by Greek: the barrier's discontinuity, the digital's delta spike, the
  Asian's damped Greeks, the lookback's doubled premium.
* :func:`barrier_paths_figure` -- sample paths coloured by what happened to
  the option (knocked or not): a barrier option is a bet on the PATH.
* :func:`digital_replication_figure` -- a digital against call spreads of
  decreasing width: the hedge a desk really trades, and why tightening it
  makes the Greeks explode.
* :func:`asian_averaging_figure` -- one path with its running average: the
  average moves less than the spot and freezes as the window fills in.

Like every figure of the library they are RETURNED (never shown), styled by
:mod:`optionlab.plotting.theme`, and take ``mode="auto" | "light" | "dark"``.
"""

from __future__ import annotations

from typing import Any, Sequence, Union

import numpy as np
import plotly.graph_objects as go

from ..exotics.asian import AsianOption
from ..exotics.barrier import BarrierOption
from ..exotics.digital import DigitalOption
from ..exotics.lookback import LookbackOption
from ..instruments import GREEK_KEYS, Instrument, Position
from ..market import Market
from ..monte_carlo import Seed, simulate_gbm_paths
from . import theme

# _ramp_colors is the package-internal, validator-checked window of the sequential ramp.
from .profiles import (
    _ramp_colors,
    compare_instruments,
    evaluate_on_grid,
    greek_axis_label,
    key_levels,
)

__all__ = [
    "vanilla_benchmark",
    "exotic_vs_vanilla",
    "barrier_paths_figure",
    "digital_replication_figure",
    "asian_averaging_figure",
]

_SINGLE_HEIGHT = 460
_MAX_DRAWN_PATHS = 500
_MAX_WIDTHS = 6  # the sequential ramp separates at most six ordered steps in every mode


def _scalar_market(mkt: Any) -> Market:
    if not isinstance(mkt, Market) or not mkt.is_scalar:
        raise ValueError("plots need a scalar base Market; the grids are built internally")
    if not mkt.spot > 0:
        raise ValueError("plots need a strictly positive base spot")
    return mkt


def _require(instrument: Any, cls: type, what: str) -> None:
    if not isinstance(instrument, cls):
        raise ValueError(f"{what} needs a {cls.__name__}, got {type(instrument).__name__}")


def _require_alive(instrument: Instrument, mkt: Market, what: str) -> float:
    """Time left to expiry, which must be positive for a path to be simulated."""
    remaining = float(instrument.expiry - mkt.t)
    if remaining <= 0:
        raise ValueError(f"{instrument.label} has expired: {what} needs some time left to simulate")
    return remaining


def _subtitle(title: str, subtitle: str) -> str:
    return f"{title}<br><sup>{subtitle}</sup>"


# ---------------------------------------------------------------------- #
# Exotic vs vanilla
# ---------------------------------------------------------------------- #
def vanilla_benchmark(exotic: Instrument, mkt: Market) -> Instrument:
    """The vanilla option an exotic is naturally compared with.

    Uses the product's own ``vanilla_equivalent()`` (same type, strike and
    expiry). A floating-strike lookback has no strike: it is compared with the
    vanilla struck at its running extreme, i.e. at the money when it is fresh
    (which needs the current spot, hence the ``mkt`` argument).
    """
    if isinstance(exotic, LookbackOption):
        return exotic.vanilla_equivalent(float(_scalar_market(mkt).spot))
    method = getattr(exotic, "vanilla_equivalent", None)
    if not callable(method):
        raise ValueError(
            f"{type(exotic).__name__} has no vanilla_equivalent(); pass vanilla=... explicitly"
        )
    return method()


def _default_spot_range(product: Instrument, mkt: Market) -> tuple[float, float]:
    """70%-130% of the strikes/barriers/spot, stopping just beyond an outer barrier.

    Far beyond a barrier nothing happens to the exotic while the vanilla keeps
    growing, which would flatten the interesting region of the price panel.
    """
    levels = key_levels(product)
    values = [level for _, level in levels] + [float(mkt.spot)]
    lo, hi = 0.7 * min(values), 1.3 * max(values)
    if levels and levels[0][0] == "B" and levels[0][1] <= min(values):
        lo = 0.9 * levels[0][1]
    if levels and levels[-1][0] == "B" and levels[-1][1] >= max(values):
        hi = 1.1 * levels[-1][1]
    return lo, hi


def exotic_vs_vanilla(
    exotic: Union[Instrument, Position],
    mkt: Market,
    greeks: Sequence[str] = ("price", "delta", "gamma", "vega"),
    x: str = "spot",
    x_range: Sequence[float] | None = None,
    n: int = 201,
    *,
    vanilla: Instrument | None = None,
    ncols: int = 2,
    trader_units: bool = True,
    mode: str = "auto",
) -> go.Figure:
    """Overlay an exotic on its vanilla benchmark, one panel per Greek.

    Parameters
    ----------
    exotic : Instrument or Position
        Any exotic (or a signed position in one; the benchmark is then scaled
        by the same quantity).
    mkt : Market
        Scalar base market.
    greeks : sequence of str, default ("price", "delta", "gamma", "vega")
        Panels: ``"price"`` and/or any Greek name.
    x, x_range, n, trader_units, mode
        As in :func:`optionlab.plotting.profiles.greek_profile`, except for the
        default spot range: 70% to 130% of the strike / barrier / spot levels,
        cut 10% beyond an outer barrier (past it the exotic is dead or has
        become the vanilla, and the vanilla's growth would flatten the rest).
    vanilla : Instrument, optional
        Benchmark to use instead of :func:`vanilla_benchmark`.
    ncols : int, default 2
        Panel columns.

    Returns
    -------
    plotly.graph_objects.Figure
        Two line traces per panel: the exotic (solid, call blue / put orange)
        and the vanilla (dashed grey).

    Notes
    -----
    What to look for:

    * **knock-out barrier** -- the price is pulled to zero (or the rebate) at
      the barrier, so delta turns NEGATIVE before it and gamma and vega change
      sign: near the barrier the holder is short gamma and short volatility;
    * **digital** -- delta is a bump centred on the strike that becomes a
      spike near expiry, gamma and vega flip sign at the strike;
    * **Asian** -- same shapes as the vanilla, scaled down: the average moves
      less than the spot, and less and less as it fills in;
    * **lookback** -- always above the vanilla; a fresh floating-strike one
      costs about twice the at-the-money option.
    """
    mkt = _scalar_market(mkt)
    if isinstance(exotic, Position):
        product, quantity = exotic.instrument, exotic.quantity
    else:
        product, quantity = exotic, 1.0
    if not isinstance(product, Instrument):
        raise ValueError(f"expected an Instrument or a Position, got {exotic!r}")
    benchmark = vanilla_benchmark(product, mkt) if vanilla is None else vanilla
    if not isinstance(benchmark, Instrument):
        raise ValueError(f"vanilla must be an Instrument, got {benchmark!r}")
    twin: Union[Instrument, Position] = benchmark
    if isinstance(exotic, Position):
        twin = Position(benchmark, quantity)

    if x == "spot" and x_range is None:
        x_range = _default_spot_range(product, mkt)
    is_put = getattr(product, "option_type", None) == "put"
    colors = (theme.COLORS["put"] if is_put else theme.COLORS["call"], theme.COLORS["neutral"])
    series = {str(exotic.label): exotic, f"Vanilla: {twin.label}": twin}
    return compare_instruments(
        series, mkt, tuple(greeks), x, x_range, n,
        colors=colors, title=f"{exotic.label} vs its vanilla benchmark",
        ncols=ncols, trader_units=trader_units, mode=mode,
    )


# ---------------------------------------------------------------------- #
# Barrier: sample paths
# ---------------------------------------------------------------------- #
def _nan_separated(times: np.ndarray, paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Many paths as ONE line trace: rows glued end to end with NaN breaks."""
    n_paths = paths.shape[0]
    x = np.tile(np.append(times, np.nan), n_paths)
    y = np.concatenate([paths, np.full((n_paths, 1), np.nan)], axis=1).ravel()
    return x, y


def barrier_paths_figure(
    barrier_option: BarrierOption,
    mkt: Market,
    n_paths: int = 60,
    seed: Seed = None,
    n_steps: int = 252,
    *,
    mode: str = "auto",
) -> go.Figure:
    """Simulated spot paths, coloured by what they do to a barrier option.

    Parameters
    ----------
    barrier_option : BarrierOption
        Must not be expired. If it is already ``knocked`` every path counts as
        knocked.
    mkt : Market
        Scalar market; paths are risk-neutral GBM from ``mkt.t`` to expiry.
    n_paths : int, default 60
        Number of paths drawn (1 to 500).
    seed : int, numpy Generator or None
        Seed of the simulation (pass one for a reproducible picture).
    n_steps : int, default 252
        Time steps; the barrier is watched at these dates only.
    mode : {"auto", "light", "dark"}

    Returns
    -------
    plotly.graph_objects.Figure
        Up to three traces -- the paths on which the option is ALIVE at expiry
        (blue), the paths on which it is DEAD (grey), and the first touch of
        the barrier on each knocked path (markers) -- plus the barrier and
        strike lines. The subtitle gives the knocked share and the prices.

    Notes
    -----
    For a knock-OUT the touching paths are the dead ones; for a knock-IN they
    are the only ones that pay. Either way the payoff depends on the whole
    path, not only on where the spot ends: two paths finishing at the same
    level can pay very different amounts. The share of touching paths is what
    the discount (knock-out) or the price (knock-in) relative to the vanilla
    is made of.
    """
    _require(barrier_option, BarrierOption, "barrier_paths_figure")
    mkt = _scalar_market(mkt)
    remaining = _require_alive(barrier_option, mkt, "barrier_paths_figure")
    n_paths, n_steps = int(n_paths), int(n_steps)
    if not 1 <= n_paths <= _MAX_DRAWN_PATHS:
        raise ValueError(f"n_paths must be between 1 and {_MAX_DRAWN_PATHS}, got {n_paths}")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    option = barrier_option
    times, paths = simulate_gbm_paths(
        mkt.spot, mkt.vol, mkt.rate, mkt.div, horizon=remaining, n_steps=n_steps,
        n_paths=n_paths, seed=seed, antithetic=False, t0=mkt.t,
    )
    beyond = paths >= option.barrier if option.is_up else paths <= option.barrier
    touched = beyond.any(axis=1) | option.knocked
    alive = touched if option.is_knock_in else ~touched

    if option.is_knock_in:
        alive_name, dead_name = "Knocked in: pays like the vanilla", "Never activated"
    else:
        alive_name, dead_name = "Survives to expiry", "Knocked out"
    fig = go.Figure()
    styles = (
        (~alive, dead_name, theme.COLORS["neutral"], 0.45),
        (alive, alive_name, theme.COLORS["call"], 0.75),
    )
    for mask, name, color, opacity in styles:  # dead paths first: the live ones stay on top
        if mask.any():
            x, y = _nan_separated(times, paths[mask])
            fig.add_trace(
                go.Scatter(
                    x=x, y=y, mode="lines", name=f"{name} ({int(mask.sum())})",
                    line=dict(color=color, width=1), opacity=opacity,
                    hovertemplate="t = %{x:.3f}y<br>spot %{y:.2f}<extra>" + name + "</extra>",
                )
            )
    first_touch = beyond.any(axis=1)
    if first_touch.any():
        columns = np.argmax(beyond[first_touch], axis=1)
        fig.add_trace(
            go.Scatter(
                x=times[columns], y=paths[first_touch, columns], mode="markers",
                name="First touch of the barrier",
                marker=dict(color=theme.COLORS["put"], size=7, line=dict(width=0)),
                hovertemplate="first touch at t = %{x:.3f}y<br>spot %{y:.2f}<extra></extra>",
            )
        )

    vanilla_price = float(option.vanilla_equivalent().price(mkt))
    share = float(touched.mean())
    subtitle = (
        f"{share:.0%} of {n_paths} sample paths touch H = {option.barrier:g} "
        f"(watched at {n_steps} dates) | model price {float(option.price(mkt)):.3f} "
        f"vs vanilla {vanilla_price:.3f} | dotted line: strike {option.strike:g}"
    )
    theme.apply_theme(
        fig, title=_subtitle(f"Sample paths: {option.label}", subtitle),
        height=_SINGLE_HEIGHT, mode=mode, hovermode="closest",
    )
    fig.update_layout(margin=dict(t=96))
    fig.update_xaxes(title_text="Calendar time (years)")
    fig.update_yaxes(title_text="Spot")
    outside = "top right" if option.is_up else "bottom right"  # label on the dead side of the barrier
    theme.add_hline(fig, option.barrier, f"barrier H {option.barrier:g}", position=outside,
                    color=theme.COLORS["put"], dash="dash", width=1.5)
    # Unlabelled (the subtitle names it): the strike runs through the thick of the paths,
    # where a label would be unreadable.
    theme.add_hline(fig, option.strike, dash="dot")
    return fig


# ---------------------------------------------------------------------- #
# Digital: replication with call spreads
# ---------------------------------------------------------------------- #
def digital_replication_figure(
    digital: DigitalOption,
    mkt: Market,
    widths: Sequence[float] | None = None,
    quantities: Sequence[str] = ("payoff", "price", "delta"),
    placement: str = "centered",
    x_range: Sequence[float] | None = None,
    n: int = 401,
    *,
    ncols: int = 3,
    trader_units: bool = True,
    mode: str = "auto",
) -> go.Figure:
    """A digital against vanilla spreads of decreasing width.

    Parameters
    ----------
    digital : DigitalOption
    mkt : Market
        Scalar base market.
    widths : sequence of float, optional
        Strike distances of the replicating spreads (at most 6). Default: 20%,
        10%, 5% and 2% of the strike. Drawn from the widest (lightest) to the
        tightest (darkest).
    quantities : sequence of str, default ("payoff", "price", "delta")
        One panel each: ``"payoff"`` (terminal payoff), ``"price"`` or any
        Greek name.
    placement : {"centered", "conservative"}, default "centered"
        Passed to :meth:`DigitalOption.replicating_call_spread`;
        ``"conservative"`` is the super-replicating spread a seller books.
    x_range : (low, high), optional
        Spot range; default 70% to 130% of the strike.
    n : int, default 401
        Spot grid size.
    ncols, trader_units, mode
        Layout, units of the Greek panels, page mode.

    Returns
    -------
    plotly.graph_objects.Figure
        ``(len(widths) + 1) * len(quantities)`` line traces; the digital is the
        bold ink line, the spreads are a light-to-dark ramp.

    Notes
    -----
    The spread ``payout / width * (C(K - w/2) - C(K + w/2))`` is a finite
    difference of the call price in the strike, and the digital is its limit
    ``-dC/dK``. Tightening the spread brings the PRICE to the digital's
    quickly (the error shrinks like ``width**2``) but multiplies the number of
    options by ``1 / width``: the peak delta of the hedge is about
    ``payout / width``. That is the trader's dilemma in one picture -- a wide
    spread is a poor replication, a tight one is an unmanageable position --
    and it is why digitals are priced and risk-managed as the (conservative)
    spread, not as the step.
    """
    _require(digital, DigitalOption, "digital_replication_figure")
    mkt = _scalar_market(mkt)
    n = int(n)
    if n < 2:
        raise ValueError(f"the number of grid points must be at least 2, got {n}")
    quantities = tuple(quantities)
    unknown = [q for q in quantities if q != "payoff" and q not in GREEK_KEYS]
    if not quantities or unknown:
        raise ValueError(f"quantities must be 'payoff' or one of {list(GREEK_KEYS)}, got {list(quantities)}")
    if widths is None:
        widths = tuple(fraction * digital.strike for fraction in (0.20, 0.10, 0.05, 0.02))
    widths = sorted({float(w) for w in widths}, reverse=True)
    if not 1 <= len(widths) <= _MAX_WIDTHS:
        raise ValueError(f"give between 1 and {_MAX_WIDTHS} widths, got {len(widths)}")
    if x_range is None:
        lo, hi = 0.7 * digital.strike, 1.3 * digital.strike
    else:
        lo, hi = (float(v) for v in x_range)
        if not 0 < lo < hi:
            raise ValueError(f"x_range must satisfy 0 < low < high, got {x_range!r}")
    grid = np.unique(np.concatenate([np.linspace(lo, hi, n), [digital.strike]]))
    grid = grid[(grid >= lo) & (grid <= hi)]

    spreads = [digital.replicating_call_spread(width, placement) for width in widths]
    series = [
        (f"Spread, width {width:g}", spread, color, 2)
        for width, spread, color in zip(widths, spreads, _ramp_colors(len(widths), mode))
    ]
    series.append((f"Digital {digital.label}", digital, theme.INK[mode]["primary"], 3))

    greek_keys = tuple(q for q in quantities if q != "payoff")
    curves = []
    for _, instrument, _, _ in series:
        values: dict[str, np.ndarray] = {}
        if greek_keys:
            values.update(evaluate_on_grid(instrument, mkt, greek_keys, trader_units=trader_units, spot=grid))
        if "payoff" in quantities:
            values["payoff"] = np.asarray(instrument.payoff(grid), dtype=float)
        curves.append(values)

    titles = [
        "Payoff at expiry" if q == "payoff" else f"{greek_axis_label(q, trader_units)} today"
        for q in quantities
    ]
    kind = "put" if not digital.is_call else "call"
    subtitle = (
        f"{placement} {kind} spreads, light = wide, dark = tight; bold line = the digital; "
        f"dashed vertical = strike {digital.strike:g}"
    )
    fig = theme.subplot_grid(
        len(quantities), ncols, titles,
        title=_subtitle(f"Replicating {digital.label} with vanilla spreads", subtitle), mode=mode,
    )
    # Five long legend entries do not fit beside the subtitle: the legend goes under the panels.
    fig.update_layout(
        margin=dict(t=104, b=96),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0.0),
    )
    n_cols = min(ncols, len(quantities))
    for panel, quantity in enumerate(quantities):
        row, col = theme.grid_position(panel, n_cols)
        for (name, _, color, width), values in zip(series, curves):
            fig.add_trace(
                go.Scatter(
                    x=grid, y=values[quantity], mode="lines", name=name,
                    line=dict(color=color, width=width), legendgroup=name, showlegend=panel == 0,
                    hovertemplate=f"{name}: %{{y:.4f}}<extra></extra>",
                ),
                row=row, col=col,
            )
        fig.update_xaxes(
            hoverformat=".2f", title_text="Spot" if panel + n_cols >= len(quantities) else None,
            row=row, col=col,
        )
    theme.add_vline(fig, digital.strike)  # after the traces, so that every panel gets one
    return fig


# ---------------------------------------------------------------------- #
# Asian: one path and its running average
# ---------------------------------------------------------------------- #
def asian_averaging_figure(
    asian: AsianOption,
    mkt: Market,
    seed: Seed = None,
    n_steps: int = 252,
    *,
    mode: str = "auto",
) -> go.Figure:
    """One simulated path with the running average an Asian option pays on.

    Parameters
    ----------
    asian : AsianOption
        Fresh, forward-starting or seasoned (the stored average is the
        starting point of the running average). Must not be expired.
    mkt : Market
        Scalar market; the path is a risk-neutral GBM from ``mkt.t`` to expiry.
    seed : int, numpy Generator or None
    n_steps : int, default 252
        Time steps of the path (the averaging dates).
    mode : {"auto", "light", "dark"}

    Returns
    -------
    plotly.graph_objects.Figure
        Two line traces -- the spot and its running average (built with
        ``asian.observe`` step by step, so exactly what a book would record)
        -- plus the strike line. The subtitle compares what the Asian and the
        vanilla pay on this path.

    Notes
    -----
    Watch the average (orange) lag behind the spot (blue) and calm down as the
    window fills in: late moves of the spot barely move it. That is why an
    Asian is cheaper than a vanilla (its effective volatility is about
    ``vol / sqrt(3)``) and why its delta and gamma fade away near expiry
    instead of exploding.
    """
    _require(asian, AsianOption, "asian_averaging_figure")
    mkt = _scalar_market(mkt)
    remaining = _require_alive(asian, mkt, "asian_averaging_figure")
    n_steps = int(n_steps)
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")

    times, paths = simulate_gbm_paths(
        mkt.spot, mkt.vol, mkt.rate, mkt.div, horizon=remaining, n_steps=n_steps,
        n_paths=1, seed=seed, antithetic=False, t0=mkt.t,
    )
    path = paths[0]
    times[-1] = asian.expiry  # exact landing on the expiry, whatever the rounding of t0 + horizon
    state, average = asian, np.full(path.shape, np.nan)
    for i, (t, spot) in enumerate(zip(times, path)):
        state = state.observe(float(spot), float(t))
        if state.running_average is not None:
            average[i] = state.running_average

    asian_payoff = float(state.payoff(path[-1]))
    vanilla_payoff = float(asian.vanilla_equivalent().payoff(path[-1]))
    subtitle = (
        f"final average {average[-1]:.2f} vs last spot {path[-1]:.2f} | strike {asian.strike:g}: "
        f"the Asian pays {asian_payoff:.2f}, the vanilla would pay {vanilla_payoff:.2f}"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=times, y=path, mode="lines", name="Spot",
                   line=dict(color=theme.PALETTE[0], width=1.5), hovertemplate="spot: %{y:.2f}<extra></extra>")
    )
    fig.add_trace(
        go.Scatter(x=times, y=average, mode="lines", name=f"Running {asian.averaging} average",
                   line=dict(color=theme.PALETTE[1], width=3), hovertemplate="average: %{y:.2f}<extra></extra>")
    )
    theme.apply_theme(
        fig, title=_subtitle(f"Averaging in action: {asian.label}", subtitle),
        height=_SINGLE_HEIGHT, mode=mode, hovermode="x unified",
    )
    fig.update_layout(margin=dict(t=96))
    fig.update_xaxes(title_text="Calendar time (years)", hoverformat=".3f")
    fig.update_yaxes(title_text="Spot / running average")
    theme.add_hline(fig, asian.strike, f"strike K {asian.strike:g}", position="bottom right")
    if asian.avg_start > mkt.t:
        theme.add_vline(fig, asian.avg_start, "averaging starts", position="top right")
    return fig
