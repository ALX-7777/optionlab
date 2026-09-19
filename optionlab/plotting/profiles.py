"""Generic Greek visualisation: profiles, dashboards, families of curves, surfaces.

Every function here takes ANY :class:`~optionlab.instruments.Instrument` (a
vanilla option, a multi-leg strategy, an exotic) or a signed
:class:`~optionlab.instruments.Position`, plus a scalar base
:class:`~optionlab.market.Market`, and returns a themed
``plotly.graph_objects.Figure``. Nothing is ever shown or written to disk.

How it works
------------
The whole module rests on one helper, :func:`evaluate_on_grid`. It builds a
*grid market* (``mkt.bumped(spot=<array>, t=<array>, ...)``) and asks the
instrument for its price or Greeks **once**: closed-form products answer with
arrays, so a 200-point profile or a 60 x 40 surface costs a single vectorised
call. Instruments whose pricer only accepts scalar markets (a Monte Carlo
product, say) are detected automatically and evaluated point by point instead,
so every figure works for every instrument -- just more slowly.

Time is always moved through the market clock (``tau = expiry - mkt.t``); the
instrument is never modified. For path-dependent products a figure therefore
answers "what if the path state stored on the instrument stayed as it is?".

Units
-----
All figures accept ``trader_units`` (default ``True``): vega per vol point,
theta and charm per calendar day, rho per 1% (see
:func:`optionlab.black_scholes.to_trader_units`). With ``trader_units=False``
the raw mathematical derivatives are plotted. The unit is always written on the
axis (or panel title) so a number is never ambiguous.

Reading guide
-------------
* :func:`greek_profile`, :func:`greek_dashboard` -- one Greek / several Greeks
  against spot, vol, time or rate, with today's market marked.
* :func:`greek_evolution` -- THE classic picture: the same profile redrawn as
  expiry approaches (or for several vol / rate levels).
* :func:`greek_vs_time` -- a Greek along calendar time for a few spot levels.
* :func:`greek_surface` -- the two previous pictures merged into a surface.
* :func:`compare_instruments`, :func:`call_put_comparison` -- overlays.
* :func:`pnl_profile` -- value today, at later dates and at expiry.
* :func:`taylor_pnl_explain`, :func:`gamma_theta_tradeoff` -- what the Greeks
  are *for*: explaining a P&L, and the rent (theta) paid for convexity (gamma).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Mapping, Sequence, Union

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

from .. import black_scholes as bs
from ..instruments import EXPIRY_TOL, EuropeanOption, Instrument, Position
from ..market import Market
from . import theme

__all__ = [
    "DEFAULT_DASHBOARD_GREEKS",
    "evaluate_on_grid",
    "greek_axis_label",
    "key_levels",
    "greek_profile",
    "greek_dashboard",
    "greek_evolution",
    "greek_vs_time",
    "greek_surface",
    "compare_instruments",
    "call_put_comparison",
    "pnl_profile",
    "taylor_pnl_explain",
    "gamma_theta_tradeoff",
]

Priceable = Union[Instrument, Position]

#: Greeks shown by the dashboards when none are requested.
DEFAULT_DASHBOARD_GREEKS: tuple[str, ...] = ("price", "delta", "gamma", "vega", "theta", "rho")

_MARKET_FIELDS = ("spot", "vol", "rate", "div", "t")
_TAU_FLOOR = 1e-4  # years (about 53 minutes): never evaluate a grid closer to expiry than this
_ONE_DAY = 1.0 / 365.0
_TENOR_LADDER = (1.0, 0.5, 0.25, 1.0 / 12.0, 1.0 / 52.0, _ONE_DAY)
_MAX_FAMILY = 8  # more ordered curves than this stop being distinguishable
_DASHES = ("solid", "dash", "dot", "dashdot")
_SINGLE_HEIGHT = 460
_SURFACE_HEIGHT = 620

# Window of the sequential ramp used for ordered families of lines. The bounds
# were chosen with the dataviz ordinal validator: both ends keep >= 2:1
# contrast against the page and up to 6 (auto), 7 (light) or 8 (dark) steps
# stay at least 0.06 OKLCH lightness apart.
_RAMP_WINDOW = {"auto": (0.24, 0.86), "light": (0.24, 1.0), "dark": (0.20, 1.0)}

# Compact unit captions (raw, trader). The long-form text lives in bs.GREEK_INFO.
_COMPACT_UNITS: dict[str, tuple[str, str]] = {
    "price": ("", ""),
    "delta": ("", ""),
    "gamma": ("per 1 spot", "per 1 spot"),
    "vega": ("per 1.00 vol", "per vol point"),
    "theta": ("per year", "per day"),
    "rho": ("per 1.00 rate", "per 1% rate"),
    "vanna": ("delta per 1.00 vol", "delta per vol point"),
    "volga": ("vega per 1.00 vol", "vega per vol point"),
    "charm": ("delta per year", "delta per day"),
    "speed": ("gamma per 1 spot", "gamma per 1 spot"),
    "color": ("gamma per year", "gamma per day"),
    "zomma": ("gamma per 1.00 vol", "gamma per vol point"),
}


# ---------------------------------------------------------------------- #
# Axes
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Axis:
    """How one market dimension is displayed."""

    key: str
    title: str
    tickformat: str  # d3-format for ticks ("" = plotly default)
    hoverformat: str
    legend_title: str
    reverse: bool = False


_AXES: dict[str, _Axis] = {
    "spot": _Axis("spot", "Spot", "", ".2f", "Spot"),
    "vol": _Axis("vol", "Volatility", ".0%", ".1%", "Volatility"),
    "tau": _Axis(
        "tau", "Time to expiry (years, axis reversed)", "", ".4f", "Time to expiry", reverse=True
    ),
    "t": _Axis("t", "Calendar time (years)", "", ".4f", "Date"),
    "rate": _Axis("rate", "Interest rate", ".1%", ".2%", "Interest rate"),
    "div": _Axis("div", "Dividend yield", ".1%", ".2%", "Dividend yield"),
}


def _axis(name: str) -> _Axis:
    if name not in _AXES:
        raise ValueError(f"axis must be one of {sorted(_AXES)}, got {name!r}")
    return _AXES[name]


def _check_axis_pair(first: str, second: str) -> None:
    """Two axes of one figure must move two DIFFERENT market dimensions."""
    clock = {"tau", "t"}
    if first == second or {first, second} <= clock:
        raise ValueError(
            f"{first!r} and {second!r} describe the same market dimension; pick two different ones"
        )


def _format_tenor(tau: float) -> str:
    """Compact maturity label: 1y, 1.5y, 6m, 1w, 46d, 12h..."""
    if tau >= 1.0 - 1e-9:
        return f"{tau:.3g}y"
    for unit, per_year in (("m", 12.0), ("w", 52.0), ("d", 365.0)):
        count = tau * per_year
        if count >= 1.0 - 1e-9 and abs(count - round(count)) < 1e-6:
            return f"{round(count):d}{unit}"
    days = tau * 365.0
    if days >= 10.0:
        return f"{days:.0f}d"
    return f"{days:.2g}d" if days >= 1.0 else f"{days * 24.0:.2g}h"


def _level_label(axis: _Axis, value: float) -> str:
    """Legend entry for one level of an axis (one member of a family of curves)."""
    if axis.key == "spot":
        return f"S = {value:g}"
    if axis.key == "tau":
        return _format_tenor(value)
    if axis.key == "t":
        return f"t = {value:.3g}y"
    prefix = {"vol": "vol", "rate": "r =", "div": "q ="}[axis.key]
    return f"{prefix} {value * 100.0:g}%"


# ---------------------------------------------------------------------- #
# Instrument introspection (duck-typed: never assumes a EuropeanOption)
# ---------------------------------------------------------------------- #
def _check_priceable(instrument: Any) -> Priceable:
    if not isinstance(instrument, (Instrument, Position)):
        raise ValueError(f"expected an Instrument or a Position, got {instrument!r}")
    return instrument


def _check_market(mkt: Any) -> Market:
    if not isinstance(mkt, Market) or not mkt.is_scalar:
        raise ValueError("plots need a scalar base Market; the grids are built internally")
    if not mkt.spot > 0:
        raise ValueError("plots need a strictly positive base spot")
    return mkt


def _check_n(n: int, minimum: int = 2) -> int:
    if int(n) < minimum:
        raise ValueError(f"the number of grid points must be at least {minimum}, got {n}")
    return int(n)


def _unwrap(instrument: Priceable) -> Instrument:
    return instrument.instrument if isinstance(instrument, Position) else instrument


def _label(instrument: Priceable) -> str:
    return str(getattr(instrument, "label", type(instrument).__name__))


def _series_color(instrument: Priceable) -> str:
    """Puts are orange and everything else is blue, as everywhere in the library."""
    is_put = getattr(_unwrap(instrument), "option_type", None) == "put"
    return theme.COLORS["put"] if is_put else theme.COLORS["call"]


def _require_expiry(instrument: Priceable, what: str) -> float:
    expiry = instrument.expiry
    if expiry is None:
        raise ValueError(f"{_label(instrument)} never expires, so it cannot be plotted against {what}")
    return float(expiry)


def key_levels(instrument: Priceable) -> list[tuple[str, float]]:
    """Spot levels worth marking on a chart: strikes (``"K"``) and barriers (``"B"``).

    The search is generic: any dataclass field whose name contains ``strike``
    or ``barrier`` and holds a positive number counts, and composites /
    positions are searched recursively. Returns ``(tag, level)`` pairs sorted
    by level, without duplicates.
    """
    found: set[tuple[str, float]] = set()

    def visit(obj: Any) -> None:
        if isinstance(obj, Position):
            visit(obj.instrument)
            return
        legs = getattr(obj, "legs", ())
        if isinstance(legs, (tuple, list)):
            for leg in legs:
                visit(leg)
        if not is_dataclass(obj):
            return
        for f in fields(obj):
            name = f.name.lower()
            tag = "K" if "strike" in name else "B" if "barrier" in name else None
            value = getattr(obj, f.name)
            if tag is None or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if math.isfinite(value) and value > 0:
                found.add((tag, float(value)))

    visit(_check_priceable(instrument))
    return sorted(found, key=lambda pair: (pair[1], pair[0]))


def _merged_levels(instruments: Sequence[Priceable]) -> list[tuple[str, float]]:
    merged = {pair for inst in instruments for pair in key_levels(inst)}
    return sorted(merged, key=lambda pair: (pair[1], pair[0]))


# ---------------------------------------------------------------------- #
# Labels and number formats
# ---------------------------------------------------------------------- #
def _greek_name(greek: str) -> str:
    info = bs.GREEK_INFO.get(greek)
    return info["label"] if info else greek.replace("_", " ").capitalize()


def greek_axis_label(greek: str, trader_units: bool = True) -> str:
    """Axis caption for a Greek, unit included: ``"Vega (per vol point)"``.

    Unknown keys (an instrument may expose extra Greeks) get a capitalised
    name and no unit.
    """
    unit = _COMPACT_UNITS.get(greek, ("", ""))[1 if trader_units else 0]
    return f"{_greek_name(greek)} ({unit})" if unit else _greek_name(greek)


def _number_format(*arrays: Any) -> str:
    """A format spec valid for BOTH d3 (hover) and Python (labels), sized to the data."""
    finite = [np.abs(a[np.isfinite(a)]) for a in (np.asarray(v, dtype=float) for v in arrays)]
    peak = max((float(a.max()) for a in finite if a.size), default=0.0)
    if peak == 0.0 or peak >= 1000.0:
        return ".2f"
    if peak >= 1.0:
        return ".3f"
    if peak >= 0.1:
        return ".4f"
    if peak >= 1e-3:
        return ".5f"
    return ".3e"


def _ramp_colors(n: int, mode: str) -> list[str]:
    """``n`` ordered colours from the sequential ramp; the LAST one is the most prominent."""
    scale = theme.sequential_scale(mode)  # raises a clear ValueError on an unknown mode
    lo, hi = _RAMP_WINDOW[mode]
    return sample_colorscale(scale, [hi] if n == 1 else list(np.linspace(lo, hi, n)))


# ---------------------------------------------------------------------- #
# THE helper: evaluate price / Greeks over any grid of market values
# ---------------------------------------------------------------------- #
def _grid_market(instrument: Priceable, mkt: Market, axes: Mapping[str, Any]) -> Market:
    """Base market with some fields replaced by (broadcastable) arrays."""
    if "tau" in axes and "t" in axes:
        raise ValueError("pass either 'tau' or 't', not both: they move the same clock")
    changes: dict[str, np.ndarray] = {}
    for name, values in axes.items():
        arr = np.asarray(values, dtype=float)
        if name == "tau":
            changes["t"] = _require_expiry(instrument, "time to expiry") - arr
        elif name in _MARKET_FIELDS:
            changes[name] = arr
        else:
            raise ValueError(f"unknown grid axis {name!r}; valid axes are {sorted(_AXES)}")
    return mkt.bumped(**changes)


def _compute(instrument: Priceable, mkt: Market, keys: tuple[str, ...]) -> dict[str, Any]:
    """Requested quantities at ``mkt`` (pricing only when nothing but the price is asked)."""
    if all(key == "price" for key in keys):
        return {"price": instrument.price(mkt)}
    greeks = instrument.greeks(mkt)
    missing = [key for key in keys if key not in greeks]
    if missing:
        raise ValueError(
            f"{_label(instrument)} has no Greek(s) {missing}; available: {list(greeks)}"
        )
    return {key: greeks[key] for key in keys}


def _try_vectorised(
    instrument: Priceable, grid_mkt: Market, keys: tuple[str, ...]
) -> dict[str, np.ndarray] | None:
    """One array call; ``None`` when the instrument cannot digest an array market."""
    try:
        values = {k: np.asarray(v, dtype=float) for k, v in _compute(instrument, grid_mkt, keys).items()}
    except Exception:  # noqa: BLE001 -- ANY failure just means "not vectorised": the scalar
        return None  # loop below re-raises genuine errors with their original message.
    if all(v.shape == grid_mkt.shape for v in values.values()):
        return values
    return None


def _scalar_loop(
    instrument: Priceable, grid_mkt: Market, keys: tuple[str, ...]
) -> dict[str, np.ndarray]:
    """Point-by-point evaluation with scalar markets (slow but universal)."""
    shape = grid_mkt.shape
    columns = {
        name: np.broadcast_to(np.asarray(getattr(grid_mkt, name), dtype=float), shape)
        for name in _MARKET_FIELDS
    }
    out = {key: np.empty(shape) for key in keys}
    for index in np.ndindex(shape):
        point = Market(**{name: float(column[index]) for name, column in columns.items()})
        for key, value in _compute(instrument, point, keys).items():
            out[key][index] = float(value)
    return out


def evaluate_on_grid(
    instrument: Priceable,
    mkt: Market,
    greeks: str | Sequence[str] = "price",
    *,
    trader_units: bool = False,
    **axes: Any,
) -> dict[str, np.ndarray]:
    """Evaluate the price and/or Greeks of any instrument over a market grid.

    Parameters
    ----------
    instrument : Instrument or Position
        Anything priceable. Never modified.
    mkt : Market
        Base market; every dimension not listed in ``axes`` keeps its value.
    greeks : str or sequence of str, default "price"
        ``"price"`` and/or any key of ``instrument.greeks(mkt)``. When only the
        price is requested the (possibly expensive) Greeks are not computed.
    trader_units : bool, default False
        Convert with :func:`optionlab.black_scholes.to_trader_units`.
    **axes : array_like
        Grid values for any of ``spot``, ``vol``, ``rate``, ``div``, ``t`` or
        ``tau`` (time to expiry, translated to ``t = expiry - tau``). Arrays
        only need to broadcast: ``spot=s[None, :], tau=taus[:, None]`` gives a
        ``(len(taus), len(s))`` surface.

    Returns
    -------
    dict[str, ndarray]
        One array of the broadcast grid shape per requested key.

    Notes
    -----
    The instrument is first called ONCE with an array market. If that fails,
    or returns arrays of the wrong shape, the instrument is assumed to be
    scalar-only (typically a Monte Carlo pricer) and the grid is walked point
    by point. A scalar-only product with numerical Greeks costs 19 repricings
    per grid point: use a coarse grid (``n`` of 20-30) for those.
    """
    _check_priceable(instrument)
    if not isinstance(mkt, Market):
        raise ValueError(f"expected a Market, got {mkt!r}")
    keys = (greeks,) if isinstance(greeks, str) else tuple(greeks)
    if not keys:
        raise ValueError("ask for at least one quantity, e.g. greeks='delta'")
    grid_mkt = _grid_market(instrument, mkt, axes)
    values = _try_vectorised(instrument, grid_mkt, keys)
    if values is None:
        values = _scalar_loop(instrument, grid_mkt, keys)
    return bs.to_trader_units(values) if trader_units else values


def _point_values(
    instrument: Priceable, mkt: Market, keys: Sequence[str], trader_units: bool
) -> dict[str, float]:
    """Requested quantities at the (scalar) base market, as floats."""
    values = evaluate_on_grid(instrument, mkt, keys, trader_units=trader_units)
    return {key: float(value) for key, value in values.items()}


# ---------------------------------------------------------------------- #
# Default ranges and grids
# ---------------------------------------------------------------------- #
def _check_range(name: str, bounds: Sequence[float]) -> tuple[float, float]:
    try:
        lo, hi = (float(b) for b in bounds)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a (low, high) pair of numbers, got {bounds!r}") from exc
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError(f"{name} must satisfy low < high, got {bounds!r}")
    return lo, hi


def _spot_grid(lo: float, hi: float, n: int, levels: Sequence[float]) -> np.ndarray:
    """Uniform grid, refined geometrically around strikes and barriers.

    Near expiry all the action (the gamma spike, the delta step) happens within
    a fraction of a percent of the strike; a uniform grid would draw it with
    three points.
    """
    grid = np.linspace(lo, hi, n)
    per_side = n // 8
    if per_side and levels:
        offsets = np.geomspace(5e-4, 0.08, per_side)
        extra = [level * (1.0 + sign * offsets) for level in levels for sign in (-1.0, 1.0)]
        grid = np.concatenate([grid, *extra, np.asarray(levels, dtype=float)])
        grid = np.unique(grid[(grid >= lo) & (grid <= hi)])
    return grid


def _remaining_life(instruments: Sequence[Priceable], mkt: Market) -> float | None:
    """Longest time to expiry among the instruments (``None`` if none ever expires)."""
    expiries = [inst.expiry for inst in instruments if inst.expiry is not None]
    return max(expiries) - mkt.t if expiries else None


def _tau_bounds(
    remaining: float | None, x_range: Sequence[float] | None, surface: bool
) -> tuple[float, float]:
    """Time-to-expiry range of a grid, floored so that expiry itself is never evaluated."""
    if x_range is not None:
        lo, hi = _check_range("the time-to-expiry range", x_range)
        lo = max(lo, _TAU_FLOOR)
        if hi <= lo:
            raise ValueError("the time-to-expiry range must extend above 1e-4 years")
        return lo, hi
    # An expired product is drawn over the year BEFORE its expiry (the clock is moved back).
    hi = remaining if remaining is not None and remaining > 2.0 * _TAU_FLOOR else 1.0
    lo = 0.02 * hi if surface else min(_ONE_DAY, 0.02 * hi)
    return max(lo, _TAU_FLOOR), hi


def _axis_grid(
    axis: _Axis,
    instruments: Sequence[Priceable],
    mkt: Market,
    x_range: Sequence[float] | None,
    n: int,
    *,
    surface: bool = False,
) -> np.ndarray:
    """Grid of ``axis`` values: the default range comes from the instruments and the market.

    Line charts get non-uniform grids where it matters (dense around strikes,
    dense near expiry); ``surface=True`` forces uniform grids and keeps a little
    further away from expiry so that one spike does not flatten the picture.
    """
    remaining = _remaining_life(instruments, mkt)
    if axis.key == "spot":
        levels = [level for _, level in _merged_levels(instruments)]
        if x_range is None:
            lo, hi = 0.5 * min([mkt.spot, *levels]), 1.5 * max([mkt.spot, *levels])
        else:
            lo, hi = _check_range("the spot range", x_range)
        if lo <= 0:
            raise ValueError("the spot range must be strictly positive")
        return np.linspace(lo, hi, n) if surface else _spot_grid(lo, hi, n, levels)

    if axis.key == "tau":
        lo, hi = _tau_bounds(remaining, x_range, surface)
        return np.linspace(lo, hi, n) if surface else np.geomspace(lo, hi, n)

    if axis.key == "t":
        if remaining is None:
            lo, hi = (mkt.t, mkt.t + 1.0) if x_range is None else _check_range("the time range", x_range)
            return np.linspace(lo, hi, n)
        expiry = mkt.t + remaining
        if x_range is None:
            if remaining <= 2.0 * _TAU_FLOOR:
                raise ValueError("the instrument has expired: nothing left to plot against calendar time")
            tau_lo, tau_hi = _tau_bounds(remaining, None, surface)
        else:
            lo, hi = _check_range("the time range", x_range)
            tau_lo, tau_hi = max(expiry - hi, _TAU_FLOOR), expiry - lo
            if tau_hi <= tau_lo:
                raise ValueError("the time range must start before the expiry")
        taus = np.linspace(tau_hi, tau_lo, n) if surface else np.geomspace(tau_hi, tau_lo, n)
        return expiry - taus

    if x_range is not None:
        lo, hi = _check_range(f"the {axis.key} range", x_range)
        if axis.key == "vol" and lo < 0:
            raise ValueError("the volatility range must be non-negative")
    elif axis.key == "vol":
        lo, hi = max(0.02, 0.25 * mkt.vol), max(0.60, 2.5 * mkt.vol)
    elif axis.key == "rate":
        lo, hi = min(0.0, mkt.rate), max(0.10, 2.0 * mkt.rate)
    else:  # div
        lo, hi = min(0.0, mkt.div), max(0.10, 2.0 * mkt.div)
    return np.linspace(lo, hi, n)


def _current_x(axis: _Axis, instrument: Priceable, mkt: Market) -> float | None:
    """Where today's market sits on ``axis`` (``None`` if the notion does not apply)."""
    if axis.key == "tau":
        return None if instrument.expiry is None else float(instrument.expiry - mkt.t)
    return float(getattr(mkt, axis.key))


# ---------------------------------------------------------------------- #
# Trace and layout building blocks
# ---------------------------------------------------------------------- #
def _line(
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    color: str,
    fmt: str,
    *,
    dash: str = "solid",
    showlegend: bool = True,
    legendgroup: str | None = None,
) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        line=dict(color=color, width=2, dash=dash),
        showlegend=showlegend,
        legendgroup=legendgroup,
        hovertemplate=f"{name}: %{{y:{fmt}}}<extra></extra>",
    )


def _now_marker(
    x: float, y: float, color: str, fmt: str, *, text: bool, slope: float = 0.0
) -> go.Scatter:
    """Dot at today's market. The label sits on the side the curve is NOT heading to."""
    position = "top center" if slope == 0.0 else "top left" if slope > 0 else "top right"
    return go.Scatter(
        x=[x],
        y=[y],
        mode="markers+text" if text else "markers",
        name="Current market",
        marker=dict(color=color, size=9),
        text=[f"now: {y:{fmt}}"] if text else None,
        textposition=position,
        cliponaxis=False,
        showlegend=False,
        hoverinfo="skip",
    )


def _local_slope(x: np.ndarray, y: np.ndarray, x0: float, reverse: bool = False) -> float:
    """On-screen direction of the curve around ``x0``: +1 rising, -1 falling, 0 flat or peak."""
    half_window = 0.05 * (x[-1] - x[0])
    rise = float(np.interp(x0 + half_window, x, y) - np.interp(x0 - half_window, x, y))
    scale = float(np.ptp(y)) or 1.0
    if abs(rise) < 0.02 * scale:
        return 0.0
    return -math.copysign(1.0, rise) if reverse else math.copysign(1.0, rise)


def _style_axis(
    fig: go.Figure, axis: _Axis, *, which: str = "x", title: bool = True, **target: Any
) -> None:
    """Tick / hover formats, title and direction of one market axis."""
    update = fig.update_xaxes if which == "x" else fig.update_yaxes
    settings: dict[str, Any] = dict(hoverformat=axis.hoverformat)
    if axis.tickformat:
        settings["tickformat"] = axis.tickformat
    if title:
        settings["title_text"] = axis.title
    if axis.reverse and which == "x":
        settings["autorange"] = "reversed"
    update(**settings, **target)


def _add_level_lines(
    fig: go.Figure,
    levels: Sequence[tuple[str, float]],
    lo: float,
    hi: float,
    *,
    label: bool = True,
    horizontal: bool = False,
    **target: Any,
) -> None:
    """Dashed reference lines at the strikes / barriers that fall inside ``[lo, hi]``."""
    visible = [(tag, level) for tag, level in levels if lo <= level <= hi]
    label = label and len(visible) <= 5
    add = theme.add_hline if horizontal else theme.add_vline
    for i, (tag, level) in enumerate(visible):
        # Inside the plot (the band above it belongs to the legend); neighbours alternate
        # between top and bottom so that close strikes do not print over each other.
        position = "top right" if i % 2 == 0 else "bottom right"
        add(fig, level, f"{tag} {level:g}" if label else None, position=position, **target)


def _panel_targets(n_panels: int, ncols: int) -> list[dict[str, int]]:
    """``row``/``col`` kwargs per panel (empty dict for a single, non-subplot figure)."""
    if n_panels == 1:
        return [{}]
    n_cols = min(ncols, n_panels)
    return [dict(zip(("row", "col"), theme.grid_position(i, n_cols))) for i in range(n_panels)]


def _subtitle(title: str, subtitle: str) -> str:
    return f"{title}<br><sup>{subtitle}</sup>"


# ---------------------------------------------------------------------- #
# Profiles, dashboards and comparisons (one shared builder)
# ---------------------------------------------------------------------- #
def _profile_figure(
    series: Sequence[tuple[str, Priceable, str, str]],
    mkt: Market,
    greeks: Sequence[str],
    x: str,
    x_range: Sequence[float] | None,
    n: int,
    ncols: int,
    trader_units: bool,
    mode: str,
    title: str,
) -> go.Figure:
    """One panel per Greek, one line per ``(name, instrument, colour, dash)`` series."""
    _check_market(mkt)
    axis, n = _axis(x), _check_n(n)
    greeks = tuple(greeks)
    if not greeks:
        raise ValueError("ask for at least one Greek")
    instruments = [_check_priceable(inst) for _, inst, _, _ in series]
    levels = _merged_levels(instruments)
    grid = _axis_grid(axis, instruments, mkt, x_range, n)
    single_series, single_panel = len(series) == 1, len(greeks) == 1

    if single_panel:
        fig = go.Figure()
    else:
        # Panels are too small for line labels: the legend of the dashes goes in the subtitle.
        notes = ["dashed verticals: strikes (K) / barriers (B)"] if levels and axis.key == "spot" else []
        notes += ["dot: current market"] if single_series else []
        fig = theme.subplot_grid(
            len(greeks), ncols, [greek_axis_label(g, trader_units) for g in greeks],
            title=_subtitle(title, "; ".join(notes)) if notes else title, mode=mode,
        )
        fig.update_layout(margin=dict(t=104), legend=dict(y=1.07))
    targets = _panel_targets(len(greeks), ncols)

    curves = [
        evaluate_on_grid(inst, mkt, greeks, trader_units=trader_units, **{x: grid})
        for inst in instruments
    ]
    # Today's market is marked only on single-instrument figures (overlays stay uncluttered).
    x_now = _current_x(axis, instruments[0], mkt) if single_series else None
    if x_now is not None and grid[0] <= x_now <= grid[-1]:
        now = _point_values(instruments[0], mkt, greeks, trader_units)
    else:
        now = None
    for panel, (greek, target) in enumerate(zip(greeks, targets)):
        fmt = _number_format(*(curve[greek] for curve in curves))
        for (name, _, color, dash), curve in zip(series, curves):
            fig.add_trace(
                _line(grid, curve[greek], name, color, fmt, dash=dash,
                      showlegend=not single_series and panel == 0, legendgroup=name),
                **target,
            )
            if now is not None:
                fig.add_trace(
                    _now_marker(x_now, now[greek], color, fmt, text=single_panel,
                                slope=_local_slope(grid, curve[greek], x_now, axis.reverse)),
                    **target,
                )

    if single_panel:
        theme.apply_theme(fig, title=title, height=_SINGLE_HEIGHT, mode=mode, hovermode="x unified")
        fig.update_yaxes(title_text=greek_axis_label(greeks[0], trader_units))
    n_cols = min(ncols, len(greeks))
    for panel, target in enumerate(targets):
        _style_axis(fig, axis, title=panel + n_cols >= len(greeks), **target)
        if axis.key == "spot":
            _add_level_lines(fig, levels, grid[0], grid[-1], label=single_panel, **target)
    return fig


def greek_profile(
    instrument: Priceable,
    mkt: Market,
    greek: str = "delta",
    x: str = "spot",
    x_range: Sequence[float] | None = None,
    n: int = 201,
    *,
    trader_units: bool = True,
    mode: str = "auto",
) -> go.Figure:
    """One Greek (or the price) of any instrument against one market variable.

    Parameters
    ----------
    instrument : Instrument or Position
        Vanilla, strategy, exotic... or a signed position (a short option shows
        the mirror image: short gamma, positive theta).
    mkt : Market
        Scalar base market; today's point is marked on the curve.
    greek : str, default "delta"
        ``"price"`` or any key of ``instrument.greeks(mkt)``.
    x : {"spot", "vol", "tau", "t", "rate", "div"}, default "spot"
        Variable on the horizontal axis. ``"tau"`` is the time to expiry (the
        axis is reversed so that time still runs left to right), ``"t"`` the
        calendar time.
    x_range : (low, high), optional
        Defaults: spot from 0.5x the lowest to 1.5x the highest of (spot,
        strikes, barriers); vol from a quarter of the current vol to 60%; time
        from now to (almost) expiry; rates from 0 to 10%.
    n : int, default 201
        Number of uniform grid points (the spot grid is additionally refined
        around strikes and barriers).
    trader_units : bool, default True
        Desk units (vega per vol point, theta per day...) instead of raw ones.
    mode : {"auto", "light", "dark"}
        Passed to :func:`optionlab.plotting.theme.apply_theme`.

    Returns
    -------
    plotly.graph_objects.Figure
        Two traces: the profile and the current-market marker.

    Notes
    -----
    A profile is a *what-if*: every other market variable is frozen. The slope
    of the price profile IS delta and its curvature IS gamma -- plotting
    ``"price"``, ``"delta"`` and ``"gamma"`` in turn is the quickest way to see
    what a derivative of a derivative means.
    """
    _check_priceable(instrument)
    title = f"{_greek_name(greek)} vs {_axis(x).legend_title.lower()}: {_label(instrument)}"
    series = [(_label(instrument), instrument, _series_color(instrument), "solid")]
    return _profile_figure(series, mkt, (greek,), x, x_range, n, 1, trader_units, mode, title)


def greek_dashboard(
    instrument: Priceable,
    mkt: Market,
    greeks: Sequence[str] = DEFAULT_DASHBOARD_GREEKS,
    x: str = "spot",
    x_range: Sequence[float] | None = None,
    n: int = 201,
    *,
    ncols: int = 3,
    trader_units: bool = True,
    mode: str = "auto",
) -> go.Figure:
    """Small multiples: one panel per Greek, all against the same variable.

    Parameters
    ----------
    instrument, mkt, x, x_range, n, trader_units, mode
        As in :func:`greek_profile`.
    greeks : sequence of str
        Panels, in order. Defaults to price, delta, gamma, vega, theta, rho.
    ncols : int, default 3
        Number of panel columns.

    Returns
    -------
    plotly.graph_objects.Figure
        Two traces per panel (profile + current-market marker). Strikes and
        barriers are drawn as unlabelled dashed lines (the panels are small);
        the subtitle says what they are.

    Notes
    -----
    Each Greek lives on its own scale, hence separate panels rather than a
    shared axis. Read them together: where gamma peaks, theta is most negative
    (convexity is paid for with time decay) and vega peaks too -- all three are
    "optionality" seen from a different angle.
    """
    title = f"Greeks vs {_axis(x).legend_title.lower()}: {_label(_check_priceable(instrument))}"
    series = [(_label(instrument), instrument, _series_color(instrument), "solid")]
    return _profile_figure(series, mkt, greeks, x, x_range, n, ncols, trader_units, mode, title)


def compare_instruments(
    instruments: Mapping[str, Priceable] | Sequence[Priceable],
    mkt: Market,
    greek: str | Sequence[str] = "delta",
    x: str = "spot",
    x_range: Sequence[float] | None = None,
    n: int = 201,
    *,
    colors: Sequence[str] | None = None,
    title: str | None = None,
    ncols: int = 3,
    trader_units: bool = True,
    mode: str = "auto",
) -> go.Figure:
    """Overlay the same Greek(s) of several instruments.

    Parameters
    ----------
    instruments : dict[str, Instrument] or sequence of Instrument
        ``{legend label: instrument}``; with a plain sequence the labels are
        the instruments' own ``label``. At most 8 (the categorical palette is
        never cycled -- beyond that, make several figures).
    mkt, x, x_range, n, trader_units, mode
        As in :func:`greek_profile`. The default spot range covers the strikes
        and barriers of ALL instruments.
    greek : str or sequence of str, default "delta"
        One Greek gives a single chart; several give one panel per Greek with
        a shared legend (click an entry to hide that instrument everywhere).
    colors : sequence of str, optional
        One ``#rrggbb`` colour per instrument; defaults to the theme palette in
        its fixed order. Lines also get distinct dash patterns.
    title : str, optional
        Figure title; a generic one is built by default.
    ncols : int, default 3
        Panel columns when several Greeks are requested.

    Returns
    -------
    plotly.graph_objects.Figure
        ``len(instruments) * len(greeks)`` line traces.

    Notes
    -----
    Typical uses: call vs put, a vanilla vs its knock-out cousin (how much
    gamma does the barrier add?), the same option at three strikes.
    """
    if isinstance(instruments, Mapping):
        named = list(instruments.items())
    else:
        named = [(_label(_check_priceable(inst)), inst) for inst in instruments]
    if not named:
        raise ValueError("compare_instruments needs at least one instrument")
    if len({name for name, _ in named}) != len(named):
        raise ValueError("instrument labels must be unique; pass a dict with explicit labels")
    if len(named) > len(theme.PALETTE):
        raise ValueError(
            f"at most {len(theme.PALETTE)} instruments per figure (colours are never recycled); "
            "split the comparison into several figures"
        )
    palette = list(colors) if colors is not None else theme.PALETTE
    if len(palette) < len(named):
        raise ValueError(f"got {len(palette)} colours for {len(named)} instruments")
    greeks = (greek,) if isinstance(greek, str) else tuple(greek)
    if title is None:
        what = _greek_name(greeks[0]) if len(greeks) == 1 else "Greeks"
        title = f"{what} vs {_axis(x).legend_title.lower()}: comparison"
    # Dash patterns back the colours up: identical curves (call and put gamma!) would
    # otherwise hide each other, and identity never rests on colour alone.
    series = [
        (str(name), inst, color, _DASHES[i % len(_DASHES)])
        for i, ((name, inst), color) in enumerate(zip(named, palette))
    ]
    return _profile_figure(series, mkt, greeks, x, x_range, n, ncols, trader_units, mode, title)


def call_put_comparison(
    strike: float,
    expiry: float,
    mkt: Market,
    greeks: Sequence[str] = DEFAULT_DASHBOARD_GREEKS,
    x: str = "spot",
    x_range: Sequence[float] | None = None,
    n: int = 201,
    *,
    ncols: int = 3,
    trader_units: bool = True,
    mode: str = "auto",
) -> go.Figure:
    """Call vs put dashboard for one strike and expiry (call blue, put orange).

    Parameters
    ----------
    strike, expiry : float
        Strike and ABSOLUTE expiry (years) shared by the two options.
    mkt, greeks, x, x_range, n, ncols, trader_units, mode
        As in :func:`greek_dashboard`.

    Returns
    -------
    plotly.graph_objects.Figure
        Two line traces per panel.

    Notes
    -----
    Put-call parity (``call - put = forward``) is visible in every panel: a
    forward has no optionality, so gamma, vega, vanna and volga are IDENTICAL
    for the call and the put, while the deltas differ by exactly the forward's
    delta ``exp(-div * tau)``.
    """
    options = {
        f"Call {strike:g}": EuropeanOption("call", strike, expiry),
        f"Put {strike:g}": EuropeanOption("put", strike, expiry),
    }
    return compare_instruments(
        options, mkt, greeks, x, x_range, n,
        colors=(theme.COLORS["call"], theme.COLORS["put"]),
        title=f"Call vs put: K = {strike:g}, expiry T = {expiry:g}",
        ncols=ncols, trader_units=trader_units, mode=mode,
    )


# ---------------------------------------------------------------------- #
# Families of curves
# ---------------------------------------------------------------------- #
def _family_figure(
    instrument: Priceable,
    mkt: Market,
    greek: str,
    x: str,
    vary: str,
    values: Sequence[float],
    colors: Sequence[str],
    x_range: Sequence[float] | None,
    n: int,
    trader_units: bool,
    mode: str,
    title: str,
) -> go.Figure:
    """One line per ``vary`` level, all evaluated in a single 2-D grid call."""
    x_axis, vary_axis = _axis(x), _axis(vary)
    grid = _axis_grid(x_axis, [instrument], mkt, x_range, _check_n(n))
    levels = np.asarray(values, dtype=float)
    surface = evaluate_on_grid(
        instrument, mkt, greek, trader_units=trader_units,
        **{x: grid[None, :], vary: levels[:, None]},
    )[greek]
    fmt = _number_format(surface)
    fig = go.Figure()
    for level, row, color in zip(levels, surface, colors):
        fig.add_trace(_line(grid, row, _level_label(vary_axis, float(level)), color, fmt))
    theme.apply_theme(fig, title=title, height=_SINGLE_HEIGHT, mode=mode, hovermode="x unified")
    fig.update_layout(legend_title_text=vary_axis.legend_title)
    fig.update_yaxes(title_text=greek_axis_label(greek, trader_units))
    _style_axis(fig, x_axis)
    if x_axis.key == "spot":
        _add_level_lines(fig, key_levels(instrument), grid[0], grid[-1])
    return fig


def _default_taus(remaining: float | None) -> tuple[float, ...]:
    """Current maturity followed by the standard ladder (1y ... 1d), six curves at most."""
    if remaining is None or remaining <= 2.0 * _TAU_FLOOR:
        return _TENOR_LADDER
    shorter = [tenor for tenor in _TENOR_LADDER if tenor < remaining * (1.0 - 1e-9)]
    if len(shorter) < 2:
        return tuple(remaining * fraction for fraction in (1.0, 0.5, 0.25, 0.1))
    return (remaining, *shorter[-5:])


def _family_values(
    vary: str, values: Sequence[float] | None, instrument: Priceable, mkt: Market
) -> np.ndarray:
    if values is None:
        if vary == "tau":
            values = _default_taus(_remaining_life([instrument], mkt))
        elif vary == "t":
            remaining = _require_expiry(instrument, "calendar time") - mkt.t
            values = tuple(instrument.expiry - tau for tau in _default_taus(remaining))
        elif vary == "vol":
            base = mkt.vol if mkt.vol > 0 else 0.2
            values = tuple(base * k for k in (0.5, 0.75, 1.0, 1.5, 2.0))
        elif vary == "spot":
            values = _moneyness_spots(instrument, mkt)
        else:
            values = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
    out = np.asarray(values, dtype=float).ravel()
    if out.size == 0 or not np.all(np.isfinite(out)):
        raise ValueError(f"the {vary} levels must be a non-empty sequence of finite numbers")
    if out.size > _MAX_FAMILY:
        raise ValueError(f"at most {_MAX_FAMILY} curves per figure, got {out.size}")
    if vary in ("tau", "spot") and np.any(out <= 0):
        raise ValueError(f"the {vary} levels must be strictly positive")
    if vary == "vol" and np.any(out < 0):
        raise ValueError("the vol levels must be non-negative")
    return out


def _moneyness_spots(instrument: Priceable, mkt: Market) -> tuple[float, ...]:
    """90% / 100% / 110% of the money level (median strike, or the spot if there is none)."""
    strikes = [level for tag, level in key_levels(instrument) if tag == "K"]
    reference = float(np.median(strikes)) if strikes else mkt.spot
    return tuple(reference * k for k in (0.9, 1.0, 1.1))


def greek_evolution(
    instrument: Priceable,
    mkt: Market,
    greek: str = "gamma",
    x: str = "spot",
    taus: Sequence[float] | None = None,
    *,
    vary: str = "tau",
    values: Sequence[float] | None = None,
    x_range: Sequence[float] | None = None,
    n: int = 201,
    trader_units: bool = True,
    mode: str = "auto",
) -> go.Figure:
    """The classic picture: one Greek profile redrawn as time (or vol, or rates) changes.

    Parameters
    ----------
    instrument, mkt, greek, x, x_range, n, trader_units, mode
        As in :func:`greek_profile`.
    taus : sequence of float, optional
        Times to expiry in years (only with ``vary="tau"``). Default: the
        current time to expiry followed by the standard ladder 1y, 6m, 3m, 1m,
        1w, 1d (entries longer than the remaining life are dropped, so by
        default time is never moved backwards).
    vary : {"tau", "vol", "rate", "div", "spot", "t"}, default "tau"
        The dimension that changes from one curve to the next.
    values : sequence of float, optional
        Levels of ``vary`` (overrides ``taus``). Defaults: multiples 0.5x-2x of
        the current vol; rates 0%-10%; 90/100/110% of the strike for spot.
        At most 8 levels.

    Returns
    -------
    plotly.graph_objects.Figure
        One line per level, coloured along a single-hue ramp from light (first
        level) to dark (last level), so the eye reads the ordering directly.

    Notes
    -----
    With ``vary="tau"`` this shows optionality being squeezed into the strike
    as expiry approaches: delta turns from a lazy S-curve into a step, gamma
    from a low hill into a spike, and vega melts away everywhere. Lowering the
    volatility (``vary="vol"``) has much the same effect as removing time,
    because only the total standard deviation ``vol * sqrt(tau)`` matters.
    """
    _check_priceable(instrument)
    _check_market(mkt)
    _check_axis_pair(x, vary)
    if taus is not None and values is None:
        if vary != "tau":
            raise ValueError("'taus' only applies with vary='tau'; use 'values' instead")
        values = taus
    levels = np.sort(_family_values(vary, values, instrument, mkt))
    if vary == "tau":
        levels = levels[::-1]  # time passes from the first (light) to the last (dark) curve
    title = (
        f"{_greek_name(greek)} vs {_axis(x).legend_title.lower()} "
        f"by {_axis(vary).legend_title.lower()}: {_label(instrument)}"
    )
    colors = _ramp_colors(len(levels), mode)
    return _family_figure(
        instrument, mkt, greek, x, vary, levels, colors, x_range, n, trader_units, mode, title
    )


def greek_vs_time(
    instrument: Priceable,
    mkt: Market,
    greek: str = "gamma",
    spots: Sequence[float] | None = None,
    x: str = "t",
    *,
    n: int = 201,
    trader_units: bool = True,
    mode: str = "auto",
) -> go.Figure:
    """A Greek along calendar time, from now to expiry, for a few frozen spot levels.

    Parameters
    ----------
    instrument, mkt, greek, trader_units, mode
        As in :func:`greek_profile`. The instrument must have an expiry and
        must not be expired.
    spots : sequence of float, optional
        Spot levels, one line each (at most 8). Default: 90%, 100% and 110% of
        the money level (the median strike, or the current spot when the
        instrument has no strike) -- i.e. one line on each side of the money
        and one at the money.
    x : {"t", "tau"}, default "t"
        Calendar time, or time to expiry on a reversed axis. Either way time
        runs left to right and the grid gets denser towards expiry.
    n : int, default 201
        Number of time points.

    Returns
    -------
    plotly.graph_objects.Figure
        One line per spot level (categorical colours).

    Notes
    -----
    This is where the at-the-money blow-up shows: ATM gamma and theta grow like
    ``1 / sqrt(tau)`` while they die for the other two lines. With
    ``greek="delta"`` the drift of the lines IS charm: the in-the-money delta
    creeps to 1, the out-of-the-money delta to 0, without any spot move.
    """
    _check_priceable(instrument)
    _check_market(mkt)
    if x not in ("t", "tau"):
        raise ValueError(f"greek_vs_time plots against 't' or 'tau', got {x!r}")
    expiry = _require_expiry(instrument, "time")
    levels = _family_values("spot", spots, instrument, mkt)
    title = f"{_greek_name(greek)} through time by spot level: {_label(instrument)}"
    fig = _family_figure(
        instrument, mkt, greek, x, "spot", levels, theme.PALETTE, None, n, trader_units, mode, title
    )
    if x == "t":
        # The lines converge on the expiry from every height, so the label lives in the axis title.
        theme.add_vline(fig, expiry)
        fig.update_xaxes(title_text=f"{_AXES['t'].title[:-1]}; dashed line: expiry at t = {expiry:g})")
    return fig


# ---------------------------------------------------------------------- #
# Surfaces
# ---------------------------------------------------------------------- #
def _color_settings(z: np.ndarray, mode: str) -> dict[str, Any]:
    """Diverging scale centred on 0 for signed data, single-hue ramp otherwise."""
    finite = z[np.isfinite(z)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
    span = max(abs(lo), abs(hi))
    if lo < -0.02 * span and hi > 0.02 * span:
        return dict(colorscale=theme.diverging_scale(mode), cmin=-span, cmax=span)
    # All-negative data (theta of a long option): the darkest colour must still mean "large".
    return dict(colorscale=theme.sequential_scale(mode), reversescale=hi <= 0.02 * span and lo < 0)


def greek_surface(
    instrument: Priceable,
    mkt: Market,
    greek: str = "gamma",
    x: str = "spot",
    y: str = "tau",
    kind: str = "surface",
    *,
    x_range: Sequence[float] | None = None,
    y_range: Sequence[float] | None = None,
    nx: int = 61,
    ny: int = 41,
    trader_units: bool = True,
    mode: str = "auto",
) -> go.Figure:
    """A Greek over two market variables, as a 3-D surface or a heatmap.

    Parameters
    ----------
    instrument, mkt, greek, trader_units, mode
        As in :func:`greek_profile`.
    x, y : {"spot", "vol", "tau", "t", "rate", "div"}
        The two grid dimensions; defaults give the (spot, time to expiry)
        surface. ``("spot", "vol")`` is the other classic.
    kind : {"surface", "heatmap"}, default "surface"
        The heatmap is easier to read precisely (and marks strikes/barriers);
        the surface shows shape.
    x_range, y_range : (low, high), optional
        Same defaults as :func:`greek_profile`, except that time axes stop at
        2% of the remaining life before expiry: the at-the-money spike of
        gamma or theta would otherwise flatten the rest of the picture.
        Grids never get closer to expiry than 1e-4 years.
    nx, ny : int
        Grid sizes (uniform).

    Returns
    -------
    plotly.graph_objects.Figure
        Two traces: the surface/heatmap and a marker at the current market
        (omitted when it lies outside the grid). Signed Greeks use a diverging
        red/blue scale centred on zero, one-signed Greeks a single-hue ramp.
    """
    _check_priceable(instrument)
    _check_market(mkt)
    _check_axis_pair(x, y)
    if kind not in ("surface", "heatmap"):
        raise ValueError(f"kind must be 'surface' or 'heatmap', got {kind!r}")
    x_axis, y_axis = _axis(x), _axis(y)
    xs = _axis_grid(x_axis, [instrument], mkt, x_range, _check_n(nx), surface=True)
    ys = _axis_grid(y_axis, [instrument], mkt, y_range, _check_n(ny), surface=True)
    z = evaluate_on_grid(
        instrument, mkt, greek, trader_units=trader_units, **{x: xs[None, :], y: ys[:, None]}
    )[greek]

    fmt = _number_format(z)
    z_label = greek_axis_label(greek, trader_units)
    hover = (
        f"{x_axis.legend_title}: %{{x:{x_axis.hoverformat}}}<br>"
        f"{y_axis.legend_title}: %{{y:{y_axis.hoverformat}}}<br>"
        f"{_greek_name(greek)}: %{{z:{fmt}}}<extra></extra>"
    )
    colors = _color_settings(z, mode)
    x_now, y_now = _current_x(x_axis, instrument, mkt), _current_x(y_axis, instrument, mkt)
    inside = (
        x_now is not None and y_now is not None
        and xs[0] <= x_now <= xs[-1] and ys[0] <= y_now <= ys[-1]
    )
    ink = theme.INK[mode]["primary"]  # ``mode`` was validated by _color_settings
    title = f"{_greek_name(greek)} surface: {_label(instrument)}"
    fig = go.Figure()

    if kind == "surface":
        fig.add_trace(go.Surface(
            x=xs, y=ys, z=z, name=z_label, hovertemplate=hover,
            colorbar=dict(title=dict(text=z_label, side="right")), **colors,
        ))
        if inside:
            z_now = _point_values(instrument, mkt, (greek,), trader_units)[greek]
            fig.add_trace(go.Scatter3d(
                x=[x_now], y=[y_now], z=[z_now], mode="markers", name="Current market",
                marker=dict(size=5, color=ink), showlegend=False,
                hovertemplate=f"Current market<br>{_greek_name(greek)}: %{{z:{fmt}}}<extra></extra>",
            ))
        theme.apply_theme(fig, title=title, height=_SURFACE_HEIGHT, mode=mode)
        scene_axes = {}
        for name, axis in (("xaxis", x_axis), ("yaxis", y_axis)):
            scene_axes[name] = dict(title=dict(text=axis.legend_title), hoverformat=axis.hoverformat)
            if axis.tickformat:
                scene_axes[name]["tickformat"] = axis.tickformat
        fig.update_layout(
            scene=dict(
                **scene_axes,
                zaxis=dict(title=dict(text=z_label)),
                aspectmode="manual",
                aspectratio=dict(x=1.25, y=1.0, z=0.7),
                camera=dict(eye=dict(x=-1.35, y=-1.55, z=0.8)),
            ),
            margin=dict(l=24, r=8, b=8),
        )
        return fig

    limits = {("zmin" if k == "cmin" else "zmax" if k == "cmax" else k): v for k, v in colors.items()}
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=z, name=z_label, hovertemplate=hover, zsmooth="best", xgap=0, ygap=0,
        colorbar=dict(title=dict(text=z_label, side="right")), **limits,
    ))
    if inside:
        fig.add_trace(go.Scatter(
            x=[x_now], y=[y_now], mode="markers", name="Current market", showlegend=False,
            marker=dict(size=10, color="#ffffff", line=dict(color="#0b0b0b", width=2)),
            hovertemplate="Current market<extra></extra>",
        ))
    theme.apply_theme(fig, title=title, height=_SINGLE_HEIGHT + 40, mode=mode, hovermode="closest")
    _style_axis(fig, x_axis, which="x")
    _style_axis(fig, y_axis, which="y", title=False)  # only an x axis is ever reversed
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False, title_text=y_axis.legend_title)
    if "spot" in (x, y):
        spot_grid = xs if x == "spot" else ys
        _add_level_lines(
            fig, key_levels(instrument), spot_grid[0], spot_grid[-1], horizontal=(y == "spot")
        )
    return fig


# ---------------------------------------------------------------------- #
# P&L views
# ---------------------------------------------------------------------- #
def _zero_crossings(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Abscissae where ``y`` changes sign (linear interpolation; flat zeros are not crossings)."""
    nonzero = np.flatnonzero(np.isfinite(y) & (y != 0.0))
    roots = []
    for i, j in zip(nonzero[:-1], nonzero[1:]):
        if y[i] * y[j] < 0:
            if j == i + 1:
                roots.append(x[i] - y[i] * (x[j] - x[i]) / (y[j] - y[i]))
            else:
                roots.append(0.5 * (x[i + 1] + x[j - 1]))
    return np.asarray(roots, dtype=float)


def _horizon_setup(instrument: Priceable, mkt: Market) -> tuple[float | None, bool]:
    """Terminal date of a P&L picture and whether it is a calendar-type package.

    A composite whose legs expire at different dates has its natural horizon at
    the FIRST expiry, where it is still worth a price (the calendar "tent"),
    not a payoff.
    """
    expiry = instrument.expiry
    first = getattr(_unwrap(instrument), "first_expiry", None)
    if expiry is not None and first is not None and first < expiry - EXPIRY_TOL:
        return float(first), True
    return (None if expiry is None else float(expiry)), False


def pnl_profile(
    instrument: Priceable,
    mkt: Market,
    entry_price: float | None = None,
    horizons: Sequence[float] | None = None,
    x_range: Sequence[float] | None = None,
    n: int = 201,
    *,
    trader_units: bool = True,
    mode: str = "auto",
) -> go.Figure:
    """P&L against spot: today, at later dates and at expiry (the payoff).

    Parameters
    ----------
    instrument : Instrument or Position
        For a short position pass ``Position(inst, -1)``: value and entry price
        both change sign and the picture flips.
    mkt : Market
        Scalar base market ("today").
    entry_price : float, optional
        What was paid for the position (negative if premium was received).
        Default: today's model value, so the "today" line goes through zero at
        the current spot. Pass ``0.0`` to plot VALUE instead of P&L.
    horizons : sequence of float, optional
        Time elapsed from now, in years, one line each (at most 7). Default:
        0, 25%, 50% and 75% of the remaining life. Horizons at or beyond the
        terminal date are dropped: the terminal line already covers them.
    x_range, n, mode
        As in :func:`greek_profile`.
    trader_units : bool
        Accepted for a uniform call signature; a P&L is in currency either way.

    Returns
    -------
    plotly.graph_objects.Figure
        One line per horizon (light to dark as time passes), the terminal line,
        a marker at today's market when the horizon 0 is drawn, and a marker
        trace at the terminal breakevens when there are any.

    Notes
    -----
    The vertical gap between the "today" curve and the expiry payoff is the
    time value: it is what theta takes away, fastest near the strike. The lines
    are pure repricings (vol and rates frozen, financing of the premium
    ignored). For path-dependent products they assume the stored path state no
    longer changes. For calendar-type packages the terminal line is the package
    VALUE at the first expiry (the "tent"), since a long-dated leg is still
    alive then.
    """
    _check_priceable(instrument)
    _check_market(mkt)
    grid = _axis_grid(_AXES["spot"], [instrument], mkt, x_range, _check_n(n))
    terminal_t, is_calendar = _horizon_setup(instrument, mkt)
    remaining = None if terminal_t is None else terminal_t - mkt.t

    if horizons is None:
        if remaining is None:
            horizons = (0.0,)
        else:
            horizons = tuple(max(remaining, 0.0) * k for k in (0.0, 0.25, 0.5, 0.75))
    elapsed = np.unique(np.asarray(horizons, dtype=float).ravel())
    if not np.all(np.isfinite(elapsed)) or np.any(elapsed < 0):
        raise ValueError("horizons are times elapsed from now, in years: finite and >= 0")
    if remaining is not None:
        elapsed = elapsed[elapsed < remaining - EXPIRY_TOL]
    if elapsed.size > _MAX_FAMILY - 1:
        raise ValueError(f"at most {_MAX_FAMILY - 1} horizons per figure, got {elapsed.size}")

    entry = (
        _point_values(instrument, mkt, ("price",), False)["price"]
        if entry_price is None else float(entry_price)
    )
    lines: list[tuple[str, np.ndarray]] = []
    if elapsed.size:
        values = evaluate_on_grid(
            instrument, mkt, "price", spot=grid[None, :], t=(mkt.t + elapsed)[:, None]
        )["price"]
        for h, row in zip(elapsed, values):
            lines.append(("today" if h == 0.0 else f"in {_format_tenor(float(h))}", row - entry))
    if terminal_t is not None:
        if is_calendar:
            terminal = evaluate_on_grid(instrument, mkt, "price", spot=grid, t=terminal_t)["price"]
            lines.append(("at first expiry (value)", terminal - entry))
        else:
            lines.append(("at expiry (payoff)", np.asarray(instrument.payoff(grid), dtype=float) - entry))

    is_pnl = entry != 0.0
    fmt = _number_format(*(y for _, y in lines))
    fig = go.Figure()
    for (name, y), color in zip(lines, _ramp_colors(len(lines), mode)):
        fig.add_trace(_line(grid, y, name, color, fmt))
    if elapsed.size and elapsed[0] == 0.0 and grid[0] <= mkt.spot <= grid[-1]:
        y_now = _point_values(instrument, mkt, ("price",), False)["price"] - entry
        fig.add_trace(_now_marker(mkt.spot, y_now, theme.COLORS["neutral"], fmt, text=False))
    if is_pnl:
        caption = f"entry price {entry:{_number_format(entry)}}"
    else:
        caption = "entry price 0: the lines are values"
    roots = _zero_crossings(grid, lines[-1][1]) if terminal_t is not None and is_pnl else np.empty(0)
    if roots.size:
        fig.add_trace(go.Scatter(
            x=roots, y=np.zeros_like(roots), mode="markers", name="Breakeven",
            marker=dict(color=theme.COLORS["neutral"], size=9, symbol="diamond"),
            hovertemplate="Breakeven: %{x:.2f}<extra></extra>",
        ))
        if roots.size <= 6:
            caption += f" | breakeven {lines[-1][0]}: " + ", ".join(f"{r:.2f}" for r in roots)

    what = "P&L" if is_pnl else "Value"
    title = _subtitle(f"{what} vs spot through time: {_label(instrument)}", caption)
    theme.apply_theme(fig, title=title, height=_SINGLE_HEIGHT, mode=mode, hovermode="x unified")
    fig.update_layout(legend_title_text="Date", margin=dict(t=96))
    fig.update_yaxes(title_text=what)
    _style_axis(fig, _AXES["spot"])
    theme.add_hline(fig, 0.0, dash="solid")
    _add_level_lines(fig, key_levels(instrument), grid[0], grid[-1])
    return fig


def _greeks_caption(greeks: Mapping[str, float], names: Sequence[str], trader_units: bool) -> str:
    """``"delta 0.637 | gamma 0.0188 per 1 spot | ..."`` in the requested units."""
    shown = bs.to_trader_units(dict(greeks)) if trader_units else greeks
    parts = []
    for name in names:
        unit = _COMPACT_UNITS.get(name, ("", ""))[1 if trader_units else 0]
        parts.append(f"{name} {shown[name]:.4g}{' ' + unit if unit else ''}")
    return " | ".join(parts)


def taylor_pnl_explain(
    instrument: Priceable,
    mkt: Market,
    x_range: Sequence[float] | None = None,
    dvol: float = 0.0,
    dt: float = 0.0,
    n: int = 201,
    *,
    trader_units: bool = True,
    mode: str = "auto",
) -> go.Figure:
    """Actual repricing P&L of a spot move against its Greek (Taylor) approximations.

    Parameters
    ----------
    instrument, mkt, n, mode
        As in :func:`greek_profile`.
    x_range : (low, high), optional
        Range of NEW spot levels; default 75% to 125% of the current spot.
    dvol : float, default 0.0
        Simultaneous change of volatility (0.01 = +1 vol point).
    dt : float, default 0.0
        Time elapsed during the move, in years (1/365 = one day).
    trader_units : bool, default True
        Units of the Greeks quoted in the subtitle (the P&L is in currency).

    Returns
    -------
    plotly.graph_objects.Figure
        Three lines -- full repricing, ``delta * dS`` and
        ``delta * dS + 0.5 * gamma * dS**2`` -- plus, when ``dvol`` or ``dt`` is
        non-zero, a fourth one adding ``vega * dvol + theta * dt``; and a marker
        at the current spot.

    Notes
    -----
    This is how a desk "explains" its P&L every day. Delta alone is a tangent:
    right for small moves, and always too pessimistic for a long option (it
    misses the convexity). Adding gamma bends the line into a parabola that
    hugs the true curve much further out. What is left over is the higher
    orders (speed, and the cross terms vanna/volga when vol moves too) -- and
    it grows fast near expiry or near a barrier, where the Greeks themselves
    change quickly.
    """
    _check_priceable(instrument)
    _check_market(mkt)
    if x_range is None:
        x_range = (0.75 * mkt.spot, 1.25 * mkt.spot)
    grid = _axis_grid(_AXES["spot"], [instrument], mkt, x_range, _check_n(n))
    dvol, dt = float(dvol), float(dt)
    if mkt.vol + dvol < 0:
        raise ValueError("dvol would make the volatility negative")
    if dt < 0:
        raise ValueError("dt is the time elapsed during the move and must be >= 0")

    names = ("price", "delta", "gamma", "vega", "theta")
    g = _point_values(instrument, mkt, names, False)
    moved = mkt.bumped(vol=mkt.vol + dvol, t=mkt.t + dt)
    actual = evaluate_on_grid(instrument, moved, "price", spot=grid)["price"] - g["price"]
    ds = grid - mkt.spot
    delta_only = g["delta"] * ds
    delta_gamma = delta_only + 0.5 * g["gamma"] * ds**2
    lines = [
        ("Full repricing", actual, "solid"),
        ("Delta", delta_only, "dot"),
        ("Delta + gamma", delta_gamma, "dash"),
    ]
    shown = ["delta", "gamma"]
    if dvol != 0.0 or dt != 0.0:
        full_taylor = delta_gamma + g["vega"] * dvol + g["theta"] * dt
        lines.append(("Delta + gamma + vega + theta", full_taylor, "dashdot"))
        shown += ["vega", "theta"]

    fmt = _number_format(*(y for _, y, _ in lines))
    fig = go.Figure()
    for (name, y, dash), color in zip(lines, theme.PALETTE):
        fig.add_trace(_line(grid, y, name, color, fmt, dash=dash))
    if grid[0] <= mkt.spot <= grid[-1]:
        y_now = float(evaluate_on_grid(instrument, moved, "price")["price"]) - g["price"]
        fig.add_trace(_now_marker(mkt.spot, y_now, theme.PALETTE[0], fmt, text=False))

    scenario = []
    if dvol != 0.0:
        scenario.append(f"vol {dvol * 100.0:+g} pts")
    if dt != 0.0:
        scenario.append(f"{_format_tenor(dt)} later")
    caption = _greeks_caption(g, shown, trader_units)
    title = _subtitle(
        f"P&L explain for a spot move: {_label(instrument)}",
        caption + (f" | scenario: {', '.join(scenario)}" if scenario else ""),
    )
    theme.apply_theme(fig, title=title, height=_SINGLE_HEIGHT, mode=mode, hovermode="x unified")
    fig.update_layout(margin=dict(t=96))
    fig.update_yaxes(title_text="P&L")
    fig.update_xaxes(title_text="Spot after the move", hoverformat=".2f")
    theme.add_hline(fig, 0.0, dash="solid")
    _add_level_lines(fig, key_levels(instrument), grid[0], grid[-1])
    return fig


def gamma_theta_tradeoff(
    instrument: Priceable,
    mkt: Market,
    dt: float = _ONE_DAY,
    x_range: Sequence[float] | None = None,
    n: int = 201,
    *,
    trader_units: bool = True,
    mode: str = "auto",
) -> go.Figure:
    """P&L of the DELTA-HEDGED position over one period: gamma earns, theta pays.

    Parameters
    ----------
    instrument, mkt, n, mode
        As in :func:`greek_profile`.
    dt : float, default 1/365
        Length of the period in years (one calendar day).
    x_range : (low, high), optional
        Range of spot levels at the end of the period; default: four standard
        deviations ``vol * spot * sqrt(dt)`` on each side of the spot.
    trader_units : bool, default True
        Units of the Greeks quoted in the subtitle (the P&L is in currency).

    Returns
    -------
    plotly.graph_objects.Figure
        Four traces: profit and loss washes (no hover), the full-repricing P&L
        ``V(S', t + dt) - V(S, t) - delta * (S' - S)`` and its approximation
        ``0.5 * gamma * dS**2 + theta * dt``. The breakeven moves are marked
        when gamma and theta have opposite signs.

    Notes
    -----
    Once delta is hedged, a long option makes money on ANY large move (gamma)
    and loses a fixed rent if nothing happens (theta). The two balance when the
    spot moves by about one standard deviation, ``vol * spot * sqrt(dt)``: a
    long-gamma position is a bet that the realised move will exceed the implied
    one. For a short option the parabola is upside down: you collect the rent
    and fear the move. (Carry on the hedge and the premium is ignored, so with
    non-zero rates the breakeven is close to, not exactly, one standard
    deviation.)
    """
    _check_priceable(instrument)
    mkt = _check_market(mkt)
    dt = float(dt)
    if not dt > 0:
        raise ValueError("dt must be strictly positive")
    sigma_move = mkt.vol * mkt.spot * math.sqrt(dt)
    if x_range is None:
        half = 4.0 * sigma_move if sigma_move > 0 else 0.05 * mkt.spot
        x_range = (max(mkt.spot - half, 0.05 * mkt.spot), mkt.spot + half)
    lo, hi = _check_range("the spot range", x_range)
    if lo <= 0:
        raise ValueError("the spot range must be strictly positive")
    grid = np.linspace(lo, hi, _check_n(n))  # uniform: the interesting point is the spot, not the strike

    g = _point_values(instrument, mkt, ("price", "delta", "gamma", "theta"), False)
    ds = grid - mkt.spot
    later = mkt.bumped(t=mkt.t + dt)
    hedged = evaluate_on_grid(instrument, later, "price", spot=grid)["price"] - g["price"] - g["delta"] * ds
    approx = 0.5 * g["gamma"] * ds**2 + g["theta"] * dt

    fmt = _number_format(hedged, approx)
    fig = go.Figure()
    for clipped, key in ((np.maximum(hedged, 0.0), "profit"), (np.minimum(hedged, 0.0), "loss")):
        fig.add_trace(go.Scatter(
            x=grid, y=clipped, mode="lines", line=dict(width=0), fill="tozeroy",
            fillcolor=theme.with_alpha(theme.COLORS[key], 0.12), name=key.capitalize(),
            showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(_line(grid, hedged, "Delta-hedged P&L (full repricing)", theme.PALETTE[0], fmt))
    approx_name = "0.5 x gamma x dS^2 + theta x dt"
    fig.add_trace(_line(grid, approx, approx_name, theme.PALETTE[1], fmt, dash="dash"))

    # The parabola 0.5 * gamma * dS^2 + theta * dt crosses zero only if gamma and theta disagree.
    has_breakeven = g["gamma"] * g["theta"] < 0
    breakeven = math.sqrt(-2.0 * g["theta"] * dt / g["gamma"]) if has_breakeven else 0.0
    caption = _greeks_caption(g, ("gamma", "theta"), trader_units)
    caption += f" | implied move over {_format_tenor(dt)}: +/-{sigma_move:.2f}"
    if has_breakeven:
        caption += f" | breakeven move: +/-{breakeven:.2f}"
    title = _subtitle(f"Gamma vs theta over {_format_tenor(dt)}: {_label(instrument)}", caption)
    theme.apply_theme(fig, title=title, height=_SINGLE_HEIGHT, mode=mode, hovermode="x unified")
    fig.update_layout(margin=dict(t=96))
    fig.update_yaxes(title_text=f"Delta-hedged P&L over {_format_tenor(dt)}")
    fig.update_xaxes(title_text=f"Spot after {_format_tenor(dt)}", hoverformat=".2f")
    theme.add_hline(fig, 0.0, dash="solid")
    theme.add_vline(fig, mkt.spot, "no move", dash="dot", position="bottom right")
    if has_breakeven:
        for level, position in ((mkt.spot - breakeven, "top left"), (mkt.spot + breakeven, "top right")):
            if lo <= level <= hi:
                theme.add_vline(fig, level, f"breakeven {level:.2f}", position=position)
    return fig
