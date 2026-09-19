"""GBM path simulation and a generic Monte Carlo pricer.

Under the risk-neutral measure the spot follows::

    dS / S = (rate - div) dt + vol dW

whose exact solution over a step ``dt`` is
``S(t + dt) = S(t) * exp((rate - div - vol**2 / 2) * dt + vol * sqrt(dt) * Z)``.
Simulating the logarithm this way has **no discretisation bias** for the spot
itself, whatever the step size; the step size only matters for products that
monitor the path (barriers, lookbacks, Asians).

Monte Carlo is the universal pricer: if you can write the payoff of a path,
you can price the product. The price to pay is statistical noise that shrinks
only like ``1 / sqrt(n_paths)``; antithetic variates (pairing every draw ``Z``
with ``-Z``) cut that noise for free.
"""

from __future__ import annotations

import inspect
import math
from typing import Union

import numpy as np

from .instruments import (
    EXPIRY_TOL,
    CompositeInstrument,
    Instrument,
    Position,
    slice_paths_to_expiry,
)
from .market import Market

__all__ = [
    "simulate_gbm_paths",
    "simulate_gbm_paths_on_grid",
    "simulate_gbm_path",
    "mc_price",
]

Seed = Union[int, np.random.Generator, None]


def _scalar(name: str, value) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 0:
        raise ValueError(f"Monte Carlo needs a scalar {name}, got an array of shape {arr.shape}")
    return float(arr)


def simulate_gbm_paths_on_grid(
    spot: float,
    vol: float,
    rate: float,
    div: float,
    times: np.ndarray,
    n_paths: int,
    seed: Seed = None,
    antithetic: bool = True,
) -> np.ndarray:
    """Simulate risk-neutral GBM paths on an arbitrary increasing time grid.

    Parameters
    ----------
    spot, vol, rate, div : float
        Initial spot, volatility, rate and dividend yield (scalars).
    times : array_like, shape (n_times,)
        Strictly increasing times; ``times[0]`` is "now" (where the path is
        worth ``spot``). Only the differences matter.
    n_paths : int
        Number of paths.
    seed : int, numpy Generator or None
        Seed (or generator) for reproducibility.
    antithetic : bool, default True
        If True the second half of the paths mirrors the first half
        (``Z -> -Z``): path ``i`` and path ``i + ceil(n_paths / 2)`` form an
        antithetic pair. With an odd ``n_paths`` the last mirror is dropped.

    Returns
    -------
    ndarray, shape (n_paths, n_times)
        Simulated spot paths; column 0 is ``spot``.
    """
    spot, vol = _scalar("spot", spot), _scalar("vol", vol)
    rate, div = _scalar("rate", rate), _scalar("div", div)
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size < 1:
        raise ValueError("times must be a 1-D array with at least one entry")
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("times must be strictly increasing")
    if spot <= 0:
        raise ValueError("spot must be strictly positive")
    if vol < 0:
        raise ValueError("vol must be non-negative")
    n_paths = int(n_paths)
    if n_paths < 1:
        raise ValueError("n_paths must be at least 1")

    rng = np.random.default_rng(seed)
    n_draws = math.ceil(n_paths / 2) if antithetic else n_paths
    z = rng.standard_normal((n_draws, dt.size))
    if antithetic:
        z = np.concatenate([z, -z], axis=0)[:n_paths]

    log_increments = (rate - div - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1
    )
    return spot * np.exp(log_paths)


def simulate_gbm_paths(
    spot: float,
    vol: float,
    rate: float = 0.0,
    div: float = 0.0,
    horizon: float = 1.0,
    n_steps: int = 252,
    n_paths: int = 10_000,
    seed: Seed = None,
    antithetic: bool = True,
    t0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate risk-neutral GBM paths on a uniform grid (exact log scheme).

    Parameters
    ----------
    spot, vol, rate, div : float
        Initial spot, volatility, rate and dividend yield.
    horizon : float
        Length of the simulation in years (> 0).
    n_steps : int
        Number of time steps (the paths have ``n_steps + 1`` columns).
    n_paths : int
        Number of paths.
    seed : int, numpy Generator or None
    antithetic : bool, default True
        See :func:`simulate_gbm_paths_on_grid`.
    t0 : float, default 0.0
        Absolute time of the first column, so that ``times`` can be fed
        directly to ``Instrument.path_payoff``.

    Returns
    -------
    times : ndarray, shape (n_steps + 1,)
        ``t0 + linspace(0, horizon, n_steps + 1)``.
    paths : ndarray, shape (n_paths, n_steps + 1)

    Notes
    -----
    To simulate the "real world" rather than the pricing measure, pass the
    expected return of the stock as ``rate`` (and keep ``div``): the drift is
    ``rate - div``.
    """
    horizon = _scalar("horizon", horizon)
    n_steps = int(n_steps)
    if horizon <= 0:
        raise ValueError("horizon must be strictly positive")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    # Simulate on the relative grid so the paths do not depend on t0 (not even by rounding).
    elapsed = np.linspace(0.0, horizon, n_steps + 1)
    paths = simulate_gbm_paths_on_grid(spot, vol, rate, div, elapsed, n_paths, seed, antithetic)
    return float(t0) + elapsed, paths


def simulate_gbm_path(
    spot: float,
    vol: float,
    rate: float = 0.0,
    div: float = 0.0,
    horizon: float = 1.0,
    n_steps: int = 252,
    seed: Seed = None,
    t0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a single GBM path; returns ``(times, path)`` with 1-D arrays."""
    times, paths = simulate_gbm_paths(
        spot, vol, rate, div, horizon, n_steps, n_paths=1, seed=seed, antithetic=False, t0=t0
    )
    return times, paths[0]


def _time_grid(t_now: float, expiries: list[float], n_steps: int) -> np.ndarray:
    """Uniform grid from now to the last expiry, with every expiry on the grid."""
    grid = np.linspace(t_now, max(expiries), n_steps + 1)
    for expiry in expiries:
        nearest = int(np.argmin(np.abs(grid - expiry)))
        if abs(grid[nearest] - expiry) <= 1e-9 and nearest > 0:
            grid[nearest] = expiry
        else:
            grid = np.insert(grid, int(np.searchsorted(grid, expiry)), expiry)
    return grid


def _leg_path_payoff(leg: Position, paths: np.ndarray, times: np.ndarray, rate: float) -> np.ndarray:
    """Position payoff per path; the discount rate is forwarded to products that ask for it.

    A payoff paid BEFORE expiry (the rebate of a knock-out, paid at the hit)
    must be carried to expiry at the risk-free rate to be discounted correctly;
    such products declare a ``rate`` keyword on their ``path_payoff``.
    """
    path_payoff = leg.instrument.path_payoff
    if "rate" in inspect.signature(path_payoff).parameters:
        return leg.quantity * np.asarray(path_payoff(paths, times, rate=rate), dtype=float)
    return leg.quantity * np.asarray(path_payoff(paths, times), dtype=float)


def mc_price(
    instrument: Union[Instrument, Position],
    mkt: Market,
    n_paths: int = 20_000,
    n_steps: int = 252,
    seed: Seed = None,
    antithetic: bool = True,
) -> tuple[float, float]:
    """Monte Carlo price of any instrument through its ``path_payoff``.

    Paths are simulated under the risk-neutral measure from ``mkt.t`` to the
    instrument's expiry and the payoff is discounted at ``mkt.rate``.

    Parameters
    ----------
    instrument : Instrument or Position
        The product. Composites are priced leg by leg **on the same paths**,
        each leg being paid and discounted at its own expiry, so calendar
        structures are handled correctly. Legs that never expire (the
        underlying) or are already expired are valued with ``leg.price(mkt)``.
    mkt : Market
        Scalar market (no array fields).
    n_paths : int, default 20000
        Number of paths (rounded up to an even number when ``antithetic``).
    n_steps : int, default 252
        Number of uniform time steps to the last expiry. Irrelevant for
        path-independent payoffs (the scheme is exact); for monitored products
        it is the monitoring frequency.
    seed : int, numpy Generator or None
    antithetic : bool, default True
        Use antithetic variates; the standard error is then computed from the
        pair averages, which are the independent samples.

    Returns
    -------
    price : float
    stderr : float
        One standard error of the estimate. The true price lies within about
        ``+/- 2 stderr`` with 95% confidence.

    Notes
    -----
    ``path_payoff(paths, times)`` receives the path **from now on**, with
    ``times`` in ABSOLUTE years (``times[0] == mkt.t``, ``times[-1]`` equal to
    the leg expiry). Whatever was accumulated before now (barrier already hit,
    running average so far...) lives in the instrument's fields. A product
    whose ``path_payoff`` accepts a ``rate`` keyword also receives ``mkt.rate``,
    which lets it carry cash paid before expiry (a knock-out rebate) forward.

    For Greeks, wrap this function in a lambda with a **fixed seed** (common
    random numbers) and pass it to :func:`optionlab.numerical.numerical_greeks`
    with larger bumps, e.g. ``spot_bump_rel=1e-2``.
    """
    if isinstance(instrument, Position):
        value, err = mc_price(instrument.instrument, mkt, n_paths, n_steps, seed, antithetic)
        return instrument.quantity * value, abs(instrument.quantity) * err
    if not isinstance(instrument, Instrument):
        raise ValueError("mc_price needs an Instrument or a Position")
    if not mkt.is_scalar:
        raise ValueError("mc_price needs a scalar Market (no array fields)")
    n_paths, n_steps = int(n_paths), int(n_steps)
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if antithetic and n_paths % 2:
        n_paths += 1

    if isinstance(instrument, CompositeInstrument):
        legs = instrument.flatten()
    else:
        legs = (Position(instrument, 1.0),)

    closed_form = 0.0
    simulated: list[Position] = []
    for leg in legs:
        if leg.expiry is None or leg.expiry - mkt.t <= EXPIRY_TOL:
            closed_form += float(leg.price(mkt))
        else:
            simulated.append(leg)
    if not simulated:
        return closed_form, 0.0

    times = _time_grid(mkt.t, sorted({leg.expiry for leg in simulated}), n_steps)
    paths = simulate_gbm_paths_on_grid(
        mkt.spot, mkt.vol, mkt.rate, mkt.div, times, n_paths, seed, antithetic
    )

    pv = np.zeros(n_paths)
    for leg in simulated:
        leg_paths, leg_times = slice_paths_to_expiry(paths, times, leg.expiry)
        payoff = _leg_path_payoff(leg, leg_paths, leg_times, float(mkt.rate))
        if payoff.shape != (n_paths,):
            raise ValueError(
                f"{type(leg.instrument).__name__}.path_payoff must return shape ({n_paths},), "
                f"got {payoff.shape}"
            )
        pv += math.exp(-mkt.rate * (leg.expiry - mkt.t)) * payoff

    samples = 0.5 * (pv[: n_paths // 2] + pv[n_paths // 2:]) if antithetic else pv
    price = closed_form + float(samples.mean())
    stderr = float(samples.std(ddof=1) / math.sqrt(samples.size)) if samples.size > 1 else 0.0
    return price, stderr
