"""Make the market move, manage a book through it, and explain the P&L.

This module is the "flight simulator" of the library. It has four parts:

* **Market scenarios** (:class:`MarketScenario`, :func:`gbm_scenario`,
  :func:`scenario_from_arrays`): a pre-generated path of spot and IMPLIED
  volatility. Two volatilities live in a scenario and they do different jobs:
  the *realised* volatility is how much the spot actually moves (it drives the
  gamma P&L against the theta bill), while the *implied* volatility is the
  level at which options are marked (its moves drive the vega P&L). Keeping
  them separate is the whole pedagogical point.
* **P&L attribution** (:func:`explain_pnl`, :func:`explain_book_pnl`): a
  Taylor expansion of the P&L on the start-of-period Greeks -- delta, gamma,
  theta, vega, vanna, volga, rho -- plus carry, fees and whatever is left
  (``unexplained``).
* **Hedging policies** (:class:`NoHedge`, :class:`DeltaHedgeEveryN`,
  :class:`DeltaBandHedge`, :class:`DeltaHedgeAtVol`) and the step-by-step
  :class:`TradingSimulator` that applies them and logs everything.
* **Experiments** (:func:`delta_hedging_experiment`,
  :func:`gamma_scalping_summary`): the classic results every options trader
  should have seen once -- the hedging error shrinks like ``1 / sqrt(N)``, and
  a delta-hedged option earns ``0.5 * Gamma * S**2 * (realised**2 -
  implied**2)`` per unit of time.

Conventions
-----------
* All Greeks are RAW derivatives (theta per YEAR, vega per 1.00 of vol), so
  ``theta * dt`` with ``dt`` in years and ``vega * dvol`` with ``dvol`` in
  absolute vol are P&L amounts without any extra scaling.
* There is no hidden state: all randomness comes from an explicit ``seed``
  (or from arrays you supply), and the simulator only mutates the book it was
  given. One call to :meth:`TradingSimulator.step` is one "next day" button.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterator, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from .book import DAYS_PER_YEAR, QUANTITY_TOL, Book, Trade, TransactionCosts
from .instruments import CompositeInstrument, Instrument, Position, Underlying
from .market import Market
from .monte_carlo import Seed, simulate_gbm_paths_on_grid

__all__ = [
    "VOL_FLOOR",
    "VOL_CAP",
    "IMPLIED_VOL_MODELS",
    "GREEK_TERMS",
    "ATTRIBUTION_TERMS",
    "HISTORY_GREEKS",
    "MarketScenario",
    "gbm_scenario",
    "scenario_from_arrays",
    "explain_pnl",
    "explain_book_pnl",
    "HedgingPolicy",
    "NoHedge",
    "DeltaHedgeEveryN",
    "DeltaBandHedge",
    "DeltaHedgeAtVol",
    "TradingSimulator",
    "attribution_totals",
    "gamma_scalping_summary",
    "delta_hedging_experiment",
    "hedging_experiment_summary",
]

#: Floor of the simulated implied volatility (1 vol point).
VOL_FLOOR: float = 0.01

#: Cap of the simulated implied volatility (keeps the toy model sane after a big jump).
VOL_CAP: float = 2.0

#: Implied-volatility dynamics understood by :func:`gbm_scenario`.
IMPLIED_VOL_MODELS: tuple[str, ...] = ("constant", "mean_reverting", "spot_correlated")

#: Market-move terms of the attribution, in reporting order.
GREEK_TERMS: tuple[str, ...] = ("delta", "gamma", "theta", "vega", "vanna", "volga", "rho")

#: Every term of the book-level attribution; they sum to the actual P&L.
ATTRIBUTION_TERMS: tuple[str, ...] = GREEK_TERMS + ("carry", "fees", "trading", "unexplained")

#: RAW book Greeks logged at the end of every step.
HISTORY_GREEKS: tuple[str, ...] = ("delta", "gamma", "vega", "theta", "rho", "vanna", "volga")

_DOLLAR_GREEKS: tuple[str, ...] = ("delta_cash", "gamma_cash", "vega_cash", "theta_cash")


# ---------------------------------------------------------------------- #
# Small validation helpers
# ---------------------------------------------------------------------- #
def _number(name: str, value: Any, *, minimum: float | None = None, strict: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None and (number <= minimum if strict else number < minimum):
        bound = ">" if strict else ">="
        raise ValueError(f"{name} must be {bound} {minimum:g}, got {number:g}")
    return number


def _whole(name: str, value: Any, minimum: int) -> int:
    number = _number(name, value)
    if number != int(number) or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(number)


def _scalar_market(mkt: Any, who: str) -> Market:
    if not isinstance(mkt, Market) or not mkt.is_scalar:
        raise ValueError(f"{who} needs a scalar Market (one spot, one vol, one time)")
    return mkt


def _readonly(name: str, values: Any) -> np.ndarray:
    try:
        arr = np.array(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MarketScenario.{name} must be numeric") from exc
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        raise ValueError(f"MarketScenario.{name} must be a 1-D array of finite numbers")
    arr.flags.writeable = False
    return arr


# ---------------------------------------------------------------------- #
# Market scenarios
# ---------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class MarketScenario:
    """A pre-generated market path: one spot and one implied vol per date.

    Parameters
    ----------
    times : array_like, shape (n_steps + 1,)
        ABSOLUTE times in years, strictly increasing. ``times[0]`` is "now".
    spot : array_like, shape (n_steps + 1,)
        Spot path (strictly positive).
    vol : array_like or float
        IMPLIED volatility path (flat surface: one number per date). A float
        means a constant implied vol.
    rate, div : float, default 0.0
        Constant rate and dividend yield.
    name : str, default "scenario"
        Display name.

    Notes
    -----
    Pre-generating the path (rather than drawing random numbers at each step)
    is what makes the simulator reproducible and replayable: the same scenario
    can be run with and without hedging, or against two different books, and
    the comparison is apples to apples. The arrays are read-only.
    """

    times: np.ndarray
    spot: np.ndarray
    vol: np.ndarray
    rate: float = 0.0
    div: float = 0.0
    name: str = "scenario"

    def __post_init__(self) -> None:
        times = _readonly("times", self.times)
        spot = _readonly("spot", self.spot)
        if times.size < 2:
            raise ValueError("a MarketScenario needs at least two dates (one step)")
        if spot.shape != times.shape:
            raise ValueError(
                f"spot has {spot.size} points but times has {times.size}: they must match"
            )
        if np.ndim(self.vol) == 0:
            vol = _readonly("vol", np.full(times.shape, _number("MarketScenario.vol", self.vol)))
        else:
            vol = _readonly("vol", self.vol)
        if vol.shape != times.shape:
            raise ValueError(f"vol has {vol.size} points but times has {times.size}: they must match")
        if np.any(np.diff(times) <= 0):
            raise ValueError("MarketScenario.times must be strictly increasing")
        if np.any(spot <= 0):
            raise ValueError("MarketScenario.spot must be strictly positive")
        if np.any(vol < 0):
            raise ValueError("MarketScenario.vol must be non-negative")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "spot", spot)
        object.__setattr__(self, "vol", vol)
        object.__setattr__(self, "rate", _number("MarketScenario.rate", self.rate))
        object.__setattr__(self, "div", _number("MarketScenario.div", self.div))
        object.__setattr__(self, "name", str(self.name))

    # --- shape ----------------------------------------------------------------
    @property
    def n_steps(self) -> int:
        """Number of steps (one less than the number of dates)."""
        return int(self.times.size - 1)

    def __len__(self) -> int:
        """Number of DATES (``n_steps + 1``)."""
        return int(self.times.size)

    @property
    def horizon(self) -> float:
        """Length of the scenario in years."""
        return float(self.times[-1] - self.times[0])

    # --- access ---------------------------------------------------------------
    def market_at(self, i: int) -> Market:
        """Scalar :class:`Market` at date ``i`` (``0 <= i <= n_steps``)."""
        index = int(i)
        if index != i or not 0 <= index <= self.n_steps:
            raise IndexError(f"scenario date index must be in [0, {self.n_steps}], got {i!r}")
        return Market(
            spot=float(self.spot[index]),
            vol=float(self.vol[index]),
            rate=self.rate,
            div=self.div,
            t=float(self.times[index]),
        )

    def markets(self) -> Iterator[Market]:
        """Iterate over the markets of every date, in order."""
        return (self.market_at(i) for i in range(len(self)))

    def dt(self, i: int) -> float:
        """Length in years of step ``i`` (from date ``i`` to date ``i + 1``)."""
        index = int(i)
        if not 0 <= index < self.n_steps:
            raise IndexError(f"step index must be in [0, {self.n_steps - 1}], got {i!r}")
        return float(self.times[index + 1] - self.times[index])

    # --- statistics -----------------------------------------------------------
    def log_returns(self) -> np.ndarray:
        """Log return of the spot over each step, shape ``(n_steps,)``."""
        return np.diff(np.log(self.spot))

    def realized_vol(self) -> float:
        """Annualised realised volatility of the spot path.

        Uses the zero-mean estimator ``sqrt(sum(r**2) / sum(dt))``, which is
        what the gamma P&L of a delta-hedged book actually "sees". Compare it
        with the implied vol you paid: that gap is the gamma-scalping P&L.
        """
        returns = self.log_returns()
        return float(math.sqrt(float(np.sum(returns**2)) / self.horizon))

    # --- export -----------------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """DataFrame with columns ``t, spot, implied_vol`` (index = date number)."""
        frame = pd.DataFrame({"t": self.times, "spot": self.spot, "implied_vol": self.vol})
        frame.index.name = "step"
        return frame

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (arrays become lists)."""
        return {
            "name": self.name,
            "times": self.times.tolist(),
            "spot": self.spot.tolist(),
            "vol": self.vol.tolist(),
            "rate": self.rate,
            "div": self.div,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MarketScenario":
        """Inverse of :meth:`to_dict`."""
        if not isinstance(data, Mapping):
            raise ValueError("MarketScenario.from_dict needs a mapping")
        known = {"name", "times", "spot", "vol", "rate", "div"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"MarketScenario.from_dict: unknown key(s) {sorted(unknown)}")
        missing = {"times", "spot", "vol"} - set(data)
        if missing:
            raise ValueError(f"MarketScenario.from_dict: missing key(s) {sorted(missing)}")
        return cls(**data)


def scenario_from_arrays(
    spot: Sequence[float],
    vol: Union[float, Sequence[float]],
    times: Sequence[float] | None = None,
    dt: float = 1.0 / 252.0,
    rate: float = 0.0,
    div: float = 0.0,
    t0: float = 0.0,
    name: str = "replay",
) -> MarketScenario:
    """Build a scenario from your own data (e.g. to replay historical prices).

    Parameters
    ----------
    spot : sequence of float
        Spot path, one value per date (at least two).
    vol : float or sequence of float
        Implied volatility: a constant, or one value per date (for a replay,
        a volatility index divided by 100 is a reasonable proxy).
    times : sequence of float, optional
        Absolute times in years. When omitted the dates are evenly spaced:
        ``t0, t0 + dt, t0 + 2 dt, ...``.
    dt : float, default 1/252
        Step used when ``times`` is omitted (one trading day).
    rate, div : float
        Constant rate and dividend yield.
    t0 : float, default 0.0
        Time of the first date when ``times`` is omitted.
    name : str, default "replay"
    """
    spot_arr = np.asarray(spot, dtype=float)
    if spot_arr.ndim != 1:
        raise ValueError("spot must be a 1-D sequence")
    if times is None:
        step = _number("dt", dt, minimum=0.0, strict=True)
        times = _number("t0", t0) + step * np.arange(spot_arr.size)
    return MarketScenario(times=times, spot=spot_arr, vol=vol, rate=rate, div=div, name=name)


def gbm_scenario(
    mkt0: Market,
    horizon: float,
    n_steps: int,
    realized_vol: float | None = None,
    drift: float | None = None,
    implied_vol_model: str = "constant",
    vol_of_vol: float = 0.8,
    kappa: float = 4.0,
    rho_spot_vol: float = -0.7,
    long_run_vol: float | None = None,
    jump_intensity: float = 0.0,
    jump_mean: float = -0.05,
    jump_std: float = 0.05,
    seed: Seed = None,
    name: str | None = None,
) -> MarketScenario:
    """Simulate one market path: GBM spot (with optional jumps) and an implied-vol path.

    The spot follows ``dS/S = drift dt + realized_vol dW`` (exact log scheme)
    plus optional compound-Poisson jumps. The implied volatility is a separate
    process -- a deliberately simple TOY model of the *level* of a flat
    surface, not a stochastic-volatility model (the spot does NOT diffuse with
    the implied vol).

    Parameters
    ----------
    mkt0 : Market
        Scalar starting market; ``mkt0.vol`` is the initial IMPLIED vol and
        ``mkt0.t`` the time of the first date.
    horizon : float
        Length of the scenario in years (> 0).
    n_steps : int
        Number of steps (e.g. ``horizon=0.25, n_steps=63`` is one quarter of
        trading days).
    realized_vol : float, optional
        Volatility at which the spot really moves. Defaults to the implied vol
        (the Black-Scholes world). Set it higher to reward long gamma, lower
        to reward short gamma.
    drift : float, optional
        Expected growth rate of the spot. Defaults to ``rate - div`` (the
        risk-neutral drift); pass a real-world ``mu`` to see that the drift
        barely matters to a delta-hedged book.
    implied_vol_model : {"constant", "mean_reverting", "spot_correlated"}
        * ``"constant"``: implied vol stays at ``mkt0.vol``.
        * ``"mean_reverting"``: ``ln(vol)`` follows an Ornstein-Uhlenbeck
          process pulled towards ``ln(long_run_vol)`` at speed ``kappa`` with
          volatility ``vol_of_vol``, independent of the spot.
        * ``"spot_correlated"``: the same process, but its shocks have
          correlation ``rho_spot_vol`` with the spot shocks, and spot jumps move
          the log-vol with the same beta (``rho * vol_of_vol / realized_vol``
          per unit of log-return). With a negative correlation this is the
          equity "leverage effect": vol jumps when the market falls.
    vol_of_vol : float, default 0.8
        Annualised volatility of ``ln(implied vol)``. With the defaults and a
        20% realised vol, a -1% day lifts the implied vol by about 2.8% of its
        level (20% -> 20.6%), the right order of magnitude for an equity index.
    kappa : float, default 4.0
        Mean-reversion speed per year (``0`` = random walk). The half-life of a
        vol shock is ``ln(2) / kappa`` years (about two months by default).
    rho_spot_vol : float, default -0.7
        Spot/vol correlation, used by ``"spot_correlated"`` only.
    long_run_vol : float, optional
        Level the implied vol reverts to (default: ``mkt0.vol``).
    jump_intensity : float, default 0.0
        Expected number of jumps per year (0 = pure diffusion).
    jump_mean, jump_std : float
        Mean and standard deviation of a jump in LOG-spot (``-0.05`` is about
        a -5% gap). The drift is compensated so that the expected growth rate
        stays ``drift``. Jumps are what delta hedging cannot handle: the spot
        gaps through your hedge and the gamma P&L arrives all at once.
    seed : int, numpy Generator or None
        Seed for reproducibility. The same seed gives the same SPOT path
        whatever the implied-vol model, so models can be compared like for like.
    name : str, optional
        Scenario name (a descriptive one is built by default).

    Returns
    -------
    MarketScenario
        ``n_steps + 1`` dates; date 0 is exactly ``mkt0``. Simulated implied
        vols are floored at :data:`VOL_FLOOR` (1%) and capped at :data:`VOL_CAP`.
    """
    mkt0 = _scalar_market(mkt0, "gbm_scenario")
    if mkt0.spot <= 0:
        raise ValueError("gbm_scenario needs a strictly positive starting spot")
    horizon = _number("horizon", horizon, minimum=0.0, strict=True)
    n_steps = _whole("n_steps", n_steps, 1)
    sigma = mkt0.vol if realized_vol is None else _number("realized_vol", realized_vol, minimum=0.0)
    mu = (mkt0.rate - mkt0.div) if drift is None else _number("drift", drift)
    if implied_vol_model not in IMPLIED_VOL_MODELS:
        raise ValueError(
            f"implied_vol_model must be one of {list(IMPLIED_VOL_MODELS)}, got {implied_vol_model!r}"
        )
    vol_of_vol = _number("vol_of_vol", vol_of_vol, minimum=0.0)
    kappa = _number("kappa", kappa, minimum=0.0)
    rho = _number("rho_spot_vol", rho_spot_vol)
    if abs(rho) > 1.0:
        raise ValueError("rho_spot_vol must be between -1 and 1")
    jump_intensity = _number("jump_intensity", jump_intensity, minimum=0.0)
    jump_mean = _number("jump_mean", jump_mean)
    jump_std = _number("jump_std", jump_std, minimum=0.0)

    dt = horizon / n_steps
    rng = np.random.default_rng(seed)
    # Fixed draw order, independent of the options, so a seed pins the spot path.
    z_spot = rng.standard_normal(n_steps)
    z_indep = rng.standard_normal(n_steps)
    n_jumps = rng.poisson(jump_intensity * dt, n_steps)
    z_jump = rng.standard_normal(n_steps)

    jumps = n_jumps * jump_mean + np.sqrt(n_jumps) * jump_std * z_jump
    compensator = jump_intensity * math.expm1(jump_mean + 0.5 * jump_std**2)
    log_returns = (mu - 0.5 * sigma**2 - compensator) * dt + sigma * math.sqrt(dt) * z_spot + jumps
    spot = mkt0.spot * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)]))

    vol = np.full(n_steps + 1, mkt0.vol)
    if implied_vol_model != "constant":
        if mkt0.vol <= 0:
            raise ValueError("a stochastic implied-vol model needs a strictly positive mkt0.vol")
        target = mkt0.vol if long_run_vol is None else _number(
            "long_run_vol", long_run_vol, minimum=0.0, strict=True
        )
        if implied_vol_model == "mean_reverting":
            rho = 0.0
        z_vol = rho * z_spot + math.sqrt(1.0 - rho**2) * z_indep
        jump_beta = rho * vol_of_vol / sigma if sigma > 0 else 0.0
        decay = math.exp(-kappa * dt)
        shock_std = vol_of_vol * (
            math.sqrt((1.0 - decay**2) / (2.0 * kappa)) if kappa > 0 else math.sqrt(dt)
        )
        level, low, high = math.log(target), math.log(VOL_FLOOR), math.log(VOL_CAP)
        x = math.log(mkt0.vol)
        for i in range(n_steps):
            x = level + (x - level) * decay + shock_std * z_vol[i] + jump_beta * jumps[i]
            x = min(max(x, low), high)
            vol[i + 1] = math.exp(x)

    if name is None:
        name = f"GBM {n_steps} steps, realised {sigma:.0%}, implied {mkt0.vol:.0%} ({implied_vol_model})"
    times = mkt0.t + np.linspace(0.0, horizon, n_steps + 1)
    return MarketScenario(times=times, spot=spot, vol=vol, rate=mkt0.rate, div=mkt0.div, name=name)


# ---------------------------------------------------------------------- #
# P&L attribution
# ---------------------------------------------------------------------- #
def explain_pnl(
    greeks_start: Mapping[str, Any], mkt_start: Market, mkt_end: Market
) -> dict[str, Any]:
    """Taylor expansion of a P&L on the START-of-period Greeks (RAW units).

    With ``dS``, ``dvol``, ``dr`` the changes in spot, implied vol and rate and
    ``dt`` the time elapsed (years)::

        delta = Delta * dS              gamma = 0.5 * Gamma * dS**2
        theta = Theta * dt              vega  = Vega * dvol
        vanna = Vanna * dS * dvol       volga = 0.5 * Volga * dvol**2
        rho   = Rho * dr

    Because the Greeks are raw derivatives (theta per YEAR, vega and rho per
    1.00) each product is directly a currency amount.

    Parameters
    ----------
    greeks_start : mapping
        RAW Greeks at the start of the period (``book.greeks(mkt_start)`` or
        ``instrument.greeks(mkt_start)``); missing keys count as 0.
    mkt_start, mkt_end : Market
        Markets at both ends of the period (arrays allowed, they broadcast).

    Returns
    -------
    dict
        One entry per name in :data:`GREEK_TERMS`; their sum is the explained
        P&L of the positions.

    Notes
    -----
    How to read it: **delta** is the directional bet; **gamma** is always
    positive for a long-option book (you earn on any move, up or down) and
    **theta** is the rent you pay for it; **vega** is the re-marking of the
    options when implied vol moves. The expansion is local, so it degrades for
    big moves, long steps and near expiry -- that gap is the ``unexplained``
    term of :func:`explain_book_pnl`.
    """
    d_spot = mkt_end.spot - mkt_start.spot
    d_vol = mkt_end.vol - mkt_start.vol
    d_rate = mkt_end.rate - mkt_start.rate
    d_time = mkt_end.t - mkt_start.t

    def greek(name: str) -> Any:
        return greeks_start.get(name, 0.0)

    return {
        "delta": greek("delta") * d_spot,
        "gamma": 0.5 * greek("gamma") * d_spot**2,
        "theta": greek("theta") * d_time,
        "vega": greek("vega") * d_vol,
        "vanna": greek("vanna") * d_spot * d_vol,
        "volga": 0.5 * greek("volga") * d_vol**2,
        "rho": greek("rho") * d_rate,
    }


def explain_book_pnl(
    greeks_start: Mapping[str, Any],
    mkt_start: Market,
    mkt_end: Market,
    actual_pnl: float,
    carry: float = 0.0,
    fees: float = 0.0,
    trading: float = 0.0,
) -> dict[str, float]:
    """Book-level attribution: Greek terms, carry, fees, trading edge and the residual.

    Parameters
    ----------
    greeks_start, mkt_start, mkt_end
        See :func:`explain_pnl`.
    actual_pnl : float
        Actual change of the book's P&L over the period (everything included).
    carry : float, default 0.0
        Interest earned on cash plus dividends received over the period.
    fees : float, default 0.0
        Fees and transaction costs PAID over the period (a positive number;
        the ``"fees"`` term of the result is its negative).
    trading : float, default 0.0
        Mark-to-model edge of trades done away from the model price
        (``quantity * (model - traded price)``; 0 when trading at the model).

    Returns
    -------
    dict
        Keys :data:`ATTRIBUTION_TERMS`. ``unexplained`` is defined as
        ``actual_pnl`` minus every other term, so **the values always sum to
        ``actual_pnl``**. A small ``unexplained`` means the Greeks told the
        whole story; a large one means higher-order effects mattered (a big
        gap, a long step, an expiry, a barrier event).
    """
    terms = {name: float(value) for name, value in explain_pnl(greeks_start, mkt_start, mkt_end).items()}
    terms["carry"] = float(carry)
    terms["fees"] = -float(fees)
    terms["trading"] = float(trading)
    terms["unexplained"] = float(actual_pnl) - sum(terms.values())
    return terms


# ---------------------------------------------------------------------- #
# Hedging policies
# ---------------------------------------------------------------------- #
class HedgingPolicy(ABC):
    """Rule deciding which hedge trades to do at each simulation date.

    Subclass it and implement :meth:`rebalance`; trade through the book
    (``book.trade`` / ``book.hedge_delta``...) so that cash, fees and the
    blotter stay consistent, and return the trades you made. Policies are
    stateless (they only look at the book, the market and the step number),
    which keeps a simulation resettable and reproducible.
    """

    @abstractmethod
    def rebalance(self, book: Book, mkt: Market, step_index: int) -> list[Trade]:
        """Trade the hedge for date ``step_index`` and return the executed trades."""

    @property
    def label(self) -> str:
        """Short description used in tables and chart titles."""
        return type(self).__name__


@dataclass(frozen=True)
class NoHedge(HedgingPolicy):
    """Do nothing: the book keeps whatever delta it has (a directional position)."""

    def rebalance(self, book: Book, mkt: Market, step_index: int) -> list[Trade]:
        return []

    @property
    def label(self) -> str:
        return "no hedge"


@dataclass(frozen=True)
class DeltaHedgeEveryN(HedgingPolicy):
    """Bring the book delta back to a target every ``n_steps`` dates (time-based hedging).

    Parameters
    ----------
    n_steps : int, default 1
        Rebalance when the step number is a multiple of ``n_steps`` (1 = every
        date). Date 0 always rebalances.
    target_delta : float, default 0.0
        Delta to come back to, in shares.
    min_trade : float, default 0.0
        Skip trades smaller than this many shares.

    Notes
    -----
    Hedging more often does not change the EXPECTED P&L, it shrinks its
    dispersion: the standard deviation of the hedging error falls like
    ``1 / sqrt(number of rebalances)``. What more hedging does cost is
    transaction fees, which grow with the number of trades.
    """

    n_steps: int = 1
    target_delta: float = 0.0
    min_trade: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_steps", _whole("n_steps", self.n_steps, 1))
        object.__setattr__(self, "target_delta", _number("target_delta", self.target_delta))
        object.__setattr__(self, "min_trade", _number("min_trade", self.min_trade, minimum=0.0))

    def rebalance(self, book: Book, mkt: Market, step_index: int) -> list[Trade]:
        if step_index % self.n_steps:
            return []
        trade = book.hedge_delta(mkt, self.target_delta, min_trade=self.min_trade, note="delta hedge")
        return [] if trade is None else [trade]

    @property
    def label(self) -> str:
        return "delta hedge every step" if self.n_steps == 1 else f"delta hedge every {self.n_steps} steps"


def _option_quantity(book: Book) -> float:
    """Gross number of non-stock units held (composite lines expanded)."""
    total = 0.0
    for position in book.positions:
        if isinstance(position.instrument, CompositeInstrument):
            legs = [leg.scaled(position.quantity) for leg in position.instrument.flatten()]
        else:
            legs = [position]
        total += sum(abs(leg.quantity) for leg in legs if not isinstance(leg.instrument, Underlying))
    return total


@dataclass(frozen=True)
class DeltaBandHedge(HedgingPolicy):
    """Re-hedge only when the delta drifts outside a band (move-based hedging).

    Parameters
    ----------
    band : float
        Half-width of the no-trade zone around ``target_delta``. In shares when
        ``relative`` is False. When ``relative`` is True it is a delta PER
        OPTION held: ``band=0.05`` on a book of 100 options re-hedges once the
        book is off by more than 5 shares ("5 deltas per contract").
    relative : bool, default False
        See ``band``.
    target_delta : float, default 0.0
        Delta the hedge comes back to when the band is breached.

    Notes
    -----
    A band trades only when the market has actually moved, so for the same
    number of trades it usually controls risk better than a fixed calendar --
    and it is the practical answer to transaction costs: the wider the band,
    the fewer the trades, the larger the residual delta risk.
    """

    band: float
    relative: bool = False
    target_delta: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "band", _number("band", self.band, minimum=0.0))
        object.__setattr__(self, "relative", bool(self.relative))
        object.__setattr__(self, "target_delta", _number("target_delta", self.target_delta))

    def rebalance(self, book: Book, mkt: Market, step_index: int) -> list[Trade]:
        width = self.band * _option_quantity(book) if self.relative else self.band
        trade = book.hedge_delta(mkt, self.target_delta, min_trade=width, note="band hedge")
        return [] if trade is None else [trade]

    @property
    def label(self) -> str:
        unit = "delta per option" if self.relative else "shares"
        return f"delta band +/-{self.band:g} {unit}"


@dataclass(frozen=True)
class DeltaHedgeAtVol(HedgingPolicy):
    """Delta hedge with the delta computed at YOUR volatility, not the market's.

    Parameters
    ----------
    hedge_vol : float
        Volatility used to compute the delta (the trades themselves happen at
        the market spot).
    n_steps : int, default 1
        Rebalance every ``n_steps`` dates.
    target_delta : float, default 0.0
        Target for the delta measured at ``hedge_vol``.

    Notes
    -----
    This is the classic experiment of Ahmad and Wilmott ("Which free lunch
    would you like today, sir?"). If you buy an option at an implied vol below
    the vol you expect to be realised, you make money either way, but HOW
    depends on the hedging vol: hedging at the (correct) realised vol locks
    in a known total profit with a noisy mark-to-market path; hedging at the
    implied vol gives a smooth daily P&L whose total depends on the path
    (it is large when the spot stays where gamma is).
    """

    hedge_vol: float
    n_steps: int = 1
    target_delta: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "hedge_vol", _number("hedge_vol", self.hedge_vol, minimum=0.0, strict=True))
        object.__setattr__(self, "n_steps", _whole("n_steps", self.n_steps, 1))
        object.__setattr__(self, "target_delta", _number("target_delta", self.target_delta))

    def rebalance(self, book: Book, mkt: Market, step_index: int) -> list[Trade]:
        if step_index % self.n_steps:
            return []
        size = self.target_delta - book.greeks(mkt.bumped(vol=self.hedge_vol))["delta"]
        if abs(size) <= QUANTITY_TOL:
            return []
        note = f"delta hedge at vol {self.hedge_vol:.1%}"
        return [book.trade(Underlying(), size, mkt, note=note)]

    @property
    def label(self) -> str:
        return f"delta hedge at vol {self.hedge_vol:.1%}"


# ---------------------------------------------------------------------- #
# The simulator
# ---------------------------------------------------------------------- #
def _advance_book(book: Book, mkt_start: Market, mkt_end: Market) -> tuple[float, list]:
    """Life cycle of one step; returns ``(carry, settlements)``.

    Carry accrues first, on the start-of-period cash and share position and at
    the start-of-period market (that is what was held over the period); then
    the new spot is shown to path-dependent products and expired lines settle.
    """
    accrued = book.accrue(mkt_start, float(mkt_end.t - mkt_start.t))
    n_before = len(book.settlements)
    book.observe(mkt_end)  # may pay cash too (the rebate of a barrier that knocks out)
    book.settle_expired(mkt_end)
    return accrued["interest"] + accrued["dividends"], book.settlements[n_before:]


def _history_columns() -> list[str]:
    columns = ["t", "spot", "implied_vol", "value", "cash", "positions_value", "pnl_step", "pnl_cum"]
    columns += [f"pnl_{term}" for term in ATTRIBUTION_TERMS]
    columns += [f"cum_pnl_{term}" for term in ATTRIBUTION_TERMS]
    columns += list(HISTORY_GREEKS) + list(_DOLLAR_GREEKS)
    columns += [
        "delta_before_hedge", "hedge_trades", "hedge_shares", "stock_position",
        "fees_paid", "n_positions", "n_settled", "settlement_cash",
    ]
    return columns


class TradingSimulator:
    """Step a book through a market scenario, hedge it, and explain every P&L.

    Parameters
    ----------
    book : Book
        The book to manage. It is mutated IN PLACE as the simulation runs (so
        the reference you hold stays live); a copy of its initial state is
        kept for :meth:`reset`.
    scenario : MarketScenario
        The market path. The book is assumed to be marked at
        ``scenario.market_at(0)`` when the simulation starts.
    policy : HedgingPolicy, optional
        Automatic hedging rule (default :class:`NoHedge`). The attribute can be
        reassigned between steps.
    costs : TransactionCosts, optional
        When given, replaces ``book.costs``: every later trade (yours and the
        policy's) pays these costs on top of the model price.
    hedge_at_start : bool, default True
        Apply the policy at date 0 as well, so that a delta-hedged run starts
        hedged. Its fees show up in the first row of the history.

    Notes
    -----
    :meth:`step` does, in this order:

    1. snapshot the start-of-step Greeks and P&L (after any trade you made
       since the previous step);
    2. move the market to the next date;
    3. life cycle: accrue carry over the elapsed period (interest on the
       start-of-period cash, dividends on the shares held), show the new spot
       to path-dependent products (``book.observe``), cash-settle what expired;
    4. measure the actual P&L and attribute it to the start-of-step Greeks;
    5. apply the hedging policy at the NEW market (model price plus costs);
    6. append one row to the history.

    Trading at the model price never creates P&L, so hedging shows up in the
    attribution only through the fees it costs and through the delta it
    removes from the NEXT step.

    Examples
    --------
    >>> from optionlab.instruments import EuropeanOption
    >>> mkt0 = Market(spot=100.0, vol=0.2)
    >>> book = Book("long gamma")
    >>> _ = book.trade(EuropeanOption("call", 100.0, 0.25), 100, mkt0)
    >>> scenario = gbm_scenario(mkt0, horizon=0.25, n_steps=63, realized_vol=0.3, seed=1)
    >>> sim = TradingSimulator(book, scenario, DeltaHedgeEveryN(1))
    >>> history = sim.run()
    >>> len(history) == scenario.n_steps + 1 and sim.is_finished
    True
    """

    def __init__(
        self,
        book: Book,
        scenario: MarketScenario,
        policy: HedgingPolicy | None = None,
        costs: TransactionCosts | None = None,
        hedge_at_start: bool = True,
    ) -> None:
        if not isinstance(book, Book):
            raise ValueError(f"TradingSimulator needs a Book, got {type(book).__name__}")
        if not isinstance(scenario, MarketScenario):
            raise ValueError(f"TradingSimulator needs a MarketScenario, got {type(scenario).__name__}")
        policy = NoHedge() if policy is None else policy
        if not isinstance(policy, HedgingPolicy):
            raise ValueError("policy must be a HedgingPolicy (or None for no hedging)")
        if costs is not None:
            if not isinstance(costs, TransactionCosts):
                raise ValueError("costs must be a TransactionCosts instance or None")
            book.costs = costs
        self.book = book
        self.scenario = scenario
        self.policy = policy
        self.hedge_at_start = bool(hedge_at_start)
        self._initial_book = book.copy()
        self._start()

    # --- state ------------------------------------------------------------------
    @property
    def step_index(self) -> int:
        """Index of the current date in the scenario (0 before the first step)."""
        return self._index

    @property
    def current_market(self) -> Market:
        """Market of the current date: trade against this one between steps."""
        return self.scenario.market_at(self._index)

    @property
    def is_finished(self) -> bool:
        """True once the last date of the scenario has been reached."""
        return self._index >= self.scenario.n_steps

    @property
    def steps_remaining(self) -> int:
        return self.scenario.n_steps - self._index

    @property
    def history(self) -> pd.DataFrame:
        """One row per date reached so far (index ``step``; row 0 is the start).

        Columns
        -------
        * ``t, spot, implied_vol``: the market of the date.
        * ``value`` (net liquidation value), ``cash``, ``positions_value``.
        * ``pnl_step`` and ``pnl_cum``: actual P&L of the step and since the
          simulation started (fees included).
        * ``pnl_<term>`` and ``cum_pnl_<term>`` for every term of
          :data:`ATTRIBUTION_TERMS`: the attribution of the step and its
          running total. Each row satisfies ``sum(pnl_<term>) == pnl_step``.
        * RAW book Greeks at the END of the step, after hedging
          (:data:`HISTORY_GREEKS`), and the dollar Greeks ``delta_cash,
          gamma_cash, vega_cash, theta_cash`` (same definitions as
          :meth:`optionlab.book.Book.dollar_greeks`).
        * ``delta_before_hedge``: book delta at the new market before the
          policy traded; ``hedge_trades`` / ``hedge_shares``: number of policy
          trades and net shares they bought; ``stock_position``: shares held
          after hedging; ``fees_paid``: fees booked in this row;
          ``n_positions``, ``n_settled``, ``settlement_cash`` (expiry
          settlements plus cash paid by path events such as a knock-out rebate).
        """
        frame = pd.DataFrame(self._rows, columns=_history_columns())
        frame.index.name = "step"
        return frame

    @property
    def last_row(self) -> dict[str, float]:
        """The most recent history row as a dict (a copy)."""
        return dict(self._rows[-1])

    # --- actions ------------------------------------------------------------------
    def trade(
        self,
        instrument: Union[Instrument, Position],
        quantity: float,
        price: float | None = None,
        fee: float = 0.0,
        note: str = "",
    ) -> Trade:
        """Trade in the book at the CURRENT market (see :meth:`optionlab.book.Book.trade`).

        The trade is picked up by the next :meth:`step`: its fees go to the
        ``fees`` term and, if ``price`` differs from the model price, the
        difference goes to the ``trading`` term.
        """
        return self.book.trade(instrument, quantity, self.current_market, price=price, fee=fee, note=note)

    def step(self) -> dict[str, float]:
        """Advance one date; returns the new history row.

        Raises
        ------
        RuntimeError
            If the scenario is finished (check :attr:`is_finished`).
        """
        if self.is_finished:
            raise RuntimeError("the scenario is finished: reset() the simulator or build a new one")
        book = self.book
        mkt_start = self.current_market

        # (1) start-of-step snapshot (includes anything traded since the last row)
        greeks_start, pnl_start = self._mark(mkt_start)
        fees_between = self._new_fees()
        edge_between = (pnl_start - self._pnl_last) + fees_between

        # (2) the market moves, (3) life cycle
        self._index += 1
        mkt_end = self.current_market
        carry, settlements = _advance_book(book, mkt_start, mkt_end)

        # (4) actual P&L of the market move and its attribution
        greeks_mid, pnl_mid = self._mark(mkt_end)
        terms = explain_book_pnl(greeks_start, mkt_start, mkt_end, pnl_mid - pnl_start, carry=carry)

        # (5) hedge at the new market, (6) log
        return self._close_row(mkt_end, greeks_mid, pnl_mid, terms, fees_between, edge_between, settlements)

    def run(self, n: int | None = None) -> pd.DataFrame:
        """Run ``n`` steps (all the remaining ones by default) and return :attr:`history`."""
        remaining = self.steps_remaining
        count = remaining if n is None else min(_whole("n", n, 0), remaining)
        for _ in range(count):
            self.step()
        return self.history

    def reset(self) -> None:
        """Go back to date 0 with the book as it was when the simulator was built.

        ``self.book`` is REBOUND to a fresh copy of the initial book (the
        object you passed in keeps its end-of-run state), so re-read
        ``sim.book`` after a reset.
        """
        self.book = self._initial_book.copy()
        self._start()

    # --- internals ------------------------------------------------------------------
    def _start(self) -> None:
        self._index = 0
        self._rows: list[dict[str, float]] = []
        self._cumulative = dict.fromkeys(ATTRIBUTION_TERMS, 0.0)
        self._pnl_cum = 0.0
        self._mark_key: tuple | None = None
        self._mark_value: tuple[dict[str, float], float] | None = None
        self._fees_seen = len(self.book.trades)
        mkt = self.current_market
        greeks, pnl = self._mark(mkt)
        self._pnl_last = pnl
        terms = dict.fromkeys(ATTRIBUTION_TERMS, 0.0)
        self._close_row(mkt, greeks, pnl, terms, 0.0, 0.0, [], hedge=self.hedge_at_start)

    def _fingerprint(self, mkt: Market) -> tuple:
        book = self.book
        return (mkt, len(book.trades), len(book.settlements), book.cash, book.initial_cash, book.positions)

    def _mark(self, mkt: Market) -> tuple[dict[str, float], float]:
        """Book Greeks and P&L at ``mkt``, recomputed only if the book or market changed."""
        key = self._fingerprint(mkt)
        if key != self._mark_key:
            greeks = self.book.greeks(mkt)
            pnl = greeks["price"] + self.book.cash - self.book.initial_cash
            self._mark_key, self._mark_value = key, (greeks, pnl)
        return self._mark_value

    def _new_fees(self) -> float:
        """Fees of the trades booked since the last call."""
        fees = sum(trade.fees for trade in self.book.trades[self._fees_seen:])
        self._fees_seen = len(self.book.trades)
        return fees

    def _close_row(
        self,
        mkt: Market,
        greeks_mid: dict[str, float],
        pnl_mid: float,
        terms: dict[str, float],
        fees_between: float,
        edge_between: float,
        settlements: list,
        hedge: bool = True,
    ) -> dict[str, float]:
        book = self.book
        n_trades = len(book.trades)
        if hedge:
            self.policy.rebalance(book, mkt, self._index)
        hedge_trades = book.trades[n_trades:]
        fees_hedge = self._new_fees()
        greeks_end, pnl_end = self._mark(mkt)
        edge_hedge = (pnl_end - pnl_mid) + fees_hedge if hedge_trades else 0.0

        terms = dict(terms)
        terms["fees"] = -(fees_between + fees_hedge)
        terms["trading"] = edge_between + edge_hedge
        pnl_step = pnl_end - self._pnl_last
        self._pnl_last = pnl_end
        self._pnl_cum += pnl_step
        for name in ATTRIBUTION_TERMS:
            self._cumulative[name] += terms[name]

        spot = float(mkt.spot)
        row: dict[str, float] = {
            "t": float(mkt.t),
            "spot": spot,
            "implied_vol": float(mkt.vol),
            "value": greeks_end["price"] + book.cash,
            "cash": book.cash,
            "positions_value": greeks_end["price"],
            "pnl_step": pnl_step,
            "pnl_cum": self._pnl_cum,
        }
        row.update({f"pnl_{name}": terms[name] for name in ATTRIBUTION_TERMS})
        row.update({f"cum_pnl_{name}": self._cumulative[name] for name in ATTRIBUTION_TERMS})
        row.update({name: greeks_end[name] for name in HISTORY_GREEKS})
        row.update(
            {
                "delta_cash": greeks_end["delta"] * spot,
                "gamma_cash": greeks_end["gamma"] * spot**2 / 100.0,
                "vega_cash": greeks_end["vega"] / 100.0,
                "theta_cash": greeks_end["theta"] / DAYS_PER_YEAR,
                "delta_before_hedge": greeks_mid["delta"],
                "hedge_trades": float(len(hedge_trades)),
                "hedge_shares": sum(
                    trade.quantity for trade in hedge_trades if isinstance(trade.instrument, Underlying)
                ),
                "stock_position": book.underlying_quantity(),
                "fees_paid": fees_between + fees_hedge,
                "n_positions": float(len(book)),
                "n_settled": float(len(settlements)),
                "settlement_cash": sum(record.cash_flow for record in settlements),
            }
        )
        self._rows.append(row)
        return dict(row)


# ---------------------------------------------------------------------- #
# Reading a history
# ---------------------------------------------------------------------- #
def _require_history(history: pd.DataFrame, columns: Sequence[str]) -> None:
    if not isinstance(history, pd.DataFrame):
        raise ValueError("history must be the DataFrame returned by TradingSimulator.history")
    missing = [c for c in columns if c not in history.columns]
    if missing:
        raise ValueError(f"history is missing column(s) {missing}")
    if history.empty:
        raise ValueError("history is empty")


def attribution_totals(history: pd.DataFrame) -> dict[str, float]:
    """Total P&L of each attribution term over a run, plus ``"actual"``.

    The terms of :data:`ATTRIBUTION_TERMS` sum to ``"actual"`` (the total
    actual P&L), which is what a desk's daily "P&L explain" report checks.
    """
    _require_history(history, ["pnl_step", *(f"pnl_{term}" for term in ATTRIBUTION_TERMS)])
    totals = {term: float(history[f"pnl_{term}"].sum()) for term in ATTRIBUTION_TERMS}
    totals["actual"] = float(history["pnl_step"].sum())
    return totals


def gamma_scalping_summary(history: pd.DataFrame) -> dict[str, float]:
    """Compare what gamma earned with what theta cost over a run.

    Parameters
    ----------
    history : pandas.DataFrame
        :attr:`TradingSimulator.history`.

    Returns
    -------
    dict
        * ``gamma_pnl``, ``theta_pnl``: cumulative gamma and theta terms;
          ``gamma_plus_theta`` is their sum, the "gamma scalping" result.
        * ``gamma_theta_ratio`` = ``gamma_pnl / |theta_pnl|``: above 1 the
          moves more than paid the rent (``nan`` when no theta was paid).
        * ``delta_pnl``, ``vega_pnl``, ``other_pnl`` (everything else, fees and
          unexplained included) and ``total_pnl``.
        * ``realized_vol``: annualised realised vol of the spot path;
          ``mean_implied_vol``: time-average implied vol; ``vol_edge`` is
          their difference.

    Notes
    -----
    For a delta-hedged vanilla book, ``gamma + theta`` is approximately
    ``sum(0.5 * Gamma * S**2 * (realised**2 - implied**2) * dt)``: a long-gamma
    book wins when the spot realises MORE than the implied vol it paid through
    theta, a short-gamma book wins when it realises less. The sign of
    ``vol_edge`` and the sign of ``gamma_plus_theta`` should agree for a
    long-gamma book (and be opposite for a short-gamma book).
    """
    _require_history(history, ["t", "spot", "implied_vol", "pnl_step", "pnl_gamma", "pnl_theta",
                               "pnl_delta", "pnl_vega"])
    gamma = float(history["pnl_gamma"].sum())
    theta = float(history["pnl_theta"].sum())
    delta = float(history["pnl_delta"].sum())
    vega = float(history["pnl_vega"].sum())
    total = float(history["pnl_step"].sum())

    times = history["t"].to_numpy(dtype=float)
    elapsed = float(times[-1] - times[0])
    if elapsed > 0:
        returns = np.diff(np.log(history["spot"].to_numpy(dtype=float)))
        realized = math.sqrt(float(np.sum(returns**2)) / elapsed)
        implied = history["implied_vol"].to_numpy(dtype=float)
        mean_implied = float(np.sum(implied[:-1] * np.diff(times)) / elapsed)
    else:
        realized, mean_implied = math.nan, float(history["implied_vol"].iloc[0])
    return {
        "gamma_pnl": gamma,
        "theta_pnl": theta,
        "gamma_plus_theta": gamma + theta,
        "gamma_theta_ratio": gamma / abs(theta) if theta != 0 else math.nan,
        "delta_pnl": delta,
        "vega_pnl": vega,
        "other_pnl": total - gamma - theta - delta - vega,
        "total_pnl": total,
        "realized_vol": realized,
        "mean_implied_vol": mean_implied,
        "vol_edge": realized - mean_implied,
    }


# ---------------------------------------------------------------------- #
# Hedging experiments
# ---------------------------------------------------------------------- #
def _is_path_independent(instrument: Instrument) -> bool:
    """True when ``observe`` is the inherited no-op for the instrument and all its legs."""
    if isinstance(instrument, CompositeInstrument):
        return all(_is_path_independent(leg.instrument) for leg in instrument.flatten())
    return type(instrument).observe is Instrument.observe


def _single_expiry(instrument: Instrument) -> bool:
    if isinstance(instrument, CompositeInstrument):
        expiries = {leg.expiry for leg in instrument.flatten() if leg.expiry is not None}
        return len(expiries) <= 1
    return True


def _vectorised_hedging_pnl(
    instrument: Instrument,
    quantity: float,
    mkt0: Market,
    times: np.ndarray,
    paths: np.ndarray,
    hedge_vol: float,
    realized_vol: float,
    costs: TransactionCosts | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Self-financing delta hedge of all paths at once; returns ``(pnl, theory)`` at expiry.

    Mirrors the book conventions exactly (interest on start-of-period cash,
    dividends ``shares * S_start * div * dt``, stock costs in bps, final unwind
    of the shares) so that it agrees with the generic book-based loop.
    """
    rate, div = float(mkt0.rate), float(mkt0.div)
    expiry = float(times[-1])
    stock_cost = 0.0 if costs is None else costs.stock_bps / 1e4
    premium = float(instrument.price(mkt0))
    entry_fee = 0.0 if costs is None else costs.cost(instrument, quantity, mkt0, premium)

    n_paths = paths.shape[0]
    cash = np.full(n_paths, -quantity * premium - entry_fee)
    shares = np.zeros(n_paths)
    hedge_premium = float(instrument.price(mkt0.bumped(vol=hedge_vol)))
    theory = np.full(n_paths, quantity * (hedge_premium - premium) * math.exp(rate * (expiry - times[0])))

    for k in range(times.size - 1):
        spot = paths[:, k]
        dt = float(times[k + 1] - times[k])
        greeks = instrument.greeks(Market(spot=spot, vol=hedge_vol, rate=rate, div=div, t=float(times[k])))
        target = -quantity * np.broadcast_to(np.asarray(greeks["delta"], dtype=float), spot.shape)
        traded = target - shares
        cash = cash - traded * spot - np.abs(traded) * spot * stock_cost
        shares = target
        gamma = np.broadcast_to(np.asarray(greeks["gamma"], dtype=float), spot.shape)
        theory = theory + (
            0.5 * quantity * gamma * spot**2 * (realized_vol**2 - hedge_vol**2) * dt
            * math.exp(rate * (expiry - float(times[k])))
        )
        cash = cash + cash * math.expm1(rate * dt) + shares * spot * div * dt

    terminal = paths[:, -1]
    payoff = np.asarray(instrument.payoff(terminal), dtype=float)
    cash = cash + quantity * payoff + shares * terminal - np.abs(shares) * terminal * stock_cost
    return cash, theory


def _generic_hedging_pnl(
    instrument: Instrument,
    quantity: float,
    mkt0: Market,
    times: np.ndarray,
    paths: np.ndarray,
    policy: HedgingPolicy,
    costs: TransactionCosts | None,
) -> np.ndarray:
    """Book-based loop over paths: works for any instrument (path-dependent included)."""
    pnl = np.empty(paths.shape[0])
    for j, path in enumerate(paths):
        scenario = MarketScenario(times, path, mkt0.vol, mkt0.rate, mkt0.div)
        book = Book("experiment", costs=costs)
        mkt = scenario.market_at(0)
        book.trade(instrument, quantity, mkt)
        policy.rebalance(book, mkt, 0)
        for i in range(1, len(scenario)):
            previous, mkt = mkt, scenario.market_at(i)
            _advance_book(book, previous, mkt)
            policy.rebalance(book, mkt, i)
        pnl[j] = book.pnl(mkt)
    return pnl


def delta_hedging_experiment(
    instrument: Union[Instrument, Position],
    mkt0: Market,
    n_paths: int = 2000,
    rebalance_steps_list: Sequence[int] = (4, 13, 52, 252),
    realized_vol: float | None = None,
    seed: Seed = None,
    quantity: float = 1.0,
    drift: float | None = None,
    hedge_vol: float | None = None,
    costs: TransactionCosts | None = None,
) -> pd.DataFrame:
    """Buy an option at the implied vol, delta hedge it to expiry, look at the final P&L.

    The same simulated paths are hedged with several rebalancing frequencies,
    so the columns differ ONLY by how often the hedge was adjusted.

    Parameters
    ----------
    instrument : Instrument or Position
        What is bought at the model price in ``mkt0`` (at ``mkt0.vol``, the
        IMPLIED vol). Must have an expiry after ``mkt0.t``.
    mkt0 : Market
        Scalar starting market.
    n_paths : int, default 2000
        Number of simulated paths (independent draws, no antithetic pairs, so
        that means and standard errors are plain i.i.d. statistics).
    rebalance_steps_list : sequence of int, default (4, 13, 52, 252)
        Numbers of equally spaced hedge adjustments until expiry (for a one-year
        option: quarterly, monthly... weekly, daily).
    realized_vol : float, optional
        Volatility of the simulated spot (default: the implied vol).
    seed : int, numpy Generator or None
    quantity : float, default 1.0
        Signed quantity: ``+1`` is long the option (long gamma, the hedge SELLS
        rallies and BUYS dips), ``-1`` is the option seller's experiment.
    drift : float, optional
        Growth rate of the spot (default ``rate - div``).
    hedge_vol : float, optional
        Volatility used to compute the hedge delta (default: the implied vol).
        See :class:`DeltaHedgeAtVol`.
    costs : TransactionCosts, optional
        Transaction costs paid at inception, on every hedge trade and on the
        final unwind of the shares.

    Returns
    -------
    pandas.DataFrame
        Long format, one row per (path, frequency): ``path``,
        ``rebalance_steps``, ``pnl`` (final P&L at expiry, in currency),
        ``pnl_pct_premium`` (P&L as a % of the premium traded),
        ``terminal_spot`` and ``theory_pnl``: the textbook prediction
        ``sum(0.5 * Gamma * S**2 * (realised**2 - hedge_vol**2) * dt)`` along
        the path, carried to expiry (``nan`` on the generic route).
        ``frame.attrs`` records ``label, quantity, premium, implied_vol,
        realized_vol, hedge_vol, horizon, vectorised``.

    Notes
    -----
    Two results to look for (see :func:`hedging_experiment_summary`):

    * with ``realized_vol == implied``, the mean P&L is ~0 for every frequency
      and its standard deviation falls like ``1 / sqrt(N)``: discrete hedging
      replicates the option only on average, and four times more trades halve
      the error;
    * with ``realized_vol != implied``, the mean is about ``0.5 * integral of
      Gamma * S**2 * (realised**2 - implied**2) dt``: the option buyer is paid
      for the vol that realises above what was paid for.

    Path-independent instruments whose Greeks accept array markets (every
    vanilla and vanilla package) are hedged on all paths at once with analytic
    deltas. Anything else (barriers, Asians, calendars...) falls back to a
    Python loop running a real :class:`~optionlab.book.Book` per path -- keep
    ``n_paths * max(rebalance_steps_list)`` modest there.
    """
    mkt0 = _scalar_market(mkt0, "delta_hedging_experiment")
    quantity = _number("quantity", quantity)
    if isinstance(instrument, Position):
        quantity *= instrument.quantity
        instrument = instrument.instrument
    if not isinstance(instrument, Instrument):
        raise ValueError("delta_hedging_experiment needs an Instrument or a Position")
    if quantity == 0:
        raise ValueError("quantity must be non-zero")
    if instrument.expiry is None or instrument.is_expired(mkt0):
        raise ValueError("the instrument must have an expiry later than mkt0.t")
    n_paths = _whole("n_paths", n_paths, 2)
    steps_list = sorted({_whole("rebalance_steps_list entries", n, 1) for n in rebalance_steps_list})
    if not steps_list:
        raise ValueError("rebalance_steps_list must not be empty")
    implied = float(mkt0.vol)
    sigma = implied if realized_vol is None else _number("realized_vol", realized_vol, minimum=0.0)
    hedge_sigma = implied if hedge_vol is None else _number("hedge_vol", hedge_vol, minimum=0.0, strict=True)
    mu = (mkt0.rate - mkt0.div) if drift is None else _number("drift", drift)
    if costs is not None and not isinstance(costs, TransactionCosts):
        raise ValueError("costs must be a TransactionCosts instance or None")

    # One set of paths on the union of all rebalancing grids (exact fractions of the horizon).
    expiry = float(instrument.expiry)
    horizon = expiry - float(mkt0.t)
    fractions = sorted({Fraction(k, n) for n in steps_list for k in range(n + 1)})
    column = {fraction: i for i, fraction in enumerate(fractions)}
    grid = float(mkt0.t) + horizon * np.array([float(f) for f in fractions])
    grid[-1] = expiry
    paths = simulate_gbm_paths_on_grid(
        mkt0.spot, sigma, mu + mkt0.div, mkt0.div, grid, n_paths, seed, antithetic=False
    )

    vectorised = _is_path_independent(instrument) and _single_expiry(instrument)
    policy = DeltaHedgeEveryN(1) if hedge_sigma == implied else DeltaHedgeAtVol(hedge_sigma)
    premium = float(instrument.price(mkt0))
    frames = []
    for n in steps_list:
        columns = [column[Fraction(k, n)] for k in range(n + 1)]
        sub_times, sub_paths = grid[columns], paths[:, columns]
        theory = np.full(n_paths, np.nan)
        pnl = None
        if vectorised:
            try:
                pnl, theory = _vectorised_hedging_pnl(
                    instrument, quantity, mkt0, sub_times, sub_paths, hedge_sigma, sigma, costs
                )
            except (TypeError, ValueError):
                vectorised = False  # Greeks not vectorised after all: use the loop
        if pnl is None:
            pnl = _generic_hedging_pnl(instrument, quantity, mkt0, sub_times, sub_paths, policy, costs)
        frames.append(
            pd.DataFrame(
                {
                    "path": np.arange(n_paths),
                    "rebalance_steps": n,
                    "pnl": pnl,
                    "pnl_pct_premium": 100.0 * pnl / abs(quantity * premium) if premium else np.nan,
                    "terminal_spot": sub_paths[:, -1],
                    "theory_pnl": theory,
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    result.attrs.update(
        {
            "label": instrument.label,
            "quantity": quantity,
            "premium": premium,
            "implied_vol": implied,
            "realized_vol": sigma,
            "hedge_vol": hedge_sigma,
            "horizon": horizon,
            "vectorised": vectorised,
        }
    )
    return result


def hedging_experiment_summary(experiment: pd.DataFrame) -> pd.DataFrame:
    """Per-frequency statistics of :func:`delta_hedging_experiment`.

    Returns
    -------
    pandas.DataFrame
        Index ``rebalance_steps``; columns ``mean``, ``stderr`` (standard error
        of the mean), ``std``, ``q05``, ``q95``, ``std_pct_premium``,
        ``std_x_sqrt_n`` (``std * sqrt(N)``: roughly constant, which IS the
        ``1 / sqrt(N)`` law) and ``theory_mean`` (mean of the textbook gamma
        formula, ``nan`` on the generic route).
    """
    required = ["rebalance_steps", "pnl", "pnl_pct_premium", "theory_pnl"]
    if not isinstance(experiment, pd.DataFrame) or any(c not in experiment.columns for c in required):
        raise ValueError(f"expected the DataFrame of delta_hedging_experiment (columns {required})")
    rows = {}
    for n, group in experiment.groupby("rebalance_steps"):
        pnl = group["pnl"].to_numpy(dtype=float)
        std = float(pnl.std(ddof=1)) if pnl.size > 1 else math.nan
        theory = group["theory_pnl"].to_numpy(dtype=float)
        rows[int(n)] = {
            "mean": float(pnl.mean()),
            "stderr": std / math.sqrt(pnl.size),
            "std": std,
            "q05": float(np.quantile(pnl, 0.05)),
            "q95": float(np.quantile(pnl, 0.95)),
            "std_pct_premium": float(group["pnl_pct_premium"].std(ddof=1)) if pnl.size > 1 else math.nan,
            "std_x_sqrt_n": std * math.sqrt(int(n)),
            "theory_mean": math.nan if np.all(np.isnan(theory)) else float(np.nanmean(theory)),
        }
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index.name = "rebalance_steps"
    return summary
