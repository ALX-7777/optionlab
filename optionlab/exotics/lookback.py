"""Lookback options with continuous monitoring.

A lookback lets its owner trade with perfect hindsight:

* **floating strike** -- the call buys at the *lowest* price seen over the
  life, ``S_T - S_min``; the put sells at the *highest*, ``S_max - S_T``.
  They are never out of the money.
* **fixed strike** -- the call is paid on the *highest* price,
  ``max(S_max - K, 0)``; the put on the lowest, ``max(K - S_min, 0)``.

Closed forms exist in the Black-Scholes world (Goldman, Sosin and Gatto, 1979
for floating strikes; Conze and Viswanathan, 1991 for fixed strikes), with a
cost of carry ``b = rate - div``.

One building block
------------------
All four products reduce to two *options on the extreme*::

    C_max(S, H) = e^{-r tau} E[(max S_u - H)^+]      for H >= S
    P_min(S, H) = e^{-r tau} E[(H - min S_u)^+]      for H <= S

    floating call = S e^{-q tau} - m e^{-r tau} + P_min(S, m)
    floating put  = M e^{-r tau} - S e^{-q tau} + C_max(S, M)
    fixed call    = e^{-r tau} max(M - K, 0) + C_max(S, max(M, K))
    fixed put     = e^{-r tau} max(K - m, 0) + P_min(S, min(m, K))

with ``m`` / ``M`` the running minimum / maximum. Each block is a vanilla
option struck at ``H`` plus a *strike-reset premium* paying for every future
improvement of the extreme. That premium carries a factor
``vol**2 / (2 b)`` which is 0/0 at zero carry; the ``b -> 0`` limit is
implemented explicitly, so futures-style markets (``rate == div``) are fine.

Trading intuition
-----------------
Hindsight is expensive: with no carry a fresh floating lookback costs about
**twice** the at-the-money vanilla (``E[max] - S = S vol sqrt(2 T / pi)``),
roughly an ATM straddle. When the spot sits on its running extreme the owner
is long a lot of gamma; once the spot has moved far away, the extreme is
unlikely to be improved and the option behaves like a forward (floating) or a
vanilla (fixed). Real contracts monitor discretely (daily closes), which
*misses* part of the extreme and makes them cheaper than these continuous
formulas: see :func:`mc_price_brownian_bridge` versus plain ``mc_price``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.special import log_ndtr, ndtr

from .. import black_scholes as bs
from ..instruments import (
    EXPIRY_TOL,
    EuropeanOption,
    Instrument,
    register_instrument,
    slice_paths_to_expiry,
)
from ..market import Market
from ..monte_carlo import Seed, simulate_gbm_paths_on_grid
from ..numerical import numerical_greeks

__all__ = ["LOOKBACK_KINDS", "LookbackOption", "mc_price_brownian_bridge"]

#: Accepted values of ``LookbackOption.kind``.
LOOKBACK_KINDS: tuple[str, ...] = ("floating", "fixed")

_SQRT_2PI = math.sqrt(2.0 * math.pi)

# Below this value of |2 b / vol**2| the zero-carry limit of the strike-reset
# premium is used: there the general formula is a 0/0 that loses all its digits.
_LAMBDA_EPS = 1e-7


def _extreme_option(spot, level, tau, vol, rate, div, eta: float) -> np.ndarray:
    """Option on the future extreme of the path, the lookback building block.

    ``eta = +1``: ``e^{-r tau} E[(max_{u<=tau} S_u - level)^+]``, needs ``level >= spot``.
    ``eta = -1``: ``e^{-r tau} E[(level - min_{u<=tau} S_u)^+]``, needs ``level <= spot``.

    With ``d1`` the Black-Scholes ``d1`` for strike ``level`` and
    ``lam = 2 b / vol**2``::

        vanilla = eta [S e^{(b-r) tau} N(eta d1) - H e^{-r tau} N(eta d2)]
        premium = S e^{-r tau} (eta / lam) [e^{b tau} N(eta d1) - (S/H)^{-lam} N(eta d3)]
        d3      = d1 - lam vol sqrt(tau)
        premium -> S e^{-r tau} vol sqrt(tau) [n(d1) + eta d1 N(eta d1)]     as b -> 0

    Without volatility the path is deterministic and the extreme is one of
    its end points; with no time left the value is 0.
    """
    spot, level, tau, vol, rate, div = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (spot, level, tau, vol, rate, div))
    )
    carry = rate - div
    tau_pos = np.maximum(tau, 0.0)
    regular = (tau > 0) & (vol > 0) & (spot > 0)

    end_point = spot * np.exp(eta * np.maximum(eta * carry * tau_pos, 0.0))
    deterministic = np.exp(-rate * tau_pos) * np.maximum(eta * (end_point - level), 0.0)

    s, h, tt, sig = (np.where(regular, v, 1.0) for v in (spot, level, tau, vol))
    sd = sig * np.sqrt(tt)
    log_ratio = np.log(s / h)
    d1 = (log_ratio + (carry + 0.5 * sig**2) * tt) / sd
    lam = 2.0 * carry / sig**2
    d3 = d1 - lam * sd
    discount = np.exp(-rate * tt)

    vanilla = eta * (s * np.exp(-div * tt) * ndtr(eta * d1) - h * discount * ndtr(eta * (d1 - sd)))
    small = np.abs(lam) < _LAMBDA_EPS
    lam_safe = np.where(small, 1.0, lam)
    # (S/H)^{-lam} N(eta d3) is evaluated in logs: the power can overflow while the product stays tame.
    general = (eta / lam_safe) * (
        np.exp(carry * tt) * ndtr(eta * d1) - np.exp(-lam_safe * log_ratio + log_ndtr(eta * d3))
    )
    zero_carry = sd * (np.exp(-0.5 * d1 * d1) / _SQRT_2PI + eta * d1 * ndtr(eta * d1))
    premium = s * discount * np.where(small, zero_carry, general)
    return np.where(regular, vanilla + premium, deterministic)


def _optional_positive(name: str, value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"LookbackOption.{name} must be a number or None") from exc
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"LookbackOption.{name} must be finite and strictly positive, got {value!r}")
    return number


@register_instrument
@dataclass(frozen=True)
class LookbackOption(Instrument):
    """Continuously monitored lookback option (floating or fixed strike).

    Parameters
    ----------
    option_type : {"call", "put"}
    expiry : float
        ABSOLUTE expiry in years.
    strike : float, optional
        Required for ``kind="fixed"``; must be ``None`` for ``kind="floating"``
        (the strike *is* the running extreme).
    kind : {"floating", "fixed"}, default "floating"
    running_min, running_max : float, optional
        Extremes observed so far. ``None`` means "start looking now": the
        extreme is initialised at the current spot when pricing. Floating
        calls and fixed puts use the minimum, floating puts and fixed calls
        the maximum; :meth:`observe` keeps both up to date.

    Notes
    -----
    * ``price`` always uses ``min(running_min, spot)`` / ``max(running_max,
      spot)``, so a spot ladder that dips below the stored minimum stays
      consistent (the ladder point itself becomes the new extreme).
    * ``greeks`` freezes the extremes at the base market before bumping, so
      the numbers are true partial derivatives. When the spot sits exactly on
      its extreme, delta has a kink (on one side the extreme is dragged along
      and the price is linear in the spot): the central difference then
      reports the average of the two one-sided gammas, and the third-order
      Greeks taken across that kink (speed, color, zomma) are bump-size
      dependent there. Treat them as "undefined at the extreme", exactly like
      the gamma of a vanilla at the strike on expiry day.

    Examples
    --------
    >>> from optionlab.market import Market
    >>> call = LookbackOption("call", 0.5, running_min=100.0)
    >>> round(call.price(Market(spot=120.0, vol=0.3, rate=0.10, div=0.06)), 3)
    25.353
    """

    option_type: str
    expiry: float
    strike: float | None = None
    kind: str = "floating"
    running_min: float | None = None
    running_max: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_type", bs.normalize_option_type(self.option_type))
        kind = self.kind.strip().lower() if isinstance(self.kind, str) else self.kind
        if kind not in LOOKBACK_KINDS:
            raise ValueError(f"kind must be one of {LOOKBACK_KINDS}, got {self.kind!r}")
        object.__setattr__(self, "kind", kind)
        try:
            expiry = float(self.expiry)
        except (TypeError, ValueError) as exc:
            raise ValueError("LookbackOption.expiry must be a number") from exc
        if not math.isfinite(expiry):
            raise ValueError(f"expiry must be finite, got {self.expiry!r}")
        strike = _optional_positive("strike", self.strike)
        if kind == "fixed" and strike is None:
            raise ValueError("a fixed-strike lookback needs a strike")
        if kind == "floating" and strike is not None:
            raise ValueError("a floating-strike lookback has no strike (use kind='fixed')")
        running_min = _optional_positive("running_min", self.running_min)
        running_max = _optional_positive("running_max", self.running_max)
        if running_min is not None and running_max is not None and running_min > running_max:
            raise ValueError("running_min cannot exceed running_max")
        for name, value in (
            ("expiry", expiry), ("strike", strike),
            ("running_min", running_min), ("running_max", running_max),
        ):
            object.__setattr__(self, name, value)

    # --- small helpers ------------------------------------------------------
    @property
    def is_call(self) -> bool:
        return self.option_type == "call"

    @property
    def is_floating(self) -> bool:
        return self.kind == "floating"

    @property
    def label(self) -> str:
        side = "C" if self.is_call else "P"
        strike = "" if self.is_floating else f" {self.strike:g}"
        return f"LB {'float' if self.is_floating else 'fixed'} {side}{strike} T={self.expiry:.2f}"

    def vanilla_equivalent(self, spot: float | None = None) -> EuropeanOption:
        """The vanilla option a lookback is naturally compared with.

        Fixed strike: same type, strike and expiry. Floating strike: the
        vanilla struck at the extreme that currently plays the role of the
        strike -- the running minimum for a call, the running maximum for a
        put -- or at ``spot`` when no extreme is stored yet (a fresh floating
        lookback is compared with the at-the-money vanilla, and costs about
        twice as much).

        Parameters
        ----------
        spot : float, optional
            Current spot; only needed for a floating lookback without stored
            extreme.
        """
        if not self.is_floating:
            return EuropeanOption(self.option_type, self.strike, self.expiry)
        level = self.running_min if self.is_call else self.running_max
        if spot is not None:
            low, high = self._extremes(spot)
            level = float(low if self.is_call else high)
        if level is None:
            raise ValueError(
                "a floating lookback without stored extreme needs the current spot "
                "to define its vanilla equivalent"
            )
        return EuropeanOption(self.option_type, level, self.expiry)

    def _extremes(self, spot) -> tuple[np.ndarray, np.ndarray]:
        """Running (min, max) once ``spot`` has been seen as well."""
        spot = np.asarray(spot, dtype=float)
        low = spot if self.running_min is None else np.minimum(self.running_min, spot)
        high = spot if self.running_max is None else np.maximum(self.running_max, spot)
        return low, high

    def _settle(self, spot_T, low, high) -> np.ndarray:
        """Payoff given the terminal spot and the extremes over the whole life."""
        if self.is_floating:
            return spot_T - low if self.is_call else high - spot_T
        if self.is_call:
            return np.maximum(high - self.strike, 0.0)
        return np.maximum(self.strike - low, 0.0)

    def _value(self, mkt: Market, low, high) -> np.ndarray:
        """Price given extremes observed *before* looking at ``mkt.spot``."""
        spot = np.asarray(mkt.spot, dtype=float)
        tau = self.expiry - np.asarray(mkt.t, dtype=float)
        tau_pos = np.maximum(tau, 0.0)
        df_rate = np.exp(-mkt.rate * tau_pos)
        df_div = np.exp(-mkt.div * tau_pos)
        low, high = np.minimum(low, spot), np.maximum(high, spot)
        args = (tau, mkt.vol, mkt.rate, mkt.div)
        if self.is_floating and self.is_call:
            value = spot * df_div - low * df_rate + _extreme_option(spot, low, *args, eta=-1.0)
        elif self.is_floating:
            value = high * df_rate - spot * df_div + _extreme_option(spot, high, *args, eta=1.0)
        elif self.is_call:
            locked = df_rate * np.maximum(high - self.strike, 0.0)
            value = locked + _extreme_option(spot, np.maximum(high, self.strike), *args, eta=1.0)
        else:
            locked = df_rate * np.maximum(self.strike - low, 0.0)
            value = locked + _extreme_option(spot, np.minimum(low, self.strike), *args, eta=-1.0)
        return np.maximum(value, 0.0)  # a lookback never has a negative payoff; this only trims rounding

    # --- valuation ------------------------------------------------------------
    def price(self, mkt: Market):
        """Closed-form value, vectorised over the market fields.

        At or after expiry this is the settlement value given the current spot
        and the stored extremes.
        """
        out = self._value(mkt, *self._extremes(mkt.spot)) + 0.0
        return float(out) if mkt.is_scalar else out

    def greeks(self, mkt: Market) -> dict[str, Any]:
        """Bump-and-reprice Greeks (RAW units) with the extremes frozen at the base market.

        Freezing matters when no extreme is stored yet: re-initialising it at
        every bumped spot would make the price linear in the spot and hide all
        the gamma of a fresh lookback.
        """
        low, high = self._extremes(mkt.spot)
        return numerical_greeks(lambda m: self._value(m, low, high), mkt, expiry=self.expiry)

    def payoff(self, spot_T):
        """Payoff versus the terminal spot if the stored extremes only change through ``spot_T``.

        With no stored extreme a floating lookback shows 0 everywhere and a
        fixed one the vanilla hockey stick: all the extra value comes from
        the *path*, which a terminal-spot diagram cannot show.
        """
        spot_T = np.asarray(spot_T, dtype=float)
        return self._settle(spot_T, *self._extremes(spot_T)) + 0.0

    # --- path dependence --------------------------------------------------------
    def observe(self, spot: float, t: float) -> "LookbackOption":
        """Update the running extremes with the spot seen at ``t`` (ignored after expiry)."""
        spot = _optional_positive("observe(spot)", spot)
        if spot is None:
            raise ValueError("observe needs a spot")
        if float(t) > self.expiry + EXPIRY_TOL:
            return self
        low, high = (float(v) for v in self._extremes(spot))
        if low == self.running_min and high == self.running_max:
            return self
        return replace(self, running_min=low, running_max=high)

    def path_payoff(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Payoff per path, the path extremes being combined with the stored ones.

        The extreme is read on the grid only, i.e. *discrete* monitoring: it
        misses the excursions between two dates, so a plain Monte Carlo price
        sits below the continuous closed form by roughly
        ``0.5826 * vol * sqrt(dt)`` of the spot (Broadie-Glasserman-Kou).
        """
        paths, _ = slice_paths_to_expiry(paths, times, self.expiry)
        low, _ = self._extremes(paths.min(axis=1))
        _, high = self._extremes(paths.max(axis=1))
        return self._settle(paths[:, -1], low, high)


def mc_price_brownian_bridge(
    lookback: LookbackOption,
    mkt: Market,
    n_paths: int = 100_000,
    n_steps: int = 8,
    seed: Seed = None,
) -> tuple[float, float]:
    """Monte Carlo price under CONTINUOUS monitoring, without discretisation bias.

    Between two simulated dates the log-spot is a Brownian bridge, whose
    maximum has a known law: given the end points ``x0``, ``x1`` and a uniform
    draw ``U``::

        max = (x0 + x1 + sqrt((x1 - x0)**2 - 2 vol**2 dt ln U)) / 2

    (mirror image for the minimum). Sampling the extreme of every step this
    way reproduces the continuously monitored extreme exactly, whatever the
    number of steps -- which is what makes it a fair referee for the closed
    forms, unlike a plain simulation that only looks at the grid dates.

    Parameters
    ----------
    lookback : LookbackOption
    mkt : Market
        Scalar market with ``spot > 0``.
    n_paths : int, default 100000
        Rounded up to an even number (antithetic pairs).
    n_steps : int, default 8
        Any value gives an unbiased price.
    seed : int, numpy Generator or None

    Returns
    -------
    price, stderr : float
    """
    if not isinstance(lookback, LookbackOption):
        raise ValueError("mc_price_brownian_bridge needs a LookbackOption")
    if not mkt.is_scalar:
        raise ValueError("mc_price_brownian_bridge needs a scalar Market (no array fields)")
    n_paths, n_steps = int(n_paths), int(n_steps)
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if lookback.is_expired(mkt):
        return float(lookback.price(mkt)), 0.0

    rng = np.random.default_rng(seed)
    n_paths += n_paths % 2
    times = np.linspace(mkt.t, lookback.expiry, n_steps + 1)
    log_paths = np.log(
        simulate_gbm_paths_on_grid(mkt.spot, mkt.vol, mkt.rate, mkt.div, times, n_paths, rng, antithetic=True)
    )
    # One uniform per step serves both extremes: each payoff only reads one of them.
    uniforms = 1.0 - rng.random((n_paths, n_steps))
    spread = np.sqrt(
        np.diff(log_paths, axis=1) ** 2 - 2.0 * mkt.vol**2 * np.diff(times) * np.log(uniforms)
    )
    mid = log_paths[:, :-1] + log_paths[:, 1:]
    path_low = np.exp(0.5 * (mid - spread).min(axis=1))
    path_high = np.exp(0.5 * (mid + spread).max(axis=1))

    low, _ = lookback._extremes(path_low)
    _, high = lookback._extremes(path_high)
    discount = math.exp(-mkt.rate * (lookback.expiry - mkt.t))
    pv = discount * lookback._settle(np.exp(log_paths[:, -1]), low, high)
    half = n_paths // 2
    samples = 0.5 * (pv[:half] + pv[half:])
    return float(samples.mean()), float(samples.std(ddof=1) / math.sqrt(half))
