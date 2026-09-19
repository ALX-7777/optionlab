"""Single-barrier options (knock-in / knock-out), continuously monitored.

A barrier option is a vanilla option that is switched **on** (knock-in) or
**off** (knock-out) the first time the spot touches a barrier level ``H``.
Because some paths are excluded, a barrier option is always cheaper than its
vanilla twin, which is the whole commercial point: "I want the call, but I am
happy to give it up if the stock first falls to 85, so I pay less".

In-out parity
    Holding a knock-in and the matching knock-out (same strike, barrier,
    expiry, no rebate) is holding the vanilla: exactly one of the two is alive
    at expiry. So ``knock-in + knock-out = vanilla``.

Regular versus reverse barriers
    A *regular* barrier sits in the out-of-the-money region (down-and-out
    call, up-and-out put): when it is hit the option is worth little anyway,
    so the price is smooth and the Greeks are tame. A *reverse* barrier sits
    in the money (up-and-out call, down-and-out put): just before the hit the
    option carries a large intrinsic value that vanishes at the touch. Close
    to the barrier the delta of an up-and-out call is strongly **negative**
    (a rally now destroys value), gamma is huge and negative, and the holder
    is short volatility. Near expiry these numbers explode; hedging them is
    what barrier traders are paid for.

Pricing
    Closed forms of Reiner & Rubinstein (1991) as laid out in Haug, *The
    Complete Guide to Option Pricing Formulas*, with cost of carry
    ``b = rate - div``. The six building blocks ``A`` to ``F`` are combined
    differently for each of the 16 cases (call/put x in/out x up/down x strike
    above/below the barrier).

Rebate convention
    * knock-**out**: the rebate is paid **at the hit** (term ``F``);
    * knock-**in**: the rebate is paid **at expiry** if the option was never
      knocked in (term ``E``).

Monitoring
    The formulas assume the barrier is watched continuously. A barrier watched
    only at discrete dates (daily closes, or the time steps of a simulation) is
    hit less often; Broadie, Glasserman & Kou (1997) showed that this is
    equivalent, to a very good approximation, to moving the barrier *away*
    from the spot by ``exp(0.5826 * vol * sqrt(dt))``, see
    :func:`bgk_adjusted_barrier`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.special import log_ndtr, ndtr

from .. import black_scholes as bs
from ..instruments import EXPIRY_TOL, GREEK_KEYS, EuropeanOption, Instrument, register_instrument
from ..market import Market
from ..numerical import numerical_greeks

__all__ = ["BARRIER_TYPES", "BGK_BETA", "BarrierOption", "bgk_adjusted_barrier"]

#: Supported barrier types.
BARRIER_TYPES: tuple[str, ...] = ("up-and-out", "up-and-in", "down-and-out", "down-and-in")

#: Broadie-Glasserman-Kou constant ``-zeta(1/2) / sqrt(2 pi)``.
BGK_BETA: float = 0.5825971579390106

_SHORT_CODES = {"uo": "up-and-out", "ui": "up-and-in", "do": "down-and-out", "di": "down-and-in"}

#: Knock-IN combination of the Reiner-Rubinstein blocks, as coefficients of
#: ``(A, B, C, D)``, keyed by ``(option_type, direction, strike >= barrier)``.
#: The knock-out combination is the vanilla ``A`` minus the knock-in one.
_KNOCK_IN_COEFFICIENTS: dict[tuple[str, str, bool], tuple[int, int, int, int]] = {
    ("call", "down", True): (0, 0, 1, 0),    # C
    ("call", "down", False): (1, -1, 0, 1),  # A - B + D
    ("call", "up", True): (1, 0, 0, 0),      # A
    ("call", "up", False): (0, 1, -1, 1),    # B - C + D
    ("put", "down", True): (0, 1, -1, 1),    # B - C + D
    ("put", "down", False): (1, 0, 0, 0),    # A
    ("put", "up", True): (1, -1, 0, 1),      # A - B + D
    ("put", "up", False): (0, 0, 1, 0),      # C
}

#: Half-width (in log-spot) of the band beyond the barrier in which the closed
#: form is continued analytically for one-sided finite differences.
_CONTINUATION_BAND = 0.05


def _normalize_barrier_type(barrier_type: str) -> str:
    if isinstance(barrier_type, str):
        key = barrier_type.strip().lower().replace("_", "-").replace(" ", "-")
        key = _SHORT_CODES.get(key, key)
        if key in BARRIER_TYPES:
            return key
    raise ValueError(f"barrier_type must be one of {BARRIER_TYPES}, got {barrier_type!r}")


def bgk_adjusted_barrier(barrier, vol, dt, direction: str, inverse: bool = False):
    """Broadie-Glasserman-Kou continuity correction of a barrier level.

    A barrier monitored every ``dt`` years is worth (almost exactly) what a
    *continuously* monitored barrier placed a little further away would be
    worth::

        H_adjusted = H * exp(+0.5826 * vol * sqrt(dt))    (up barrier)
        H_adjusted = H * exp(-0.5826 * vol * sqrt(dt))    (down barrier)

    Intuition: between two observation dates the spot can sneak across the
    barrier and come back unseen; on average it overshoots by about
    ``0.5826 * vol * sqrt(dt)`` (in log terms) before being caught.

    Parameters
    ----------
    barrier : float or ndarray
        Contractual barrier level (> 0).
    vol : float or ndarray
        Volatility used for the shift.
    dt : float or ndarray
        Time between two monitoring dates, in years (daily: ``1 / 252``).
    direction : {"up", "down"}
        Whether the barrier is above or below the spot.
    inverse : bool, default False
        If True the shift is applied the other way (towards the spot). Use it
        to estimate a *continuous* barrier price with a simulation that only
        observes every ``dt``: simulate the option with the inverse-adjusted
        barrier.

    Returns
    -------
    float or ndarray
        The shifted barrier, to be used in the continuous-monitoring formula.
    """
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    barrier = np.asarray(barrier, dtype=float)
    vol = np.asarray(vol, dtype=float)
    dt = np.asarray(dt, dtype=float)
    if np.any(barrier <= 0):
        raise ValueError("barrier must be strictly positive")
    if np.any(vol < 0) or np.any(dt < 0):
        raise ValueError("vol and dt must be non-negative")
    sign = 1.0 if direction == "up" else -1.0
    if inverse:
        sign = -sign
    out = barrier * np.exp(sign * BGK_BETA * vol * np.sqrt(dt))
    return float(out) if out.ndim == 0 else out


@register_instrument
@dataclass(frozen=True)
class BarrierOption(Instrument):
    """European single-barrier option with continuous monitoring.

    Parameters
    ----------
    option_type : {"call", "put"}
    strike : float
        Strike of the underlying vanilla payoff (> 0).
    barrier : float
        Barrier level ``H`` (> 0).
    expiry : float
        ABSOLUTE expiry in years.
    barrier_type : {"up-and-out", "up-and-in", "down-and-out", "down-and-in"}
        Case-insensitive; ``_`` or spaces instead of ``-`` and the short codes
        ``"uo"``, ``"ui"``, ``"do"``, ``"di"`` are accepted.
    rebate : float, default 0.0
        Consolation cash amount (>= 0). Knock-out: paid **at the hit**.
        Knock-in: paid **at expiry** if the barrier was never touched.
    knocked : bool, default False
        Path state: True once the barrier has been touched. Updated by
        :meth:`observe`.

    Notes
    -----
    **Life cycle.** While ``knocked`` is False and the spot is on the live side
    of the barrier, the price is the Reiner-Rubinstein closed form. Once
    knocked, a knock-in *is* the vanilla option (see
    :meth:`vanilla_equivalent`) and a knock-out is worth **zero**: its rebate
    is paid at the hit. :meth:`observation_cash_flow` reports that payment and
    a :class:`~optionlab.book.Book` credits it when :meth:`observe` flips the
    flag, so the P&L is continuous through the knock-out.

    **Argument order.** ``barrier`` comes BEFORE ``expiry`` (unlike
    ``EuropeanOption(option_type, strike, expiry)``); both are plain numbers,
    so a swapped call cannot be detected. Prefer keywords:
    ``BarrierOption("call", strike=100, barrier=120, expiry=1.0,
    barrier_type="up-and-out")``.

    If ``price`` is asked for a spot already beyond the barrier while
    ``knocked`` is still False, that spot is treated as the hit happening
    *now*: a knock-out is worth its rebate, a knock-in is worth the vanilla.
    This makes price-versus-spot plots correct on both sides of the barrier.

    **Greeks** are numerical (bump and reprice) but computed on the analytic
    continuation of the closed form, so that a bump never straddles the
    barrier: on the live side they are the true one-sided sensitivities right
    up to the barrier, beyond it they are the Greeks of what the option has
    become (nothing, or a vanilla). Expect a jump at the barrier -- for a
    reverse knock-out such as an up-and-out call, the delta goes from strongly
    negative to zero and gamma is enormous just before the touch. Calling
    :func:`optionlab.numerical.numerical_greeks` directly on the instrument
    instead differentiates *across* the jump when the spot is within one bump
    of the barrier, and returns the (large, bump-dependent) numbers one would
    expect from differentiating a discontinuity.

    Examples
    --------
    >>> doc = BarrierOption("call", 90.0, 95.0, 0.5, "down-and-out", rebate=3.0)
    >>> round(doc.price(Market(spot=100.0, vol=0.25, rate=0.08, div=0.04)), 4)
    9.0246
    """

    option_type: str
    strike: float
    barrier: float
    expiry: float
    barrier_type: str
    rebate: float = 0.0
    knocked: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_type", bs.normalize_option_type(self.option_type))
        object.__setattr__(self, "barrier_type", _normalize_barrier_type(self.barrier_type))
        try:
            strike, barrier = float(self.strike), float(self.barrier)
            expiry, rebate = float(self.expiry), float(self.rebate)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "BarrierOption strike, barrier, expiry and rebate must be numbers"
            ) from exc
        if not (math.isfinite(strike) and strike > 0):
            raise ValueError(f"strike must be a finite positive number, got {self.strike!r}")
        if not (math.isfinite(barrier) and barrier > 0):
            raise ValueError(f"barrier must be a finite positive number, got {self.barrier!r}")
        if not math.isfinite(expiry):
            raise ValueError(f"expiry must be finite, got {self.expiry!r}")
        if not (math.isfinite(rebate) and rebate >= 0):
            raise ValueError(f"rebate must be finite and non-negative, got {self.rebate!r}")
        if not isinstance(self.knocked, (bool, np.bool_)):
            raise ValueError(f"knocked must be a bool, got {self.knocked!r}")
        object.__setattr__(self, "strike", strike)
        object.__setattr__(self, "barrier", barrier)
        object.__setattr__(self, "expiry", expiry)
        object.__setattr__(self, "rebate", rebate)
        object.__setattr__(self, "knocked", bool(self.knocked))

    # --- descriptors ------------------------------------------------------
    @property
    def is_call(self) -> bool:
        return self.option_type == "call"

    @property
    def is_up(self) -> bool:
        """True for an up barrier (hit from below), False for a down barrier."""
        return self.barrier_type.startswith("up")

    @property
    def is_knock_in(self) -> bool:
        return self.barrier_type.endswith("in")

    @property
    def direction(self) -> str:
        """``"up"`` or ``"down"``."""
        return "up" if self.is_up else "down"

    @property
    def is_reverse(self) -> bool:
        """True when the barrier sits on the in-the-money side (up call, down put)."""
        return self.is_up == self.is_call

    @property
    def label(self) -> str:
        code = ("U" if self.is_up else "D") + ("I" if self.is_knock_in else "O")
        kind = "C" if self.is_call else "P"
        text = f"{code} {kind} {self.strike:g} H={self.barrier:g} T={self.expiry:.2f}"
        return text + " [knocked]" if self.knocked else text

    def vanilla_equivalent(self) -> EuropeanOption:
        """The vanilla option with the same type, strike and expiry.

        It is what a knock-in becomes once knocked, and the upper bound of the
        barrier option's price (for a zero rebate).
        """
        return EuropeanOption(self.option_type, self.strike, self.expiry)

    def discrete_barrier_adjusted(
        self,
        vol: float,
        dt: float | None = None,
        n_monitoring_per_year: float | None = None,
    ) -> "BarrierOption":
        """Continuous-monitoring proxy for this option monitored discretely.

        Returns a copy whose barrier is pushed away from the spot by the
        Broadie-Glasserman-Kou factor (:func:`bgk_adjusted_barrier`). Its
        closed-form price approximates the price of *this* contract when the
        barrier is only observed every ``dt`` -- which is also what a Monte
        Carlo with time step ``dt`` computes.

        Parameters
        ----------
        vol : float
            Volatility used for the shift (the pricing vol).
        dt : float, optional
            Time between monitoring dates in years.
        n_monitoring_per_year : float, optional
            Alternative to ``dt`` (``dt = 1 / n_monitoring_per_year``). Exactly
            one of the two must be given.
        """
        if (dt is None) == (n_monitoring_per_year is None):
            raise ValueError("give exactly one of dt and n_monitoring_per_year")
        if dt is None:
            if not n_monitoring_per_year > 0:
                raise ValueError("n_monitoring_per_year must be strictly positive")
            dt = 1.0 / float(n_monitoring_per_year)
        if not (np.ndim(vol) == 0 and np.ndim(dt) == 0):
            raise ValueError("vol and dt must be scalars (the barrier is a single number)")
        return replace(self, barrier=bgk_adjusted_barrier(self.barrier, vol, dt, self.direction))

    # --- path state ---------------------------------------------------------
    def _breached(self, spot) -> np.ndarray:
        """True where ``spot`` is at or beyond the barrier (``>=`` up, ``<=`` down)."""
        spot = np.asarray(spot, dtype=float)
        return spot >= self.barrier if self.is_up else spot <= self.barrier

    def observe(self, spot: float, t: float) -> "BarrierOption":
        """Flip ``knocked`` if ``spot`` touches the barrier at time ``t``.

        The barrier is live up to and including the expiry date; observations
        after expiry are ignored. For a knock-out with a rebate, the rebate
        is due at this moment: the returned instrument is worth zero and the
        cash flow is reported by :meth:`observation_cash_flow`, which
        :meth:`optionlab.book.Book.observe` books automatically.
        """
        if self.knocked or t > self.expiry + EXPIRY_TOL or not bool(self._breached(spot)):
            return self
        return replace(self, knocked=True)

    def observation_cash_flow(self, observed: Instrument) -> float:
        """The rebate, when ``observed`` is this knock-out right after its barrier was hit.

        A knock-in pays its rebate at expiry (it is part of the settlement
        value), so only knock-outs generate a cash flow at the hit.
        """
        just_knocked = (
            isinstance(observed, BarrierOption) and observed.knocked and not self.knocked
        )
        return self.rebate if just_knocked and not self.is_knock_in else 0.0

    # --- closed form --------------------------------------------------------
    def _reiner_rubinstein(self, S, tau, vol, r, q, check: np.ndarray) -> np.ndarray:
        """Haug's formula on broadcast arrays with ``tau > 0``, ``vol > 0``, ``S > 0``.

        ``check`` flags the entries whose result is actually used (the others
        hold harmless placeholder inputs). Powers of ``H / S`` are multiplied
        with normal probabilities in log space, so tiny volatilities do not
        overflow.
        """
        phi = 1.0 if self.is_call else -1.0
        eta = -1.0 if self.is_up else 1.0
        X, H, R = self.strike, self.barrier, self.rebate

        s = vol * np.sqrt(tau)
        mu = (r - q - 0.5 * vol**2) / vol**2
        log_hs = np.log(H / S)
        drift = (1.0 + mu) * s
        x1 = np.log(S / X) / s + drift
        x2 = -log_hs / s + drift
        y1 = np.log(H * H / (S * X)) / s + drift
        y2 = log_hs / s + drift
        df_r, df_q = np.exp(-r * tau), np.exp(-q * tau)

        def power_cdf(exponent, arg):
            """``(H / S)**exponent * N(arg)``."""
            return np.exp(exponent * log_hs + log_ndtr(arg))

        def direct(x):  # blocks A (x1) and B (x2)
            return phi * (S * df_q * ndtr(phi * x) - X * df_r * ndtr(phi * (x - s)))

        def reflected(y):  # blocks C (y1) and D (y2)
            return phi * (
                S * df_q * power_cdf(2.0 * (mu + 1.0), eta * y)
                - X * df_r * power_cdf(2.0 * mu, eta * (y - s))
            )

        with np.errstate(over="ignore", invalid="ignore"):
            coefficients = _KNOCK_IN_COEFFICIENTS[(self.option_type, self.direction, X >= H)]
            if not self.is_knock_in:  # knock-out = vanilla (block A) - knock-in
                coefficients = tuple(v - k for v, k in zip((1, 0, 0, 0), coefficients))
            blocks = ((direct, x1), (direct, x2), (reflected, y1), (reflected, y2))  # A, B, C, D
            value = np.zeros(np.shape(S))
            for coefficient, (block, arg) in zip(coefficients, blocks):
                if coefficient:
                    value = value + coefficient * block(arg)

            if R > 0 and self.is_knock_in:
                # E: rebate at expiry if the barrier was never touched.
                value = value + R * df_r * (
                    ndtr(eta * (x2 - s)) - power_cdf(2.0 * mu, eta * (y2 - s))
                )
            elif R > 0:
                # F: rebate at the hit (discounted over the random hitting time).
                discriminant = mu**2 + 2.0 * r / vol**2
                if np.any(check & (discriminant < 0)):
                    raise ValueError(
                        "the rebate-at-hit formula needs mu**2 + 2 * rate / vol**2 >= 0; "
                        "this combination of negative rate, dividend yield and vol is not supported"
                    )
                lam = np.sqrt(np.maximum(discriminant, 0.0))
                z = log_hs / s + lam * s
                value = value + R * (
                    power_cdf(mu + lam, eta * z) + power_cdf(mu - lam, eta * (z - 2.0 * lam * s))
                )
        return value

    def _deterministic_value(self, S, tau, r, q, vanilla, where: np.ndarray) -> np.ndarray:
        """Value when the spot cannot diffuse (``vol = 0`` or ``spot = 0``), ``tau > 0``.

        The spot then rolls along its forward curve ``S * exp((r - q) * u)`` and
        either reaches the barrier at a known date or never does. Only the
        entries flagged by ``where`` (live side of the barrier) are meaningful.
        """
        H, R = self.barrier, self.rebate
        b = r - q
        forward = S * np.exp(b * tau)
        hits = where & ((forward >= H) if self.is_up else (forward <= H))
        b_safe = np.where(hits, b, 1.0)
        S_safe = np.where(hits, S, H)
        t_hit = np.where(hits, np.log(H / S_safe) / b_safe, 0.0)
        if self.is_knock_in:
            return np.where(hits, vanilla, R * np.exp(-r * tau))
        return np.where(hits, R * np.exp(-r * t_hit), vanilla)

    def _value(self, mkt: Market, continuation: bool = False) -> np.ndarray:
        """Price on the broadcast market grid.

        With ``continuation=True`` the closed form is also used in a thin band
        *beyond* the barrier (it is a smooth function of spot there), which is
        what lets :meth:`greeks` take one-sided finite differences.
        """
        raw = [np.asarray(x, dtype=float) for x in (mkt.spot, mkt.vol, mkt.rate, mkt.div, mkt.t)]
        S, vol, r, q, t = np.broadcast_arrays(*raw)
        nan_mask = np.isnan(S) | np.isnan(vol) | np.isnan(r) | np.isnan(q) | np.isnan(t)
        vanilla = np.asarray(self.vanilla_equivalent().price(mkt), dtype=float)
        vanilla = np.broadcast_to(vanilla, S.shape)

        if self.knocked:
            value = vanilla if self.is_knock_in else np.zeros(S.shape)
            return np.where(nan_mask, np.nan, value)

        H = self.barrier
        tau = self.expiry - t
        alive = tau > 0
        diffusive = alive & (vol > 0) & (S > 0)
        breached = self._breached(S)
        live = diffusive & ~breached
        if continuation:
            S_pos = np.where(S > 0, S, H)
            live = live | (diffusive & (np.abs(np.log(S_pos / H)) < _CONTINUATION_BAND))

        # Placeholders (a spot 10% inside the live region, tau = 1, vol = 20%) keep
        # the formula finite on the entries that np.where discards below.
        S_safe = np.where(live, S, H * (0.9 if self.is_up else 1.1))
        tau_safe = np.where(live, tau, 1.0)
        vol_safe = np.where(live, vol, 0.2)
        formula = self._reiner_rubinstein(S_safe, tau_safe, vol_safe, r, q, check=live)
        if not continuation:
            formula = np.maximum(formula, 0.0)

        if self.is_knock_in:
            value_at_hit, unhit_settlement = vanilla, np.full(S.shape, self.rebate)
        else:
            value_at_hit, unhit_settlement = np.full(S.shape, self.rebate), vanilla
        frozen = alive & ~diffusive & ~breached
        tau_frozen = np.where(frozen, tau, 0.0)
        deterministic = self._deterministic_value(S, tau_frozen, r, q, vanilla, frozen)

        value = np.where(
            live,
            formula,
            np.where(breached, value_at_hit, np.where(alive, deterministic, unhit_settlement)),
        )
        return np.where(nan_mask, np.nan, value)

    # --- valuation ----------------------------------------------------------------
    def price(self, mkt: Market):
        """Reiner-Rubinstein value, vectorised over the market.

        * knocked knock-in: the vanilla price; knocked knock-out: 0;
        * spot at or beyond the barrier (not yet flagged): value at the hit,
          i.e. the rebate (knock-out) or the vanilla price (knock-in);
        * at or after expiry: the settlement value -- vanilla payoff for a
          surviving knock-out, the rebate for a knock-in that never knocked.
        """
        value = self._value(mkt) + 0.0
        return float(value) if value.ndim == 0 else value

    def greeks(self, mkt: Market) -> dict[str, Any]:
        """Price and Greeks (RAW units), one-sided at the barrier.

        On the live side the closed form is bumped and repriced on its analytic
        continuation, so no bump ever straddles the barrier. Beyond the barrier
        (or once knocked) the Greeks are those of the vanilla (knock-in) or
        zero (knock-out).
        """
        vanilla = self.vanilla_equivalent().greeks(mkt)
        if self.is_knock_in:
            after_hit = {k: np.asarray(vanilla[k], dtype=float) for k in GREEK_KEYS}
        else:
            after_hit = {k: np.zeros(mkt.shape) for k in GREEK_KEYS}
            if not self.knocked:
                after_hit["price"] = np.full(mkt.shape, self.rebate)
        if self.knocked:
            merged = after_hit
        else:
            live = numerical_greeks(
                lambda m: self._value(m, continuation=True), mkt, expiry=self.expiry
            )
            breached = np.broadcast_to(self._breached(mkt.spot), mkt.shape)
            merged = {k: np.where(breached, after_hit[k], live[k]) for k in GREEK_KEYS}
            merged["price"] = self._value(mkt)

        out: dict[str, Any] = {}
        for key in GREEK_KEYS:
            value = np.asarray(merged[key], dtype=float) + 0.0
            out[key] = float(value) if value.ndim == 0 else value
        return out

    def payoff(self, spot_T):
        """Terminal payoff assuming the ``knocked`` state no longer changes.

        A terminal spot beyond the barrier necessarily means a hit, so for an
        option that has not knocked yet: a knock-out pays the vanilla payoff
        on the live side and the rebate beyond the barrier; a knock-in pays
        the vanilla payoff beyond the barrier and the rebate on the live side.
        """
        spot_T = np.asarray(spot_T, dtype=float)
        vanilla = bs.intrinsic_value(spot_T, self.strike, self.option_type)
        if self.knocked:
            return vanilla + 0.0 if self.is_knock_in else np.zeros(spot_T.shape)
        breached = self._breached(spot_T)
        if self.is_knock_in:
            return np.where(breached, vanilla, self.rebate)
        return np.where(breached, self.rebate, vanilla)

    def path_payoff(self, paths: np.ndarray, times: np.ndarray, rate: float = 0.0) -> np.ndarray:
        """Payoff per simulated path, the barrier being watched at every column.

        Parameters
        ----------
        paths : ndarray, shape (n_paths, n_times)
            Spot paths from now (column 0 = current spot) to expiry.
        times : ndarray, shape (n_times,)
            ABSOLUTE times of the columns.
        rate : float, default 0.0
            Risk-free rate used to carry a knock-out rebate from the hit date
            to expiry. The rebate is paid at the hit, but a Monte Carlo pricer
            discounts every payoff from expiry, so this method returns
            ``rebate * exp(rate * (expiry - t_hit))``.
            :func:`optionlab.monte_carlo.mc_price` passes ``mkt.rate``
            automatically; with the default ``0.0`` the rebate is valued as if
            it were paid at expiry. Irrelevant when ``rebate == 0`` and for
            knock-ins.

        Notes
        -----
        The stored ``knocked`` flag is honoured: an already knocked-in option
        pays the vanilla payoff on every path and an already knocked-out one
        pays nothing (its rebate was paid in the past). Monitoring is
        *discrete* (the path columns), so the result converges to the
        closed form only as the time step shrinks -- or compare it with
        ``self.discrete_barrier_adjusted(vol, dt).price(mkt)``.
        """
        paths = np.atleast_2d(np.asarray(paths, dtype=float))
        times = np.asarray(times, dtype=float)
        if times.shape != (paths.shape[1],):
            raise ValueError("times must have one entry per path column")
        vanilla = bs.intrinsic_value(paths[:, -1], self.strike, self.option_type)
        if self.knocked:
            return vanilla + 0.0 if self.is_knock_in else np.zeros(paths.shape[0])

        breached = self._breached(paths)
        hit = breached.any(axis=1)
        if self.is_knock_in:
            return np.where(hit, vanilla, self.rebate)
        t_hit = times[np.argmax(breached, axis=1)]
        rebate_at_expiry = self.rebate * np.exp(float(rate) * (times[-1] - t_hit))
        return np.where(hit, rebate_at_expiry, vanilla)
