"""Analytic Black-Scholes-Merton prices, Greeks and implied volatility.

Model
-----
The underlying follows a geometric Brownian motion with flat volatility
``vol``, continuously compounded rate ``rate`` and continuous dividend yield
``div`` (Merton's extension). With ``tau`` the time to maturity in years::

    d1 = (ln(S / K) + (r - q + vol**2 / 2) * tau) / (vol * sqrt(tau))
    d2 = d1 - vol * sqrt(tau)
    call = S * exp(-q * tau) * N(d1) - K * exp(-r * tau) * N(d2)
    put  = K * exp(-r * tau) * N(-d2) - S * exp(-q * tau) * N(-d1)

Conventions
-----------
* Every function is a pure function, fully vectorised with numpy
  broadcasting: any argument except ``option_type`` may be an array.
  All-scalar input returns a Python ``float``.
* ``option_type`` is the string ``"call"`` or ``"put"`` (case-insensitive).
* Greeks are **raw mathematical derivatives**: vega is per 1.00 of volatility
  (not per vol point), theta is ``dV/dt`` per *year* of calendar time (so it
  is negative for a long vanilla option), rho is per 1.00 of rate. Use
  :func:`to_trader_units` to convert to desk conventions.
* Edge cases never emit warnings or NaN: for ``tau <= 0`` the price is the
  intrinsic value, delta is a step function and every other Greek is 0; for
  ``vol <= 0`` the option is worth its discounted forward intrinsic value.
  NaN inputs propagate to NaN outputs.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, NamedTuple

import numpy as np
from scipy.special import ndtr

__all__ = [
    "GREEK_NAMES",
    "GREEK_INFO",
    "TRADER_UNIT_SCALES",
    "normalize_option_type",
    "d1_d2",
    "price",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "vanna",
    "volga",
    "vomma",
    "charm",
    "speed",
    "color",
    "zomma",
    "dual_delta",
    "greeks",
    "intrinsic_value",
    "to_trader_units",
    "implied_vol",
]

_SQRT_2PI = math.sqrt(2.0 * math.pi)

#: Names of the Greeks returned by :func:`greeks` (after ``"price"``), in order.
GREEK_NAMES: tuple[str, ...] = (
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "vanna",
    "volga",
    "charm",
    "speed",
    "color",
    "zomma",
)

#: Multipliers turning raw derivatives into trader units (see ``to_trader_units``).
TRADER_UNIT_SCALES: dict[str, float] = {
    "vega": 1.0 / 100.0,
    "theta": 1.0 / 365.0,
    "rho": 1.0 / 100.0,
    "vanna": 1.0 / 100.0,
    "volga": 1.0 / 10_000.0,
    "charm": 1.0 / 365.0,
    "color": 1.0 / 365.0,
    "zomma": 1.0 / 100.0,
}

#: Human-readable metadata for each Greek (labels, units, plain-English meaning).
GREEK_INFO: dict[str, dict[str, str]] = {
    "price": {
        "label": "Price",
        "symbol": "V",
        "raw_unit": "currency",
        "trader_unit": "currency",
        "description": (
            "Model value of the position today. Everything else on this list "
            "describes how this number moves when the market moves."
        ),
    },
    "delta": {
        "label": "Delta",
        "symbol": "dV/dS",
        "raw_unit": "per 1.00 of spot",
        "trader_unit": "per 1.00 of spot",
        "description": (
            "How much the value changes when the spot moves by 1. It is also the "
            "number of shares to sell to be hedged against small spot moves, and a "
            "rough probability-like measure of finishing in the money."
        ),
    },
    "gamma": {
        "label": "Gamma",
        "symbol": "d2V/dS2",
        "raw_unit": "delta per 1.00 of spot",
        "trader_unit": "delta per 1.00 of spot",
        "description": (
            "How fast delta changes when the spot moves. Long gamma means your "
            "delta grows when the market rallies and shrinks when it falls, so "
            "re-hedging makes you buy low and sell high; you pay for it with theta."
        ),
    },
    "vega": {
        "label": "Vega",
        "symbol": "dV/dsigma",
        "raw_unit": "per 1.00 of volatility (100 vol points)",
        "trader_unit": "per 1 vol point (0.01)",
        "description": (
            "Sensitivity to implied volatility. Long options are long vega: they "
            "gain when the market starts pricing bigger future moves. Vega is "
            "largest at the money and for long maturities."
        ),
    },
    "theta": {
        "label": "Theta",
        "symbol": "dV/dt",
        "raw_unit": "per year of calendar time",
        "trader_unit": "per calendar day (1/365 year)",
        "description": (
            "Value change from the pure passage of time, everything else frozen. "
            "A long option usually bleeds theta every day: that is the rent paid "
            "for owning gamma."
        ),
    },
    "rho": {
        "label": "Rho",
        "symbol": "dV/dr",
        "raw_unit": "per 1.00 of rate (10,000 bp)",
        "trader_unit": "per 1% of rate (0.01)",
        "description": (
            "Sensitivity to the interest rate. Higher rates lift the forward, "
            "which helps calls and hurts puts; the effect grows with maturity."
        ),
    },
    "vanna": {
        "label": "Vanna",
        "symbol": "d2V/dS dsigma",
        "raw_unit": "delta per 1.00 of volatility",
        "trader_unit": "delta per 1 vol point (0.01)",
        "description": (
            "How delta changes when volatility moves (equivalently, how vega "
            "changes when the spot moves). It tells you how a vol shock will "
            "un-hedge a delta-neutral book."
        ),
    },
    "volga": {
        "label": "Volga (vomma)",
        "symbol": "d2V/dsigma2",
        "raw_unit": "vega per 1.00 of volatility",
        "trader_unit": "vega (per point) per 1 vol point",
        "description": (
            "Convexity in volatility: how vega changes when volatility moves. "
            "Out-of-the-money options are long volga, so they benefit from "
            "volatility of volatility."
        ),
    },
    "charm": {
        "label": "Charm",
        "symbol": "d(delta)/dt",
        "raw_unit": "delta per year",
        "trader_unit": "delta per calendar day",
        "description": (
            "Delta decay: how delta drifts as time passes with the spot unchanged. "
            "It is the reason a hedge that was flat on Friday is off on Monday."
        ),
    },
    "speed": {
        "label": "Speed",
        "symbol": "d3V/dS3",
        "raw_unit": "gamma per 1.00 of spot",
        "trader_unit": "gamma per 1.00 of spot",
        "description": (
            "How gamma changes when the spot moves. Useful to anticipate how "
            "the re-hedging need evolves in a large move."
        ),
    },
    "color": {
        "label": "Color",
        "symbol": "d(gamma)/dt",
        "raw_unit": "gamma per year",
        "trader_unit": "gamma per calendar day",
        "description": (
            "Gamma decay: how gamma changes as time passes. Near expiry, "
            "at-the-money gamma explodes while away-from-the-money gamma dies."
        ),
    },
    "zomma": {
        "label": "Zomma",
        "symbol": "d(gamma)/dsigma",
        "raw_unit": "gamma per 1.00 of volatility",
        "trader_unit": "gamma per 1 vol point (0.01)",
        "description": (
            "How gamma changes when volatility moves. Lower volatility "
            "concentrates gamma around the strike."
        ),
    },
}


# ---------------------------------------------------------------------- #
# Option type handling
# ---------------------------------------------------------------------- #
def normalize_option_type(option_type: str) -> str:
    """Validate an option type and return it lower-cased.

    Parameters
    ----------
    option_type : str
        ``"call"`` or ``"put"``, in any letter case.

    Returns
    -------
    str
        ``"call"`` or ``"put"``.

    Raises
    ------
    ValueError
        If the value is anything else.
    """
    if isinstance(option_type, str):
        lowered = option_type.strip().lower()
        if lowered in ("call", "put"):
            return lowered
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _omega(option_type: str) -> float:
    """+1 for a call, -1 for a put."""
    return 1.0 if normalize_option_type(option_type) == "call" else -1.0


# ---------------------------------------------------------------------- #
# Shared intermediate quantities
# ---------------------------------------------------------------------- #
class _Core(NamedTuple):
    """Broadcast inputs and shared Black-Scholes building blocks.

    In the *regular* region (``tau > 0``, ``vol > 0``, ``spot > 0``) the usual
    formulas apply. Elsewhere ``pdf1`` is 0 and ``cdf1 = cdf2`` is the
    in-the-money indicator, which makes every formula collapse to the right
    limit without ever dividing by zero.
    """

    omega: float
    S: np.ndarray
    K: np.ndarray
    T: np.ndarray  # time to maturity floored at 0
    sig: np.ndarray
    r: np.ndarray
    q: np.ndarray
    alive: np.ndarray  # tau > 0
    regular: np.ndarray  # tau > 0, vol > 0, spot > 0
    S_safe: np.ndarray  # safe denominators (1 outside the regular region)
    T_safe: np.ndarray
    sig_safe: np.ndarray
    sig_sqrt: np.ndarray  # vol * sqrt(tau), 1 outside the regular region
    sqrt_T: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    pdf1: np.ndarray  # phi(d1)
    cdf1: np.ndarray  # N(omega * d1)
    cdf2: np.ndarray  # N(omega * d2)
    df_q: np.ndarray  # exp(-q * tau)
    df_r: np.ndarray  # exp(-r * tau)
    nan_mask: np.ndarray
    scalar: bool


def _core(spot, strike, tau, vol, rate, div, option_type: str) -> _Core:
    omega = _omega(option_type)
    raw = [np.asarray(x, dtype=float) for x in (spot, strike, tau, vol, rate, div)]
    scalar = all(a.ndim == 0 for a in raw)
    S, K, T, sig, r, q = np.broadcast_arrays(*raw)
    if np.any(K <= 0):
        raise ValueError("strike must be strictly positive")

    nan_mask = np.isnan(S) | np.isnan(K) | np.isnan(T) | np.isnan(sig) | np.isnan(r) | np.isnan(q)
    alive = T > 0
    regular = alive & (sig > 0) & (S > 0)

    T_eff = np.where(alive, T, 0.0)
    sqrt_T = np.sqrt(T_eff)
    S_safe = np.where(regular, S, 1.0)
    T_safe = np.where(regular, T, 1.0)
    sig_safe = np.where(regular, sig, 1.0)
    sig_sqrt = np.where(regular, sig * sqrt_T, 1.0)

    d1 = np.where(
        regular,
        (np.log(S_safe / K) + (r - q + 0.5 * sig_safe**2) * T_eff) / sig_sqrt,
        0.0,
    )
    d2 = np.where(regular, d1 - sig_sqrt, 0.0)

    df_q = np.exp(-q * T_eff)
    df_r = np.exp(-r * T_eff)
    itm = (omega * (S * df_q - K * df_r) > 0).astype(float)

    pdf1 = np.where(regular, np.exp(-0.5 * d1 * d1) / _SQRT_2PI, 0.0)
    cdf1 = np.where(regular, ndtr(omega * d1), itm)
    cdf2 = np.where(regular, ndtr(omega * d2), itm)

    return _Core(
        omega, S, K, T_eff, sig, r, q, alive, regular, S_safe, T_safe, sig_safe,
        sig_sqrt, sqrt_T, d1, d2, pdf1, cdf1, cdf2, df_q, df_r, nan_mask, scalar,
    )


def _finish(value: np.ndarray, c: _Core):
    """Apply NaN propagation and return a float for all-scalar input."""
    out = np.where(c.nan_mask, np.nan, value) + 0.0  # "+ 0.0" turns -0.0 into 0.0
    return float(out) if c.scalar else out


# ---------------------------------------------------------------------- #
# Formulas on the shared core (each returns an ndarray)
# ---------------------------------------------------------------------- #
def _price(c: _Core) -> np.ndarray:
    return c.omega * (c.S * c.df_q * c.cdf1 - c.K * c.df_r * c.cdf2)


def _delta(c: _Core) -> np.ndarray:
    return c.omega * c.df_q * c.cdf1


def _gamma(c: _Core) -> np.ndarray:
    return c.df_q * c.pdf1 / (c.S_safe * c.sig_sqrt)


def _vega(c: _Core) -> np.ndarray:
    return c.S * c.df_q * c.pdf1 * c.sqrt_T


def _theta(c: _Core) -> np.ndarray:
    # dV/dt = -dV/dtau. The first term is the time decay of the optionality,
    # the second is the carry: dividends earned on the delta-equivalent stock
    # (+q S e^{-q tau} N(d1) for a call) minus interest on the strike
    # (-r K e^{-r tau} N(d2) for a call); both flip sign for a put.
    decay = -c.S * c.df_q * c.pdf1 * c.sig_safe / (2.0 * np.sqrt(c.T_safe))
    carry = c.omega * (c.q * c.S * c.df_q * c.cdf1 - c.r * c.K * c.df_r * c.cdf2)
    return np.where(c.alive, decay + carry, 0.0)


def _rho(c: _Core) -> np.ndarray:
    return c.omega * c.K * c.T * c.df_r * c.cdf2


def _vanna(c: _Core) -> np.ndarray:
    return -c.df_q * c.pdf1 * c.d2 / c.sig_safe


def _volga(c: _Core) -> np.ndarray:
    return _vega(c) * c.d1 * c.d2 / c.sig_safe


def _dd1_dtau(c: _Core) -> np.ndarray:
    """Partial derivative of d1 with respect to tau (0 outside the regular region)."""
    return (2.0 * (c.r - c.q) * c.T - c.d2 * c.sig_sqrt) / (2.0 * c.T_safe * c.sig_sqrt)


def _charm(c: _Core) -> np.ndarray:
    # d(delta)/dt = -d(delta)/dtau with delta = omega e^{-q tau} N(omega d1).
    value = c.omega * c.q * c.df_q * c.cdf1 - c.df_q * c.pdf1 * _dd1_dtau(c)
    return np.where(c.alive, value, 0.0)


def _speed(c: _Core) -> np.ndarray:
    return -_gamma(c) / c.S_safe * (c.d1 / c.sig_sqrt + 1.0)


def _color(c: _Core) -> np.ndarray:
    # d(gamma)/dt = -d(gamma)/dtau; ln(gamma) = -q tau - d1^2/2 - ln(tau)/2 + const.
    return _gamma(c) * (c.q + 0.5 / c.T_safe + c.d1 * _dd1_dtau(c))


def _zomma(c: _Core) -> np.ndarray:
    return _gamma(c) * (c.d1 * c.d2 - 1.0) / c.sig_safe


def _dual_delta(c: _Core) -> np.ndarray:
    return -c.omega * c.df_r * c.cdf2


_FORMULAS: dict[str, Callable[[_Core], np.ndarray]] = {
    "price": _price,
    "delta": _delta,
    "gamma": _gamma,
    "vega": _vega,
    "theta": _theta,
    "rho": _rho,
    "vanna": _vanna,
    "volga": _volga,
    "charm": _charm,
    "speed": _speed,
    "color": _color,
    "zomma": _zomma,
}


def _evaluate(name_fn, spot, strike, tau, vol, rate, div, option_type):
    c = _core(spot, strike, tau, vol, rate, div, option_type)
    return _finish(name_fn(c), c)


# ---------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------- #
def d1_d2(spot, strike, tau, vol, rate=0.0, div=0.0):
    """Return the Black-Scholes ``(d1, d2)`` pair.

    ``d2`` is the standardised distance between the forward and the strike
    measured in standard deviations: ``N(d2)`` is the risk-neutral probability
    that a call finishes in the money.

    In degenerate cases (``tau <= 0``, ``vol <= 0`` or ``spot <= 0``) both
    numbers are ``+inf`` / ``-inf`` depending on the sign of the forward
    moneyness, and ``0`` exactly at the (forward) money.
    """
    c = _core(spot, strike, tau, vol, rate, div, "call")
    moneyness = np.sign(c.S * c.df_q - c.K * c.df_r)
    limit = np.where(moneyness == 0, 0.0, np.where(moneyness > 0, np.inf, -np.inf))
    d1 = np.where(c.regular, c.d1, limit)
    d2 = np.where(c.regular, c.d2, limit)
    return _finish(d1, c), _finish(d2, c)


def intrinsic_value(spot, strike, option_type: str = "call"):
    """Exercise value ``max(S - K, 0)`` (call) or ``max(K - S, 0)`` (put)."""
    omega = _omega(option_type)
    S = np.asarray(spot, dtype=float)
    K = np.asarray(strike, dtype=float)
    out = np.maximum(omega * (S - K), 0.0)
    return float(out) if out.ndim == 0 else out


def price(spot, strike, tau, vol, rate=0.0, div=0.0, option_type: str = "call"):
    """Black-Scholes-Merton price of a European option.

    Parameters
    ----------
    spot, strike : float or ndarray
        Underlying price and strike (strike must be > 0).
    tau : float or ndarray
        Time to maturity in years. ``tau <= 0`` returns the intrinsic value.
    vol : float or ndarray
        Annualised volatility. ``vol <= 0`` returns the discounted forward
        intrinsic value ``max(omega * (S e^{-q tau} - K e^{-r tau}), 0)``.
    rate, div : float or ndarray, default 0.0
        Continuous risk-free rate and dividend yield.
    option_type : {"call", "put"}

    Returns
    -------
    float or ndarray
        Option value, broadcast over all inputs.

    Notes
    -----
    Reading the call formula as a hedge: hold ``e^{-q tau} N(d1)`` shares and
    borrow ``K e^{-r tau} N(d2)`` of cash. The option is worth the cost of
    that replicating portfolio.
    """
    return _evaluate(_price, spot, strike, tau, vol, rate, div, option_type)


def delta(spot, strike, tau, vol, rate=0.0, div=0.0, option_type: str = "call"):
    """Delta ``dV/dS``: ``e^{-q tau} N(d1)`` for a call, ``-e^{-q tau} N(-d1)`` for a put.

    Delta is the hedge ratio: short ``delta`` shares per long option and the
    position is insensitive to small spot moves. At or after expiry it is the
    step function ``1{S > K}`` (call) or ``-1{S < K}`` (put).
    """
    return _evaluate(_delta, spot, strike, tau, vol, rate, div, option_type)


def gamma(spot, strike, tau, vol, rate=0.0, div=0.0, option_type: str = "call"):
    """Gamma ``d2V/dS2 = e^{-q tau} phi(d1) / (S vol sqrt(tau))`` (same for call and put).

    Gamma measures how quickly the hedge goes stale. It peaks near the strike
    and blows up as expiry approaches at the money.
    """
    return _evaluate(_gamma, spot, strike, tau, vol, rate, div, option_type)


def vega(spot, strike, tau, vol, rate=0.0, div=0.0, option_type: str = "call"):
    """Vega ``dV/dvol = S e^{-q tau} phi(d1) sqrt(tau)`` per 1.00 of vol (same for call and put).

    Divide by 100 for the usual "per vol point" number. Vega grows with the
    square root of maturity: long-dated options are volatility instruments,
    short-dated ones are gamma instruments.
    """
    return _evaluate(_vega, spot, strike, tau, vol, rate, div, option_type)


def theta(spot, strike, tau, vol, rate=0.0, div=0.0, option_type: str = "call"):
    """Theta ``dV/dt`` per year of calendar time (``= -dV/dtau``).

    For a call::

        theta = -S e^{-q tau} phi(d1) vol / (2 sqrt(tau))
                + q S e^{-q tau} N(d1) - r K e^{-r tau} N(d2)

    and for a put the two carry terms flip sign and use ``N(-d1)``, ``N(-d2)``.
    The first term is the price of optionality melting away; the carry terms
    can make theta *positive* for deep in-the-money puts (you are waiting to
    receive the strike) or calls on high-dividend stocks. Divide by 365 for
    the daily number.
    """
    return _evaluate(_theta, spot, strike, tau, vol, rate, div, option_type)


def rho(spot, strike, tau, vol, rate=0.0, div=0.0, option_type: str = "call"):
    """Rho ``dV/dr = omega K tau e^{-r tau} N(omega d2)`` per 1.00 of rate.

    A call is a leveraged long position financed at the risk-free rate, so it
    gains when rates rise; a put is the opposite.
    """
    return _evaluate(_rho, spot, strike, tau, vol, rate, div, option_type)


def vanna(spot, strike, tau, vol, rate=0.0, div=0.0, option_type: str = "call"):
    """Vanna ``d2V/(dS dvol) = -e^{-q tau} phi(d1) d2 / vol`` (same for call and put).

    Positive for out-of-the-money calls, negative for out-of-the-money puts:
    a vol spike makes OTM calls "more in play" (delta rises) and OTM puts too
    (delta becomes more negative).
    """
    return _evaluate(_vanna, spot, strike, tau, vol, rate, div, option_type)


def volga(spot, strike, tau, vol, rate=0.0, div=0.0, option_type: str = "call"):
    """Volga / vomma ``d2V/dvol2 = vega d1 d2 / vol`` (same for call and put).

    Near the money ``d1 d2`` is about zero, so the price is almost linear in
    vol; in the wings it is convex, which is why wings are the way to trade
    vol-of-vol.
    """
    return _evaluate(_volga, spot, strike, tau, vol, rate, div, option_type)


vomma = volga


def charm(spot, strike, tau, vol, rate=0.0, div=0.0, option_type: str = "call"):
    """Charm ``d(delta)/dt`` per year (delta decay)::

        charm = omega q e^{-q tau} N(omega d1)
                - e^{-q tau} phi(d1) (2 (r - q) tau - d2 vol sqrt(tau)) / (2 tau vol sqrt(tau))

    As expiry approaches, in-the-money deltas drift towards +/-1 and
    out-of-the-money deltas towards 0; charm is the speed of that drift.
    """
    return _evaluate(_charm, spot, strike, tau, vol, rate, div, option_type)


def speed(spot, strike, tau, vol, rate=0.0, div=0.0, option_type: str = "call"):
    """Speed ``d3V/dS3 = -gamma / S * (d1 / (vol sqrt(tau)) + 1)`` (same for call and put)."""
    return _evaluate(_speed, spot, strike, tau, vol, rate, div, option_type)


def color(spot, strike, tau, vol, rate=0.0, div=0.0, option_type: str = "call"):
    """Color ``d(gamma)/dt`` per year (gamma decay; same for call and put)::

        color = gamma * (q + 1 / (2 tau) + d1 * dd1/dtau)

    At the money gamma *grows* as time passes (positive color); in the wings
    it dies out.
    """
    return _evaluate(_color, spot, strike, tau, vol, rate, div, option_type)


def zomma(spot, strike, tau, vol, rate=0.0, div=0.0, option_type: str = "call"):
    """Zomma ``d(gamma)/dvol = gamma (d1 d2 - 1) / vol`` (same for call and put)."""
    return _evaluate(_zomma, spot, strike, tau, vol, rate, div, option_type)


def dual_delta(spot, strike, tau, vol, rate=0.0, div=0.0, option_type: str = "call"):
    """Dual delta ``dV/dK = -omega e^{-r tau} N(omega d2)``.

    Minus the discounted risk-neutral probability of finishing in the money;
    it is the slope of the price as a function of strike, which is what a
    tight call spread replicates.
    """
    return _evaluate(_dual_delta, spot, strike, tau, vol, rate, div, option_type)


def greeks(spot, strike, tau, vol, rate=0.0, div=0.0, option_type: str = "call") -> dict[str, Any]:
    """Price and all Greeks in one pass (``d1``/``d2`` computed once).

    Returns
    -------
    dict
        Keys ``"price"`` followed by every name in :data:`GREEK_NAMES`
        (delta, gamma, vega, theta, rho, vanna, volga, charm, speed, color,
        zomma), in RAW units. Values are floats for all-scalar input and
        arrays otherwise.
    """
    c = _core(spot, strike, tau, vol, rate, div, option_type)
    return {name: _finish(fn(c), c) for name, fn in _FORMULAS.items()}


def to_trader_units(greeks_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Convert raw Greeks to the units quoted on a trading desk.

    * vega, vanna, zomma: per 1 vol point (divide by 100)
    * volga: per vol point squared (divide by 10,000)
    * theta, charm, color: per calendar day (divide by 365)
    * rho: per 1% of rate (divide by 100)
    * price, delta, gamma, speed and any unknown key: passed through untouched
      (so a table row mixing Greeks with labels or ``None`` entries is fine)

    The input mapping is not modified.
    """
    return {
        k: v * TRADER_UNIT_SCALES[k] if k in TRADER_UNIT_SCALES else v
        for k, v in greeks_dict.items()
    }


# ---------------------------------------------------------------------- #
# Implied volatility
# ---------------------------------------------------------------------- #
def implied_vol(
    price,
    spot,
    strike,
    tau,
    rate=0.0,
    div=0.0,
    option_type: str = "call",
    *,
    tol: float = 1e-12,
    max_iter: int = 100,
    vol_max: float = 64.0,
):
    """Volatility that reproduces an option price (vectorised).

    The solver is Newton's method started at the Manaster-Koehler point and
    safeguarded by a bisection bracket, so it cannot diverge; it typically
    converges in a handful of iterations.

    Parameters
    ----------
    price : float or ndarray
        Observed option price(s).
    spot, strike, tau, rate, div, option_type
        As in :func:`price`.
    tol : float
        Relative tolerance on the volatility.
    max_iter : int
        Maximum number of safeguarded Newton iterations.
    vol_max : float
        Largest volatility searched (6400% by default).

    Returns
    -------
    float or ndarray
        Implied volatility; ``nan`` where the price violates the no-arbitrage
        bounds (below the discounted forward intrinsic value, or at/above the
        discounted spot for a call / discounted strike for a put), where
        ``tau <= 0``, or where no volatility below ``vol_max`` matches.
        A price exactly on the lower bound maps to ``0.0``.

    Notes
    -----
    Implied vol is the market's price quoted in a more comparable unit: two
    options with wildly different premiums can be compared through the vol
    that each premium implies.
    """
    omega = _omega(option_type)
    raw = [np.asarray(x, dtype=float) for x in (price, spot, strike, tau, rate, div)]
    scalar = all(a.ndim == 0 for a in raw)
    P, S, K, T, r, q = (np.array(a, dtype=float) for a in np.broadcast_arrays(*raw))
    if np.any(K <= 0):
        raise ValueError("strike must be strictly positive")

    T_pos = np.where(T > 0, T, 0.0)
    disc_spot = S * np.exp(-q * T_pos)
    disc_strike = K * np.exp(-r * T_pos)
    lower = np.maximum(omega * (disc_spot - disc_strike), 0.0)
    upper = disc_spot if omega > 0 else disc_strike
    eps = 1e-14 * np.maximum(np.maximum(disc_spot, disc_strike), 1.0)

    valid = (T > 0) & (S > 0) & (P >= lower - eps) & (P < upper)
    at_lower = valid & (P <= lower + eps)
    solve = valid & ~at_lower

    out = np.full(P.shape, np.nan)
    out[at_lower] = 0.0

    if np.any(solve):
        out[solve] = _solve_implied_vol(
            P[solve], S[solve], K[solve], T[solve], r[solve], q[solve],
            option_type, tol, max_iter, vol_max,
        )
    return float(out) if scalar else out


def _solve_implied_vol(P, S, K, T, r, q, option_type, tol, max_iter, vol_max):
    """Safeguarded Newton on 1-D arrays of bracketable problems."""

    def value_and_vega(sig):
        c = _core(S, K, T, sig, r, q, option_type)
        return _price(c) - P, _vega(c)

    # Upper bracket: double until the model price exceeds the target.
    lo = np.zeros_like(P)
    hi = np.ones_like(P)
    f_hi, _ = value_and_vega(hi)
    while np.any((f_hi < 0) & (hi < vol_max)):
        hi = np.where((f_hi < 0) & (hi < vol_max), np.minimum(2.0 * hi, vol_max), hi)
        f_hi, _ = value_and_vega(hi)
    bracketed = f_hi >= 0

    # Manaster-Koehler start: the inflection point of price(vol), from which
    # Newton converges monotonically.
    x = np.sqrt(2.0 * np.abs(np.log(S / K) + (r - q) * T) / T)
    x = np.clip(np.where(x > 1e-4, x, 0.2), 1e-4, hi)

    done = ~bracketed
    for _ in range(max_iter):
        f, v = value_and_vega(x)
        lo = np.where(f < 0, x, lo)
        hi = np.where(f > 0, x, hi)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            newton = x - f / v
        inside = np.isfinite(newton) & (newton > lo) & (newton < hi)
        x_new = np.where(inside, newton, 0.5 * (lo + hi))
        converged = (np.abs(x_new - x) <= tol * (1.0 + x)) | (f == 0)
        x = np.where(done, x, x_new)
        done = done | converged
        if np.all(done):
            break
    return np.where(bracketed, x, np.nan)
