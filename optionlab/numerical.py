"""Generic bump-and-reprice Greeks.

Any product that can be *priced* can be *risk-managed*: bump one market input,
reprice, and take a finite difference. That is exactly how most real trading
systems compute the Greeks of exotic books, and it is how every instrument in
this library gets its Greeks unless it overrides them with closed forms.

The engine only needs a function ``Market -> price``; it never imports the
instrument classes, so it can also differentiate a Monte Carlo pricer or any
ad-hoc lambda.
"""

from __future__ import annotations

from typing import Any, Callable, Union

import numpy as np

from .market import Market

__all__ = ["NUMERICAL_GREEK_KEYS", "numerical_greeks"]

PriceFn = Callable[[Market], Any]

#: Keys of the dict returned by :func:`numerical_greeks`, in order.
NUMERICAL_GREEK_KEYS: tuple[str, ...] = (
    "price",
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


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``num / den`` with 0 wherever ``den == 0`` (no warnings)."""
    den = np.asarray(den, dtype=float)
    ok = den != 0
    return np.where(ok, num / np.where(ok, den, 1.0), 0.0)


def numerical_greeks(
    instrument_or_price_fn: Union[PriceFn, Any],
    mkt: Market,
    spot_bump_rel: float = 1e-3,
    vol_bump: float = 1e-3,
    t_bump: float = 1e-4,
    rate_bump: float = 1e-4,
    expiry: float | None = None,
) -> dict[str, Any]:
    """Finite-difference Greeks of any priceable object, in RAW units.

    Parameters
    ----------
    instrument_or_price_fn : Instrument or callable
        Either an object with a ``price(mkt)`` method (its ``expiry`` attribute
        is used when present) or a plain function ``mkt -> price``.
    mkt : Market
        Market at which to differentiate; fields may be arrays, in which case
        every Greek is an array of the broadcast shape. ``spot`` must be > 0.
    spot_bump_rel : float, default 1e-3
        Spot bump as a fraction of spot (central differences; the third
        derivative also uses ``+/- 2`` bumps). The default suits closed forms;
        use ``1e-2`` or more for Monte Carlo pricers, whose noise is amplified
        by ``1 / bump**2`` in gamma.
    vol_bump : float, default 1e-3
        Absolute volatility bump (central; shrunk to ``vol / 2`` when the vol
        is smaller than the bump; vol Greeks are reported as 0 at ``vol = 0``).
    t_bump : float, default 1e-4
        Time bump in years. Time only moves forward (path-dependent products
        cannot be "un-observed"), so theta, charm and color use a second-order
        *forward* difference at ``t + h`` and ``t + 2h``. ``h`` is clipped to
        a quarter of the remaining life so ``t + 2h`` never reaches expiry.
    rate_bump : float, default 1e-4
        Absolute rate bump (central).
    expiry : float, optional
        Absolute expiry used to clip the time bump. Defaults to
        ``instrument.expiry`` when available; ``None`` means "never expires".

    Returns
    -------
    dict
        Keys :data:`NUMERICAL_GREEK_KEYS`: price, delta, gamma, vega, theta,
        rho, vanna, volga, charm, speed, color, zomma. Once expired
        (``t >= expiry``) everything is 0 except ``price`` and ``delta``.

    Notes
    -----
    Nothing is mutated: each bump creates a new :class:`Market` through
    ``mkt.bumped``. A full set costs 19 repricings, all vectorised.

    Second derivatives divide a tiny price difference by ``bump**2``: too small
    a bump amplifies rounding noise, too large a bump smears the curvature.
    Around discontinuities (barriers, digital strikes, expiry) no bump size is
    "right" -- that instability is a real feature of those products' risk.
    """
    if hasattr(instrument_or_price_fn, "price"):
        price_fn: PriceFn = instrument_or_price_fn.price
        if expiry is None:
            expiry = getattr(instrument_or_price_fn, "expiry", None)
    elif callable(instrument_or_price_fn):
        price_fn = instrument_or_price_fn
    else:
        raise ValueError("numerical_greeks needs an object with .price(mkt) or a callable mkt -> price")
    for name, bump in (("spot_bump_rel", spot_bump_rel), ("vol_bump", vol_bump),
                       ("t_bump", t_bump), ("rate_bump", rate_bump)):
        if not bump > 0:
            raise ValueError(f"{name} must be strictly positive")

    spot = np.asarray(mkt.spot, dtype=float)
    vol = np.asarray(mkt.vol, dtype=float)
    t = np.asarray(mkt.t, dtype=float)
    if np.any(spot <= 0):
        raise ValueError("numerical_greeks requires a strictly positive spot")

    def reprice(**changes: Any) -> np.ndarray:
        return np.asarray(price_fn(mkt.bumped(**changes)), dtype=float)

    def second_diff(up: np.ndarray, mid: np.ndarray, dn: np.ndarray) -> np.ndarray:
        return (up - 2.0 * mid + dn) / hs**2

    def forward_diff(f0: np.ndarray, f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
        return _safe_div(-3.0 * f0 + 4.0 * f1 - f2, 2.0 * ht)

    base = reprice()

    # --- bump sizes ------------------------------------------------------
    hs = spot_bump_rel * spot
    hv = np.minimum(vol_bump, 0.5 * vol)
    if expiry is None:
        alive = np.ones(np.shape(t), dtype=bool)
        ht = np.full(np.shape(t), float(t_bump))
    else:
        tau = expiry - t
        alive = tau > 0
        ht = np.where(alive, np.minimum(t_bump, 0.25 * np.where(alive, tau, 0.0)), 0.0)
    hr = rate_bump

    # --- spot ladder -----------------------------------------------------
    s_up, s_dn = reprice(spot=spot + hs), reprice(spot=spot - hs)
    s_up2, s_dn2 = reprice(spot=spot + 2.0 * hs), reprice(spot=spot - 2.0 * hs)
    delta = (s_up - s_dn) / (2.0 * hs)
    gamma = second_diff(s_up, base, s_dn)
    speed = (s_up2 - 2.0 * s_up + 2.0 * s_dn - s_dn2) / (2.0 * hs**3)

    # --- volatility (and spot/vol crosses) -------------------------------
    v_up, v_dn = reprice(vol=vol + hv), reprice(vol=vol - hv)
    uu, ud = reprice(spot=spot + hs, vol=vol + hv), reprice(spot=spot + hs, vol=vol - hv)
    du, dd = reprice(spot=spot - hs, vol=vol + hv), reprice(spot=spot - hs, vol=vol - hv)
    vega = _safe_div(v_up - v_dn, 2.0 * hv)
    volga = _safe_div(v_up - 2.0 * base + v_dn, hv**2)
    vanna = _safe_div(uu - ud - du + dd, 4.0 * hs * hv)
    zomma = _safe_div(second_diff(uu, v_up, du) - second_diff(ud, v_dn, dd), 2.0 * hv)

    # --- time (forward only) and spot/time crosses -----------------------
    t1, t2 = t + ht, t + 2.0 * ht
    p1, p2 = reprice(t=t1), reprice(t=t2)
    p1_up, p1_dn = reprice(spot=spot + hs, t=t1), reprice(spot=spot - hs, t=t1)
    p2_up, p2_dn = reprice(spot=spot + hs, t=t2), reprice(spot=spot - hs, t=t2)
    theta = forward_diff(base, p1, p2)
    charm = forward_diff(delta, (p1_up - p1_dn) / (2.0 * hs), (p2_up - p2_dn) / (2.0 * hs))
    color = forward_diff(gamma, second_diff(p1_up, p1, p1_dn), second_diff(p2_up, p2, p2_dn))

    # --- rate --------------------------------------------------------------
    rate = np.asarray(mkt.rate, dtype=float)
    rho = (reprice(rate=rate + hr) - reprice(rate=rate - hr)) / (2.0 * hr)

    raw = {
        "price": base,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
        "vanna": vanna,
        "volga": volga,
        "charm": charm,
        "speed": speed,
        "color": color,
        "zomma": zomma,
    }
    out: dict[str, Any] = {}
    for key in NUMERICAL_GREEK_KEYS:
        value = np.asarray(raw[key], dtype=float)
        if key not in ("price", "delta"):
            value = np.where(alive, value, 0.0)
        out[key] = float(value) if value.ndim == 0 else value
    return out
