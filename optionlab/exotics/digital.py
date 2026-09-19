"""Digital (binary) options: cash-or-nothing and asset-or-nothing.

A digital pays a fixed amount -- or one share -- if the option finishes in the
money, and nothing otherwise. The payoff is a *step* instead of a hockey
stick, which gives the product a very different risk profile from a vanilla:

* The price of a cash digital is the discounted risk-neutral probability of
  finishing in the money: ``payout * exp(-r tau) * N(omega * d2)``.
* As expiry approaches, the price as a function of spot tends to the step
  itself. Its slope (delta) becomes a tall spike centred on the strike and its
  curvature (gamma) a positive spike just below the strike glued to a negative
  one just above. A trader who is short a digital with the spot sitting on the
  strike the day before expiry has to hold an enormous, violently changing
  stock hedge to cover a bounded payout: this is *pin risk*.
* Vega changes sign at the (forward) strike: an out-of-the-money digital wants
  movement (long vega), an in-the-money one wants calm (short vega).

Because of the exploding Greeks, desks do not hedge a digital as such. They
book it as a tight **call spread**: ``payout / width`` calls struck just below
the strike against as many calls struck just above. The spread's payoff is a
ramp instead of a step, so its delta is capped at ``payout / width``. As the
width goes to zero the spread converges to the digital (it is the finite
difference version of ``-dC/dK = exp(-r tau) N(d2)``), see
:meth:`DigitalOption.replicating_call_spread`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import ndtr

from .. import black_scholes as bs
from ..instruments import (
    GREEK_KEYS,
    CompositeInstrument,
    EuropeanOption,
    Instrument,
    Position,
    register_instrument,
)
from ..market import Market

__all__ = ["DIGITAL_KINDS", "DigitalOption"]

#: Supported payout conventions.
DIGITAL_KINDS: tuple[str, ...] = ("cash", "asset")

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _market_arrays(mkt: Market) -> tuple[np.ndarray, ...]:
    """Market fields ``(spot, vol, rate, div, t)`` broadcast to the market shape."""
    raw = [np.asarray(x, dtype=float) for x in (mkt.spot, mkt.vol, mkt.rate, mkt.div, mkt.t)]
    return tuple(np.broadcast_arrays(*raw))


def _finish(value: np.ndarray, nan_mask: np.ndarray):
    """Propagate NaN inputs, drop ``-0.0`` and return a float for a scalar market."""
    out = np.where(nan_mask, np.nan, value) + 0.0
    return float(out) if out.ndim == 0 else out


@register_instrument
@dataclass(frozen=True)
class DigitalOption(Instrument):
    """European digital (binary) option.

    Parameters
    ----------
    option_type : {"call", "put"}
        A digital call pays when ``S_T > strike``, a digital put when
        ``S_T < strike`` (case-insensitive; stored lower-case).
    strike : float
        Strike price (> 0).
    expiry : float
        ABSOLUTE expiry in years (time to maturity is ``expiry - mkt.t``).
    payout : float, default 1.0
        For ``kind="cash"``: the cash amount paid. For ``kind="asset"``: the
        number of shares delivered. Must be finite and non-negative.
    kind : {"cash", "asset"}, default "cash"
        ``"cash"`` = cash-or-nothing, ``"asset"`` = asset-or-nothing.

    Notes
    -----
    Closed forms (``omega = +1`` call, ``-1`` put)::

        cash  = payout * exp(-r tau) * N(omega * d2)
        asset = payout * S * exp(-q tau) * N(omega * d1)

    Two identities worth remembering (both are tested):

    * cash call + cash put = ``payout * exp(-r tau)``: together they pay the
      cash for sure, so they are worth a zero-coupon bond.
    * asset call - ``K`` x cash call = vanilla call: a vanilla call is "receive
      the share, pay the strike, both only if in the money". More generally
      ``asset = omega * vanilla + K * cash``, which is how the asset digital's
      Greeks are computed here.

    All twelve Greeks are analytic (RAW units). At or after expiry the price is
    the settlement value at the current spot and every Greek is zero, except
    the delta of an in-the-money asset digital, which has become ``payout``
    shares.

    Examples
    --------
    >>> put = DigitalOption("put", 80.0, 0.75, payout=10.0)
    >>> round(put.price(Market(spot=100.0, vol=0.35, rate=0.06, div=0.06)), 4)
    2.671
    """

    option_type: str
    strike: float
    expiry: float
    payout: float = 1.0
    kind: str = "cash"

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_type", bs.normalize_option_type(self.option_type))
        try:
            strike, expiry, payout = float(self.strike), float(self.expiry), float(self.payout)
        except (TypeError, ValueError) as exc:
            raise ValueError("DigitalOption strike, expiry and payout must be numbers") from exc
        if not (math.isfinite(strike) and strike > 0):
            raise ValueError(f"strike must be a finite positive number, got {self.strike!r}")
        if not math.isfinite(expiry):
            raise ValueError(f"expiry must be finite, got {self.expiry!r}")
        if not (math.isfinite(payout) and payout >= 0):
            raise ValueError(f"payout must be finite and non-negative, got {self.payout!r}")
        kind = self.kind.strip().lower() if isinstance(self.kind, str) else self.kind
        if kind not in DIGITAL_KINDS:
            raise ValueError(f"kind must be one of {DIGITAL_KINDS}, got {self.kind!r}")
        object.__setattr__(self, "strike", strike)
        object.__setattr__(self, "expiry", expiry)
        object.__setattr__(self, "payout", payout)
        object.__setattr__(self, "kind", kind)

    # --- small helpers ----------------------------------------------------
    @property
    def is_call(self) -> bool:
        return self.option_type == "call"

    @property
    def is_cash(self) -> bool:
        return self.kind == "cash"

    @property
    def _omega(self) -> float:
        return 1.0 if self.is_call else -1.0

    @property
    def label(self) -> str:
        prefix = "Dig" if self.is_cash else "AoN"
        return f"{prefix} {'C' if self.is_call else 'P'} {self.strike:g} T={self.expiry:.2f}"

    def vanilla_equivalent(self) -> EuropeanOption:
        """The vanilla option with the same type, strike and expiry."""
        return EuropeanOption(self.option_type, self.strike, self.expiry)

    # --- cash-or-nothing building block -------------------------------------
    def _unit_cash(
        self, mkt: Market, price_only: bool = False
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Price and Greeks of the cash digital paying 1, plus the NaN-input mask.

        Outside the regular region (``tau <= 0``, ``vol <= 0`` or ``spot <= 0``)
        the density term vanishes and ``N(omega d2)`` becomes the (forward)
        in-the-money indicator, exactly like :mod:`optionlab.black_scholes`.
        With ``price_only`` the returned dict holds the ``"price"`` entry only.
        """
        omega, K = self._omega, self.strike
        S, vol, r, q, t = _market_arrays(mkt)
        tau = self.expiry - t
        nan_mask = np.isnan(S) | np.isnan(vol) | np.isnan(r) | np.isnan(q) | np.isnan(t)

        alive = tau > 0
        regular = alive & (vol > 0) & (S > 0)
        T = np.where(alive, tau, 0.0)
        T_safe = np.where(regular, tau, 1.0)
        S_safe = np.where(regular, S, 1.0)
        vol_safe = np.where(regular, vol, 1.0)
        s = vol_safe * np.sqrt(T_safe)  # vol * sqrt(tau)

        d1_raw, d2_raw = bs.d1_d2(S_safe, K, T_safe, vol_safe, r, q)
        d1 = np.where(regular, d1_raw, 0.0)
        d2 = np.where(regular, d2_raw, 0.0)

        df = np.exp(-r * T)
        itm = (omega * (S * np.exp(-q * T) - K * df) > 0).astype(float)
        price = df * np.where(regular, ndtr(omega * d2), itm)
        if price_only:
            return {"price": price}, nan_mask

        # w = omega * exp(-r tau) * phi(d2): the common factor of every derivative.
        w = np.where(regular, omega * df * np.exp(-0.5 * d2 * d2) / _SQRT_2PI, 0.0)

        b = r - q
        dd1_dtau = (2.0 * b * T_safe - d2 * s) / (2.0 * T_safe * s)
        dd2_dtau = (2.0 * b * T_safe - d1 * s) / (2.0 * T_safe * s)

        delta = w / (S_safe * s)
        gamma = -w * d1 / (S_safe * s) ** 2
        out = {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "vega": -w * d1 / vol_safe,
            "theta": np.where(alive, r * price - w * dd2_dtau, 0.0),
            "rho": -T * price + w * np.sqrt(T_safe) / vol_safe,
            "vanna": w * (d1 * d2 - 1.0) / (S_safe * vol_safe * s),
            "volga": -w * (d1 * d1 * d2 - d1 - d2) / vol_safe**2,
            "charm": delta * (r + d2 * dd2_dtau + 0.5 / T_safe),
            "speed": w * ((d1 * d2 - 1.0) / s + 2.0 * d1) / (S_safe**3 * s**2),
            "color": gamma * (r + d2 * dd2_dtau + 1.0 / T_safe) + w * dd1_dtau / (S_safe * s) ** 2,
            "zomma": -w * (d1 * d1 * d2 - d2 - 2.0 * d1) / ((S_safe * s) ** 2 * vol_safe),
        }
        return out, nan_mask

    # --- valuation ------------------------------------------------------------
    def price(self, mkt: Market):
        """Closed-form value; the settlement value at the current spot once expired."""
        cash, nan_mask = self._unit_cash(mkt, price_only=True)
        if self.is_cash:
            return _finish(self.payout * cash["price"], nan_mask)
        vanilla = np.asarray(self.vanilla_equivalent().price(mkt), dtype=float)
        value = self._omega * vanilla + self.strike * cash["price"]
        return _finish(self.payout * np.maximum(value, 0.0), nan_mask)

    def greeks(self, mkt: Market) -> dict[str, Any]:
        """Analytic price and Greeks in RAW units, keys :data:`GREEK_KEYS`.

        The asset-or-nothing Greeks use ``asset = omega * vanilla + K * cash``.
        Near expiry and near the strike these numbers are huge and change sign
        within a fraction of a percent of spot: that is the product, not a bug.
        """
        cash, nan_mask = self._unit_cash(mkt)
        if self.is_cash:
            values = {k: self.payout * cash[k] for k in GREEK_KEYS}
        else:
            vanilla = self.vanilla_equivalent().greeks(mkt)
            values = {
                k: self.payout * (self._omega * np.asarray(vanilla[k]) + self.strike * cash[k])
                for k in GREEK_KEYS
            }
            values["price"] = np.maximum(values["price"], 0.0)
        return {k: _finish(v, nan_mask) for k, v in values.items()}

    def probability_itm(self, mkt: Market):
        """Risk-neutral probability of finishing in the money, ``N(omega * d2)``.

        It is the undiscounted price of the cash digital paying 1. It is *not*
        the real-world probability: it is computed with the stock drifting at
        ``rate - div`` instead of its true expected return.
        """
        cash, nan_mask = self._unit_cash(mkt, price_only=True)
        _, _, r, _, t = _market_arrays(mkt)
        growth = np.exp(r * np.maximum(self.expiry - t, 0.0))
        return _finish(np.clip(cash["price"] * growth, 0.0, 1.0), nan_mask)

    def payoff(self, spot_T):
        """``payout`` (cash) or ``payout * S_T`` (asset) if in the money, else 0.

        The inequality is strict: a digital finishing exactly at the strike
        pays nothing.
        """
        spot_T = np.asarray(spot_T, dtype=float)
        in_the_money = self._omega * (spot_T - self.strike) > 0
        amount = self.payout if self.is_cash else self.payout * spot_T
        return np.where(in_the_money, amount, 0.0)

    # --- teaching: static replication -------------------------------------------
    def replicating_call_spread(
        self, width: float, placement: str = "centered"
    ) -> CompositeInstrument:
        """Vanilla spread that approximates this digital (what a desk really hedges).

        A cash digital call is replaced by ``payout / width`` call spreads
        (long the lower strike, short the upper strike); a digital put by as
        many put spreads (long the upper strike, short the lower one). An
        asset-or-nothing digital additionally holds ``omega * payout`` vanilla
        options at the strike, because ``asset = omega * vanilla + K * cash``.

        Parameters
        ----------
        width : float
            Distance between the two strikes (> 0). The smaller the width, the
            closer the spread is to the digital *and* the larger its Greeks:
            the maximum delta of the hedge is about ``payout / width``.
        placement : {"centered", "conservative"}, default "centered"
            ``"centered"`` straddles the strike (``K - w/2``, ``K + w/2``): the
            most accurate, the pricing error shrinks like ``width**2``.
            ``"conservative"`` puts the whole ramp on the out-of-the-money side
            (``K - w``, ``K`` for a call; ``K``, ``K + w`` for a put), so the
            spread pays at least as much as the digital everywhere. This is
            how a seller prices and hedges in practice: the extra premium is
            the price of never being short the step.

        Returns
        -------
        CompositeInstrument
            Same expiry as the digital; price, Greeks and payoff can be
            compared leg for leg with the digital's.
        """
        try:
            width = float(width)
        except (TypeError, ValueError) as exc:
            raise ValueError("width must be a number") from exc
        if not (math.isfinite(width) and width > 0):
            raise ValueError(f"width must be finite and strictly positive, got {width!r}")
        if placement == "centered":
            low, high = self.strike - 0.5 * width, self.strike + 0.5 * width
        elif placement == "conservative":
            low, high = (
                (self.strike - width, self.strike) if self.is_call
                else (self.strike, self.strike + width)
            )
        else:
            raise ValueError(f"placement must be 'centered' or 'conservative', got {placement!r}")
        if low <= 0:
            raise ValueError(
                f"width {width:g} is too large for strike {self.strike:g}: "
                "the lower strike must stay positive"
            )

        long_strike, short_strike = (low, high) if self.is_call else (high, low)
        cash_amount = self.payout if self.is_cash else self.payout * self.strike
        units = cash_amount / width
        legs = [
            Position(EuropeanOption(self.option_type, long_strike, self.expiry), units),
            Position(EuropeanOption(self.option_type, short_strike, self.expiry), -units),
        ]
        if not self.is_cash:
            legs.append(Position(self.vanilla_equivalent(), self._omega * self.payout))
        name = f"Spread replication of {self.label} (width {width:g})"
        return CompositeInstrument(name, tuple(legs))
