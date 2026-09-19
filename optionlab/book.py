"""The trading book: positions, cash, risk and life cycle.

A :class:`Book` is what a trader actually manages. It holds

* **positions**: a netted ``{instrument: signed quantity}`` map (buying 2 calls
  and selling 2 of the same calls leaves no line at all),
* **cash**: every trade is paid for out of (or paid into) the cash account, so
  the book is *self-financing*: ``value = positions + cash`` only changes
  because the market moves, never because you traded at the model price,
* a **blotter** (the list of :class:`Trade`) and the list of
  :class:`Settlement` records created when instruments expire.

On top of that state the book answers the three questions of a risk manager:
*what is it worth* (``value``, ``pnl``, ``positions_frame``), *what happens if
the market moves* (``greeks``, ``dollar_greeks``, ``spot_ladder``,
``scenario_grid``, ``stress_tests`` -- all by FULL repricing) and *what do I
trade to fix it* (``hedge_delta``, ``neutralise``, ``solve_hedge``).

Conventions
-----------
* Greeks are RAW derivatives unless a method says ``trader_units`` (see
  :func:`optionlab.black_scholes.to_trader_units`).
* The book has no clock of its own: time lives in the :class:`Market` passed
  to each method (``mkt.t``). A simulator steps the market and calls
  ``observe -> settle_expired -> accrue`` at each step.
* State-changing methods (``trade``, ``observe``, ``settle_expired``,
  ``accrue``, hedges) need a scalar market. Valuation methods accept array
  markets and stay vectorised; instruments whose pricer cannot handle arrays
  are transparently repriced in a Python loop.
* The book is a plain mutable object with JSON persistence, so it can live in
  ``st.session_state`` and be saved/restored by an app.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, Union

import numpy as np
import pandas as pd

from . import black_scholes as bs
from .instruments import (
    EXPIRY_TOL,
    GREEK_KEYS,
    CompositeInstrument,
    Instrument,
    Position,
    Underlying,
    instrument_from_dict,
)
from .market import Market

__all__ = [
    "QUANTITY_TOL",
    "MIN_VOL",
    "DAYS_PER_YEAR",
    "DEFAULT_SPOT_SHOCKS",
    "DEFAULT_VOL_SHOCKS",
    "DEFAULT_SCENARIOS",
    "shocked_market",
    "TransactionCosts",
    "Trade",
    "Settlement",
    "Book",
]

#: Net quantities smaller than this (in absolute value) are treated as flat.
QUANTITY_TOL: float = 1e-12

#: Floor applied to a *shocked* volatility (a -10 point shock on a 8% vol
#: would otherwise go negative).
MIN_VOL: float = 1e-4

#: Day count used to convert ``days`` into years (calendar days).
DAYS_PER_YEAR: float = 365.0

#: Relative spot shocks of the default risk ladder.
DEFAULT_SPOT_SHOCKS: tuple[float, ...] = (
    -0.20, -0.15, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15, 0.20,
)

#: Absolute volatility shocks (0.05 = +5 vol points) of the default scenario grid.
DEFAULT_VOL_SHOCKS: tuple[float, ...] = (-0.10, -0.05, 0.0, 0.05, 0.10)

#: Default stress scenarios. ``spot`` is a relative move, ``vol`` and ``rate``
#: are absolute moves, ``days`` is the number of calendar days that pass.
#: Spot and vol are shocked TOGETHER in the directional scenarios because that
#: is how equity markets behave: vol jumps in a crash and leaks in a rally.
DEFAULT_SCENARIOS: dict[str, dict[str, float]] = {
    "Crash: spot -20%, vol +15 pts": {"spot": -0.20, "vol": 0.15},
    "Sell-off: spot -10%, vol +8 pts": {"spot": -0.10, "vol": 0.08},
    "Gap down: spot -5%": {"spot": -0.05},
    "Gap up: spot +5%": {"spot": 0.05},
    "Rally: spot +10%, vol -5 pts": {"spot": 0.10, "vol": -0.05},
    "Melt-up: spot +20%, vol -8 pts": {"spot": 0.20, "vol": -0.08},
    "Vol spike: vol +10 pts": {"vol": 0.10},
    "Vol crush: vol -10 pts": {"vol": -0.10},
    "One week passes": {"days": 7.0},
    "One month passes": {"days": 30.0},
    "Rates +100 bp": {"rate": 0.01},
}

_MARKET_FIELDS: tuple[str, ...] = ("spot", "vol", "rate", "div", "t")
_SCENARIO_KEYS: tuple[str, ...] = ("spot", "vol", "rate", "days")


# ---------------------------------------------------------------------- #
# Market helpers
# ---------------------------------------------------------------------- #
def shocked_market(mkt: Market, spot=0.0, vol=0.0, rate=0.0, days=0.0) -> Market:
    """Apply a scenario to a market and return the shocked copy.

    Parameters
    ----------
    mkt : Market
        Starting point.
    spot : float or ndarray, default 0.0
        RELATIVE spot move (``-0.10`` = the spot falls 10%). Must be > -1.
    vol : float or ndarray, default 0.0
        ABSOLUTE volatility move (``0.05`` = +5 vol points). The shocked vol
        is floored at :data:`MIN_VOL`.
    rate : float or ndarray, default 0.0
        ABSOLUTE rate move (``0.01`` = +100 bp).
    days : float or ndarray, default 0.0
        Calendar days that pass (``t`` moves forward by ``days / 365``). Time
        only moves forward, so this must be >= 0.

    Returns
    -------
    Market
        Arrays are allowed for every shock (they must broadcast), which is how
        ladders and grids are repriced in a single vectorised call.
    """
    spot_shock = np.asarray(spot, dtype=float)
    days_shift = np.asarray(days, dtype=float)
    if np.any(~(spot_shock > -1.0)):
        raise ValueError("relative spot shocks must be greater than -1 (-100%)")
    if np.any(~(days_shift >= 0.0)):
        raise ValueError("days must be non-negative: time only moves forward")
    floor = np.minimum(mkt.vol, MIN_VOL)
    return mkt.bumped(
        spot=mkt.spot * (1.0 + spot_shock),
        vol=np.maximum(mkt.vol + np.asarray(vol, dtype=float), floor),
        rate=mkt.rate + np.asarray(rate, dtype=float),
        t=mkt.t + days_shift / DAYS_PER_YEAR,
    )


def _require_scalar(mkt: Market, action: str) -> None:
    if not isinstance(mkt, Market):
        raise ValueError(f"{action} needs a Market, got {type(mkt).__name__}")
    if not mkt.is_scalar:
        raise ValueError(f"{action} needs a scalar Market (one spot, one vol, one time), not arrays")


def _scalar_markets(mkt: Market) -> Iterator[tuple[tuple[int, ...], Market]]:
    """Yield ``(index, scalar market)`` for every element of an array market."""
    shape = mkt.shape
    grids = {name: np.broadcast_to(getattr(mkt, name), shape) for name in _MARKET_FIELDS}
    for index in np.ndindex(shape):
        yield index, Market(**{name: float(grid[index]) for name, grid in grids.items()})


def _price(instrument: Instrument, mkt: Market):
    """``instrument.price(mkt)``, vectorised when possible, looped otherwise."""
    if mkt.is_scalar:
        return float(instrument.price(mkt))
    try:
        return np.broadcast_to(np.asarray(instrument.price(mkt), dtype=float), mkt.shape)
    except (TypeError, ValueError):
        out = np.empty(mkt.shape)
        for index, scalar_mkt in _scalar_markets(mkt):
            out[index] = float(instrument.price(scalar_mkt))
        return out


def _greeks(instrument: Instrument, mkt: Market) -> dict[str, Any]:
    """``instrument.greeks(mkt)`` on :data:`GREEK_KEYS`, with the same loop fallback."""
    if mkt.is_scalar:
        raw = instrument.greeks(mkt)
        return {key: float(raw.get(key, 0.0)) for key in GREEK_KEYS}
    try:
        raw = instrument.greeks(mkt)
        return {
            key: np.broadcast_to(np.asarray(raw.get(key, 0.0), dtype=float), mkt.shape)
            for key in GREEK_KEYS
        }
    except (TypeError, ValueError):
        out = {key: np.zeros(mkt.shape) for key in GREEK_KEYS}
        for index, scalar_mkt in _scalar_markets(mkt):
            raw = instrument.greeks(scalar_mkt)
            for key in GREEK_KEYS:
                out[key][index] = float(raw.get(key, 0.0))
        return out


def _zeros(mkt: Market):
    return 0.0 if mkt.is_scalar else np.zeros(mkt.shape)


def _finite(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _shock_vector(name: str, shocks: Any) -> np.ndarray:
    values = np.atleast_1d(np.asarray(shocks, dtype=float))
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a non-empty 1-D sequence of finite numbers")
    return values


# ---------------------------------------------------------------------- #
# Transaction costs
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class TransactionCosts:
    """Simple proportional transaction-cost model (paid on every trade, both ways).

    Parameters
    ----------
    stock_bps : float, default 0.0
        Cost of trading the underlying, in basis points of the traded notional
        (``5`` means 0.05% of ``|quantity| * price``).
    option_pct : float, default 0.0
        Cost of trading any other instrument, as a fraction of the traded
        premium (``0.01`` means 1% of ``|quantity| * |price|``).
    option_vol_spread : float, default 0.0
        Half bid-ask spread of options expressed in VOLATILITY (``0.005`` =
        half a vol point). Options are quoted in vol, so crossing the spread
        costs ``|quantity| * |vega| * option_vol_spread``.

    Notes
    -----
    ``option_pct`` and ``option_vol_spread`` add up when both are set. A
    multi-leg package pays the cost of each of its legs (at model prices): you
    cross one spread per leg, which is why a four-leg condor is expensive to
    trade and why a delta hedge that re-trades the stock every day slowly
    bleeds the ``stock_bps``.
    """

    stock_bps: float = 0.0
    option_pct: float = 0.0
    option_vol_spread: float = 0.0

    def __post_init__(self) -> None:
        for f in fields(self):
            value = _finite(f"TransactionCosts.{f.name}", getattr(self, f.name))
            if value < 0:
                raise ValueError(f"TransactionCosts.{f.name} must be non-negative")
            object.__setattr__(self, f.name, value)

    def cost(
        self, instrument: Instrument, quantity: float, mkt: Market, price: float | None = None
    ) -> float:
        """Cost (always >= 0) of trading ``quantity`` units of ``instrument``.

        ``price`` is the traded unit price (model price when ``None``); it is
        ignored for packages, whose legs are costed one by one at model prices.
        """
        quantity = abs(float(quantity))
        if isinstance(instrument, CompositeInstrument):
            return sum(
                self.cost(leg.instrument, quantity * leg.quantity, mkt)
                for leg in instrument.flatten()
            )
        unit_price = abs(float(instrument.price(mkt)) if price is None else float(price))
        if isinstance(instrument, Underlying):
            return quantity * unit_price * self.stock_bps / 1e4
        cost = quantity * unit_price * self.option_pct
        if self.option_vol_spread > 0:
            cost += quantity * abs(float(instrument.greeks(mkt)["vega"])) * self.option_vol_spread
        return cost

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TransactionCosts":
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"TransactionCosts: unknown field(s) {sorted(unknown)}")
        return cls(**data)


# ---------------------------------------------------------------------- #
# Records
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class Trade:
    """One line of the blotter.

    Parameters
    ----------
    t : float
        Market time of the trade (years).
    instrument : Instrument
        What was traded (a package traded as one ticket stays one line here,
        even when the book stores its legs separately).
    quantity : float
        Signed quantity: positive = bought, negative = sold.
    price : float
        Price per unit of the instrument.
    fees : float, default 0.0
        Total fees and transaction costs paid (always reduces cash).
    note : str, default ""
        Free text ("delta hedge", "open straddle"...).
    """

    t: float
    instrument: Instrument
    quantity: float
    price: float
    fees: float = 0.0
    note: str = ""

    @property
    def cash_flow(self) -> float:
        """Cash impact of the trade: ``-(quantity * price) - fees``."""
        return -self.quantity * self.price - self.fees

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "instrument": self.instrument.to_dict(),
            "quantity": self.quantity,
            "price": self.price,
            "fees": self.fees,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Trade":
        return cls(
            t=float(data["t"]),
            instrument=instrument_from_dict(data["instrument"]),
            quantity=float(data["quantity"]),
            price=float(data["price"]),
            fees=float(data.get("fees", 0.0)),
            note=str(data.get("note", "")),
        )


@dataclass(frozen=True)
class Settlement:
    """Cash paid by a position: settlement at expiry, or a path event before it.

    ``unit_value`` is the amount paid per unit -- the instrument's price at
    expiry (e.g. the intrinsic value of a vanilla), or the cash due at a path
    event (the rebate of a barrier option that has just knocked out). The
    book receives ``cash_flow = quantity * unit_value`` (a short position pays).
    """

    t: float
    instrument: Instrument
    quantity: float
    unit_value: float
    spot: float

    @property
    def cash_flow(self) -> float:
        """Cash received by the book: ``quantity * unit_value``."""
        return self.quantity * self.unit_value + 0.0  # "+ 0.0" turns -0.0 into 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "instrument": self.instrument.to_dict(),
            "quantity": self.quantity,
            "unit_value": self.unit_value,
            "spot": self.spot,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Settlement":
        return cls(
            t=float(data["t"]),
            instrument=instrument_from_dict(data["instrument"]),
            quantity=float(data["quantity"]),
            unit_value=float(data["unit_value"]),
            spot=float(data["spot"]),
        )


# ---------------------------------------------------------------------- #
# The book
# ---------------------------------------------------------------------- #
class Book:
    """A self-financing trading book: netted positions plus a cash account.

    Parameters
    ----------
    name : str, default "book"
        Display name.
    cash : float, default 0.0
        Starting cash. It is remembered as ``initial_cash`` so that
        ``pnl = value - initial_cash``. Cash may go negative (you borrow).
    costs : TransactionCosts, optional
        Cost model applied to every trade (none by default).

    Attributes
    ----------
    cash : float
        Current cash balance.
    initial_cash : float
        Capital put into the book (starting cash plus later deposits).
    trades : list of Trade
        The blotter, in execution order.
    settlements : list of Settlement
        Cash settlements of expired positions.
    interest_accrued, dividends_accrued : float
        Cumulative carry booked by :meth:`accrue` (already included in ``cash``).

    Examples
    --------
    >>> from optionlab.instruments import EuropeanOption
    >>> mkt = Market(spot=100.0, vol=0.2)
    >>> book = Book("demo", cash=1_000.0)
    >>> _ = book.trade(EuropeanOption("call", 100.0, 1.0), 10, mkt)
    >>> round(book.pnl(mkt), 10)          # trading at the model price is P&L neutral
    0.0
    >>> _ = book.hedge_delta(mkt)
    >>> abs(book.greeks(mkt)["delta"]) < 1e-12
    True
    """

    def __init__(
        self, name: str = "book", cash: float = 0.0, costs: TransactionCosts | None = None
    ) -> None:
        if costs is not None and not isinstance(costs, TransactionCosts):
            raise ValueError("costs must be a TransactionCosts instance or None")
        self.name = str(name)
        self.cash = _finite("cash", cash)
        self.initial_cash = self.cash
        self.costs = costs
        self.trades: list[Trade] = []
        self.settlements: list[Settlement] = []
        self.interest_accrued = 0.0
        self.dividends_accrued = 0.0
        self._positions: dict[Instrument, float] = {}

    def __repr__(self) -> str:
        return f"Book(name={self.name!r}, cash={self.cash:.2f}, lines={len(self._positions)})"

    # ------------------------------------------------------------------ #
    # Positions
    # ------------------------------------------------------------------ #
    @property
    def positions(self) -> tuple[Position, ...]:
        """Snapshot of the open lines as :class:`Position` objects (insertion order)."""
        return tuple(Position(inst, qty) for inst, qty in self._positions.items())

    def quantity(self, instrument: Instrument) -> float:
        """Net signed quantity held in ``instrument`` (0 if there is no line)."""
        return self._positions.get(instrument, 0.0)

    def __len__(self) -> int:
        return len(self._positions)

    def __contains__(self, instrument: object) -> bool:
        return instrument in self._positions

    @property
    def is_empty(self) -> bool:
        """True when the book holds no position (cash does not count)."""
        return not self._positions

    def underlying_quantity(self) -> float:
        """Net number of shares held, including shares inside package lines."""
        total = 0.0
        for position in self._leaf_positions():
            if isinstance(position.instrument, Underlying):
                total += position.quantity
        return total

    def _leaf_positions(self) -> list[Position]:
        leaves: list[Position] = []
        for inst, qty in self._positions.items():
            if isinstance(inst, CompositeInstrument):
                leaves.extend(leg.scaled(qty) for leg in inst.flatten())
            else:
                leaves.append(Position(inst, qty))
        return leaves

    def _add(self, instrument: Instrument, quantity: float) -> None:
        net = self._positions.get(instrument, 0.0) + quantity
        if abs(net) <= QUANTITY_TOL:
            self._positions.pop(instrument, None)
        else:
            self._positions[instrument] = net

    # ------------------------------------------------------------------ #
    # Trading
    # ------------------------------------------------------------------ #
    def trade(
        self,
        instrument: Union[Instrument, Position],
        quantity: float,
        mkt: Market,
        price: float | None = None,
        fee: float = 0.0,
        note: str = "",
        flatten: bool = True,
    ) -> Trade:
        """Buy (``quantity > 0``) or sell (``quantity < 0``) an instrument.

        Parameters
        ----------
        instrument : Instrument or Position
            What to trade. A :class:`Position` is accepted for convenience (its
            own quantity multiplies ``quantity``).
        quantity : float
            Signed number of units; must be non-zero.
        mkt : Market
            Scalar market at the time of the trade (``mkt.t`` stamps the blotter).
        price : float, optional
            Unit price. ``None`` (default) trades at the model price
            ``instrument.price(mkt)``, which leaves the book value unchanged.
            Pass your own price to practise "buying cheap / selling rich": the
            difference with the model price is an immediate mark-to-model P&L.
        fee : float, default 0.0
            Fixed fee for this ticket, added to the book's
            :class:`TransactionCosts` (if any).
        note : str, default ""
            Free text stored on the blotter.
        flatten : bool, default True
            A :class:`CompositeInstrument` is stored leg by leg, so its legs
            net against existing lines and expire independently (essential for
            calendar spreads). With ``False`` the package stays a single line.
            Either way the blotter shows ONE trade at the package price.

        Returns
        -------
        Trade
            The blotter record. Cash moves by ``-(quantity * price) - fees``.

        Notes
        -----
        The life of a path-dependent product starts at the trade: the line is
        stored as ``instrument.observe(mkt.spot, mkt.t)``, so a fresh lookback
        starts looking and a fresh Asian starts averaging from the traded
        spot. This does not change its value, but it makes the risk right (on
        a spot ladder a lookback bought at 100 keeps its extreme at 100) and
        the first averaging step exact. The stored instrument then differs
        from the fresh contract: read it back from :attr:`positions` when you
        need it for :meth:`quantity` or :meth:`close_position`. Vanillas,
        digitals and un-hit barriers are stored unchanged.
        """
        _require_scalar(mkt, "Book.trade")
        if isinstance(instrument, Position):
            quantity = _finite("quantity", quantity) * instrument.quantity
            instrument = instrument.instrument
        if not isinstance(instrument, Instrument):
            raise ValueError(f"can only trade an Instrument or a Position, got {instrument!r}")
        quantity = _finite("quantity", quantity)
        if quantity == 0.0:
            raise ValueError("trade quantity must be non-zero")
        fee = _finite("fee", fee)
        if fee < 0:
            raise ValueError("fee must be non-negative")
        if instrument.is_expired(mkt):
            raise ValueError(f"cannot trade {instrument.label!r}: it has already expired")

        unit_price = float(instrument.price(mkt)) if price is None else _finite("price", price)
        fees = fee
        if self.costs is not None:
            fees += self.costs.cost(instrument, quantity, mkt, unit_price)

        record = Trade(float(mkt.t), instrument, quantity, unit_price, fees, str(note))
        self.cash += record.cash_flow
        self.trades.append(record)

        if flatten and isinstance(instrument, CompositeInstrument):
            legs = instrument.flatten()
        else:
            legs = (Position(instrument, 1.0),)
        for leg in legs:
            size = leg.quantity * quantity
            self._add(self._observed(leg.instrument, size, mkt), size)
        return record

    def close_position(
        self,
        instrument: Instrument,
        mkt: Market,
        price: float | None = None,
        fee: float = 0.0,
        note: str = "close",
    ) -> Trade:
        """Trade the opposite of the current line in ``instrument`` (back to flat)."""
        held = self.quantity(instrument)
        if held == 0.0:
            raise ValueError(f"no open position in {getattr(instrument, 'label', instrument)!r}")
        return self.trade(instrument, -held, mkt, price=price, fee=fee, note=note, flatten=False)

    def liquidate(self, mkt: Market, note: str = "liquidate") -> list[Trade]:
        """Close every line at the model price; afterwards ``value == cash``."""
        return [self.close_position(p.instrument, mkt, note=note) for p in self.positions]

    def deposit(self, amount: float) -> None:
        """Add (or withdraw, if negative) capital. P&L is unaffected.

        Both ``cash`` and ``initial_cash`` move, because new capital is not a gain.
        """
        amount = _finite("amount", amount)
        self.cash += amount
        self.initial_cash += amount

    # ------------------------------------------------------------------ #
    # Valuation
    # ------------------------------------------------------------------ #
    def positions_value(self, mkt: Market):
        """Mark-to-model value of all positions, ``sum(quantity * price)`` (cash excluded)."""
        total = _zeros(mkt)
        for inst, qty in self._positions.items():
            total = total + qty * _price(inst, mkt)
        return total

    def value(self, mkt: Market):
        """Net liquidation value: positions marked to model plus cash.

        This is what you would walk away with if you closed everything at the
        model price right now.
        """
        return self.positions_value(mkt) + self.cash

    def pnl(self, mkt: Market):
        """Profit since inception: ``value(mkt) - initial_cash``."""
        return self.value(mkt) - self.initial_cash

    def greeks(self, mkt: Market) -> dict[str, Any]:
        """Aggregated Greeks of the book in RAW units, ``sum(quantity * greek)``.

        Keys are :data:`optionlab.instruments.GREEK_KEYS`; ``"price"`` is the
        positions value (cash has no Greeks and is excluded). Greeks are
        additive across positions, which is the whole point of running a
        book: you hedge the NET exposure, not each trade.
        """
        total = {key: _zeros(mkt) for key in GREEK_KEYS}
        for inst, qty in self._positions.items():
            inst_greeks = _greeks(inst, mkt)
            for key in GREEK_KEYS:
                total[key] = total[key] + qty * inst_greeks[key]
        return total

    def dollar_greeks(self, mkt: Market) -> dict[str, Any]:
        """Greeks expressed in currency for standard market moves.

        Returns
        -------
        dict
            * ``"delta_cash"`` = ``delta * S``: the stock-equivalent exposure in
              currency. A +1% spot move earns about ``delta_cash / 100``.
            * ``"gamma_cash"`` = ``gamma * S**2 / 100``: how much ``delta_cash``
              changes for a +1% spot move. The gamma P&L of a 1% move is
              ``gamma_cash / 200`` (that is ``0.5 * gamma * (0.01 * S)**2``).
            * ``"vega_cash"`` = ``vega / 100``: P&L for +1 volatility point.
            * ``"theta_cash"`` = ``theta / 365``: P&L of one calendar day
              passing with nothing else moving (negative = the book bleeds).
            * ``"rho_cash"`` = ``rho / 100``: P&L for a +1% (100 bp) rate move.
        """
        raw = self.greeks(mkt)
        spot = mkt.spot
        return {
            "delta_cash": raw["delta"] * spot,
            "gamma_cash": raw["gamma"] * spot**2 / 100.0,
            "vega_cash": raw["vega"] / 100.0,
            "theta_cash": raw["theta"] / DAYS_PER_YEAR,
            "rho_cash": raw["rho"] / 100.0,
        }

    def breakeven_move(self, mkt: Market, days: float = 1.0):
        """Relative spot move over ``days`` at which gamma P&L offsets theta.

        Solves ``0.5 * gamma * dS**2 + theta * dt = 0`` and returns ``|dS| / S``.
        Long gamma pays theta: the spot must move MORE than this for the day to
        be profitable; short gamma collects theta and needs the spot to move
        LESS. For a delta-hedged vanilla book with zero rates the result is the
        implied volatility scaled to the period, ``vol * sqrt(days / 365)``.

        Returns ``nan`` when gamma and theta have the same sign (or gamma is
        0): there is then no trade-off between the two.
        """
        raw = self.greeks(mkt)
        gamma = np.asarray(raw["gamma"], dtype=float)
        theta = np.asarray(raw["theta"], dtype=float)
        ratio = np.full(gamma.shape, np.nan)
        np.divide(-2.0 * theta * (days / DAYS_PER_YEAR), gamma, out=ratio, where=gamma != 0)
        ratio = np.where(ratio >= 0, ratio, np.nan)
        out = np.sqrt(ratio) / mkt.spot
        return float(out) if np.ndim(out) == 0 else out

    def snapshot(self, mkt: Market) -> dict[str, float]:
        """One flat record of the book's state (handy to log at every simulation step).

        Keys: ``t, spot, vol, cash, positions_value, value, pnl`` and the RAW
        aggregated ``delta, gamma, vega, theta``.
        """
        _require_scalar(mkt, "Book.snapshot")
        raw = self.greeks(mkt)
        return {
            "t": mkt.t,
            "spot": mkt.spot,
            "vol": mkt.vol,
            "cash": self.cash,
            "positions_value": raw["price"],
            "value": raw["price"] + self.cash,
            "pnl": raw["price"] + self.cash - self.initial_cash,
            "delta": raw["delta"],
            "gamma": raw["gamma"],
            "vega": raw["vega"],
            "theta": raw["theta"],
        }

    def positions_frame(
        self, mkt: Market, trader_units: bool = True, include_cash: bool = False
    ) -> pd.DataFrame:
        """Per-position valuation and risk table, with a final ``TOTAL`` row.

        Parameters
        ----------
        mkt : Market
            Scalar market.
        trader_units : bool, default True
            Greeks in desk units (vega per vol point, theta per day, rho per
            1%...) instead of raw derivatives. ``frame.attrs["units"]`` records
            the choice (``"trader"`` or ``"raw"``).
        include_cash : bool, default False
            Add a ``CASH`` row before the total, so that the total ``value``
            is the net liquidation value instead of the positions value.

        Returns
        -------
        pandas.DataFrame
            Columns ``label, quantity, expiry, unit_price, value`` followed by
            one column per Greek. Greek columns are POSITION Greeks (already
            multiplied by the quantity), so the ``TOTAL`` row is their sum.
        """
        _require_scalar(mkt, "Book.positions_frame")
        greek_names = GREEK_KEYS[1:]
        rows: list[dict[str, Any]] = []
        for inst, qty in self._positions.items():
            inst_greeks = _greeks(inst, mkt)
            position_greeks = {key: qty * inst_greeks[key] for key in greek_names}
            if trader_units:
                position_greeks = bs.to_trader_units(position_greeks)
            rows.append(
                {
                    "label": inst.label,
                    "quantity": qty,
                    "expiry": math.nan if inst.expiry is None else inst.expiry,
                    "unit_price": inst_greeks["price"],
                    "value": qty * inst_greeks["price"],
                    **position_greeks,
                }
            )
        blank = {"quantity": math.nan, "expiry": math.nan, "unit_price": math.nan}
        if include_cash:
            rows.append({"label": "CASH", **blank, "value": self.cash, **dict.fromkeys(greek_names, 0.0)})
        total = {key: sum(row[key] for row in rows) for key in ("value",) + greek_names}
        rows.append({"label": "TOTAL", **blank, **total})
        frame = pd.DataFrame(
            rows, columns=["label", "quantity", "expiry", "unit_price", "value", *greek_names]
        )
        frame.attrs["units"] = "trader" if trader_units else "raw"
        return frame

    def blotter_frame(self) -> pd.DataFrame:
        """The trade blotter as a DataFrame (one row per :class:`Trade`)."""
        columns = ["t", "instrument", "quantity", "price", "fees", "cash_flow", "note"]
        rows = [
            {
                "t": tr.t,
                "instrument": tr.instrument.label,
                "quantity": tr.quantity,
                "price": tr.price,
                "fees": tr.fees,
                "cash_flow": tr.cash_flow,
                "note": tr.note,
            }
            for tr in self.trades
        ]
        return pd.DataFrame(rows, columns=columns)

    def settlements_frame(self) -> pd.DataFrame:
        """Expiry settlements as a DataFrame (one row per :class:`Settlement`)."""
        columns = ["t", "instrument", "quantity", "spot", "unit_value", "cash_flow"]
        rows = [
            {
                "t": s.t,
                "instrument": s.instrument.label,
                "quantity": s.quantity,
                "spot": s.spot,
                "unit_value": s.unit_value,
                "cash_flow": s.cash_flow,
            }
            for s in self.settlements
        ]
        return pd.DataFrame(rows, columns=columns)

    # ------------------------------------------------------------------ #
    # Risk: full-repricing ladders, grids and stress tests
    # ------------------------------------------------------------------ #
    def spot_ladder(
        self,
        mkt: Market,
        shocks: Sequence[float] = DEFAULT_SPOT_SHOCKS,
        trader_units: bool = True,
    ) -> pd.DataFrame:
        """P&L and Greeks of the book under relative spot shocks (full repricing).

        A ladder shows what the Greeks cannot: delta and gamma are local, so
        for a 10% gap the only honest number is the repriced book. It also
        shows how the Greeks THEMSELVES move (a short-gamma book gets longer
        delta as the market falls -- exactly when you do not want it).

        Parameters
        ----------
        mkt : Market
            Scalar base market.
        shocks : sequence of float
            Relative spot moves (``-0.1`` = -10%), each > -1.
        trader_units : bool, default True
            Units of the Greek columns (recorded in ``frame.attrs["units"]``).

        Returns
        -------
        pandas.DataFrame
            Index ``spot_shock``; columns ``spot``, ``value`` (net liquidation
            value), ``pnl`` (change versus the unshocked market, cash being
            unchanged) and one column per Greek. Everything is repriced in one
            vectorised call; instruments that cannot price arrays are looped.
        """
        _require_scalar(mkt, "Book.spot_ladder")
        shocks = _shock_vector("shocks", shocks)
        ladder_mkt = shocked_market(mkt, spot=shocks)
        raw = self.greeks(ladder_mkt)
        risk = {key: raw[key] for key in GREEK_KEYS[1:]}
        if trader_units:
            risk = bs.to_trader_units(risk)
        frame = pd.DataFrame(
            {
                "spot": np.asarray(ladder_mkt.spot),
                "value": raw["price"] + self.cash,
                "pnl": raw["price"] - self.positions_value(mkt),
                **risk,
            },
            index=pd.Index(shocks, name="spot_shock"),
        )
        frame.attrs["units"] = "trader" if trader_units else "raw"
        return frame

    def scenario_grid(
        self,
        mkt: Market,
        spot_shocks: Sequence[float] = DEFAULT_SPOT_SHOCKS,
        vol_shocks: Sequence[float] = DEFAULT_VOL_SHOCKS,
        horizon_days: float = 0.0,
        quantity: str = "pnl",
        trader_units: bool = True,
    ) -> pd.DataFrame:
        """Spot x vol scenario matrix by full repricing.

        Parameters
        ----------
        mkt : Market
            Scalar base market.
        spot_shocks : sequence of float
            Relative spot moves (rows).
        vol_shocks : sequence of float
            Absolute vol moves, ``0.05`` = +5 points (columns). The shocked vol
            is floored at :data:`MIN_VOL`.
        horizon_days : float, default 0.0
            Calendar days that pass before the shock is applied (``t`` moves
            forward). Instruments expiring inside the horizon are worth their
            settlement value at the shocked spot. Carry on the cash account is
            NOT included: this is a repricing of the positions only.
        quantity : str, default "pnl"
            ``"pnl"`` (change in value versus today's unshocked market),
            ``"value"`` (net liquidation value) or any Greek name.
        trader_units : bool, default True
            Units used when ``quantity`` is a Greek.

        Returns
        -------
        pandas.DataFrame
            Index ``spot_shock``, columns ``vol_shock``. Spot and vol are
            shocked together because their joint move is what hurts: a short
            put loses on delta AND on vega in a crash.
        """
        _require_scalar(mkt, "Book.scenario_grid")
        spot_shocks = _shock_vector("spot_shocks", spot_shocks)
        vol_shocks = _shock_vector("vol_shocks", vol_shocks)
        grid_mkt = shocked_market(
            mkt, spot=spot_shocks[:, None], vol=vol_shocks[None, :], days=_finite("horizon_days", horizon_days)
        )
        if quantity == "pnl":
            values = self.positions_value(grid_mkt) - self.positions_value(mkt)
        elif quantity == "value":
            values = self.value(grid_mkt)
        elif quantity in GREEK_KEYS[1:]:
            values = self.greeks(grid_mkt)[quantity]
            if trader_units:
                values = values * bs.TRADER_UNIT_SCALES.get(quantity, 1.0)
        else:
            raise ValueError(
                f"quantity must be 'pnl', 'value' or one of {list(GREEK_KEYS[1:])}, got {quantity!r}"
            )
        values = np.broadcast_to(values, (spot_shocks.size, vol_shocks.size))
        return pd.DataFrame(
            np.array(values),
            index=pd.Index(spot_shocks, name="spot_shock"),
            columns=pd.Index(vol_shocks, name="vol_shock"),
        )

    def stress_tests(
        self, mkt: Market, scenarios: Mapping[str, Mapping[str, float]] | None = None
    ) -> pd.DataFrame:
        """P&L of named stress scenarios by full repricing.

        Parameters
        ----------
        mkt : Market
            Scalar base market.
        scenarios : mapping, optional
            ``{name: {"spot": rel, "vol": abs, "rate": abs, "days": n}}`` (every
            key optional, see :func:`shocked_market`). Defaults to
            :data:`DEFAULT_SCENARIOS`.

        Returns
        -------
        pandas.DataFrame
            Index ``scenario``; columns ``spot_shock, vol_shock, rate_shock,
            days`` (the inputs), ``spot, vol`` (the shocked levels), ``value``
            (net liquidation value) and ``pnl`` (versus the current market;
            carry on cash is not included in time scenarios).

        Notes
        -----
        Greeks answer "what if the market moves a little"; stress tests answer
        "what if it moves a lot, in several dimensions at once". A book that
        looks flat on every Greek can still lose money in a crash scenario.
        """
        _require_scalar(mkt, "Book.stress_tests")
        scenarios = DEFAULT_SCENARIOS if scenarios is None else scenarios
        if not scenarios:
            raise ValueError("scenarios must contain at least one scenario")
        for name, shocks in scenarios.items():
            unknown = set(shocks) - set(_SCENARIO_KEYS)
            if unknown:
                raise ValueError(
                    f"scenario {name!r}: unknown key(s) {sorted(unknown)}; valid keys are {list(_SCENARIO_KEYS)}"
                )
        inputs = {
            key: np.array([float(shocks.get(key, 0.0)) for shocks in scenarios.values()])
            for key in _SCENARIO_KEYS
        }
        stressed = shocked_market(mkt, **inputs)
        positions_value = self.positions_value(stressed)
        return pd.DataFrame(
            {
                "spot_shock": inputs["spot"],
                "vol_shock": inputs["vol"],
                "rate_shock": inputs["rate"],
                "days": inputs["days"],
                "spot": np.asarray(stressed.spot),
                "vol": np.asarray(stressed.vol),
                "value": positions_value + self.cash,
                "pnl": positions_value - self.positions_value(mkt),
            },
            index=pd.Index(list(scenarios), name="scenario"),
        )

    # ------------------------------------------------------------------ #
    # Life cycle
    # ------------------------------------------------------------------ #
    def observe(self, mkt: Market) -> dict[Instrument, Instrument]:
        """Show the current spot to every path-dependent instrument.

        Each line is replaced by ``instrument.observe(mkt.spot, mkt.t)`` with
        its quantity preserved; lines that become identical are netted.
        Instruments whose expiry is strictly in the past are left untouched,
        so a late observation can never change a settlement value.

        Some path events pay cash on the spot: a knock-out barrier option pays
        its rebate when the barrier is touched. That amount
        (``instrument.observation_cash_flow(new_instrument)`` per unit) is
        credited to the cash account and recorded as a :class:`Settlement`, so
        the book value stays continuous through the event. The knocked-out
        line itself stays in the book, worth zero, until its expiry.

        Returns
        -------
        dict
            ``{old instrument: new instrument}`` for the lines that changed
            (e.g. a barrier option that has just knocked out).
        """
        _require_scalar(mkt, "Book.observe")
        replaced: dict[Instrument, Instrument] = {}
        updated: dict[Instrument, float] = {}
        for inst, qty in self._positions.items():
            observed = self._observed(inst, qty, mkt)
            if observed is not inst:
                replaced[inst] = observed
            updated[observed] = updated.get(observed, 0.0) + qty
        self._positions = {i: q for i, q in updated.items() if abs(q) > QUANTITY_TOL}
        return replaced

    def _observed(self, inst: Instrument, qty: float, mkt: Market) -> Instrument:
        """``inst`` after seeing the market (``inst`` itself if nothing changed).

        Books the cash that the observation event pays to a holder of ``qty``
        units. Instruments whose expiry is strictly in the past are not observed.
        """
        spot, t = float(mkt.spot), float(mkt.t)
        if inst.expiry is not None and t > inst.expiry + EXPIRY_TOL:
            return inst
        observed = inst.observe(spot, t)
        if not isinstance(observed, Instrument):
            raise ValueError(f"{inst.label!r}.observe must return an Instrument")
        if observed is inst or observed == inst:
            return inst
        unit_cash = float(inst.observation_cash_flow(observed))
        if unit_cash != 0.0:
            record = Settlement(t, inst, qty, unit_cash, spot)
            self.cash += record.cash_flow
            self.settlements.append(record)
        return observed

    def settle_expired(self, mkt: Market) -> list[Settlement]:
        """Cash-settle every position that has reached its expiry.

        An expired instrument's ``price(mkt)`` is its settlement value (e.g.
        ``max(S - K, 0)`` for a call), so the book receives ``quantity *
        price`` in cash and the line disappears. Because the mark just before
        expiry converges to that same value, the net liquidation value is
        continuous through expiry: settlement converts value into cash, it
        does not create P&L.

        Package lines (``flatten=False``) whose legs expire at different dates
        are split at the first expiry: expired legs are settled and surviving
        legs stay in the book as separate lines.

        Call this at each simulation step AFTER :meth:`observe`; settling late
        would use a spot observed after the expiry date.

        Returns
        -------
        list of Settlement
            The new records (also appended to ``self.settlements``).
        """
        _require_scalar(mkt, "Book.settle_expired")
        survivors: dict[Instrument, float] = {}
        records: list[Settlement] = []
        for inst, qty in self._positions.items():
            units = [Position(inst, qty)]
            if isinstance(inst, CompositeInstrument):
                legs = [leg.scaled(qty) for leg in inst.flatten()]
                if any(leg.is_expired(mkt) for leg in legs):
                    units = legs
            for unit in units:
                if unit.is_expired(mkt):
                    records.append(
                        Settlement(
                            t=float(mkt.t),
                            instrument=unit.instrument,
                            quantity=unit.quantity,
                            unit_value=float(unit.instrument.price(mkt)),
                            spot=float(mkt.spot),
                        )
                    )
                else:
                    survivors[unit.instrument] = survivors.get(unit.instrument, 0.0) + unit.quantity
        self._positions = {i: q for i, q in survivors.items() if abs(q) > QUANTITY_TOL}
        self.cash += sum(record.cash_flow for record in records)
        self.settlements.extend(records)
        return records

    def accrue(self, mkt: Market, dt: float) -> dict[str, float]:
        """Book the carry earned over a period of ``dt`` years.

        Conventions (kept deliberately simple and explicit):

        * **Interest**: the cash balance at the start of the period compounds
          continuously at ``mkt.rate``: ``cash *= exp(rate * dt)``. Negative
          cash pays interest at the same rate (borrowing = lending rate).
        * **Dividends**: the underlying pays a continuous yield ``mkt.div``.
          Holding ``n`` shares over ``dt`` pays ``n * mkt.spot * mkt.div * dt``
          in cash; a short position PAYS that amount. Shares inside package
          lines count. Dividends received do not earn interest within the period.
        * Pass the market that prevailed over the period (a simulator uses the
          market at the START of the step, before the spot moves).

        These two cash flows are exactly what the Black-Scholes replication
        argument assumes, so a continuously delta-hedged option book that
        accrues this way replicates the option.

        Returns
        -------
        dict
            ``{"interest": ..., "dividends": ...}`` booked by this call (the
            running totals are ``interest_accrued`` and ``dividends_accrued``).
        """
        _require_scalar(mkt, "Book.accrue")
        dt = _finite("dt", dt)
        if dt < 0:
            raise ValueError("dt must be non-negative")
        interest = self.cash * math.expm1(float(mkt.rate) * dt)
        dividends = self.underlying_quantity() * float(mkt.spot) * float(mkt.div) * dt
        self.cash += interest + dividends
        self.interest_accrued += interest
        self.dividends_accrued += dividends
        return {"interest": interest, "dividends": dividends}

    # ------------------------------------------------------------------ #
    # Hedging
    # ------------------------------------------------------------------ #
    def delta_hedge_trade_size(self, mkt: Market, target_delta: float = 0.0) -> float:
        """Number of shares to BUY (negative = sell) to bring the book delta to the target.

        The stock has a delta of exactly 1 and no other Greek, so the answer
        is simply ``target_delta - book_delta``.
        """
        _require_scalar(mkt, "Book.delta_hedge_trade_size")
        return _finite("target_delta", target_delta) - self.greeks(mkt)["delta"]

    def hedge_delta(
        self,
        mkt: Market,
        target_delta: float = 0.0,
        min_trade: float = 0.0,
        fee: float = 0.0,
        note: str = "delta hedge",
    ) -> Trade | None:
        """Trade the underlying so that the book delta equals ``target_delta``.

        Parameters
        ----------
        min_trade : float, default 0.0
            Do nothing (return ``None``) when the required trade is not larger
            than this many shares. A no-trade band is the practical answer to
            transaction costs: hedging less often saves costs but leaves more
            delta risk.
        """
        size = self.delta_hedge_trade_size(mkt, target_delta)
        if abs(size) <= max(float(min_trade), QUANTITY_TOL):
            return None
        return self.trade(Underlying(), size, mkt, fee=fee, note=note)

    def hedge_size(
        self, mkt: Market, greek: str, hedge_instrument: Instrument, target: float = 0.0
    ) -> float:
        """Quantity of ``hedge_instrument`` that brings one book Greek to ``target``.

        ``quantity = (target - book_greek) / instrument_greek``. For example
        ``book.hedge_size(mkt, "vega", some_option)`` is the vega hedge. Note
        that the hedge brings its OWN other Greeks (an option used to kill
        vega also adds delta and gamma) -- see :meth:`solve_hedge` to handle
        several Greeks at once.
        """
        return self.solve_hedge(mkt, [hedge_instrument], {greek: target})[hedge_instrument]

    def neutralise(
        self,
        mkt: Market,
        greek: str,
        with_instrument: Instrument,
        target: float = 0.0,
        fee: float = 0.0,
        note: str | None = None,
    ) -> Trade | None:
        """Execute :meth:`hedge_size` at the model price (``None`` if already on target)."""
        size = self.hedge_size(mkt, greek, with_instrument, target)
        if abs(size) <= QUANTITY_TOL:
            return None
        return self.trade(with_instrument, size, mkt, fee=fee, note=note or f"{greek} hedge")

    def solve_hedge(
        self,
        mkt: Market,
        instruments: Sequence[Instrument],
        targets: Union[Mapping[str, float], Sequence[str]] = ("delta", "gamma"),
    ) -> dict[Instrument, float]:
        """Quantities of ``n`` hedge instruments that set ``n`` Greeks to their targets.

        Solves the linear system ``sum_j q_j * greek_i(instrument_j) = target_i
        - book_greek_i`` (RAW units). The classic cases:

        * delta-gamma: ``book.solve_hedge(mkt, [Underlying(), option])`` -- the
          option is sized to kill the gamma, then the stock mops up the delta
          (of the book AND of the hedge option);
        * delta-vega: ``targets=("delta", "vega")``;
        * delta-gamma-vega: three instruments, e.g. the stock and two options
          with different expiries (gamma lives in short-dated options, vega in
          long-dated ones, which is what makes the system solvable).

        Parameters
        ----------
        mkt : Market
            Scalar market.
        instruments : sequence of Instrument
            Distinct hedge instruments, as many as there are targets.
        targets : mapping or sequence of str
            ``{greek: target}``, or Greek names to be brought to 0.

        Returns
        -------
        dict
            ``{instrument: quantity to trade}`` (nothing is executed; see
            :meth:`hedge`).

        Raises
        ------
        ValueError
            If the system is not square or is singular (e.g. trying to hedge
            gamma with the underlying alone, which has none).
        """
        _require_scalar(mkt, "Book.solve_hedge")
        instruments = list(instruments)
        target_map = dict(targets) if isinstance(targets, Mapping) else {g: 0.0 for g in targets}
        unknown = [g for g in target_map if g not in GREEK_KEYS[1:]]
        if unknown:
            raise ValueError(f"unknown Greek(s) {unknown}; valid names are {list(GREEK_KEYS[1:])}")
        if not instruments or len(instruments) != len(target_map):
            raise ValueError(
                f"need as many hedge instruments as targets, got {len(instruments)} "
                f"instrument(s) for {len(target_map)} target(s)"
            )
        if len(set(instruments)) != len(instruments):
            raise ValueError("hedge instruments must be distinct")
        for inst in instruments:
            if not isinstance(inst, Instrument):
                raise ValueError(f"hedge instruments must be Instruments, got {inst!r}")
            if inst.is_expired(mkt):
                raise ValueError(f"cannot hedge with {inst.label!r}: it has already expired")

        book_greeks = self.greeks(mkt)
        hedge_greeks = [_greeks(inst, mkt) for inst in instruments]
        matrix = np.array([[g[greek] for g in hedge_greeks] for greek in target_map])
        gaps = np.array([_finite(f"target {k}", v) - book_greeks[k] for k, v in target_map.items()])
        row_scale = np.abs(matrix).max(axis=1)
        if np.any(row_scale == 0) or np.linalg.matrix_rank(matrix / row_scale[:, None]) < len(instruments):
            raise ValueError(
                "the hedge cannot be solved: these instruments do not span the requested Greeks "
                "(the underlying has no gamma or vega; two options with the same expiry have "
                "proportional gamma and vega)"
            )
        quantities = np.linalg.solve(matrix, gaps)
        return {inst: float(q) for inst, q in zip(instruments, quantities)}

    def hedge(
        self,
        mkt: Market,
        instruments: Sequence[Instrument],
        targets: Union[Mapping[str, float], Sequence[str]] = ("delta", "gamma"),
        fee: float = 0.0,
        note: str = "hedge",
    ) -> list[Trade]:
        """Execute the solution of :meth:`solve_hedge` at model prices (``fee`` per ticket)."""
        sizes = self.solve_hedge(mkt, instruments, targets)
        return [
            self.trade(inst, size, mkt, fee=fee, note=note)
            for inst, size in sizes.items()
            if abs(size) > QUANTITY_TOL
        ]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def copy(self, name: str | None = None) -> "Book":
        """Independent copy (instruments are immutable, so they are shared safely).

        Useful for what-if analysis: copy the book, try a trade on the copy,
        compare the risk, and only then trade for real.
        """
        clone = Book(self.name if name is None else name, self.cash, self.costs)
        clone.initial_cash = self.initial_cash
        clone.trades = list(self.trades)
        clone.settlements = list(self.settlements)
        clone.interest_accrued = self.interest_accrued
        clone.dividends_accrued = self.dividends_accrued
        clone._positions = dict(self._positions)
        return clone

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict holding the complete state of the book."""
        return {
            "name": self.name,
            "cash": self.cash,
            "initial_cash": self.initial_cash,
            "interest_accrued": self.interest_accrued,
            "dividends_accrued": self.dividends_accrued,
            "costs": None if self.costs is None else self.costs.to_dict(),
            "positions": [p.to_dict() for p in self.positions],
            "trades": [tr.to_dict() for tr in self.trades],
            "settlements": [s.to_dict() for s in self.settlements],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Book":
        """Rebuild a book from :meth:`to_dict` output.

        Custom instrument classes must be imported (hence registered) first.
        """
        if not isinstance(data, Mapping):
            raise ValueError("Book.from_dict needs a mapping")
        known = {
            "name", "cash", "initial_cash", "interest_accrued", "dividends_accrued",
            "costs", "positions", "trades", "settlements",
        }
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Book.from_dict: unknown key(s) {sorted(unknown)}")
        costs = data.get("costs")
        book = cls(
            name=data.get("name", "book"),
            cash=data.get("cash", 0.0),
            costs=None if costs is None else TransactionCosts.from_dict(costs),
        )
        book.initial_cash = _finite("initial_cash", data.get("initial_cash", book.cash))
        book.interest_accrued = _finite("interest_accrued", data.get("interest_accrued", 0.0))
        book.dividends_accrued = _finite("dividends_accrued", data.get("dividends_accrued", 0.0))
        for entry in data.get("positions", ()):
            position = Position.from_dict(entry)
            book._add(position.instrument, position.quantity)
        book.trades = [Trade.from_dict(entry) for entry in data.get("trades", ())]
        book.settlements = [Settlement.from_dict(entry) for entry in data.get("settlements", ())]
        return book

    def to_json(self, path: Union[str, Path, None] = None, indent: int | None = 2) -> str:
        """Serialise to JSON; also write the text to ``path`` when one is given."""
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, source: Union[str, Path]) -> "Book":
        """Load a book from a JSON string or from the path of a JSON file."""
        if isinstance(source, Path) or not str(source).lstrip().startswith("{"):
            source = Path(source).read_text(encoding="utf-8")
        try:
            data = json.loads(source)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not a valid book JSON document: {exc}") from exc
        return cls.from_dict(data)
