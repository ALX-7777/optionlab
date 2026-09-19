"""Instrument interface, vanilla products, positions and multi-leg composites.

Design
------
* Every concrete instrument is a **frozen dataclass**: immutable, hashable and
  usable as a dict key. Anything the product has "seen" along the path (a
  barrier hit, a running maximum, a running average...) is stored as a field,
  and :meth:`Instrument.observe` returns a *new* instrument with updated state.
* Instruments carry an **absolute** ``expiry`` (in years); the time to maturity
  is ``expiry - mkt.t``. At or after expiry, ``price`` returns the settlement
  value given the current spot and the stored path state.
* ``price`` and ``greeks`` are vectorised over the :class:`~optionlab.market.Market`
  fields for every closed-form product.
* Greeks are RAW derivatives (see :mod:`optionlab.black_scholes`).
* Instruments serialise to plain dicts (``to_dict``) and come back through
  :func:`instrument_from_dict`, which looks the class up in
  :data:`INSTRUMENT_REGISTRY` (filled by the :func:`register_instrument`
  decorator).

Writing a new product means subclassing :class:`Instrument`, implementing
``price`` and ``payoff`` (plus ``observe`` / ``path_payoff`` if it is
path-dependent) and decorating the class with ``@register_instrument``. Greeks
then come for free from the numerical engine.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Mapping

import numpy as np

from . import black_scholes as bs
from .market import Market
from .numerical import numerical_greeks

__all__ = [
    "GREEK_KEYS",
    "EXPIRY_TOL",
    "INSTRUMENT_REGISTRY",
    "register_instrument",
    "instrument_from_dict",
    "slice_paths_to_expiry",
    "Instrument",
    "Underlying",
    "EuropeanOption",
    "Position",
    "CompositeInstrument",
]

#: Keys returned by every ``greeks(mkt)`` in the library, in order.
GREEK_KEYS: tuple[str, ...] = ("price",) + bs.GREEK_NAMES

#: Tolerance (in years, about 3 ms) used when comparing the clock with an expiry,
#: so that a simulation stepping ``t += dt`` still lands "on" the expiry date.
EXPIRY_TOL: float = 1e-10

#: Maps a class name to the instrument class, for deserialisation.
INSTRUMENT_REGISTRY: dict[str, type["Instrument"]] = {}


# ---------------------------------------------------------------------- #
# Registry and (de)serialisation
# ---------------------------------------------------------------------- #
def register_instrument(cls: type["Instrument"]) -> type["Instrument"]:
    """Class decorator adding an instrument class to :data:`INSTRUMENT_REGISTRY`.

    The key is the class name, which is also the ``"type"`` entry written by
    :meth:`Instrument.to_dict`. Registering a class with an existing name
    replaces the previous entry (this keeps module reloads harmless).
    """
    if not (isinstance(cls, type) and issubclass(cls, Instrument)):
        raise ValueError("register_instrument can only decorate Instrument subclasses")
    INSTRUMENT_REGISTRY[cls.__name__] = cls
    return cls


def instrument_from_dict(data: Mapping[str, Any]) -> "Instrument":
    """Rebuild an instrument from the dict produced by :meth:`Instrument.to_dict`.

    Raises
    ------
    ValueError
        If ``data`` has no ``"type"`` entry or the type is not registered
        (usually because the module defining it has not been imported).
    """
    if not isinstance(data, Mapping) or "type" not in data:
        raise ValueError("an instrument dict must be a mapping with a 'type' entry")
    type_name = data["type"]
    cls = INSTRUMENT_REGISTRY.get(type_name)
    if cls is None:
        raise ValueError(
            f"Unknown instrument type {type_name!r}. Registered types: "
            f"{sorted(INSTRUMENT_REGISTRY)}. Import the module defining it first."
        )
    return cls.from_dict(data)


def _encode(value: Any) -> Any:
    """Turn a field value into something ``json.dumps`` accepts."""
    if isinstance(value, (Instrument, Position)):
        return value.to_dict()
    if isinstance(value, (tuple, list)):
        return [_encode(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _decode(value: Any) -> Any:
    """Inverse of :func:`_encode` (lists come back as tuples, to stay hashable)."""
    if isinstance(value, Mapping) and "type" in value:
        if value["type"] == "Position":
            return Position.from_dict(value)
        return instrument_from_dict(value)
    if isinstance(value, (list, tuple)):
        return tuple(_decode(v) for v in value)
    return value


def _init_kwargs(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    """Decode the dataclass constructor arguments stored in ``data``."""
    names = {f.name for f in fields(cls) if f.init}
    payload = {k: v for k, v in data.items() if k != "type"}
    unknown = set(payload) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown field(s) {sorted(unknown)} in serialised data")
    return {k: _decode(v) for k, v in payload.items()}


def slice_paths_to_expiry(
    paths: np.ndarray, times: np.ndarray, expiry: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """Cut simulated paths at an instrument's expiry.

    Parameters
    ----------
    paths : ndarray, shape (n_paths, n_times)
    times : ndarray, shape (n_times,)
        Absolute times of the path columns (increasing).
    expiry : float or None
        Absolute expiry; ``None`` keeps everything.

    Returns
    -------
    (paths, times)
        Columns with ``times <= expiry`` (at least the first column).
    """
    paths = np.atleast_2d(np.asarray(paths, dtype=float))
    times = np.asarray(times, dtype=float)
    if expiry is None:
        return paths, times
    n_keep = max(int(np.searchsorted(times, expiry + EXPIRY_TOL, side="right")), 1)
    return paths[:, :n_keep], times[:n_keep]


def _full(mkt: Market, value: float):
    """``value`` broadcast to the market shape (a float for a scalar market)."""
    shape = mkt.shape
    return float(value) if shape == () else np.full(shape, float(value))


# ---------------------------------------------------------------------- #
# Abstract base class
# ---------------------------------------------------------------------- #
class Instrument(ABC):
    """Interface shared by every tradable product.

    Subclasses are frozen dataclasses and must provide an ``expiry`` attribute
    (a dataclass field or a property): the absolute expiry in years, or
    ``None`` for products that never expire (the underlying).
    """

    expiry: float | None

    # --- valuation ------------------------------------------------------
    @abstractmethod
    def price(self, mkt: Market):
        """Value of ONE unit of the instrument in market ``mkt``.

        Vectorised over the market fields. At or after expiry this is the
        settlement value given the current spot and the stored path state.
        """

    @abstractmethod
    def payoff(self, spot_T):
        """Terminal payoff as a function of the terminal spot.

        For path-dependent products this is the payoff *if the path state
        stored on the instrument no longer changes* (e.g. a knock-out that has
        not knocked yet is assumed to survive).
        """

    def greeks(self, mkt: Market) -> dict[str, Any]:
        """Price and Greeks in RAW units, keys :data:`GREEK_KEYS`.

        The default implementation bumps and reprices
        (:func:`optionlab.numerical.numerical_greeks`); closed-form products
        override it with analytic formulas.
        """
        return numerical_greeks(self, mkt)

    # --- path dependence --------------------------------------------------
    def observe(self, spot: float, t: float) -> "Instrument":
        """Return the instrument after observing ``spot`` at time ``t``.

        Path-dependent products return a NEW instrument with updated state
        (barrier flag, running extremum, running average...). Path-independent
        products return ``self``.
        """
        return self

    def observation_cash_flow(self, observed: "Instrument") -> float:
        """Cash paid to the holder of ONE unit when :meth:`observe` turned ``self`` into ``observed``.

        Most path events only change the state of the product, but some pay
        cash on the spot -- the rebate of a knock-out barrier option is due the
        moment the barrier is touched. Whoever runs the life cycle (the
        :class:`~optionlab.book.Book`) must book this amount when it replaces
        ``self`` by ``observed``; otherwise the rebate would silently vanish
        from the P&L. The default is 0.
        """
        return 0.0

    def path_payoff(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Payoff for each simulated path, shape ``(n_paths,)``.

        Parameters
        ----------
        paths : ndarray, shape (n_paths, n_times)
            Spot paths starting at "now" (first column = current spot) and
            ending at this instrument's expiry.
        times : ndarray, shape (n_times,)
            ABSOLUTE times of the columns (``times[0]`` is the current time,
            ``times[-1]`` the expiry).

        Notes
        -----
        The path covers only the future. Whatever happened before ``times[0]``
        already lives in the instrument's fields (that is what ``observe`` is
        for), so implementations must combine stored state and simulated path.
        The default is path-independent: ``payoff(paths[:, -1])``.
        """
        paths = np.atleast_2d(np.asarray(paths, dtype=float))
        return np.asarray(self.payoff(paths[:, -1]), dtype=float)

    # --- life cycle -------------------------------------------------------
    def tau(self, mkt: Market):
        """Time to maturity ``expiry - mkt.t`` (``inf`` if the product never expires)."""
        if self.expiry is None:
            return _full(mkt, math.inf)
        return self.expiry - mkt.t

    def is_expired(self, mkt: Market):
        """True once the market clock has reached the expiry (array if ``mkt.t`` is)."""
        if self.expiry is None:
            return False
        expired = np.asarray(mkt.t) >= self.expiry - EXPIRY_TOL
        return bool(expired) if expired.ndim == 0 else expired

    @property
    def label(self) -> str:
        """Short human-readable name used in tables and legends."""
        return type(self).__name__

    # --- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict: ``{"type": <ClassName>, **fields}``."""
        if not is_dataclass(self):
            raise ValueError(f"{type(self).__name__} must be a dataclass to use the default to_dict")
        out: dict[str, Any] = {"type": type(self).__name__}
        for f in fields(self):
            if f.init:
                out[f.name] = _encode(getattr(self, f.name))
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Instrument":
        """Build an instance from :meth:`to_dict` output (``"type"`` is ignored)."""
        try:
            return cls(**_init_kwargs(cls, data))
        except TypeError as exc:
            raise ValueError(f"cannot build {cls.__name__} from {dict(data)!r}: {exc}") from exc

    # --- sugar: 2 * option -> Position --------------------------------------
    def __mul__(self, quantity: float) -> "Position":
        if isinstance(quantity, (int, float, np.integer, np.floating)):
            return Position(self, float(quantity))
        return NotImplemented

    __rmul__ = __mul__

    def __neg__(self) -> "Position":
        return Position(self, -1.0)


# ---------------------------------------------------------------------- #
# Concrete vanilla instruments
# ---------------------------------------------------------------------- #
@register_instrument
@dataclass(frozen=True)
class Underlying(Instrument):
    """One unit of the underlying stock.

    Price is the spot, delta is 1 and every other Greek is 0: the stock is the
    pure delta instrument, which is why it is the natural hedge for the delta
    of an option book (it adds no gamma, vega or theta).
    """

    @property
    def expiry(self) -> None:
        return None

    def price(self, mkt: Market):
        return mkt.spot + _full(mkt, 0.0)

    def greeks(self, mkt: Market) -> dict[str, Any]:
        out = {key: _full(mkt, 0.0) for key in GREEK_KEYS}
        out["price"] = self.price(mkt)
        out["delta"] = _full(mkt, 1.0)
        return out

    def payoff(self, spot_T):
        return np.asarray(spot_T, dtype=float)

    @property
    def label(self) -> str:
        return "Underlying"


@register_instrument
@dataclass(frozen=True)
class EuropeanOption(Instrument):
    """European call or put, priced with Black-Scholes-Merton.

    Parameters
    ----------
    option_type : {"call", "put"}
        Case-insensitive; stored lower-case.
    strike : float
        Strike price (> 0).
    expiry : float
        ABSOLUTE expiry time in years (time to maturity is ``expiry - mkt.t``).

    Notes
    -----
    A call is the right to buy at the strike: limited loss (the premium),
    unlimited upside. A put is the right to sell: insurance against a fall.

    Examples
    --------
    >>> call = EuropeanOption("call", 100.0, 1.0)
    >>> round(call.price(Market(spot=100.0, vol=0.2, rate=0.05)), 4)
    10.4506
    """

    option_type: str
    strike: float
    expiry: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_type", bs.normalize_option_type(self.option_type))
        try:
            strike, expiry = float(self.strike), float(self.expiry)
        except (TypeError, ValueError) as exc:
            raise ValueError("EuropeanOption strike and expiry must be numbers") from exc
        if not (math.isfinite(strike) and strike > 0):
            raise ValueError(f"strike must be a finite positive number, got {self.strike!r}")
        if not math.isfinite(expiry):
            raise ValueError(f"expiry must be finite, got {self.expiry!r}")
        object.__setattr__(self, "strike", strike)
        object.__setattr__(self, "expiry", expiry)

    def _args(self, mkt: Market) -> tuple:
        return (mkt.spot, self.strike, self.expiry - mkt.t, mkt.vol, mkt.rate, mkt.div, self.option_type)

    def price(self, mkt: Market):
        return bs.price(*self._args(mkt))

    def greeks(self, mkt: Market) -> dict[str, Any]:
        """Analytic price and Greeks (RAW units), keys :data:`GREEK_KEYS`."""
        return bs.greeks(*self._args(mkt))

    def payoff(self, spot_T):
        return bs.intrinsic_value(np.asarray(spot_T, dtype=float), self.strike, self.option_type)

    def implied_vol(self, price, mkt: Market):
        """Volatility at which the model matches ``price`` (``mkt.vol`` is ignored)."""
        return bs.implied_vol(
            price, mkt.spot, self.strike, self.expiry - mkt.t, mkt.rate, mkt.div, self.option_type
        )

    @property
    def is_call(self) -> bool:
        return self.option_type == "call"

    @property
    def label(self) -> str:
        return f"{'C' if self.is_call else 'P'} {self.strike:g} T={self.expiry:.2f}"


# ---------------------------------------------------------------------- #
# Position
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class Position:
    """A signed quantity of an instrument.

    Parameters
    ----------
    instrument : Instrument
    quantity : float
        Positive = long, negative = short, in units of the underlying
        (contract multiplier 1).

    Notes
    -----
    All valuation methods mirror the instrument's and are simply scaled by
    ``quantity``; shorting an option flips the sign of every Greek (short
    gamma, short vega, *long* theta).
    """

    instrument: Instrument
    quantity: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.instrument, Instrument):
            raise ValueError(f"Position.instrument must be an Instrument, got {self.instrument!r}")
        try:
            quantity = float(self.quantity)
        except (TypeError, ValueError) as exc:
            raise ValueError("Position.quantity must be a number") from exc
        if not math.isfinite(quantity):
            raise ValueError("Position.quantity must be finite")
        object.__setattr__(self, "quantity", quantity)

    @property
    def expiry(self) -> float | None:
        return self.instrument.expiry

    @property
    def label(self) -> str:
        return f"{self.quantity:+g} x {self.instrument.label}"

    def price(self, mkt: Market):
        """Market value of the position: ``quantity * instrument.price(mkt)``."""
        return self.quantity * self.instrument.price(mkt)

    def greeks(self, mkt: Market) -> dict[str, Any]:
        """Position Greeks (instrument Greeks times ``quantity``; ``"price"`` is the value)."""
        return {k: self.quantity * v for k, v in self.instrument.greeks(mkt).items()}

    def payoff(self, spot_T):
        return self.quantity * np.asarray(self.instrument.payoff(spot_T), dtype=float)

    def path_payoff(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        return self.quantity * self.instrument.path_payoff(paths, times)

    def observe(self, spot: float, t: float) -> "Position":
        observed = self.instrument.observe(spot, t)
        return self if observed is self.instrument else Position(observed, self.quantity)

    def is_expired(self, mkt: Market):
        return self.instrument.is_expired(mkt)

    def scaled(self, k: float) -> "Position":
        """Same instrument, quantity multiplied by ``k`` (``k = -1`` reverses the trade)."""
        return Position(self.instrument, self.quantity * k)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "Position",
            "instrument": self.instrument.to_dict(),
            "quantity": self.quantity,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Position":
        if "instrument" not in data:
            raise ValueError("a Position dict needs an 'instrument' entry")
        return cls(instrument_from_dict(data["instrument"]), data.get("quantity", 1.0))


def _as_position(leg: Any) -> Position:
    if isinstance(leg, Position):
        return leg
    if isinstance(leg, Instrument):
        return Position(leg, 1.0)
    if isinstance(leg, (tuple, list)) and len(leg) == 2 and isinstance(leg[0], Instrument):
        return Position(leg[0], leg[1])
    raise ValueError(
        f"a composite leg must be a Position, an Instrument or an (Instrument, quantity) pair, got {leg!r}"
    )


# ---------------------------------------------------------------------- #
# Composite (multi-leg) instrument
# ---------------------------------------------------------------------- #
@register_instrument
@dataclass(frozen=True)
class CompositeInstrument(Instrument):
    """A named package of positions that behaves as a single instrument.

    Spreads, straddles, butterflies, collars... are all composites: their
    price, Greeks and payoff are the quantity-weighted sums of their legs.

    Parameters
    ----------
    name : str
        Display name (e.g. ``"Bull call spread 95/105"``).
    legs : sequence
        Each leg is a :class:`Position`, an :class:`Instrument` (quantity 1)
        or an ``(instrument, quantity)`` pair. Legs may themselves be
        composites and may have different expiries (calendar spreads): an
        expired leg contributes its settlement value at the current spot.

    Examples
    --------
    >>> straddle = CompositeInstrument("Straddle", (
    ...     Position(EuropeanOption("call", 100, 1.0), 1),
    ...     Position(EuropeanOption("put", 100, 1.0), 1)))
    >>> straddle.payoff([90.0, 100.0, 120.0]).tolist()
    [10.0, 0.0, 20.0]
    """

    name: str
    legs: tuple[Position, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError("CompositeInstrument.name must be a string")
        if isinstance(self.legs, (Position, Instrument)):
            raw_legs: tuple = (self.legs,)
        else:
            raw_legs = tuple(self.legs)
        legs = tuple(_as_position(leg) for leg in raw_legs)
        if not legs:
            raise ValueError("a CompositeInstrument needs at least one leg")
        object.__setattr__(self, "legs", legs)

    # --- expiries ---------------------------------------------------------
    @property
    def expiry(self) -> float | None:
        """Latest leg expiry (``None`` if no leg ever expires)."""
        expiries = [leg.expiry for leg in self.legs if leg.expiry is not None]
        return max(expiries) if expiries else None

    @property
    def first_expiry(self) -> float | None:
        """Earliest leg expiry -- the natural horizon of a calendar spread."""
        expiries = [leg.expiry for leg in self.legs if leg.expiry is not None]
        return min(expiries) if expiries else None

    # --- valuation ----------------------------------------------------------
    def price(self, mkt: Market):
        return sum(leg.price(mkt) for leg in self.legs)

    def greeks(self, mkt: Market) -> dict[str, Any]:
        """Sum of leg Greeks (keys common to every leg, in :data:`GREEK_KEYS` order first)."""
        leg_greeks = [leg.greeks(mkt) for leg in self.legs]
        common = set(leg_greeks[0]).intersection(*leg_greeks[1:])
        ordered = [k for k in GREEK_KEYS if k in common] + sorted(common - set(GREEK_KEYS))
        return {k: sum(g[k] for g in leg_greeks) for k in ordered}

    def payoff(self, spot_T):
        """Sum of leg payoffs, every leg being settled at the same terminal spot.

        For a calendar spread this is only the *long-dated* picture; the
        classic calendar "tent" is the package **price** at the first expiry:
        ``composite.price(mkt.bumped(spot=grid, t=composite.first_expiry))``.
        """
        spot_T = np.asarray(spot_T, dtype=float)
        return sum(leg.payoff(spot_T) for leg in self.legs)

    def path_payoff(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Sum of leg payoffs along each path, each leg read at its own expiry.

        Cash flows paid at different dates are added without reinvestment;
        :func:`optionlab.monte_carlo.mc_price` discounts each leg from its own
        expiry instead of calling this method.
        """
        total = np.zeros(np.atleast_2d(paths).shape[0])
        for leg in self.legs:
            leg_paths, leg_times = slice_paths_to_expiry(paths, times, leg.expiry)
            total = total + leg.path_payoff(leg_paths, leg_times)
        return total

    def observe(self, spot: float, t: float) -> "CompositeInstrument":
        """Observe every leg; subclasses (extra dataclass fields) keep their type and metadata."""
        legs = tuple(leg.observe(spot, t) for leg in self.legs)
        if all(new is old for new, old in zip(legs, self.legs)):
            return self
        return replace(self, legs=legs)

    def observation_cash_flow(self, observed: "Instrument") -> float:
        """Sum of the legs' observation cash flows (``observed`` comes from :meth:`observe`)."""
        if not isinstance(observed, CompositeInstrument) or len(observed.legs) != len(self.legs):
            return 0.0
        return sum(
            old.quantity * old.instrument.observation_cash_flow(new.instrument)
            for old, new in zip(self.legs, observed.legs)
        )

    # --- structure ------------------------------------------------------------
    def flatten(self, merge: bool = False) -> tuple[Position, ...]:
        """Elementary positions, nested composites expanded recursively.

        Quantities are multiplied down the tree, so ``2 x (1 x call - 1 x put)``
        flattens to ``+2 call, -2 put``.

        Parameters
        ----------
        merge : bool, default False
            If True, positions in the same instrument are netted (first
            appearance order is kept; legs netting to zero are dropped).
        """
        flat: list[Position] = []
        for leg in self.legs:
            if isinstance(leg.instrument, CompositeInstrument):
                flat.extend(p.scaled(leg.quantity) for p in leg.instrument.flatten())
            else:
                flat.append(leg)
        if not merge:
            return tuple(flat)
        netted: dict[Instrument, float] = {}
        for p in flat:
            netted[p.instrument] = netted.get(p.instrument, 0.0) + p.quantity
        return tuple(Position(inst, qty) for inst, qty in netted.items() if qty != 0.0)

    def scaled(self, k: float) -> "CompositeInstrument":
        """Same package (same type and metadata) with every leg quantity multiplied by ``k``."""
        return replace(self, legs=tuple(leg.scaled(k) for leg in self.legs))

    @property
    def label(self) -> str:
        return self.name
