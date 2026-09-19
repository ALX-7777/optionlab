"""Predefined option strategies: factories, a UI-friendly registry and analytics.

A *strategy* is a named package of vanilla legs (calls, puts, the stock) that
expresses a market view: direction (bullish / bearish), volatility (long /
short vol), income or protection. This module provides

* :class:`Strategy` -- a :class:`~optionlab.instruments.CompositeInstrument`
  with teaching metadata (``description``, ``view`` tags, registry ``key``).
  It is a regular instrument: it prices, has Greeks, serialises and can be put
  in a book like any other product.
* one factory per classic strategy (``bull_call_spread``, ``iron_condor``,
  ``calendar_spread``...), each validating its strikes and expiries;
* :data:`STRATEGY_REGISTRY` -- one :class:`StrategySpec` per factory, with a
  parameter schema whose defaults are expressed RELATIVE to the spot, so that
  a user interface can build its input widgets automatically;
* analytics that work on ANY instrument: :func:`net_premium`,
  :func:`payoff_at_expiry`, :func:`pnl_at_expiry`, :func:`breakevens`,
  :func:`max_profit`, :func:`max_loss`, :func:`probability_of_profit`,
  :func:`legs_table` and :func:`summary`.

Conventions
-----------
* Quantities are signed (+ long / - short), one option is on ONE unit of the
  underlying. The ``quantity`` argument of a factory scales every leg.
* A premium is a cost: **debit > 0** (you pay), **credit < 0** (you receive).
* "At expiry" means at the *analysis horizon*: the common expiry for ordinary
  strategies, the FRONT expiry for calendars and diagonals, where the legs
  still alive are valued with Black-Scholes (flat vol and rates taken from the
  market passed in).
* P&L at expiry = value at the horizon - premium paid today. Financing is
  deliberately ignored: the premium is NOT capitalised to the horizon and
  dividends received on a stock leg are not added. Simple, and what every
  textbook payoff diagram shows.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ndtr

from .black_scholes import normalize_option_type, to_trader_units
from .instruments import (
    EXPIRY_TOL,
    CompositeInstrument,
    EuropeanOption,
    Instrument,
    Position,
    Underlying,
    register_instrument,
)
from .market import Market

__all__ = [
    "VIEW_TAGS",
    "PARAM_KINDS",
    "Strategy",
    "ParamSpec",
    "StrategySpec",
    "STRATEGY_REGISTRY",
    "list_strategies",
    "get_strategy_spec",
    "build_strategy",
    "analysis_horizon",
    "reference_levels",
    "spot_grid",
    "net_premium",
    "payoff_at_expiry",
    "pnl_at_expiry",
    "breakevens",
    "max_profit",
    "max_loss",
    "probability_of_profit",
    "legs_table",
    "summary",
    "long_call",
    "long_put",
    "short_call",
    "short_put",
    "covered_call",
    "protective_put",
    "collar",
    "bull_call_spread",
    "bear_put_spread",
    "bull_put_spread",
    "bear_call_spread",
    "long_straddle",
    "short_straddle",
    "long_strangle",
    "short_strangle",
    "strip",
    "strap",
    "long_call_butterfly",
    "long_put_butterfly",
    "iron_butterfly",
    "long_call_condor",
    "iron_condor",
    "call_ratio_spread",
    "put_ratio_spread",
    "call_backspread",
    "put_backspread",
    "risk_reversal",
    "seagull",
    "jade_lizard",
    "synthetic_long",
    "synthetic_short",
    "conversion",
    "reversal",
    "box_spread",
    "calendar_spread",
    "diagonal_spread",
    "double_calendar",
]

#: Market-view tags used by the predefined strategies.
VIEW_TAGS: tuple[str, ...] = (
    "bullish",
    "bearish",
    "neutral",
    "long vol",
    "short vol",
    "income",
    "hedge",
    "arbitrage",
)

#: Kinds of factory parameters (they tell a UI which widget to draw).
PARAM_KINDS: tuple[str, ...] = ("strike", "expiry", "quantity", "option_type")

_SLOPE_TOL = 1e-7  # |dP&L/dS| below this at very high spot counts as "flat"
_ANALYSIS_POINTS = 2001  # grid size used by the breakeven / extremum searches


# ---------------------------------------------------------------------- #
# Strategy: a composite with teaching metadata
# ---------------------------------------------------------------------- #
@register_instrument
@dataclass(frozen=True)
class Strategy(CompositeInstrument):
    """A named multi-leg package with a description and market-view tags.

    Parameters
    ----------
    name : str
        Display name, e.g. ``"Bull call spread 95/105"``.
    legs : sequence
        Legs, as for :class:`~optionlab.instruments.CompositeInstrument`.
    description : str, default ""
        What the strategy is, the view it expresses, who uses it, main risks.
    view : tuple of str, default ()
        Market-view tags (see :data:`VIEW_TAGS`; free text is accepted so that
        custom strategies can carry their own tags).
    key : str, default ""
        Name of the factory in :data:`STRATEGY_REGISTRY` ("" for a custom
        strategy), so a UI can find the parameter schema back.

    Notes
    -----
    A strategy is just a linear combination of its legs: its price, Greeks and
    payoff are quantity-weighted sums. That is the whole trick behind options
    structuring -- you *add* simple payoffs until the sum matches your view.
    """

    description: str = ""
    view: tuple[str, ...] = ()
    key: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.description, str) or not isinstance(self.key, str):
            raise ValueError("Strategy.description and Strategy.key must be strings")
        tags = (self.view,) if isinstance(self.view, str) else tuple(self.view)
        if not all(isinstance(tag, str) for tag in tags):
            raise ValueError("Strategy.view must be a sequence of strings")
        object.__setattr__(self, "view", tags)

    @property
    def is_multi_expiry(self) -> bool:
        """True for calendars / diagonals: the legs do not all expire together."""
        return self.first_expiry != self.expiry


# ---------------------------------------------------------------------- #
# Registry: parameter schema + specs
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParamSpec:
    """Description of one factory parameter, for automatic UI building.

    Parameters
    ----------
    name : str
        Keyword name in the factory signature.
    kind : {"strike", "expiry", "quantity", "option_type"}
        What the parameter is (which widget to draw).
    default : float or str
        RELATIVE default: a strike is a multiple of the spot (``0.95`` means
        95% of spot), an expiry is a multiple of the reference time to expiry
        (``2.0`` means twice as far), a quantity is an absolute number and an
        option type is ``"call"`` or ``"put"``.
    description : str
        One-line help text.
    """

    name: str
    kind: str
    default: float | str
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in PARAM_KINDS:
            raise ValueError(f"ParamSpec.kind must be one of {PARAM_KINDS}, got {self.kind!r}")

    def resolve(self, spot: float, expiry: float, t: float = 0.0) -> float | str:
        """Concrete default value for a given spot, reference expiry and current time."""
        if self.kind == "strike":
            return float(self.default) * spot
        if self.kind == "expiry":
            return t + float(self.default) * (expiry - t)
        return self.default


@dataclass(frozen=True)
class StrategySpec:
    """Registry entry of a predefined strategy.

    Attributes
    ----------
    name : str
        Registry key = factory function name (e.g. ``"iron_condor"``).
    title : str
        Human-readable name (e.g. ``"Iron condor"``).
    factory : callable
        The factory function; ``factory(**params) -> Strategy``.
    category : str
        Family used to group strategies in a menu.
    view : tuple of str
        Market-view tags.
    description : str
        Teaching text (taken from the factory docstring).
    params : tuple of ParamSpec
        Parameter schema, in the order of the factory signature.
    """

    name: str
    title: str
    factory: Callable[..., Strategy]
    category: str
    view: tuple[str, ...]
    description: str
    params: tuple[ParamSpec, ...]

    @property
    def schema(self) -> dict[str, ParamSpec]:
        """Parameter name -> :class:`ParamSpec`."""
        return {p.name: p for p in self.params}

    def default_params(self, spot: float, expiry: float = 1.0, t: float = 0.0) -> dict[str, Any]:
        """Default factory arguments for a given spot and reference expiry.

        Parameters
        ----------
        spot : float
            Current spot (> 0); strike defaults are multiples of it.
        expiry : float, default 1.0
            ABSOLUTE reference expiry (the front expiry of a calendar).
        t : float, default 0.0
            Current time; expiry multiples apply to ``expiry - t``.
        """
        spot, expiry, t = float(spot), float(expiry), float(t)
        if not (math.isfinite(spot) and spot > 0):
            raise ValueError(f"spot must be a finite positive number, got {spot!r}")
        if not expiry > t:
            raise ValueError(f"expiry ({expiry:g}) must be later than the current time t ({t:g})")
        return {p.name: p.resolve(spot, expiry, t) for p in self.params}

    def build_default(
        self, spot: float, expiry: float = 1.0, t: float = 0.0, **overrides: Any
    ) -> Strategy:
        """Build the strategy with its default parameters (optionally overridden)."""
        params = self.default_params(spot, expiry, t)
        unknown = set(overrides) - set(params)
        if unknown:
            raise ValueError(
                f"{self.name}: unknown parameter(s) {sorted(unknown)}; valid: {list(params)}"
            )
        params.update(overrides)
        return self.factory(**params)


#: Registry key (factory name) -> :class:`StrategySpec`, in menu order.
STRATEGY_REGISTRY: dict[str, StrategySpec] = {}


def _docstring_description(func: Callable[..., Any]) -> str:
    """Teaching text of a factory: its docstring up to the ``Parameters`` section."""
    doc = inspect.getdoc(func) or ""
    return " ".join(doc.split("\nParameters\n", 1)[0].split())


def _register(
    *, title: str, category: str, view: Iterable[str], params: Iterable[ParamSpec]
) -> Callable[[Callable[..., Strategy]], Callable[..., Strategy]]:
    """Decorator adding a factory to :data:`STRATEGY_REGISTRY`.

    The schema must list exactly the factory's parameters, in order -- checked
    at import time so the UI schema can never drift from the code.
    """
    view, params = tuple(view), tuple(params)
    unknown_tags = set(view) - set(VIEW_TAGS)
    if unknown_tags:
        raise ValueError(f"unknown view tag(s) {sorted(unknown_tags)}; valid: {VIEW_TAGS}")

    def decorator(factory: Callable[..., Strategy]) -> Callable[..., Strategy]:
        signature = list(inspect.signature(factory).parameters)
        if signature != [p.name for p in params]:
            raise ValueError(
                f"{factory.__name__}: schema {[p.name for p in params]} does not match "
                f"the signature {signature}"
            )
        STRATEGY_REGISTRY[factory.__name__] = StrategySpec(
            name=factory.__name__,
            title=title,
            factory=factory,
            category=category,
            view=view,
            description=_docstring_description(factory) or title,
            params=params,
        )
        return factory

    return decorator


def _normalize_key(name: str) -> str:
    return "_".join(str(name).strip().lower().replace("-", " ").replace("_", " ").split())


def get_strategy_spec(name: str) -> StrategySpec:
    """Look a spec up by name (case, spaces and hyphens are forgiven)."""
    spec = STRATEGY_REGISTRY.get(_normalize_key(name))
    if spec is None:
        raise ValueError(f"Unknown strategy {name!r}. Available: {list(STRATEGY_REGISTRY)}")
    return spec


def list_strategies(category: str | None = None, view: str | None = None) -> list[str]:
    """Names of the predefined strategies, optionally filtered.

    Parameters
    ----------
    category : str, optional
        Keep one family only (e.g. ``"Vertical spread"``), case-insensitive.
    view : str, optional
        Keep strategies carrying this view tag (e.g. ``"long vol"``).
    """
    names = []
    for name, spec in STRATEGY_REGISTRY.items():
        if category is not None and spec.category.lower() != category.strip().lower():
            continue
        if view is not None and view.strip().lower() not in spec.view:
            continue
        names.append(name)
    return names


def build_strategy(name: str, **params: Any) -> Strategy:
    """Build a predefined strategy by name: ``build_strategy("collar", put_strike=95, ...)``.

    Raises
    ------
    ValueError
        Unknown strategy, unknown parameter or missing required parameter
        (the message lists what the factory expects).
    """
    spec = get_strategy_spec(name)
    signature = inspect.signature(spec.factory).parameters
    unknown = set(params) - set(signature)
    if unknown:
        raise ValueError(
            f"{spec.name}: unknown parameter(s) {sorted(unknown)}; expected {list(signature)}"
        )
    missing = [
        p.name for p in signature.values() if p.default is inspect.Parameter.empty and p.name not in params
    ]
    if missing:
        raise ValueError(f"{spec.name}: missing parameter(s) {missing}; expected {list(signature)}")
    return spec.factory(**params)


# ---------------------------------------------------------------------- #
# Validation and assembly helpers
# ---------------------------------------------------------------------- #
def _positive(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return number


def _ascending(**strikes: Any) -> tuple[float, ...]:
    """Validate strikes given in the order they must be sorted (strictly increasing)."""
    values = [(name, _positive(name, value)) for name, value in strikes.items()]
    for (low_name, low), (high_name, high) in zip(values, values[1:]):
        if not low < high:
            raise ValueError(
                f"strikes must be strictly increasing: {low_name}={low:g} "
                f"must be below {high_name}={high:g}"
            )
    return tuple(value for _, value in values)


def _expiry(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _near_far(near_expiry: Any, far_expiry: Any) -> tuple[float, float]:
    near, far = _expiry("near_expiry", near_expiry), _expiry("far_expiry", far_expiry)
    if not near < far:
        raise ValueError(f"near_expiry ({near:g}) must be earlier than far_expiry ({far:g})")
    return near, far


def _ratio(value: Any) -> float:
    ratio = _positive("ratio", value)
    if not ratio > 1:
        raise ValueError(f"ratio must be greater than 1 (a 1 x ratio spread), got {value!r}")
    return ratio


def _call(strike: float, expiry: float) -> EuropeanOption:
    return EuropeanOption("call", strike, expiry)


def _put(strike: float, expiry: float) -> EuropeanOption:
    return EuropeanOption("put", strike, expiry)


_STOCK = Underlying()


def _assemble(
    key: str, detail: str, legs: Iterable[tuple[Instrument, float]], quantity: Any
) -> Strategy:
    """Build the :class:`Strategy` of a registered factory (metadata come from its spec)."""
    size = _positive("quantity", quantity)
    spec = STRATEGY_REGISTRY[key]
    name = f"{spec.title} {detail}" + ("" if size == 1 else f" (x{size:g})")
    return Strategy(
        name=name,
        legs=tuple(Position(instrument, weight * size) for instrument, weight in legs),
        description=spec.description,
        view=spec.view,
        key=key,
    )


def _strike_param(name: str, moneyness: float, description: str) -> ParamSpec:
    return ParamSpec(name, "strike", moneyness, description)


_EXPIRY = ParamSpec("expiry", "expiry", 1.0, "Absolute expiry of every leg, in years.")
_NEAR = ParamSpec("near_expiry", "expiry", 1.0, "Expiry of the option sold (front month).")
_FAR = ParamSpec("far_expiry", "expiry", 2.0, "Expiry of the option bought (back month).")
_QUANTITY = ParamSpec("quantity", "quantity", 1.0, "Number of packages (> 0); scales every leg.")
_RATIO = ParamSpec("ratio", "quantity", 2.0, "Options on the multiple side for ONE on the other (> 1).")
_OPTION_TYPE = ParamSpec("option_type", "option_type", "call", "Build it with calls or with puts.")


# ---------------------------------------------------------------------- #
# Single legs
# ---------------------------------------------------------------------- #
@_register(
    title="Long call",
    category="Single leg",
    view=("bullish", "long vol"),
    params=(_strike_param("strike", 1.0, "Strike of the call bought."), _EXPIRY, _QUANTITY),
)
def long_call(strike: float, expiry: float, quantity: float = 1.0) -> Strategy:
    """Buy a call: the right to buy the stock at the strike.

    The simplest bullish bet with a built-in stop-loss: the most you can lose
    is the premium, while the upside is unlimited. You are long delta, long
    gamma and long vega, and you pay for all of it through time decay (theta),
    so you need the move to come before expiry, not just eventually.

    Parameters
    ----------
    strike, expiry : float
        Strike (> 0) and absolute expiry in years.
    quantity : float, default 1.0
        Number of calls (> 0).
    """
    (strike,) = _ascending(strike=strike)
    return _assemble("long_call", f"{strike:g}", [(_call(strike, expiry), 1)], quantity)


@_register(
    title="Long put",
    category="Single leg",
    view=("bearish", "long vol"),
    params=(_strike_param("strike", 1.0, "Strike of the put bought."), _EXPIRY, _QUANTITY),
)
def long_put(strike: float, expiry: float, quantity: float = 1.0) -> Strategy:
    """Buy a put: the right to sell the stock at the strike.

    A bearish bet with limited risk, or insurance when you already own the
    stock. The loss is capped at the premium and the gain grows as the stock
    falls (up to the strike, since a price cannot go below zero). Like every
    long option it is long gamma and vega and bleeds theta every day.

    Parameters
    ----------
    strike, expiry : float
        Strike (> 0) and absolute expiry in years.
    quantity : float, default 1.0
        Number of puts (> 0).
    """
    (strike,) = _ascending(strike=strike)
    return _assemble("long_put", f"{strike:g}", [(_put(strike, expiry), 1)], quantity)


@_register(
    title="Short call",
    category="Single leg",
    view=("bearish", "short vol", "income"),
    params=(_strike_param("strike", 1.05, "Strike of the call sold."), _EXPIRY, _QUANTITY),
)
def short_call(strike: float, expiry: float, quantity: float = 1.0) -> Strategy:
    """Sell a call without owning the stock (a "naked" call).

    You collect the premium and keep it if the stock stays below the strike,
    so you earn theta and profit from falling volatility. The risk is the
    mirror image of the long call: the loss is UNLIMITED if the stock rallies,
    which is why brokers ask for a large margin on this trade.

    Parameters
    ----------
    strike, expiry : float
        Strike (> 0) and absolute expiry in years.
    quantity : float, default 1.0
        Number of calls sold (> 0).
    """
    (strike,) = _ascending(strike=strike)
    return _assemble("short_call", f"{strike:g}", [(_call(strike, expiry), -1)], quantity)


@_register(
    title="Short put",
    category="Single leg",
    view=("bullish", "short vol", "income"),
    params=(_strike_param("strike", 0.95, "Strike of the put sold."), _EXPIRY, _QUANTITY),
)
def short_put(strike: float, expiry: float, quantity: float = 1.0) -> Strategy:
    """Sell a put: get paid to promise to buy the stock at the strike.

    Used by investors happy to buy the stock lower ("cash-secured put") and by
    volatility sellers. The gain is capped at the premium; the loss grows as
    the stock falls, down to strike minus premium if it goes to zero. It has
    the same risk profile as a covered call (put-call parity).

    Parameters
    ----------
    strike, expiry : float
        Strike (> 0) and absolute expiry in years.
    quantity : float, default 1.0
        Number of puts sold (> 0).
    """
    (strike,) = _ascending(strike=strike)
    return _assemble("short_put", f"{strike:g}", [(_put(strike, expiry), -1)], quantity)


# ---------------------------------------------------------------------- #
# Stock + options
# ---------------------------------------------------------------------- #
@_register(
    title="Covered call",
    category="Stock + option",
    view=("bullish", "income", "short vol"),
    params=(_strike_param("strike", 1.05, "Strike of the call sold."), _EXPIRY, _QUANTITY),
)
def covered_call(strike: float, expiry: float, quantity: float = 1.0) -> Strategy:
    """Own the stock and sell a call against it.

    The classic income strategy of long-term holders: the premium is earned
    whatever happens, in exchange for giving away the upside above the strike.
    It does NOT protect against a fall -- the downside is the stock's, only
    cushioned by the premium -- and by put-call parity it is economically a
    short put.

    Parameters
    ----------
    strike, expiry : float
        Strike of the call sold (usually above the spot) and absolute expiry.
    quantity : float, default 1.0
        Number of shares / calls (> 0).
    """
    (strike,) = _ascending(strike=strike)
    legs = [(_STOCK, 1), (_call(strike, expiry), -1)]
    return _assemble("covered_call", f"{strike:g}", legs, quantity)


@_register(
    title="Protective put",
    category="Stock + option",
    view=("bullish", "hedge", "long vol"),
    params=(_strike_param("strike", 0.95, "Strike of the put bought."), _EXPIRY, _QUANTITY),
)
def protective_put(strike: float, expiry: float, quantity: float = 1.0) -> Strategy:
    """Own the stock and buy a put as insurance.

    The put sets a floor under the position: below the strike every euro lost
    on the stock is made back on the put. The upside stays open, minus the
    insurance premium, which is the drag on performance if nothing happens.
    By put-call parity the package behaves like a long call.

    Parameters
    ----------
    strike, expiry : float
        Strike of the put (the floor) and absolute expiry.
    quantity : float, default 1.0
        Number of shares / puts (> 0).
    """
    (strike,) = _ascending(strike=strike)
    legs = [(_STOCK, 1), (_put(strike, expiry), 1)]
    return _assemble("protective_put", f"{strike:g}", legs, quantity)


@_register(
    title="Collar",
    category="Stock + option",
    view=("bullish", "hedge"),
    params=(
        _strike_param("put_strike", 0.95, "Strike of the put bought (the floor)."),
        _strike_param("call_strike", 1.05, "Strike of the call sold (the cap)."),
        _EXPIRY,
        _QUANTITY,
    ),
)
def collar(put_strike: float, call_strike: float, expiry: float, quantity: float = 1.0) -> Strategy:
    """Own the stock, buy a put below the spot and finance it by selling a call above.

    The stock position is locked between a floor (put strike) and a cap (call
    strike); choosing the strikes so that both premiums cancel gives the
    popular "zero-cost collar". Used by shareholders who want protection
    without paying for it, the price being the upside they give away.

    Parameters
    ----------
    put_strike, call_strike : float
        Floor and cap (``put_strike < call_strike``).
    expiry : float
        Absolute expiry in years.
    quantity : float, default 1.0
        Number of shares / collars (> 0).
    """
    low, high = _ascending(put_strike=put_strike, call_strike=call_strike)
    legs = [(_STOCK, 1), (_put(low, expiry), 1), (_call(high, expiry), -1)]
    return _assemble("collar", f"{low:g}/{high:g}", legs, quantity)


# ---------------------------------------------------------------------- #
# Vertical spreads
# ---------------------------------------------------------------------- #
_LOW_HIGH = (
    _strike_param("low_strike", 0.95, "Lower strike."),
    _strike_param("high_strike", 1.05, "Higher strike."),
)


@_register(
    title="Bull call spread",
    category="Vertical spread",
    view=("bullish",),
    params=(*_LOW_HIGH, _EXPIRY, _QUANTITY),
)
def bull_call_spread(
    low_strike: float, high_strike: float, expiry: float, quantity: float = 1.0
) -> Strategy:
    """Buy a call and sell a higher-strike call (a debit spread).

    A cheaper bullish bet than the outright call: the call sold pays for part
    of the call bought, and caps the profit at the distance between the
    strikes minus the debit. Both the maximum gain and the maximum loss (the
    debit) are known at inception, and the vega and theta of the two legs
    largely offset each other.

    Parameters
    ----------
    low_strike, high_strike : float
        Strike bought and strike sold (``low_strike < high_strike``).
    expiry : float
        Absolute expiry in years.
    quantity : float, default 1.0
        Number of spreads (> 0).
    """
    low, high = _ascending(low_strike=low_strike, high_strike=high_strike)
    legs = [(_call(low, expiry), 1), (_call(high, expiry), -1)]
    return _assemble("bull_call_spread", f"{low:g}/{high:g}", legs, quantity)


@_register(
    title="Bear put spread",
    category="Vertical spread",
    view=("bearish",),
    params=(*_LOW_HIGH, _EXPIRY, _QUANTITY),
)
def bear_put_spread(
    low_strike: float, high_strike: float, expiry: float, quantity: float = 1.0
) -> Strategy:
    """Buy a put and sell a lower-strike put (a debit spread).

    The bearish twin of the bull call spread: you pay a reduced premium for a
    gain capped at the distance between the strikes minus the debit. Popular
    when puts are expensive (steep skew), because the put sold is the one the
    market overpays the most.

    Parameters
    ----------
    low_strike, high_strike : float
        Strike sold and strike bought (``low_strike < high_strike``).
    expiry : float
        Absolute expiry in years.
    quantity : float, default 1.0
        Number of spreads (> 0).
    """
    low, high = _ascending(low_strike=low_strike, high_strike=high_strike)
    legs = [(_put(high, expiry), 1), (_put(low, expiry), -1)]
    return _assemble("bear_put_spread", f"{low:g}/{high:g}", legs, quantity)


@_register(
    title="Bull put spread",
    category="Vertical spread",
    view=("bullish", "income"),
    params=(*_LOW_HIGH, _EXPIRY, _QUANTITY),
)
def bull_put_spread(
    low_strike: float, high_strike: float, expiry: float, quantity: float = 1.0
) -> Strategy:
    """Sell a put and buy a lower-strike put as protection (a credit spread).

    You receive a net credit and keep it if the stock finishes above the
    higher strike: a bullish-to-neutral income trade. The put bought turns the
    open-ended risk of a short put into a known maximum loss, the distance
    between the strikes minus the credit.

    Parameters
    ----------
    low_strike, high_strike : float
        Strike bought and strike sold (``low_strike < high_strike``).
    expiry : float
        Absolute expiry in years.
    quantity : float, default 1.0
        Number of spreads (> 0).
    """
    low, high = _ascending(low_strike=low_strike, high_strike=high_strike)
    legs = [(_put(high, expiry), -1), (_put(low, expiry), 1)]
    return _assemble("bull_put_spread", f"{low:g}/{high:g}", legs, quantity)


@_register(
    title="Bear call spread",
    category="Vertical spread",
    view=("bearish", "income"),
    params=(*_LOW_HIGH, _EXPIRY, _QUANTITY),
)
def bear_call_spread(
    low_strike: float, high_strike: float, expiry: float, quantity: float = 1.0
) -> Strategy:
    """Sell a call and buy a higher-strike call as protection (a credit spread).

    A bearish-to-neutral income trade: the credit is kept if the stock
    finishes below the lower strike. The call bought caps the otherwise
    unlimited risk of the short call at the distance between the strikes
    minus the credit.

    Parameters
    ----------
    low_strike, high_strike : float
        Strike sold and strike bought (``low_strike < high_strike``).
    expiry : float
        Absolute expiry in years.
    quantity : float, default 1.0
        Number of spreads (> 0).
    """
    low, high = _ascending(low_strike=low_strike, high_strike=high_strike)
    legs = [(_call(low, expiry), -1), (_call(high, expiry), 1)]
    return _assemble("bear_call_spread", f"{low:g}/{high:g}", legs, quantity)


# ---------------------------------------------------------------------- #
# Volatility strategies
# ---------------------------------------------------------------------- #
_ATM = _strike_param("strike", 1.0, "Common strike of the call and the put.")
_STRANGLE = (
    _strike_param("put_strike", 0.9, "Strike of the put (below the spot)."),
    _strike_param("call_strike", 1.1, "Strike of the call (above the spot)."),
)


@_register(
    title="Long straddle",
    category="Volatility",
    view=("neutral", "long vol"),
    params=(_ATM, _EXPIRY, _QUANTITY),
)
def long_straddle(strike: float, expiry: float, quantity: float = 1.0) -> Strategy:
    """Buy a call and a put with the same strike.

    A pure volatility bet: you do not care which way the stock goes as long
    as it moves by more than the total premium, or implied volatility rises.
    It is the position with the most gamma and vega per euro -- and the most
    theta to pay, so a quiet market is the enemy. Typically bought ahead of
    events (earnings, elections) when the move is expected to be large.

    Parameters
    ----------
    strike, expiry : float
        Common strike (usually at the money) and absolute expiry.
    quantity : float, default 1.0
        Number of straddles (> 0).
    """
    (strike,) = _ascending(strike=strike)
    legs = [(_call(strike, expiry), 1), (_put(strike, expiry), 1)]
    return _assemble("long_straddle", f"{strike:g}", legs, quantity)


@_register(
    title="Short straddle",
    category="Volatility",
    view=("neutral", "short vol", "income"),
    params=(_ATM, _EXPIRY, _QUANTITY),
)
def short_straddle(strike: float, expiry: float, quantity: float = 1.0) -> Strategy:
    """Sell a call and a put with the same strike.

    You collect two premiums and keep them if the stock stays near the
    strike: the trade earns theta and gains when implied volatility falls.
    The risk is unlimited on the upside and very large on the downside, and
    being short gamma means every hedge is a buy-high / sell-low trade.

    Parameters
    ----------
    strike, expiry : float
        Common strike (usually at the money) and absolute expiry.
    quantity : float, default 1.0
        Number of straddles sold (> 0).
    """
    (strike,) = _ascending(strike=strike)
    legs = [(_call(strike, expiry), -1), (_put(strike, expiry), -1)]
    return _assemble("short_straddle", f"{strike:g}", legs, quantity)


@_register(
    title="Long strangle",
    category="Volatility",
    view=("neutral", "long vol"),
    params=(*_STRANGLE, _EXPIRY, _QUANTITY),
)
def long_strangle(
    put_strike: float, call_strike: float, expiry: float, quantity: float = 1.0
) -> Strategy:
    """Buy an out-of-the-money put and an out-of-the-money call.

    A cheaper cousin of the straddle: less premium at risk, but the stock has
    to travel further before the trade pays (the breakevens are wider). The
    maximum loss, the total premium, is suffered anywhere between the two
    strikes.

    Parameters
    ----------
    put_strike, call_strike : float
        Strikes of the put and of the call (``put_strike < call_strike``).
    expiry : float
        Absolute expiry in years.
    quantity : float, default 1.0
        Number of strangles (> 0).
    """
    low, high = _ascending(put_strike=put_strike, call_strike=call_strike)
    legs = [(_put(low, expiry), 1), (_call(high, expiry), 1)]
    return _assemble("long_strangle", f"{low:g}/{high:g}", legs, quantity)


@_register(
    title="Short strangle",
    category="Volatility",
    view=("neutral", "short vol", "income"),
    params=(*_STRANGLE, _EXPIRY, _QUANTITY),
)
def short_strangle(
    put_strike: float, call_strike: float, expiry: float, quantity: float = 1.0
) -> Strategy:
    """Sell an out-of-the-money put and an out-of-the-money call.

    The bread-and-butter trade of volatility sellers: the full credit is kept
    as long as the stock stays between the strikes, a wider comfort zone than
    the short straddle for a smaller premium. The tails are unprotected:
    unlimited loss on the upside, very large on the downside.

    Parameters
    ----------
    put_strike, call_strike : float
        Strikes of the put and of the call (``put_strike < call_strike``).
    expiry : float
        Absolute expiry in years.
    quantity : float, default 1.0
        Number of strangles sold (> 0).
    """
    low, high = _ascending(put_strike=put_strike, call_strike=call_strike)
    legs = [(_put(low, expiry), -1), (_call(high, expiry), -1)]
    return _assemble("short_strangle", f"{low:g}/{high:g}", legs, quantity)


@_register(
    title="Strip",
    category="Volatility",
    view=("bearish", "long vol"),
    params=(_ATM, _EXPIRY, _QUANTITY),
)
def strip(strike: float, expiry: float, quantity: float = 1.0) -> Strategy:
    """Buy one call and TWO puts with the same strike.

    A straddle with a bearish tilt: you expect a big move and think a fall is
    more likely than a rally, so the downside pays twice as fast. It costs
    more than a straddle and starts with a negative delta.

    Parameters
    ----------
    strike, expiry : float
        Common strike and absolute expiry.
    quantity : float, default 1.0
        Number of strips (> 0).
    """
    (strike,) = _ascending(strike=strike)
    legs = [(_call(strike, expiry), 1), (_put(strike, expiry), 2)]
    return _assemble("strip", f"{strike:g}", legs, quantity)


@_register(
    title="Strap",
    category="Volatility",
    view=("bullish", "long vol"),
    params=(_ATM, _EXPIRY, _QUANTITY),
)
def strap(strike: float, expiry: float, quantity: float = 1.0) -> Strategy:
    """Buy TWO calls and one put with the same strike.

    A straddle with a bullish tilt: you expect a big move and think a rally
    is more likely than a fall, so the upside pays twice as fast. It costs
    more than a straddle and starts with a positive delta.

    Parameters
    ----------
    strike, expiry : float
        Common strike and absolute expiry.
    quantity : float, default 1.0
        Number of straps (> 0).
    """
    (strike,) = _ascending(strike=strike)
    legs = [(_call(strike, expiry), 2), (_put(strike, expiry), 1)]
    return _assemble("strap", f"{strike:g}", legs, quantity)


# ---------------------------------------------------------------------- #
# Butterflies and condors
# ---------------------------------------------------------------------- #
_FLY = (
    _strike_param("low_strike", 0.9, "Lower wing."),
    _strike_param("mid_strike", 1.0, "Body (where the profit peaks)."),
    _strike_param("high_strike", 1.1, "Upper wing."),
)
_CONDOR = (
    _strike_param("strike_1", 0.85, "Lowest strike (long wing)."),
    _strike_param("strike_2", 0.95, "Lower short strike."),
    _strike_param("strike_3", 1.05, "Upper short strike."),
    _strike_param("strike_4", 1.15, "Highest strike (long wing)."),
)


@_register(
    title="Long call butterfly",
    category="Butterfly & condor",
    view=("neutral", "short vol"),
    params=(*_FLY, _EXPIRY, _QUANTITY),
)
def long_call_butterfly(
    low_strike: float, mid_strike: float, high_strike: float, expiry: float, quantity: float = 1.0
) -> Strategy:
    """Buy one low call, sell two middle calls, buy one high call.

    A cheap, limited-risk bet that the stock will finish near the middle
    strike: the payoff is a tent peaking there. It is a bull call spread plus
    a bear call spread, so the most you can lose is the small debit. With
    unequal wings it becomes a "broken-wing" butterfly whose payoff does not
    return to zero on one side.

    Parameters
    ----------
    low_strike, mid_strike, high_strike : float
        Strictly increasing strikes (wings usually symmetric around the body).
    expiry : float
        Absolute expiry in years.
    quantity : float, default 1.0
        Number of butterflies (> 0).
    """
    low, mid, high = _ascending(low_strike=low_strike, mid_strike=mid_strike, high_strike=high_strike)
    legs = [(_call(low, expiry), 1), (_call(mid, expiry), -2), (_call(high, expiry), 1)]
    return _assemble("long_call_butterfly", f"{low:g}/{mid:g}/{high:g}", legs, quantity)


@_register(
    title="Long put butterfly",
    category="Butterfly & condor",
    view=("neutral", "short vol"),
    params=(*_FLY, _EXPIRY, _QUANTITY),
)
def long_put_butterfly(
    low_strike: float, mid_strike: float, high_strike: float, expiry: float, quantity: float = 1.0
) -> Strategy:
    """Buy one low put, sell two middle puts, buy one high put.

    Same tent-shaped payoff as the call butterfly -- with symmetric wings
    put-call parity makes the two interchangeable, and any price difference
    is an arbitrage. Traders pick whichever side is more liquid or cheaper
    after bid-ask costs.

    Parameters
    ----------
    low_strike, mid_strike, high_strike : float
        Strictly increasing strikes (wings usually symmetric around the body).
    expiry : float
        Absolute expiry in years.
    quantity : float, default 1.0
        Number of butterflies (> 0).
    """
    low, mid, high = _ascending(low_strike=low_strike, mid_strike=mid_strike, high_strike=high_strike)
    legs = [(_put(low, expiry), 1), (_put(mid, expiry), -2), (_put(high, expiry), 1)]
    return _assemble("long_put_butterfly", f"{low:g}/{mid:g}/{high:g}", legs, quantity)


@_register(
    title="Iron butterfly",
    category="Butterfly & condor",
    view=("neutral", "short vol", "income"),
    params=(*_FLY, _EXPIRY, _QUANTITY),
)
def iron_butterfly(
    low_strike: float, mid_strike: float, high_strike: float, expiry: float, quantity: float = 1.0
) -> Strategy:
    """Sell a straddle at the middle strike and buy a strangle around it.

    A short straddle with insurance on both tails: you receive a net credit,
    keep most of it if the stock pins the middle strike, and the wings cap the
    loss at the wing width minus the credit. The payoff is the same tent as a
    long butterfly, shifted down by a constant -- you get paid up front
    instead of at expiry.

    Parameters
    ----------
    low_strike, mid_strike, high_strike : float
        Put wing, short straddle strike and call wing (strictly increasing).
    expiry : float
        Absolute expiry in years.
    quantity : float, default 1.0
        Number of iron butterflies (> 0).
    """
    low, mid, high = _ascending(low_strike=low_strike, mid_strike=mid_strike, high_strike=high_strike)
    legs = [
        (_put(low, expiry), 1),
        (_put(mid, expiry), -1),
        (_call(mid, expiry), -1),
        (_call(high, expiry), 1),
    ]
    return _assemble("iron_butterfly", f"{low:g}/{mid:g}/{high:g}", legs, quantity)


@_register(
    title="Long call condor",
    category="Butterfly & condor",
    view=("neutral", "short vol"),
    params=(*_CONDOR, _EXPIRY, _QUANTITY),
)
def long_call_condor(
    strike_1: float,
    strike_2: float,
    strike_3: float,
    strike_4: float,
    expiry: float,
    quantity: float = 1.0,
) -> Strategy:
    """Buy the lowest and highest calls, sell the two middle calls.

    A butterfly with a flat roof: the maximum profit is earned anywhere
    between the two middle strikes instead of at a single point. You pay a
    small debit for a range bet with limited risk; the wider profit zone costs
    a lower peak than the butterfly's.

    Parameters
    ----------
    strike_1, strike_2, strike_3, strike_4 : float
        Strictly increasing strikes: long, short, short, long.
    expiry : float
        Absolute expiry in years.
    quantity : float, default 1.0
        Number of condors (> 0).
    """
    k1, k2, k3, k4 = _ascending(strike_1=strike_1, strike_2=strike_2, strike_3=strike_3, strike_4=strike_4)
    legs = [
        (_call(k1, expiry), 1),
        (_call(k2, expiry), -1),
        (_call(k3, expiry), -1),
        (_call(k4, expiry), 1),
    ]
    return _assemble("long_call_condor", f"{k1:g}/{k2:g}/{k3:g}/{k4:g}", legs, quantity)


@_register(
    title="Iron condor",
    category="Butterfly & condor",
    view=("neutral", "short vol", "income"),
    params=(*_CONDOR, _EXPIRY, _QUANTITY),
)
def iron_condor(
    strike_1: float,
    strike_2: float,
    strike_3: float,
    strike_4: float,
    expiry: float,
    quantity: float = 1.0,
) -> Strategy:
    """Sell an out-of-the-money put spread and an out-of-the-money call spread.

    The favourite of income traders: a short strangle with both tails bought
    back. The credit is kept in full if the stock finishes between the two
    short strikes, and the worst case is the wider spread width minus the
    credit. The trade wins often and small, and loses rarely but bigger.

    Parameters
    ----------
    strike_1, strike_2, strike_3, strike_4 : float
        Strictly increasing strikes: long put, short put, short call, long call.
    expiry : float
        Absolute expiry in years.
    quantity : float, default 1.0
        Number of iron condors (> 0).
    """
    k1, k2, k3, k4 = _ascending(strike_1=strike_1, strike_2=strike_2, strike_3=strike_3, strike_4=strike_4)
    legs = [
        (_put(k1, expiry), 1),
        (_put(k2, expiry), -1),
        (_call(k3, expiry), -1),
        (_call(k4, expiry), 1),
    ]
    return _assemble("iron_condor", f"{k1:g}/{k2:g}/{k3:g}/{k4:g}", legs, quantity)


# ---------------------------------------------------------------------- #
# Ratio spreads and backspreads
# ---------------------------------------------------------------------- #
_RATIO_STRIKES = (
    _strike_param("low_strike", 1.0, "Lower strike."),
    _strike_param("high_strike", 1.1, "Higher strike."),
)
_PUT_RATIO_STRIKES = (
    _strike_param("low_strike", 0.9, "Lower strike."),
    _strike_param("high_strike", 1.0, "Higher strike."),
)


@_register(
    title="Call ratio spread",
    category="Ratio & backspread",
    view=("bullish", "short vol"),
    params=(*_RATIO_STRIKES, _EXPIRY, _RATIO, _QUANTITY),
)
def call_ratio_spread(
    low_strike: float,
    high_strike: float,
    expiry: float,
    ratio: float = 2.0,
    quantity: float = 1.0,
) -> Strategy:
    """Buy one call and sell MORE higher-strike calls (1 x 2 by default).

    A bull call spread financed by an extra short call: very cheap, sometimes
    even a credit, with the best result if the stock drifts up to the short
    strike and stops there. The extra calls are naked, so a strong rally
    produces an UNLIMITED loss: this is a short-volatility trade in disguise.

    Parameters
    ----------
    low_strike, high_strike : float
        Strike bought and strike sold (``low_strike < high_strike``).
    expiry : float
        Absolute expiry in years.
    ratio : float, default 2.0
        Calls sold for one call bought (> 1).
    quantity : float, default 1.0
        Number of ratio spreads (> 0).
    """
    low, high = _ascending(low_strike=low_strike, high_strike=high_strike)
    ratio = _ratio(ratio)
    legs = [(_call(low, expiry), 1), (_call(high, expiry), -ratio)]
    return _assemble("call_ratio_spread", f"{low:g}/{high:g} 1x{ratio:g}", legs, quantity)


@_register(
    title="Put ratio spread",
    category="Ratio & backspread",
    view=("bearish", "short vol"),
    params=(*_PUT_RATIO_STRIKES, _EXPIRY, _RATIO, _QUANTITY),
)
def put_ratio_spread(
    low_strike: float,
    high_strike: float,
    expiry: float,
    ratio: float = 2.0,
    quantity: float = 1.0,
) -> Strategy:
    """Buy one put and sell MORE lower-strike puts (1 x 2 by default).

    A bear put spread financed by an extra short put: the ideal outcome is a
    gentle fall to the short strike. Below it the extra puts are naked and
    the loss grows all the way down to a stock price of zero, so a crash is
    the scenario to fear.

    Parameters
    ----------
    low_strike, high_strike : float
        Strike sold and strike bought (``low_strike < high_strike``).
    expiry : float
        Absolute expiry in years.
    ratio : float, default 2.0
        Puts sold for one put bought (> 1).
    quantity : float, default 1.0
        Number of ratio spreads (> 0).
    """
    low, high = _ascending(low_strike=low_strike, high_strike=high_strike)
    ratio = _ratio(ratio)
    legs = [(_put(high, expiry), 1), (_put(low, expiry), -ratio)]
    return _assemble("put_ratio_spread", f"{low:g}/{high:g} 1x{ratio:g}", legs, quantity)


@_register(
    title="Call backspread",
    category="Ratio & backspread",
    view=("bullish", "long vol"),
    params=(*_RATIO_STRIKES, _EXPIRY, _RATIO, _QUANTITY),
)
def call_backspread(
    low_strike: float,
    high_strike: float,
    expiry: float,
    ratio: float = 2.0,
    quantity: float = 1.0,
) -> Strategy:
    """Sell one call and buy MORE higher-strike calls (the reverse ratio spread).

    The call sold finances the calls bought, so the trade is cheap and gains
    without limit in a strong rally -- and usually a little in a sell-off too.
    The worst case is a slow drift up to the long strike at expiry, where the
    short call is in the money and the long calls expire worthless. It is
    long gamma and long vega: a bet on a big move, preferably upwards.

    Parameters
    ----------
    low_strike, high_strike : float
        Strike sold and strike bought (``low_strike < high_strike``).
    expiry : float
        Absolute expiry in years.
    ratio : float, default 2.0
        Calls bought for one call sold (> 1).
    quantity : float, default 1.0
        Number of backspreads (> 0).
    """
    low, high = _ascending(low_strike=low_strike, high_strike=high_strike)
    ratio = _ratio(ratio)
    legs = [(_call(low, expiry), -1), (_call(high, expiry), ratio)]
    return _assemble("call_backspread", f"{low:g}/{high:g} 1x{ratio:g}", legs, quantity)


@_register(
    title="Put backspread",
    category="Ratio & backspread",
    view=("bearish", "long vol"),
    params=(*_PUT_RATIO_STRIKES, _EXPIRY, _RATIO, _QUANTITY),
)
def put_backspread(
    low_strike: float,
    high_strike: float,
    expiry: float,
    ratio: float = 2.0,
    quantity: float = 1.0,
) -> Strategy:
    """Sell one put and buy MORE lower-strike puts (the reverse put ratio spread).

    A cheap crash hedge: the put sold pays for the puts bought, the package
    gains heavily in a collapse and costs little if the market rallies. The
    pain point is a slow slide to the long strike at expiry. Long gamma and
    long vega, it benefits from the volatility spike that comes with a crash.

    Parameters
    ----------
    low_strike, high_strike : float
        Strike bought and strike sold (``low_strike < high_strike``).
    expiry : float
        Absolute expiry in years.
    ratio : float, default 2.0
        Puts bought for one put sold (> 1).
    quantity : float, default 1.0
        Number of backspreads (> 0).
    """
    low, high = _ascending(low_strike=low_strike, high_strike=high_strike)
    ratio = _ratio(ratio)
    legs = [(_put(high, expiry), -1), (_put(low, expiry), ratio)]
    return _assemble("put_backspread", f"{low:g}/{high:g} 1x{ratio:g}", legs, quantity)


# ---------------------------------------------------------------------- #
# Directional combinations
# ---------------------------------------------------------------------- #
@_register(
    title="Risk reversal",
    category="Directional combo",
    view=("bullish",),
    params=(
        _strike_param("put_strike", 0.9, "Strike of the put sold."),
        _strike_param("call_strike", 1.1, "Strike of the call bought."),
        _EXPIRY,
        _QUANTITY,
    ),
)
def risk_reversal(
    put_strike: float, call_strike: float, expiry: float, quantity: float = 1.0
) -> Strategy:
    """Sell an out-of-the-money put to pay for an out-of-the-money call.

    A leveraged bullish position for (almost) no premium: it behaves like a
    stretched synthetic long, flat between the strikes and stock-like outside.
    Its price is a direct reading of the skew, which is why the "25-delta risk
    reversal" is a quoted market parameter. The risk is the short put: you own
    the downside below the put strike.

    Parameters
    ----------
    put_strike, call_strike : float
        Strike sold and strike bought (``put_strike < call_strike``).
    expiry : float
        Absolute expiry in years.
    quantity : float, default 1.0
        Number of risk reversals (> 0).
    """
    low, high = _ascending(put_strike=put_strike, call_strike=call_strike)
    legs = [(_put(low, expiry), -1), (_call(high, expiry), 1)]
    return _assemble("risk_reversal", f"{low:g}/{high:g}", legs, quantity)


@_register(
    title="Seagull",
    category="Directional combo",
    view=("bullish",),
    params=(
        _strike_param("put_strike", 0.9, "Strike of the put sold."),
        _strike_param("call_strike", 1.0, "Strike of the call bought."),
        _strike_param("cap_strike", 1.1, "Strike of the call sold (the cap)."),
        _EXPIRY,
        _QUANTITY,
    ),
)
def seagull(
    put_strike: float,
    call_strike: float,
    cap_strike: float,
    expiry: float,
    quantity: float = 1.0,
) -> Strategy:
    """Buy a call spread and finance it by selling a lower-strike put (bullish seagull).

    A risk reversal whose upside is capped: selling both the put and the far
    call makes the bullish call spread very cheap or free. The profit is
    limited to the width of the call spread, while the short put leaves you
    exposed to a sharp fall. Common in FX and commodity hedging, where the
    name comes from the shape of the payoff.

    Parameters
    ----------
    put_strike, call_strike, cap_strike : float
        Put sold, call bought, call sold (strictly increasing).
    expiry : float
        Absolute expiry in years.
    quantity : float, default 1.0
        Number of seagulls (> 0).
    """
    low, mid, high = _ascending(put_strike=put_strike, call_strike=call_strike, cap_strike=cap_strike)
    legs = [(_put(low, expiry), -1), (_call(mid, expiry), 1), (_call(high, expiry), -1)]
    return _assemble("seagull", f"{low:g}/{mid:g}/{high:g}", legs, quantity)


@_register(
    title="Jade lizard",
    category="Directional combo",
    view=("neutral", "bullish", "short vol", "income"),
    params=(
        _strike_param("put_strike", 0.95, "Strike of the put sold."),
        _strike_param("call_strike", 1.05, "Strike of the call sold."),
        _strike_param("wing_strike", 1.08, "Strike of the call bought (upside protection)."),
        _EXPIRY,
        _QUANTITY,
    ),
)
def jade_lizard(
    put_strike: float,
    call_strike: float,
    wing_strike: float,
    expiry: float,
    quantity: float = 1.0,
) -> Strategy:
    """Sell an out-of-the-money put and an out-of-the-money call spread.

    A short strangle with the upside tail bought back. When the total credit
    exceeds the width of the call spread there is NO risk on the upside at
    all, which is the whole point of the structure; the entire risk sits in
    the short put. A neutral-to-bullish income trade for high implied
    volatility.

    Parameters
    ----------
    put_strike, call_strike, wing_strike : float
        Put sold, call sold, call bought (strictly increasing).
    expiry : float
        Absolute expiry in years.
    quantity : float, default 1.0
        Number of jade lizards (> 0).
    """
    low, mid, high = _ascending(put_strike=put_strike, call_strike=call_strike, wing_strike=wing_strike)
    legs = [(_put(low, expiry), -1), (_call(mid, expiry), -1), (_call(high, expiry), 1)]
    return _assemble("jade_lizard", f"{low:g}/{mid:g}/{high:g}", legs, quantity)


# ---------------------------------------------------------------------- #
# Synthetics and arbitrage packages
# ---------------------------------------------------------------------- #
_PARITY_STRIKE = _strike_param("strike", 1.0, "Common strike of the call and the put.")


@_register(
    title="Synthetic long",
    category="Synthetic & arbitrage",
    view=("bullish",),
    params=(_PARITY_STRIKE, _EXPIRY, _QUANTITY),
)
def synthetic_long(strike: float, expiry: float, quantity: float = 1.0) -> Strategy:
    """Buy a call and sell a put with the same strike: a synthetic forward.

    At expiry you end up buying the stock at the strike whatever happens
    (you exercise the call or get assigned on the put), so the package is a
    forward purchase. Put-call parity follows: call - put = S e^(-qT) -
    K e^(-rT). Delta is close to 1 while gamma and vega cancel: stock
    exposure with very little cash.

    Parameters
    ----------
    strike, expiry : float
        Common strike and absolute expiry.
    quantity : float, default 1.0
        Number of synthetic forwards (> 0).
    """
    (strike,) = _ascending(strike=strike)
    legs = [(_call(strike, expiry), 1), (_put(strike, expiry), -1)]
    return _assemble("synthetic_long", f"{strike:g}", legs, quantity)


@_register(
    title="Synthetic short",
    category="Synthetic & arbitrage",
    view=("bearish",),
    params=(_PARITY_STRIKE, _EXPIRY, _QUANTITY),
)
def synthetic_short(strike: float, expiry: float, quantity: float = 1.0) -> Strategy:
    """Sell a call and buy a put with the same strike: a synthetic short forward.

    You will sell the stock at the strike at expiry whatever happens. It is a
    way to be short without borrowing the shares, useful when the stock is
    hard or expensive to borrow. The loss is unlimited on the upside, exactly
    as for a short stock position.

    Parameters
    ----------
    strike, expiry : float
        Common strike and absolute expiry.
    quantity : float, default 1.0
        Number of synthetic short forwards (> 0).
    """
    (strike,) = _ascending(strike=strike)
    legs = [(_call(strike, expiry), -1), (_put(strike, expiry), 1)]
    return _assemble("synthetic_short", f"{strike:g}", legs, quantity)


@_register(
    title="Conversion",
    category="Synthetic & arbitrage",
    view=("neutral", "arbitrage"),
    params=(_PARITY_STRIKE, _EXPIRY, _QUANTITY),
)
def conversion(strike: float, expiry: float, quantity: float = 1.0) -> Strategy:
    """Own the stock, buy a put and sell a call with the same strike.

    Long stock plus a synthetic short: the package is worth exactly the
    strike at expiry wherever the stock ends, so it is a disguised
    zero-coupon bond and must cost K e^(-rT) (plus the dividends you will
    collect). Arbitrageurs put it on when calls are too expensive relative to
    puts; every Greek of the package is essentially zero.

    Parameters
    ----------
    strike, expiry : float
        Common strike and absolute expiry.
    quantity : float, default 1.0
        Number of conversions (> 0).
    """
    (strike,) = _ascending(strike=strike)
    legs = [(_STOCK, 1), (_put(strike, expiry), 1), (_call(strike, expiry), -1)]
    return _assemble("conversion", f"{strike:g}", legs, quantity)


@_register(
    title="Reversal",
    category="Synthetic & arbitrage",
    view=("neutral", "arbitrage"),
    params=(_PARITY_STRIKE, _EXPIRY, _QUANTITY),
)
def reversal(strike: float, expiry: float, quantity: float = 1.0) -> Strategy:
    """Short the stock, sell a put and buy a call with the same strike.

    The mirror image of the conversion: short stock plus a synthetic long.
    You owe exactly the strike at expiry, so the package is a disguised loan:
    you receive K e^(-rT) today. Put on when puts are too expensive relative
    to calls; in practice the stock-borrow cost eats most of the edge.

    Parameters
    ----------
    strike, expiry : float
        Common strike and absolute expiry.
    quantity : float, default 1.0
        Number of reversals (> 0).
    """
    (strike,) = _ascending(strike=strike)
    legs = [(_STOCK, -1), (_put(strike, expiry), -1), (_call(strike, expiry), 1)]
    return _assemble("reversal", f"{strike:g}", legs, quantity)


@_register(
    title="Box spread",
    category="Synthetic & arbitrage",
    view=("neutral", "arbitrage"),
    params=(*_LOW_HIGH, _EXPIRY, _QUANTITY),
)
def box_spread(
    low_strike: float, high_strike: float, expiry: float, quantity: float = 1.0
) -> Strategy:
    """Buy a bull call spread and a bear put spread on the same strikes.

    A synthetic long at the low strike plus a synthetic short at the high
    strike: the payoff is the distance between the strikes in EVERY scenario,
    so the box is a zero-coupon bond worth (K2 - K1) e^(-rT). Traders use it
    to lend or borrow cash through the options market; it has no delta, gamma
    or vega.

    Parameters
    ----------
    low_strike, high_strike : float
        The two strikes (``low_strike < high_strike``).
    expiry : float
        Absolute expiry in years.
    quantity : float, default 1.0
        Number of boxes (> 0).
    """
    low, high = _ascending(low_strike=low_strike, high_strike=high_strike)
    legs = [
        (_call(low, expiry), 1),
        (_call(high, expiry), -1),
        (_put(high, expiry), 1),
        (_put(low, expiry), -1),
    ]
    return _assemble("box_spread", f"{low:g}/{high:g}", legs, quantity)


# ---------------------------------------------------------------------- #
# Time spreads
# ---------------------------------------------------------------------- #
@_register(
    title="Calendar spread",
    category="Time spread",
    view=("neutral", "long vol"),
    params=(_PARITY_STRIKE, _NEAR, _FAR, _OPTION_TYPE, _QUANTITY),
)
def calendar_spread(
    strike: float,
    near_expiry: float,
    far_expiry: float,
    option_type: str = "call",
    quantity: float = 1.0,
) -> Strategy:
    """Sell a short-dated option and buy a longer-dated one with the same strike.

    The near option decays faster than the far one, so the spread earns theta
    while the stock stays near the strike: at the front expiry the value is a
    tent centred on the strike. Unlike a butterfly it is LONG vega (the far
    option has more vega) and short gamma. The loss is limited to the debit;
    the main risks are a large move either way or a fall in implied
    volatility.

    Parameters
    ----------
    strike : float
        Common strike.
    near_expiry, far_expiry : float
        Absolute expiries of the option sold and bought (``near < far``).
    option_type : {"call", "put"}, default "call"
        Type of both options.
    quantity : float, default 1.0
        Number of calendars (> 0).
    """
    (strike,) = _ascending(strike=strike)
    near, far = _near_far(near_expiry, far_expiry)
    kind = normalize_option_type(option_type)
    legs = [(EuropeanOption(kind, strike, near), -1), (EuropeanOption(kind, strike, far), 1)]
    detail = f"{kind} {strike:g} T={near:.2f}/{far:.2f}"
    return _assemble("calendar_spread", detail, legs, quantity)


@_register(
    title="Diagonal spread",
    category="Time spread",
    view=("long vol", "income"),
    params=(
        _strike_param("near_strike", 1.05, "Strike of the short-dated option sold."),
        _strike_param("far_strike", 0.95, "Strike of the long-dated option bought."),
        _NEAR,
        _FAR,
        _OPTION_TYPE,
        _QUANTITY,
    ),
)
def diagonal_spread(
    near_strike: float,
    far_strike: float,
    near_expiry: float,
    far_expiry: float,
    option_type: str = "call",
    quantity: float = 1.0,
) -> Strategy:
    """Sell a short-dated option and buy a longer-dated one with a DIFFERENT strike.

    A calendar spread with a directional tilt. The popular version buys a
    long-dated in-the-money call and repeatedly sells short-dated
    out-of-the-money calls against it (the "poor man's covered call"): the
    long call replaces the stock for a fraction of the capital, and each call
    sold brings in income. The position stays long vega through the back leg.

    Parameters
    ----------
    near_strike, far_strike : float
        Strike sold (near expiry) and strike bought (far expiry); any order.
    near_expiry, far_expiry : float
        Absolute expiries of the option sold and bought (``near < far``).
    option_type : {"call", "put"}, default "call"
        Type of both options.
    quantity : float, default 1.0
        Number of diagonals (> 0).
    """
    near_strike = _positive("near_strike", near_strike)
    far_strike = _positive("far_strike", far_strike)
    near, far = _near_far(near_expiry, far_expiry)
    kind = normalize_option_type(option_type)
    legs = [(EuropeanOption(kind, near_strike, near), -1), (EuropeanOption(kind, far_strike, far), 1)]
    detail = f"{kind} {near_strike:g}/{far_strike:g} T={near:.2f}/{far:.2f}"
    return _assemble("diagonal_spread", detail, legs, quantity)


@_register(
    title="Double calendar",
    category="Time spread",
    view=("neutral", "long vol"),
    params=(
        _strike_param("put_strike", 0.95, "Strike of the put calendar."),
        _strike_param("call_strike", 1.05, "Strike of the call calendar."),
        _NEAR,
        _FAR,
        _QUANTITY,
    ),
)
def double_calendar(
    put_strike: float,
    call_strike: float,
    near_expiry: float,
    far_expiry: float,
    quantity: float = 1.0,
) -> Strategy:
    """A put calendar below the spot plus a call calendar above it.

    Two tents side by side give a wider profit zone than a single calendar:
    the trade earns theta as long as the stock stays between (or near) the two
    strikes and, being long two back-month options, gains if implied
    volatility rises. Often put on before earnings to play the front-month
    volatility crush.

    Parameters
    ----------
    put_strike, call_strike : float
        Strikes of the put and call calendars (``put_strike < call_strike``).
    near_expiry, far_expiry : float
        Absolute expiries of the options sold and bought (``near < far``).
    quantity : float, default 1.0
        Number of double calendars (> 0).
    """
    low, high = _ascending(put_strike=put_strike, call_strike=call_strike)
    near, far = _near_far(near_expiry, far_expiry)
    legs = [
        (_put(low, near), -1),
        (_put(low, far), 1),
        (_call(high, near), -1),
        (_call(high, far), 1),
    ]
    detail = f"{low:g}/{high:g} T={near:.2f}/{far:.2f}"
    return _assemble("double_calendar", detail, legs, quantity)


# ---------------------------------------------------------------------- #
# Analytics (work on any Instrument or Position)
# ---------------------------------------------------------------------- #
def _as_instrument(obj: Any) -> Instrument:
    if isinstance(obj, Instrument):
        return obj
    if isinstance(obj, Position):
        return CompositeInstrument(obj.label, (obj,))
    raise ValueError(f"expected an Instrument or a Position, got {obj!r}")


def _flat_legs(instrument: Instrument) -> tuple[Position, ...]:
    if isinstance(instrument, CompositeInstrument):
        return instrument.flatten()
    return (Position(instrument, 1.0),)


def _scalar_market(mkt: Any) -> Market:
    if not isinstance(mkt, Market):
        raise ValueError(f"expected a Market, got {mkt!r}")
    if not mkt.is_scalar:
        raise ValueError("strategy analytics need a scalar Market (no array fields)")
    return mkt


def analysis_horizon(strategy: Instrument | Position, mkt: Market | None = None) -> float | None:
    """Date at which the "at expiry" analytics are evaluated.

    It is the FIRST leg expiry still ahead of ``mkt.t`` (simply the first leg
    expiry when no market is given): the common expiry of an ordinary
    strategy, the front expiry of a calendar. Returns ``None`` when no leg
    ever expires (a pure stock position).
    """
    expiries = sorted(
        {leg.expiry for leg in _flat_legs(_as_instrument(strategy)) if leg.expiry is not None}
    )
    if not expiries:
        return None
    if mkt is not None:
        now = float(_scalar_market(mkt).t)
        upcoming = [expiry for expiry in expiries if expiry > now + EXPIRY_TOL]
        return upcoming[0] if upcoming else expiries[-1]
    return expiries[0]


def reference_levels(strategy: Instrument | Position, mkt: Market | None = None) -> list[float]:
    """Sorted price levels that structure the payoff: strikes, barriers and the spot."""
    levels: set[float] = set()
    for leg in _flat_legs(_as_instrument(strategy)):
        for attribute in ("strike", "barrier"):
            value = getattr(leg.instrument, attribute, None)
            if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
                levels.add(float(value))
    if mkt is not None and float(_scalar_market(mkt).spot) > 0:
        levels.add(float(mkt.spot))
    return sorted(levels)


def spot_grid(
    strategy: Instrument | Position,
    mkt: Market,
    n_points: int = 401,
    spot_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Spot axis for profiles and payoff diagrams.

    The default range covers every strike and the current spot with a margin
    that grows with ``vol * sqrt(time to horizon)`` (between 25% and 60%).
    Strikes and the spot are inserted exactly, so payoff kinks stay sharp.

    Parameters
    ----------
    strategy : Instrument or Position
    mkt : Market
        Scalar market (spot, vol and clock are used to size the range).
    n_points : int, default 401
        Number of evenly spaced points (before the exact levels are added).
    spot_range : (float, float), optional
        Explicit ``(low, high)`` range, ``0 < low < high``.
    """
    mkt = _scalar_market(mkt)
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    levels = reference_levels(strategy, mkt) or [100.0]
    if spot_range is None:
        horizon = analysis_horizon(strategy, mkt)
        tau = 1.0 if horizon is None else max(horizon - float(mkt.t), 0.0)
        margin = float(np.clip(2.5 * float(mkt.vol) * math.sqrt(tau), 0.25, 0.6))
        low, high = levels[0] * (1.0 - margin), levels[-1] * (1.0 + margin)
    else:
        low, high = float(spot_range[0]), float(spot_range[1])
        if not 0 < low < high:
            raise ValueError(f"spot_range must satisfy 0 < low < high, got {spot_range!r}")
    inside = [level for level in levels if low <= level <= high]
    return np.unique(np.concatenate([np.linspace(low, high, int(n_points)), inside]))


def net_premium(
    strategy: Instrument | Position, mkt: Market, include_underlying: bool = True
) -> float:
    """Cost of putting the package on today: **debit > 0, credit < 0**.

    Parameters
    ----------
    strategy : Instrument or Position
    mkt : Market
        Scalar market used to price every leg.
    include_underlying : bool, default True
        If True (default) the stock legs are part of the cost, e.g. a covered
        call costs ``spot - call premium``; this is the entry cost used by the
        P&L functions. If False only the option legs are counted, which gives
        the "premium received" an income trader talks about.
    """
    mkt = _scalar_market(mkt)
    legs = _flat_legs(_as_instrument(strategy))
    if not include_underlying:
        legs = tuple(leg for leg in legs if not isinstance(leg.instrument, Underlying))
    return float(sum(leg.price(mkt) for leg in legs))


def payoff_at_expiry(
    strategy: Instrument | Position,
    spots: Any,
    mkt: Market | None = None,
    horizon: float | None = None,
) -> np.ndarray:
    """Value of the package at the analysis horizon, as a function of the spot then.

    For an ordinary strategy this is the terminal payoff and no market is
    needed. For a calendar or diagonal the horizon is the FRONT expiry: the
    expired legs are worth their intrinsic value and the legs still alive are
    priced with Black-Scholes using the vol, rate and dividend yield of
    ``mkt`` -- which is an assumption: the back-month implied vol on that day
    is unknown today, and it is the main risk of the trade.

    Parameters
    ----------
    strategy : Instrument or Position
    spots : array_like
        Spot levels at the horizon.
    mkt : Market, optional
        Scalar market; required only when some leg outlives the horizon.
    horizon : float, optional
        Absolute evaluation date; default :func:`analysis_horizon`. Legs that
        expire before the horizon are counted at their intrinsic value for
        the same spot (the library's settlement convention).

    Returns
    -------
    ndarray
        Same shape as ``spots``.
    """
    instrument = _as_instrument(strategy)
    spots = np.asarray(spots, dtype=float)
    if horizon is None:
        horizon = analysis_horizon(instrument, mkt)
    last_expiry = instrument.expiry
    if horizon is None or last_expiry is None or last_expiry <= horizon + EXPIRY_TOL:
        return np.asarray(instrument.payoff(spots), dtype=float) + np.zeros_like(spots)
    if mkt is None:
        raise ValueError(
            "this strategy has legs expiring after the analysis horizon (calendar / diagonal): "
            "pass a Market so they can be valued at the front expiry"
        )
    at_horizon = _scalar_market(mkt).bumped(spot=spots, t=float(horizon))
    return np.asarray(instrument.price(at_horizon), dtype=float) + np.zeros_like(spots)


def pnl_at_expiry(
    strategy: Instrument | Position,
    spots: Any,
    mkt: Market | None = None,
    entry_cost: float | None = None,
    horizon: float | None = None,
) -> np.ndarray:
    """Profit and loss at the analysis horizon: ``payoff_at_expiry - entry cost``.

    The entry cost defaults to today's :func:`net_premium`. Financing is
    ignored on purpose: the premium is not capitalised to the horizon and
    dividends on stock legs are not added (the textbook payoff diagram).

    Parameters
    ----------
    strategy, spots, mkt, horizon
        See :func:`payoff_at_expiry`.
    entry_cost : float, optional
        Price actually paid for the package (debit > 0, credit < 0). Needed
        if ``mkt`` is not given.
    """
    if entry_cost is None:
        if mkt is None:
            raise ValueError("pnl_at_expiry needs a Market or an explicit entry_cost")
        entry_cost = net_premium(strategy, mkt)
    return payoff_at_expiry(strategy, spots, mkt, horizon) - float(entry_cost)


@dataclass(frozen=True)
class _TerminalProfile:
    """Terminal P&L sampled on the analysis grid, plus its slope at very high spot."""

    function: Callable[[np.ndarray], np.ndarray]
    grid: np.ndarray
    values: np.ndarray
    slope_at_infinity: float
    far_value: float


def _terminal_profile(
    strategy: Instrument | Position, mkt: Market | None, entry_cost: float | None, horizon: float | None
) -> _TerminalProfile:
    instrument = _as_instrument(strategy)
    if entry_cost is None:
        if mkt is None:
            raise ValueError("a Market or an explicit entry_cost is required")
        entry_cost = net_premium(instrument, mkt)
    if horizon is None:
        horizon = analysis_horizon(instrument, mkt)

    def function(spots: np.ndarray) -> np.ndarray:
        return pnl_at_expiry(instrument, spots, mkt, entry_cost, horizon)

    levels = reference_levels(instrument, mkt) or [100.0]
    top = levels[-1]
    grid = np.unique(np.concatenate([np.linspace(0.0, 3.0 * top, _ANALYSIS_POINTS), levels]))
    # The payoff of vanilla legs is exactly linear beyond the last strike, and
    # Black-Scholes values are linear to machine precision this far out.
    far = np.array([1.0e4 * top, 2.0e4 * top])
    # Spots of 0 and 10,000 strikes are probed on purpose; a leg whose pricer
    # cannot cope there (some exotics) must not break the analysis.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        far_values = function(far)
        values = function(grid)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        raise ValueError("the P&L at the horizon could not be evaluated on the analysis grid")
    grid, values = grid[finite], values[finite]
    slope = float((far_values[1] - far_values[0]) / (far[1] - far[0]))
    if not math.isfinite(slope):
        slope = float((values[-1] - values[-2]) / (grid[-1] - grid[-2]))
        far_values = values[-1:]
    scale = sum(abs(leg.quantity) for leg in _flat_legs(instrument)) or 1.0
    if abs(slope) <= _SLOPE_TOL * scale:
        slope = 0.0
    return _TerminalProfile(function, grid, values, slope, float(far_values[0]))


def breakevens(
    strategy: Instrument | Position,
    mkt: Market | None = None,
    entry_cost: float | None = None,
    horizon: float | None = None,
) -> list[float]:
    """Spot levels at the horizon where the P&L changes sign, in increasing order.

    Sign changes are located on a fine grid (every strike included) and
    refined with Brent's method; a breakeven beyond the grid is found from
    the slope of the P&L at very high spot. If the P&L is exactly zero over a
    whole interval, the ends of that interval are returned; a strategy whose
    P&L never changes sign (a box spread) has no breakeven.

    Parameters
    ----------
    strategy, mkt, entry_cost, horizon
        See :func:`pnl_at_expiry`.
    """
    profile = _terminal_profile(strategy, mkt, entry_cost, horizon)
    grid, values = profile.grid, profile.values

    def scalar(spot: float) -> float:
        return float(profile.function(np.array([spot]))[0])

    eps = 1e-9 * grid[-1]
    signs = np.where(np.abs(values) <= eps, 0.0, np.sign(values))
    roots: list[float] = []
    i, last = 0, len(grid) - 1
    while i <= last:
        if signs[i] == 0:
            j = i
            while j < last and signs[j + 1] == 0:
                j += 1
            # ends of the flat-zero stretch, ignoring ends that are just the grid limits
            roots.extend(grid[k] for k in {i, j} if 0 < k < last)
            i = j + 1
            continue
        if i < last and signs[i] * signs[i + 1] < 0:
            roots.append(brentq(scalar, grid[i], grid[i + 1], xtol=1e-12, rtol=1e-14))
        i += 1

    slope = profile.slope_at_infinity
    if slope != 0.0 and signs[-1] != 0 and signs[-1] * slope < 0:
        upper = grid[-1] + 2.0 * abs(values[-1] / slope) + 1.0
        if scalar(upper) * values[-1] < 0:
            roots.append(brentq(scalar, grid[-1], upper, xtol=1e-12, rtol=1e-14))

    unique: list[float] = []
    for root in sorted(float(r) for r in roots):
        if not unique or root - unique[-1] > 1e-7 * max(1.0, root):
            unique.append(root)
    return unique


def _extreme_pnl(profile: _TerminalProfile, sign: float) -> float:
    """Largest (``sign=+1``) or smallest (``sign=-1``) P&L, ``inf`` if unbounded."""
    if sign * profile.slope_at_infinity > 0:
        return sign * math.inf
    grid, values = profile.grid, profile.values
    best_index = int(np.argmax(sign * values))
    candidates = [sign * float(values[best_index])]
    if profile.slope_at_infinity == 0.0:
        candidates.append(sign * profile.far_value)
    low, high = grid[max(best_index - 1, 0)], grid[min(best_index + 1, len(grid) - 1)]
    if high > low:  # refine smooth peaks (calendar tents) between the neighbouring grid points
        refined = minimize_scalar(
            lambda spot: -sign * float(profile.function(np.array([spot]))[0]),
            bounds=(low, high),
            method="bounded",
            options={"xatol": 1e-10},
        )
        candidates.append(-float(refined.fun))
    return sign * max(candidates)


def max_profit(
    strategy: Instrument | Position,
    mkt: Market | None = None,
    entry_cost: float | None = None,
    horizon: float | None = None,
) -> float:
    """Best possible P&L at the horizon; ``math.inf`` when the profit is unlimited.

    "Unlimited" is decided from the SLOPE of the terminal P&L at very high
    spot (positive slope: the profit grows forever), never from the ends of a
    plotting grid. The spot cannot fall below zero, so the downside is always
    bounded and is read at ``spot = 0``.
    """
    return _extreme_pnl(_terminal_profile(strategy, mkt, entry_cost, horizon), +1.0)


def max_loss(
    strategy: Instrument | Position,
    mkt: Market | None = None,
    entry_cost: float | None = None,
    horizon: float | None = None,
) -> float:
    """Worst possible P&L at the horizon, as a SIGNED number; ``-math.inf`` when unlimited.

    A long call returns ``-premium``; a short call returns ``-inf`` (negative
    slope at very high spot); a box spread, which cannot lose, returns its
    (positive) locked-in P&L.

    Notes
    -----
    For calendars the value at the front expiry includes a Black-Scholes
    priced back leg. With a dividend yield, a deep in-the-money long-dated
    call grows like ``S e^(-q tau)`` while the expiring short call costs
    ``S``: the slope is slightly negative and the loss is, strictly speaking,
    unbounded -- the function reports it as such.
    """
    return _extreme_pnl(_terminal_profile(strategy, mkt, entry_cost, horizon), -1.0)


def probability_of_profit(
    strategy: Instrument | Position, mkt: Market, entry_cost: float | None = None
) -> float:
    """Risk-neutral probability that the P&L at the horizon is positive.

    The spot at the horizon is lognormal with drift ``rate - div`` and the
    market's flat volatility; the probability mass of every interval between
    breakevens where the P&L is positive is added up.

    Notes
    -----
    This is the probability *implied by the pricing model*, not a forecast:
    option sellers typically show a high probability of profit precisely
    because their rare losses are large. Always read it next to the maximum
    loss.
    """
    mkt = _scalar_market(mkt)
    horizon = analysis_horizon(strategy, mkt)
    if horizon is None:
        raise ValueError("probability_of_profit needs at least one leg with an expiry")
    profile = _terminal_profile(strategy, mkt, entry_cost, horizon)

    def is_profit(spot: float) -> bool:
        return float(profile.function(np.array([spot]))[0]) > 0.0

    spot, vol, tau = float(mkt.spot), float(mkt.vol), max(horizon - float(mkt.t), 0.0)
    forward = spot * math.exp((float(mkt.rate) - float(mkt.div)) * tau)
    std = vol * math.sqrt(tau)
    if std <= 0 or spot <= 0:
        return 1.0 if is_profit(forward) else 0.0

    def cdf(level: float) -> float:
        if level <= 0:
            return 0.0
        if math.isinf(level):
            return 1.0
        return float(ndtr((math.log(level / forward) + 0.5 * std**2) / std))

    edges = [0.0, *breakevens(strategy, mkt, entry_cost, horizon), math.inf]
    probability = 0.0
    for low, high in zip(edges, edges[1:]):
        probe = 2.0 * low + 1.0 if math.isinf(high) else 0.5 * (low + high)
        if high > low and is_profit(probe):
            probability += cdf(high) - cdf(low)
    return min(max(probability, 0.0), 1.0)


_LEG_GREEKS = ("delta", "gamma", "vega", "theta", "rho")


def _leg_rows(instrument: Instrument, mkt: Market) -> list[dict[str, Any]]:
    rows = []
    for leg in _flat_legs(instrument):
        greeks = leg.greeks(mkt)
        rows.append(
            {
                "leg": leg.label,
                "side": "long" if leg.quantity > 0 else "short",
                "quantity": leg.quantity,
                "instrument": type(leg.instrument).__name__,
                "option_type": getattr(leg.instrument, "option_type", None),
                "strike": getattr(leg.instrument, "strike", None),
                "expiry": leg.expiry,
                "unit_price": float(leg.instrument.price(mkt)),
                "value": float(greeks["price"]),
                **{name: float(greeks[name]) for name in _LEG_GREEKS},
            }
        )
    return rows


def legs_table(
    strategy: Instrument | Position, mkt: Market, trader_units: bool = False
) -> pd.DataFrame:
    """One row per elementary leg: what it is, what it costs, what it contributes.

    Columns: ``leg, side, quantity, instrument, option_type, strike, expiry,
    unit_price, value, delta, gamma, vega, theta, rho``. ``value`` and the
    Greeks are POSITION numbers (already multiplied by the signed quantity),
    so each column sums to the strategy total.

    Parameters
    ----------
    trader_units : bool, default False
        If True, vega is per vol point, theta per calendar day and rho per 1%
        of rate; otherwise RAW derivatives.
    """
    rows = _leg_rows(_as_instrument(strategy), _scalar_market(mkt))
    if trader_units:
        rows = [to_trader_units(row) for row in rows]
    return pd.DataFrame(rows)


def summary(strategy: Instrument | Position, mkt: Market) -> dict[str, Any]:
    """Everything a student wants to know about a strategy, in one dict.

    Keys
    ----
    ``name``, ``key``, ``description``, ``view`` : identification (``key``,
        ``description`` and ``view`` are empty for a non-:class:`Strategy`).
    ``net_premium`` : entry cost of the whole package (debit > 0, credit < 0).
    ``options_premium`` : same, option legs only (stock legs excluded).
    ``premium_type`` : ``"debit"``, ``"credit"`` or ``"zero cost"``.
    ``horizon`` : analysis horizon (front expiry); ``is_multi_expiry`` : bool.
    ``breakevens`` : list of floats; ``max_profit`` / ``max_loss`` : signed
        P&L, possibly ``inf`` / ``-inf``.
    ``probability_of_profit`` : risk-neutral, ``None`` if no leg expires.
    ``greeks`` : RAW units; ``greeks_trader`` : trader units.
    ``legs`` : list of per-leg dicts (the rows of :func:`legs_table`, RAW units).
    """
    mkt = _scalar_market(mkt)
    instrument = _as_instrument(strategy)
    premium = net_premium(instrument, mkt)
    horizon = analysis_horizon(instrument, mkt)
    greeks = {name: float(value) for name, value in instrument.greeks(mkt).items()}
    if abs(premium) <= 1e-12:
        premium_type = "zero cost"
    else:
        premium_type = "debit" if premium > 0 else "credit"
    return {
        "name": instrument.label,
        "key": getattr(instrument, "key", ""),
        "description": getattr(instrument, "description", ""),
        "view": tuple(getattr(instrument, "view", ())),
        "net_premium": premium,
        "options_premium": net_premium(instrument, mkt, include_underlying=False),
        "premium_type": premium_type,
        "horizon": horizon,
        "is_multi_expiry": instrument.expiry != horizon and horizon is not None,
        "breakevens": breakevens(instrument, mkt, premium, horizon),
        "max_profit": max_profit(instrument, mkt, premium, horizon),
        "max_loss": max_loss(instrument, mkt, premium, horizon),
        "probability_of_profit": (
            None if horizon is None else probability_of_profit(instrument, mkt, premium)
        ),
        "greeks": greeks,
        "greeks_trader": to_trader_units(greeks),
        "legs": _leg_rows(instrument, mkt),
    }
