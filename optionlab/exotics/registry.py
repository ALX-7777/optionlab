"""UI-friendly catalogue of the exotic options.

:data:`EXOTIC_REGISTRY` plays for exotics the role that
:data:`optionlab.strategies.STRATEGY_REGISTRY` plays for strategies: one
:class:`ExoticSpec` per product, holding a teaching description and a
parameter schema whose defaults are expressed RELATIVE to the market (a strike
is a multiple of the spot, an expiry a multiple of the reference time to
expiry), so that a user interface can draw its input widgets automatically and
always start from a sensible contract::

    spec = EXOTIC_REGISTRY["up_and_out_barrier"]
    option = spec.build_default(spot=250.0, expiry=0.5)      # K=250, H=300
    option = spec.build_default(250.0, 0.5, barrier=280.0)   # override one field
    option = build_exotic("asian", option_type="put", strike=95, expiry=1.0)

Every product is built with KEYWORD arguments, which removes the classic trap
of :class:`~optionlab.exotics.barrier.BarrierOption` (``barrier`` comes before
``expiry`` in its positional order).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..instruments import Instrument
from .asian import AVERAGING_TYPES, AsianOption
from .barrier import BarrierOption
from .digital import DigitalOption
from .lookback import LookbackOption

__all__ = [
    "EXOTIC_PARAM_KINDS",
    "ExoticParam",
    "ExoticSpec",
    "EXOTIC_REGISTRY",
    "list_exotics",
    "get_exotic_spec",
    "build_exotic",
]

#: Kinds of parameters (they tell a UI which widget to draw and how the
#: RELATIVE default becomes a concrete value, see :meth:`ExoticParam.resolve`).
EXOTIC_PARAM_KINDS: tuple[str, ...] = (
    "strike",       # price level, default = multiple of the spot
    "barrier",      # price level, default = multiple of the spot
    "expiry",       # absolute date, default = multiple of the reference time to expiry
    "start_time",   # absolute date, default = fraction of the time to expiry (0 = now)
    "amount",       # absolute number (payout, rebate)
    "option_type",  # "call" / "put"
    "choice",       # one of ``choices``
)


@dataclass(frozen=True)
class ExoticParam:
    """Description of one constructor parameter, for automatic UI building.

    Parameters
    ----------
    name : str
        Keyword name in the instrument constructor.
    kind : str
        One of :data:`EXOTIC_PARAM_KINDS`.
    default : float or str
        RELATIVE default (see :data:`EXOTIC_PARAM_KINDS`); absolute for
        ``"amount"``, a plain string for ``"option_type"`` and ``"choice"``.
    description : str
        One-line help text.
    choices : tuple of str
        Allowed values of a ``"choice"`` parameter.
    """

    name: str
    kind: str
    default: float | str
    description: str = ""
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in EXOTIC_PARAM_KINDS:
            raise ValueError(f"ExoticParam.kind must be one of {EXOTIC_PARAM_KINDS}, got {self.kind!r}")
        if self.kind == "choice" and self.default not in self.choices:
            raise ValueError(f"{self.name}: default {self.default!r} is not one of {self.choices}")

    def resolve(self, spot: float, expiry: float, t: float = 0.0) -> float | str:
        """Concrete default value for a given spot, reference expiry and current time."""
        if self.kind in ("strike", "barrier"):
            return float(self.default) * spot
        if self.kind in ("expiry", "start_time"):
            return t + float(self.default) * (expiry - t)
        return self.default


@dataclass(frozen=True)
class ExoticSpec:
    """Registry entry of an exotic product.

    Attributes
    ----------
    name : str
        Registry key, e.g. ``"up_and_out_barrier"``.
    title : str
        Human-readable name.
    instrument_class : type
        The :class:`~optionlab.instruments.Instrument` subclass that is built.
    category : str
        Family used to group products in a menu (``"Digital"``, ``"Barrier"``,
        ``"Asian"``, ``"Lookback"``).
    description : str
        Teaching text: what it pays, who buys it, where the risk hides.
    params : tuple of ExoticParam
        Parameters the user chooses.
    fixed : tuple of (name, value) pairs
        Constructor arguments that define the product and are not exposed
        (e.g. ``(("barrier_type", "up-and-out"),)``); a tuple keeps the spec
        hashable.
    """

    name: str
    title: str
    instrument_class: type[Instrument]
    category: str
    description: str
    params: tuple[ExoticParam, ...]
    fixed: tuple[tuple[str, Any], ...] = ()

    @property
    def schema(self) -> dict[str, ExoticParam]:
        """Parameter name -> :class:`ExoticParam`."""
        return {p.name: p for p in self.params}

    def default_params(self, spot: float, expiry: float = 1.0, t: float = 0.0) -> dict[str, Any]:
        """Default constructor arguments for a given spot and ABSOLUTE reference expiry."""
        spot, expiry, t = float(spot), float(expiry), float(t)
        if not (math.isfinite(spot) and spot > 0):
            raise ValueError(f"spot must be a finite positive number, got {spot!r}")
        if not expiry > t:
            raise ValueError(f"expiry ({expiry:g}) must be later than the current time t ({t:g})")
        return {p.name: p.resolve(spot, expiry, t) for p in self.params}

    def build(self, **params: Any) -> Instrument:
        """Build the product from explicit parameters (validated by the instrument itself)."""
        unknown = set(params) - set(self.schema)
        if unknown:
            raise ValueError(
                f"{self.name}: unknown parameter(s) {sorted(unknown)}; valid: {list(self.schema)}"
            )
        try:
            return self.instrument_class(**dict(self.fixed), **params)
        except TypeError as exc:  # a required constructor argument is missing
            raise ValueError(f"{self.name}: {exc}; expected {list(self.schema)}") from exc

    def build_default(
        self, spot: float, expiry: float = 1.0, t: float = 0.0, **overrides: Any
    ) -> Instrument:
        """Build the product with its default parameters (optionally overridden)."""
        params = self.default_params(spot, expiry, t)
        unknown = set(overrides) - set(params)
        if unknown:
            raise ValueError(
                f"{self.name}: unknown parameter(s) {sorted(unknown)}; valid: {list(params)}"
            )
        params.update(overrides)
        return self.build(**params)


# ---------------------------------------------------------------------- #
# Shared parameters
# ---------------------------------------------------------------------- #
_EXPIRY = ExoticParam("expiry", "expiry", 1.0, "Absolute expiry, in years.")


def _option_type(default: str) -> ExoticParam:
    return ExoticParam("option_type", "option_type", default, "Call or put.")


def _strike(moneyness: float = 1.0, description: str = "Strike price.") -> ExoticParam:
    return ExoticParam("strike", "strike", moneyness, description)


def _barrier_params(option_type: str, barrier: float) -> tuple[ExoticParam, ...]:
    return (
        _option_type(option_type),
        _strike(1.0, "Strike of the underlying vanilla payoff."),
        ExoticParam("barrier", "barrier", barrier, "Barrier level H, watched continuously."),
        _EXPIRY,
        ExoticParam(
            "rebate", "amount", 0.0,
            "Consolation cash: paid at the hit (knock-out) or at expiry if never hit (knock-in).",
        ),
    )


_SPECS: tuple[ExoticSpec, ...] = (
    ExoticSpec(
        name="cash_digital",
        title="Cash-or-nothing digital",
        instrument_class=DigitalOption,
        category="Digital",
        description=(
            "Pays a fixed cash amount if the option finishes in the money, nothing otherwise: "
            "a pure bet on WHERE the spot ends, not on how far. Its price is the discounted "
            "risk-neutral probability of finishing in the money. Near expiry the step payoff "
            "makes delta a spike and gamma flip sign at the strike (pin risk), which is why "
            "desks book and hedge it as a tight call spread."
        ),
        params=(
            _option_type("call"),
            _strike(1.0, "Level the spot must finish above (call) or below (put)."),
            _EXPIRY,
            ExoticParam("payout", "amount", 10.0, "Cash paid when in the money."),
        ),
        fixed=(("kind", "cash"),),
    ),
    ExoticSpec(
        name="asset_digital",
        title="Asset-or-nothing digital",
        instrument_class=DigitalOption,
        category="Digital",
        description=(
            "Delivers the share itself if the option finishes in the money. It is the other "
            "half of a vanilla: asset-or-nothing call minus strike x cash-or-nothing call "
            "equals the vanilla call ('receive the share, pay the strike, both only if in the "
            "money'). The jump at the strike is as large as the strike, so the pin risk is "
            "even bigger than for a cash digital."
        ),
        params=(
            _option_type("call"),
            _strike(1.0, "Level the spot must finish above (call) or below (put)."),
            _EXPIRY,
            ExoticParam("payout", "amount", 1.0, "Number of shares delivered when in the money."),
        ),
        fixed=(("kind", "asset"),),
    ),
    ExoticSpec(
        name="up_and_out_barrier",
        title="Up-and-out barrier",
        instrument_class=BarrierOption,
        category="Barrier",
        description=(
            "A vanilla that DIES the first time the spot trades at or above the barrier. For a "
            "call this is a 'reverse' knock-out: the barrier sits in the money, so the option "
            "loses its whole intrinsic value at the touch. Much cheaper than the vanilla, but "
            "close to the barrier delta turns negative, gamma is large and negative and the "
            "holder is short volatility -- the classic hard-to-hedge exotic."
        ),
        params=_barrier_params("call", 1.20),
        fixed=(("barrier_type", "up-and-out"),),
    ),
    ExoticSpec(
        name="up_and_in_barrier",
        title="Up-and-in barrier",
        instrument_class=BarrierOption,
        category="Barrier",
        description=(
            "A vanilla that only COMES ALIVE if the spot trades at or above the barrier before "
            "expiry. Knock-in plus knock-out (same strike and barrier, no rebate) is the "
            "vanilla, so this is the complement of the up-and-out: an up-and-in call keeps "
            "almost all the value of the vanilla call, because the paths that pay are mostly "
            "the ones that rallied through the barrier."
        ),
        params=_barrier_params("call", 1.20),
        fixed=(("barrier_type", "up-and-in"),),
    ),
    ExoticSpec(
        name="down_and_out_barrier",
        title="Down-and-out barrier",
        instrument_class=BarrierOption,
        category="Barrier",
        description=(
            "A vanilla that dies the first time the spot trades at or below the barrier. For a "
            "call this is a 'regular' barrier: it sits out of the money, where the option is "
            "worth little anyway, so the discount to the vanilla is modest and the Greeks stay "
            "tame. 'I want the call, but I am happy to give it up if the stock first falls 15%'."
        ),
        params=_barrier_params("call", 0.85),
        fixed=(("barrier_type", "down-and-out"),),
    ),
    ExoticSpec(
        name="down_and_in_barrier",
        title="Down-and-in barrier",
        instrument_class=BarrierOption,
        category="Barrier",
        description=(
            "A vanilla that only comes alive if the spot trades at or below the barrier. The "
            "down-and-in PUT is the engine of reverse convertibles and autocallables: the "
            "investor sells crash protection that only activates after a large fall, and is "
            "paid a coupon for it. Its value and its vega explode as the spot approaches the "
            "barrier."
        ),
        params=_barrier_params("put", 0.80),
        fixed=(("barrier_type", "down-and-in"),),
    ),
    ExoticSpec(
        name="asian",
        title="Asian (average price)",
        instrument_class=AsianOption,
        category="Asian",
        description=(
            "Pays on the AVERAGE spot over a window instead of the spot on the last day. The "
            "standard product of commodity and FX hedgers, whose real exposure is an average "
            "price. An average moves less than its last point (about vol / sqrt(3)), so the "
            "option is cheaper than the vanilla, and its Greeks fade as the average fills in: "
            "the opposite of a vanilla, whose gamma explodes into expiry."
        ),
        params=(
            _option_type("call"),
            _strike(1.0, "Strike compared with the average."),
            _EXPIRY,
            ExoticParam(
                "averaging", "choice", "arithmetic",
                "Arithmetic is what trades (lognormal approximation); geometric is exact.",
                choices=AVERAGING_TYPES,
            ),
            ExoticParam("avg_start", "start_time", 0.0, "Absolute start of the averaging window."),
        ),
    ),
    ExoticSpec(
        name="floating_lookback",
        title="Floating-strike lookback",
        instrument_class=LookbackOption,
        category="Lookback",
        description=(
            "Trading with perfect hindsight: the call buys at the LOWEST price seen over the "
            "life (pays S_T - S_min), the put sells at the highest. It is never out of the "
            "money, and hindsight is expensive: a fresh one costs about twice the at-the-money "
            "vanilla. While the spot sits on its running extreme the holder is long a lot of "
            "gamma; far from it the option behaves like a forward."
        ),
        params=(_option_type("call"), _EXPIRY),
        fixed=(("kind", "floating"),),
    ),
    ExoticSpec(
        name="fixed_lookback",
        title="Fixed-strike lookback",
        instrument_class=LookbackOption,
        category="Lookback",
        description=(
            "A vanilla paid on the best price of the whole life instead of the last one: the "
            "call pays max(S_max - K, 0), the put max(K - S_min, 0). Whatever has been reached "
            "is locked in, so the option never gives back its gains; it is always worth more "
            "than the vanilla with the same strike."
        ),
        params=(_option_type("call"), _strike(1.0, "Strike compared with the running extreme."), _EXPIRY),
        fixed=(("kind", "fixed"),),
    ),
)

#: Registry key -> :class:`ExoticSpec`, in menu order.
EXOTIC_REGISTRY: dict[str, ExoticSpec] = {spec.name: spec for spec in _SPECS}


def _normalize_key(name: str) -> str:
    return "_".join(str(name).strip().lower().replace("-", " ").replace("_", " ").split())


def get_exotic_spec(name: str) -> ExoticSpec:
    """Look a spec up by name (case, spaces and hyphens are forgiven)."""
    spec = EXOTIC_REGISTRY.get(_normalize_key(name))
    if spec is None:
        raise ValueError(f"Unknown exotic {name!r}. Available: {list(EXOTIC_REGISTRY)}")
    return spec


def list_exotics(category: str | None = None) -> list[str]:
    """Registry names, optionally restricted to one family (case-insensitive)."""
    return [
        name for name, spec in EXOTIC_REGISTRY.items()
        if category is None or spec.category.lower() == category.strip().lower()
    ]


def build_exotic(name: str, **params: Any) -> Instrument:
    """Build an exotic by registry name: ``build_exotic("cash_digital", strike=100, expiry=1.0)``.

    Parameters left out take the constructor's own defaults; a missing
    required parameter, an unknown one or an invalid value raises ``ValueError``.
    """
    return get_exotic_spec(name).build(**params)
