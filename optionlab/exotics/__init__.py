"""Exotic options: digitals, barriers, Asians and lookbacks.

Every product is a regular :class:`~optionlab.instruments.Instrument` -- it
prices, has Greeks, serialises, goes into a :class:`~optionlab.book.Book` and
through the simulator like a vanilla. Path-dependent ones carry their path
state (barrier hit, running average, running extremes) as fields updated by
``observe``.

Importing this package registers all four classes for deserialisation and
exposes :data:`EXOTIC_REGISTRY`, the UI-friendly catalogue (teaching
description + parameter schema + ``build_default(spot, expiry)``).
"""

from .asian import AVERAGING_TYPES, AsianOption, mc_price_with_control_variate
from .barrier import BARRIER_TYPES, BGK_BETA, BarrierOption, bgk_adjusted_barrier
from .digital import DIGITAL_KINDS, DigitalOption
from .lookback import LOOKBACK_KINDS, LookbackOption, mc_price_brownian_bridge
from .registry import (
    EXOTIC_PARAM_KINDS,
    EXOTIC_REGISTRY,
    ExoticParam,
    ExoticSpec,
    build_exotic,
    get_exotic_spec,
    list_exotics,
)

__all__ = [
    "AVERAGING_TYPES",
    "BARRIER_TYPES",
    "BGK_BETA",
    "DIGITAL_KINDS",
    "LOOKBACK_KINDS",
    "EXOTIC_PARAM_KINDS",
    "AsianOption",
    "BarrierOption",
    "DigitalOption",
    "LookbackOption",
    "bgk_adjusted_barrier",
    "mc_price_with_control_variate",
    "mc_price_brownian_bridge",
    "ExoticParam",
    "ExoticSpec",
    "EXOTIC_REGISTRY",
    "list_exotics",
    "get_exotic_spec",
    "build_exotic",
]
