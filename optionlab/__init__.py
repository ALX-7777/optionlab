"""optionlab -- a Black-Scholes options lab for learning and trading practice.

The package is organised in layers:

* :mod:`optionlab.market` -- the :class:`Market` snapshot (spot, flat vol,
  rate, dividend yield, current time).
* :mod:`optionlab.black_scholes` -- vectorised analytic prices, Greeks and
  implied volatility (``import optionlab.black_scholes as bs``).
* :mod:`optionlab.instruments` -- the instrument interface, vanilla products,
  positions and composite (multi-leg) instruments.
* :mod:`optionlab.numerical` -- generic bump-and-reprice Greeks.
* :mod:`optionlab.monte_carlo` -- GBM path simulation and a generic Monte
  Carlo pricer.
* :mod:`optionlab.strategies` -- predefined option strategies (factories,
  :data:`STRATEGY_REGISTRY`, payoff analytics).
* :mod:`optionlab.exotics` -- digitals, barriers, Asians, lookbacks and
  :data:`EXOTIC_REGISTRY`.
* :mod:`optionlab.book` -- the trading book: positions, cash, aggregated
  Greeks, risk ladders, scenario grids, hedging, JSON persistence.
* :mod:`optionlab.simulator` -- market scenarios, the step-by-step trading
  simulator, hedging policies, P&L attribution and hedging experiments.
* :mod:`optionlab.plotting` -- plotly figures. It is NOT imported here, so
  ``import optionlab`` never pays for plotly; use
  ``from optionlab.plotting import greek_dashboard`` when you need a figure.

Conventions shared by every module: ``Market.t`` is the current time and
instruments carry an ABSOLUTE expiry (``tau = expiry - mkt.t``); option types
are ``"call"`` / ``"put"``; all Greeks are RAW derivatives
(:func:`to_trader_units` converts to desk units); every ``greeks()`` dict has
the keys :data:`GREEK_KEYS`.

Importing the package registers every built-in instrument class, so
:func:`instrument_from_dict` and :meth:`Book.from_json` can always rebuild
vanillas, strategies and exotics.

Examples
--------
>>> import optionlab as ol
>>> mkt = ol.Market(spot=100.0, vol=0.2, rate=0.05)
>>> call = ol.EuropeanOption("call", 100.0, 1.0)
>>> round(call.price(mkt), 4)
10.4506
>>> condor = ol.STRATEGY_REGISTRY["iron_condor"].build_default(spot=100.0, expiry=0.5)
>>> knock_out = ol.EXOTIC_REGISTRY["up_and_out_barrier"].build_default(spot=100.0, expiry=0.5)
>>> book = ol.Book("demo")
>>> _ = book.trade(condor, -10, mkt)
>>> _ = book.trade(knock_out, 5, mkt)
>>> ol.Book.from_json(book.to_json()).value(mkt) == book.value(mkt)
True
"""

from . import black_scholes, exotics, monte_carlo, numerical, simulator, strategies
from .black_scholes import GREEK_INFO, GREEK_NAMES, implied_vol, to_trader_units
from .book import Book, Settlement, Trade, TransactionCosts, shocked_market
from .exotics import (
    EXOTIC_REGISTRY,
    AsianOption,
    BarrierOption,
    DigitalOption,
    LookbackOption,
    build_exotic,
    list_exotics,
)
from .instruments import (
    GREEK_KEYS,
    INSTRUMENT_REGISTRY,
    CompositeInstrument,
    EuropeanOption,
    Instrument,
    Position,
    Underlying,
    instrument_from_dict,
    register_instrument,
)
from .market import Market
from .monte_carlo import mc_price, simulate_gbm_paths
from .numerical import numerical_greeks
from .simulator import (
    DeltaBandHedge,
    DeltaHedgeAtVol,
    DeltaHedgeEveryN,
    HedgingPolicy,
    MarketScenario,
    NoHedge,
    TradingSimulator,
    delta_hedging_experiment,
    explain_pnl,
    gbm_scenario,
    scenario_from_arrays,
)
from .strategies import STRATEGY_REGISTRY, Strategy, build_strategy, list_strategies

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # sub-modules worth using through their namespace
    "black_scholes",
    "numerical",
    "monte_carlo",
    "strategies",
    "exotics",
    "simulator",
    # market and pricing
    "Market",
    "GREEK_KEYS",
    "GREEK_NAMES",
    "GREEK_INFO",
    "to_trader_units",
    "implied_vol",
    "numerical_greeks",
    "simulate_gbm_paths",
    "mc_price",
    # instruments
    "Instrument",
    "Underlying",
    "EuropeanOption",
    "Position",
    "CompositeInstrument",
    "INSTRUMENT_REGISTRY",
    "register_instrument",
    "instrument_from_dict",
    # strategies
    "Strategy",
    "STRATEGY_REGISTRY",
    "list_strategies",
    "build_strategy",
    # exotics
    "DigitalOption",
    "BarrierOption",
    "AsianOption",
    "LookbackOption",
    "EXOTIC_REGISTRY",
    "list_exotics",
    "build_exotic",
    # book
    "Book",
    "Trade",
    "Settlement",
    "TransactionCosts",
    "shocked_market",
    # simulator
    "MarketScenario",
    "gbm_scenario",
    "scenario_from_arrays",
    "TradingSimulator",
    "HedgingPolicy",
    "NoHedge",
    "DeltaHedgeEveryN",
    "DeltaBandHedge",
    "DeltaHedgeAtVol",
    "explain_pnl",
    "delta_hedging_experiment",
]
