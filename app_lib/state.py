"""Session state: the market every page shares, the book and the simulator.

Streamlit reruns the whole script on every interaction, so anything that must
survive a click lives in ``st.session_state``. This module is the only place
that touches those keys; pages call the accessors below.

The market inputs are stored in the units a trader types (volatility, rate and
dividend yield in **percent**) and converted to the decimals the library expects
by :func:`current_market`.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from optionlab import Book, Market
from optionlab.simulator import TradingSimulator

DEFAULT_SPOT = 100.0
DEFAULT_VOL_PCT = 20.0
DEFAULT_RATE_PCT = 3.0
DEFAULT_DIV_PCT = 1.0
DEFAULT_CASH = 100_000.0

_MARKET_DEFAULTS: dict[str, float] = {
    "mkt_spot": DEFAULT_SPOT,
    "mkt_vol_pct": DEFAULT_VOL_PCT,
    "mkt_rate_pct": DEFAULT_RATE_PCT,
    "mkt_div_pct": DEFAULT_DIV_PCT,
}


# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #
def init_session() -> None:
    """Create every session key once, at the start of each run."""
    for key, value in _MARKET_DEFAULTS.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("trader_units", True)
    st.session_state.setdefault("book", Book("My book", cash=DEFAULT_CASH))
    st.session_state.setdefault("sim", None)
    st.session_state.setdefault("sim_settings", {})


# --------------------------------------------------------------------------- #
# The shared market
# --------------------------------------------------------------------------- #
def current_market() -> Market:
    """The market currently set in the sidebar, in library units (decimals)."""
    return Market(
        spot=float(st.session_state.mkt_spot),
        vol=float(st.session_state.mkt_vol_pct) / 100.0,
        rate=float(st.session_state.mkt_rate_pct) / 100.0,
        div=float(st.session_state.mkt_div_pct) / 100.0,
    )


def spot() -> float:
    """Shortcut for the current spot, used to place default strikes."""
    return float(st.session_state.mkt_spot)


def trader_units() -> bool:
    """True when Greeks should be shown in trader units (the default)."""
    return bool(st.session_state.trader_units)


def _reset_market() -> None:
    """Callback: restore the default market before the rerun."""
    for key, value in _MARKET_DEFAULTS.items():
        st.session_state[key] = value


def sidebar_market() -> Market:
    """Render the market controls in the sidebar and return the resulting market.

    Every page sees the same market, so a strategy, an exotic and the book are
    always priced in the same world — which is the whole point of the sidebar.
    """
    with st.sidebar:
        st.subheader("Market", divider=False)
        st.caption("Shared by every page: the world all instruments are priced in.")

        st.number_input(
            "Spot",
            min_value=1.0,
            max_value=100_000.0,
            step=1.0,
            key="mkt_spot",
            help="Price of the underlying today. Strike defaults follow it, so a strike of "
            "105 with a spot of 100 is 5% out of the money.",
        )
        st.slider(
            "Implied volatility",
            min_value=1.0,
            max_value=150.0,
            step=0.5,
            format="%.1f%%",
            key="mkt_vol_pct",
            help="The volatility the options are quoted at, annualised. A 20% vol means a "
            "one-standard-deviation move of about 20% over a year, or roughly 1.25% a day.",
        )
        st.slider(
            "Interest rate",
            min_value=0.0,
            max_value=10.0,
            step=0.25,
            format="%.2f%%",
            key="mkt_rate_pct",
            help="Continuously compounded risk-free rate. It sets the forward price and the "
            "cost of carrying a hedge.",
        )
        st.slider(
            "Dividend yield",
            min_value=0.0,
            max_value=10.0,
            step=0.25,
            format="%.2f%%",
            key="mkt_div_pct",
            help="Continuous dividend yield of the underlying. It lowers the forward, so it "
            "makes calls cheaper and puts dearer.",
        )

        with st.container(horizontal=True):
            st.button(
                "Reset market",
                icon=":material/restart_alt:",
                on_click=_reset_market,
                help="Back to spot 100, vol 20%, rate 3%, dividend 1%.",
            )

        st.toggle(
            "Trader units",
            key="trader_units",
            help="On: vega per 1 volatility point, theta per calendar day, rho per 1%. "
            "Off: raw mathematical derivatives. Traders quote the first, textbooks the second.",
        )

    return current_market()


# --------------------------------------------------------------------------- #
# The book
# --------------------------------------------------------------------------- #
def get_book() -> Book:
    """The book the user is building, kept across pages and reruns."""
    return st.session_state.book


def set_book(book: Book) -> None:
    st.session_state.book = book


def reset_book(cash: float = DEFAULT_CASH) -> None:
    """Start again with an empty book and fresh cash. Also drops the simulation."""
    st.session_state.book = Book("My book", cash=cash)
    clear_sim()


# --------------------------------------------------------------------------- #
# The simulator
# --------------------------------------------------------------------------- #
def get_sim() -> TradingSimulator | None:
    """The running simulation, or None when none has been started."""
    return st.session_state.sim


def set_sim(sim: TradingSimulator | None, settings: dict[str, Any] | None = None) -> None:
    st.session_state.sim = sim
    if settings is not None:
        st.session_state.sim_settings = settings


def sim_settings() -> dict[str, Any]:
    """The inputs the running simulation was created with (for replay and display)."""
    return st.session_state.sim_settings


def clear_sim() -> None:
    st.session_state.sim = None
    st.session_state.sim_settings = {}
