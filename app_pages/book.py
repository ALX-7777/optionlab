"""My book — build a portfolio and read its risk the way a trading desk does.

This is the practice ground of the app. Stock, vanillas, the 37 strategies and
the 9 exotics all go into a single netted, self-financing book, and the page
then answers the three questions a risk manager asks every morning: *what is it
worth* (value, P&L, dollar Greeks), *what happens if the market moves* (spot
ladder, spot x vol grid, P&L by horizon, stress tests) and *what do I trade to
fix it* (delta hedge, second-Greek hedge).

Every number on the page is computed by :mod:`optionlab.book` through FULL
repricing — no Taylor approximation, no hand-made formula.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app_lib import state, ui
from optionlab import (
    EXOTIC_REGISTRY,
    GREEK_INFO,
    GREEK_KEYS,
    STRATEGY_REGISTRY,
    Book,
    EuropeanOption,
    Market,
    Underlying,
    to_trader_units,
)
from optionlab.book import DEFAULT_SCENARIOS
from optionlab.plotting import (
    book_greeks_breakdown,
    book_pnl_by_horizon,
    book_risk_profile,
    book_scenario_heatmap,
    stress_test_chart,
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
STOCK, VANILLA, STRATEGY, EXOTIC = "Stock", "Vanilla option", "Strategy", "Exotic"
FAMILIES = (STOCK, VANILLA, STRATEGY, EXOTIC)

MODEL_PRICE, MY_PRICE = "Model price", "My price"

#: Greek columns shown in the positions table unless the user asks for all of them.
HEADLINE_COLUMNS = ("delta", "gamma", "vega", "theta", "rho")

#: The five risk reports, in the order a desk would look at them.
REPORTS = (
    "Risk profile",
    "Spot x vol grid",
    "Greeks by line",
    "P&L by horizon",
    "Stress tests",
)

#: Greeks that can be neutralised with one option on top of the delta hedge.
HEDGEABLE = ("gamma", "vega", "theta", "rho")


# --------------------------------------------------------------------------- #
# Cached risk computations
#
# The cache key is the book's own JSON plus the market: both are primitives, and
# the book is rebuilt inside, so a cached frame can never hold a stale object.
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=32)
def spot_ladder_frame(
    book_json: str,
    spot: float,
    vol: float,
    rate: float,
    div: float,
    shocks: tuple[float, ...],
    trader_units: bool,
) -> pd.DataFrame:
    """Book value, P&L and Greeks repriced at each relative spot shock."""
    book = Book.from_json(book_json)
    market = Market(spot=spot, vol=vol, rate=rate, div=div)
    return book.spot_ladder(market, list(shocks), trader_units=trader_units)


@st.cache_data(show_spinner=False, max_entries=32)
def scenario_grid_frame(
    book_json: str,
    spot: float,
    vol: float,
    rate: float,
    div: float,
    spot_shocks: tuple[float, ...],
    vol_shocks: tuple[float, ...],
    horizon_days: float,
) -> pd.DataFrame:
    """P&L of the book on the spot x vol grid, ``horizon_days`` forward."""
    book = Book.from_json(book_json)
    market = Market(spot=spot, vol=vol, rate=rate, div=div)
    return book.scenario_grid(
        market, list(spot_shocks), list(vol_shocks), horizon_days=horizon_days, quantity="pnl"
    )


@st.cache_data(show_spinner=False, max_entries=32)
def stress_frame(book_json: str, spot: float, vol: float, rate: float, div: float) -> pd.DataFrame:
    """P&L of the book in each of the library's named stress scenarios."""
    book = Book.from_json(book_json)
    return book.stress_tests(Market(spot=spot, vol=vol, rate=rate, div=div))


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def quantity_label(name: str) -> str:
    """Display name of a plotted quantity ("pnl" -> "P&L", "vega" -> "Vega")."""
    if name == "pnl":
        return "P&L"
    if name == "value":
        return "Value"
    return GREEK_INFO.get(name, {}).get("label", name.capitalize())


def scenario_sentence(name: str, shocks: dict) -> str:
    """One plain sentence describing a stress scenario, from the library's own dict."""
    parts = []
    if shocks.get("spot"):
        parts.append(f"the spot moves {shocks['spot'] * 100:+.0f}%")
    if shocks.get("vol"):
        parts.append(f"implied volatility moves {shocks['vol'] * 100:+.0f} points")
    if shocks.get("rate"):
        parts.append(f"rates move {shocks['rate'] * 10_000:+.0f} bp")
    if shocks.get("days"):
        parts.append(f"{shocks['days']:g} calendar days pass")
    return f"**{name}** — " + (" and ".join(parts) if parts else "nothing moves") + "."


def position_labels(book: Book) -> list[str]:
    """One unique, readable label per open line, used by the close-a-line picker."""
    return [
        f"{index + 1}. {position.quantity:+,.6g} x {position.instrument.label}"
        for index, position in enumerate(book.positions)
    ]


def greek_comparison(before: dict, after: dict, *, in_trader_units: bool) -> pd.DataFrame:
    """Before / after table of the headline Greeks, in the units the sidebar asks for."""
    left = to_trader_units(dict(before)) if in_trader_units else dict(before)
    right = to_trader_units(dict(after)) if in_trader_units else dict(after)
    return pd.DataFrame(
        [
            {
                "Greek": GREEK_INFO[key]["label"],
                "Now": float(left[key]),
                "After the hedge": float(right[key]),
                "Change": float(right[key]) - float(left[key]),
            }
            for key in ("delta", "gamma", "vega", "theta")
        ]
    )


def desk_read(raw: dict, dollars: dict) -> str:
    """The book described the way a trader would say it out loud, from its own Greeks."""

    def side(value: float, tolerance: float) -> str | None:
        return None if abs(value) <= tolerance else ("long" if value > 0 else "short")

    delta = side(float(raw["delta"]), 0.5)
    parts = ["delta flat" if delta is None else f"{delta} delta"]
    for name, value in (("gamma", dollars["gamma_cash"]), ("vega", dollars["vega_cash"])):
        exposure = side(float(value), 0.005)
        parts.append(f"no {name}" if exposure is None else f"{exposure} {name}")
    theta = float(dollars["theta_cash"])
    if abs(theta) <= 0.005:
        parts.append("no theta")
    else:
        parts.append("collecting theta" if theta > 0 else "paying theta")
    return ", ".join(parts)


def gamma_sentence(gamma_cash: float) -> str:
    """The trade-off implied by the sign of gamma, in one sentence."""
    if gamma_cash < 0:
        return (
            "You are paid to stand still: the book earns while the spot does nothing and loses on "
            "every large move, in either direction."
        )
    if gamma_cash > 0:
        return (
            "You are paid when the market moves: the book earns on large moves in either direction "
            "and bleeds while nothing happens."
        )
    return "With no gamma, the spot has to move a long way before the delta changes at all."


def flash(kind: str, message: str) -> None:
    """Queue a message to be shown at the top of the page after the rerun."""
    st.session_state["book_flash"] = (kind, message)


# --------------------------------------------------------------------------- #
# Reading the add-position controls back from session state
#
# The controls are read from session state (never from values captured at render
# time), so a click that lands in the same message as an edited input still
# trades what the user can see on screen.
# --------------------------------------------------------------------------- #
def build_from_state() -> tuple[object, float, str]:
    """``(instrument, signed quantity, blotter note)`` from the add-position controls."""
    family = st.session_state.get("book_family") or VANILLA
    spot = state.spot()

    if family == STOCK:
        return Underlying(), float(st.session_state["book_stock_qty"]), "stock"

    if family == VANILLA:
        option = EuropeanOption(
            str(st.session_state["book_vanilla_type"]),
            float(st.session_state["book_vanilla_strike"]),
            float(st.session_state["book_vanilla_days"]) / 365.0,
        )
        return option, float(st.session_state["book_vanilla_qty"]), option.label

    if family == STRATEGY:
        spec = STRATEGY_REGISTRY[str(st.session_state["book_strategy_key"])]
        expiry = float(st.session_state["book_strategy_days"]) / 365.0
        strategy = spec.build_default(spot, expiry)
        return strategy, float(st.session_state["book_strategy_qty"]), spec.title

    spec = EXOTIC_REGISTRY[str(st.session_state["book_exotic_key"])]
    params = {}
    for param in spec.params:
        raw = st.session_state[f"book_{spec.name}_{param.name}"]
        params[param.name] = float(raw) / 365.0 if param.kind in ("expiry", "start_time") else raw
    return spec.build(**params), float(st.session_state["book_exotic_qty"]), spec.title


def execution_from_state() -> tuple[float | None, float]:
    """``(unit price or None for the model price, fee)`` from the execution controls."""
    manual = st.session_state.get("book_price_mode") == MY_PRICE
    price = float(st.session_state.get("book_price", 0.0)) if manual else None
    return price, float(st.session_state.get("book_fee", 0.0))


# --------------------------------------------------------------------------- #
# Callbacks: the only places where the book is mutated
# --------------------------------------------------------------------------- #
def add_position() -> None:
    """Trade the instrument currently described by the add-position controls."""
    try:
        instrument, quantity, note = build_from_state()
    except (KeyError, ValueError) as exc:
        flash("error", f"This contract cannot be built: {exc}")
        return
    if quantity == 0.0:
        flash("warning", "Quantity is zero. Use a positive quantity to buy, a negative one to sell.")
        return

    price, fee = execution_from_state()
    try:
        trade = state.get_book().trade(
            instrument, quantity, state.current_market(), price=price, fee=fee, note=note
        )
    except ValueError as exc:
        flash("error", f"Trade rejected: {exc}")
        return

    side = "Bought" if quantity > 0 else "Sold"
    flash(
        "success",
        f"{side} {abs(quantity):,.6g} x {instrument.label} at {ui.money(trade.price)} per unit. "
        f"Cash {ui.money(trade.cash_flow, signed=True)}.",
    )


def close_line() -> None:
    """Trade the opposite of the selected line, back to flat."""
    book = state.get_book()
    labels = position_labels(book)
    pick = st.session_state.get("book_close_pick")
    if pick not in labels:
        flash("warning", "That line is no longer in the book.")
        return
    position = book.positions[labels.index(pick)]
    try:
        trade = book.close_position(position.instrument, state.current_market())
    except ValueError as exc:
        flash("error", f"Could not close the line: {exc}")
        return
    flash(
        "success",
        f"Closed {position.quantity:+,.6g} x {position.instrument.label} at "
        f"{ui.money(trade.price)} per unit. Cash {ui.money(trade.cash_flow, signed=True)}.",
    )


def hedge_delta() -> None:
    """Buy or sell the underlying so that the book delta reaches the target."""
    book = state.get_book()
    market = state.current_market()
    target = float(st.session_state.get("book_hedge_target", 0.0))
    band = float(st.session_state.get("book_hedge_band", 0.0))
    try:
        trade = book.hedge_delta(market, target_delta=target, min_trade=band)
    except ValueError as exc:
        flash("error", f"Delta hedge rejected: {exc}")
        return
    if trade is None:
        reason = (
            "the book is already at the target delta"
            if band <= 0
            else f"the hedge is smaller than the {band:,.0f}-share no-trade band"
        )
        flash("warning", f"Nothing done: {reason}.")
        return
    side = "Bought" if trade.quantity > 0 else "Sold"
    flash("success", f"{side} {abs(trade.quantity):,.2f} shares at {ui.money(trade.price)}. Delta is now {target:,.2f}.")


def hedge_second_greek() -> None:
    """Neutralise a second Greek with the chosen option (and re-hedge delta if asked)."""
    book = state.get_book()
    market = state.current_market()
    greek = str(st.session_state.get("book_hedge_greek", "vega"))
    try:
        option = EuropeanOption(
            str(st.session_state["book_hedge_type"]),
            float(st.session_state["book_hedge_strike"]),
            float(st.session_state["book_hedge_days"]) / 365.0,
        )
    except (KeyError, ValueError) as exc:
        flash("error", f"The hedge option cannot be built: {exc}")
        return

    try:
        if st.session_state.get("book_hedge_with_delta", True):
            trades = book.hedge(market, [Underlying(), option], targets=("delta", greek))
        else:
            trade = book.neutralise(market, greek, option)
            trades = [] if trade is None else [trade]
    except ValueError as exc:
        flash("error", f"Hedge rejected: {exc}")
        return

    if not trades:
        flash("warning", f"Nothing done: the book is already flat on {greek}.")
        return
    done = ", ".join(f"{trade.quantity:+,.2f} x {trade.instrument.label}" for trade in trades)
    flash("success", f"{greek.capitalize()} hedge executed: {done}.")


def load_sample_book() -> None:
    """Fill the book with a small, realistic income position, delta-hedged."""
    book = state.get_book()
    market = state.current_market()
    spot = state.spot()
    try:
        book.trade(
            STRATEGY_REGISTRY["iron_condor"].build_default(spot, 0.25),
            200,
            market,
            note="income: three-month iron condors",
        )
        book.trade(
            EuropeanOption("put", round(spot * 0.90, 2), 0.25), 80, market, note="tail hedge: crash puts"
        )
        book.hedge_delta(market, note="opening delta hedge")
    except ValueError as exc:
        flash("error", f"Could not build the sample book: {exc}")
        return
    flash(
        "success",
        "Sample book loaded: 200 three-month iron condors for income, 80 crash puts as a tail "
        "hedge, and a delta hedge in stock. Look at the headline Greeks to see what you now own.",
    )


def load_uploaded_book() -> None:
    """Replace the book with the uploaded JSON file."""
    uploaded = st.session_state.get("book_upload")
    if uploaded is None:
        flash("warning", "Choose a JSON file first.")
        return
    try:
        loaded = Book.from_json(uploaded.getvalue().decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        flash("error", f"That file is not a book: {exc}")
        return
    state.set_book(loaded)
    flash("success", f"Loaded {loaded.name!r}: {len(loaded)} line(s), cash {ui.money(loaded.cash)}.")


def reset_book() -> None:
    """Start again with an empty book and fresh cash."""
    cash = float(st.session_state.get("book_reset_cash", 100_000.0))
    state.reset_book(cash=cash)
    flash("success", f"Book reset with {ui.money(cash)} of cash. The trading simulation was cleared too.")


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
mkt = state.current_market()
spot = state.spot()
trader_units = state.trader_units()
book = state.get_book()

ui.page_header(
    "My book",
    "Trade stock, vanillas, strategies and exotics into one netted book, then read its risk "
    "the way a desk does: value, dollar Greeks, ladders, scenarios and hedges.",
)

ui.how_to_use(
    """
The book is the position you carry across the whole app: anything you trade here, on the
Strategies page or on the Exotics page lands in the same place, and the trading simulator
starts from it.

- **Positions** — add stock, a vanilla, one of the 37 strategies or one of the 9 exotics.
  The book *nets*, so buying and selling the same contract leaves no line at all.
- **Risk reports** — five ways of asking what happens if the market moves, each a full
  repricing rather than a Taylor approximation. Only the selected report is computed.
- **Hedging** — what to trade to flatten the delta, and what a second instrument would
  have to do to flatten a second Greek.
- **Blotter and file** — every ticket in execution order, and the book as a JSON document
  you can download and reload before an interview.

Start with *Load a sample book* if the book is empty: it opens an income position with
enough risk to make every report worth reading.
"""
)

pending = st.session_state.pop("book_flash", None)
if pending:
    kind, message = pending
    icons = {
        "success": ":material/check_circle:",
        "warning": ":material/warning:",
        "error": ":material/error:",
    }
    getattr(st, kind)(message, icon=icons.get(kind, ":material/info:"))

# --- Headline -------------------------------------------------------------- #
raw_greeks = book.greeks(mkt)
dollars = book.dollar_greeks(mkt)
ui.metric_row(
    [
        (
            "Net liquidation value",
            ui.money(book.value(mkt)),
            "Positions marked to model plus cash: what you would walk away with if you closed "
            "everything right now at the model price.",
        ),
        (
            "Cash",
            ui.money(book.cash),
            "The cash account. Every trade is paid for out of it, so the book is self-financing: "
            "buying an option moves value from cash into the position, it does not create P&L.",
        ),
        (
            "P&L since inception",
            ui.money(book.pnl(mkt), signed=True),
            "Net liquidation value minus the capital put in. Trading at the model price is P&L "
            "neutral, so this only moves when the market moves or when you deal away from the model.",
        ),
        ("Open lines", f"{len(book)}", "Number of netted instrument lines. Cash is not a line."),
    ]
)
ui.metric_row(
    [
        (
            "Delta cash",
            ui.money(dollars["delta_cash"], decimals=0, signed=True),
            "delta x spot: the stock-equivalent exposure in currency. A +1% spot move earns about "
            "delta cash / 100.",
        ),
        (
            "Gamma cash (per 1%)",
            ui.money(dollars["gamma_cash"], decimals=0, signed=True),
            "gamma x spot² / 100: how much delta cash changes for a +1% spot move. The gamma P&L "
            "of a 1% move is gamma cash / 200.",
        ),
        (
            "Vega (per vol point)",
            ui.money(dollars["vega_cash"], decimals=0, signed=True),
            "vega / 100: P&L for +1 volatility point (for example implied going from 20% to 21%).",
        ),
        (
            "Theta (per day)",
            ui.money(dollars["theta_cash"], decimals=0, signed=True),
            "theta / 365: P&L of one calendar day passing with nothing else moving. Negative means "
            "the book bleeds; that is the rent you pay for owning gamma.",
        ),
    ]
)
st.caption(
    "Dollar Greeks are always in currency for a standard market move, whatever the sidebar units "
    "toggle says. They are the numbers a desk quotes to a risk manager."
)

if not book.is_empty:
    with st.container(border=True):
        st.markdown(
            f":green-badge[:material/record_voice_over: Say it out loud] "
            f"**{desk_read(raw_greeks, dollars).capitalize()}.** "
            f"{gamma_sentence(float(dollars['gamma_cash']))}"
        )

if book.is_empty:
    ui.empty_state(
        "Your book is empty, so there is nothing to risk-manage yet. Add a position below — stock, "
        "a vanilla, one of the 37 strategies or one of the 9 exotics. The Strategies and Exotics "
        "pages are good places to decide what you want first.",
        icon=":material/inventory_2:",
    )
    with st.container(horizontal=True):
        st.button(
            "Load a sample book",
            icon=":material/playlist_add:",
            type="primary",
            key="book_sample",
            on_click=load_sample_book,
            help="An income book: 200 three-month iron condors, 80 crash puts as a tail hedge and "
            "a delta hedge in stock. Enough risk to make every report on this page worth reading.",
        )
        st.page_link("app_pages/strategies.py", label="Design a strategy", icon=":material/account_tree:")
        st.page_link("app_pages/exotics.py", label="Pick an exotic", icon=":material/diamond:")

# Plain sentence-case tab labels, like every other page in the app: icons live on
# buttons, expanders and callouts, never inside a tab label.
positions_tab, risk_tab, hedging_tab, file_tab = st.tabs(
    ["Positions", "Risk reports", "Hedging", "Blotter and file"]
)

# --------------------------------------------------------------------------- #
# Positions tab
# --------------------------------------------------------------------------- #
with positions_tab:
    ui.section(
        "Add a position",
        "Pick what you want to trade. The book nets it against what you already hold, and a "
        "multi-leg package is stored leg by leg so the legs can expire independently.",
    )

    st.segmented_control(
        "Instrument family",
        FAMILIES,
        default=VANILLA,
        required=True,
        key="book_family",
        help="Stock is the pure delta instrument. A vanilla is one call or one put. A strategy is "
        "a named package of vanilla legs. An exotic is a path-dependent product.",
    )
    family = st.session_state.get("book_family") or VANILLA

    if family == STOCK:
        st.number_input(
            "Shares (+ buy, - sell)",
            value=100.0,
            step=10.0,
            key="book_stock_qty",
            help="The stock has a delta of exactly 1 and no other Greek, which is why it is the "
            "natural hedge for the delta of an option book.",
        )
        st.caption(f"One share is worth the spot, {ui.money(spot)}.")

    elif family == VANILLA:
        columns = st.columns(4)
        columns[0].segmented_control(
            "Type", ["call", "put"], default="call", required=True, key="book_vanilla_type"
        )
        strike = columns[1].number_input(
            "Strike",
            min_value=0.01,
            value=float(round(spot, 2)),
            step=max(0.5, round(spot / 100.0, 2)),
            key="book_vanilla_strike",
        )
        columns[2].slider(
            "Days to expiry", min_value=1, max_value=1095, value=90, step=1, key="book_vanilla_days"
        )
        columns[3].number_input(
            "Quantity (+ long, - short)",
            value=10.0,
            step=1.0,
            key="book_vanilla_qty",
            help="One option is on one unit of the underlying (no contract multiplier).",
        )
        vanilla_type = str(st.session_state.get("book_vanilla_type") or "call")
        st.caption(f"Strike {ui.money(strike)} is {ui.moneyness_caption(strike, spot, vanilla_type)}.")

    elif family == STRATEGY:
        keys = list(STRATEGY_REGISTRY)
        columns = st.columns([2, 1, 1])
        strategy_key = columns[0].selectbox(
            "Strategy",
            keys,
            index=keys.index("iron_condor"),
            format_func=lambda name: f"{STRATEGY_REGISTRY[name].title}  ({STRATEGY_REGISTRY[name].category})",
            key="book_strategy_key",
            help="All 37 strategies of the library, built with their default strikes around the "
            "current spot. Use the Strategies page to design one leg by leg.",
        )
        columns[1].slider(
            "Days to expiry", min_value=7, max_value=1095, value=90, step=1, key="book_strategy_days"
        )
        columns[2].number_input(
            "Packages (+ long, - short)",
            value=10.0,
            step=1.0,
            key="book_strategy_qty",
            help="Negative sells the whole package. Watch the sign: several factories already "
            "describe the SOLD structure — the iron condor sells both wings — so a positive "
            "quantity is the income trade there.",
        )
        spec = STRATEGY_REGISTRY[strategy_key]
        st.markdown(" ".join(f":blue-badge[{tag}]" for tag in spec.view))
        st.caption(spec.description)

    else:
        keys = list(EXOTIC_REGISTRY)
        columns = st.columns([2, 1])
        exotic_key = columns[0].selectbox(
            "Exotic",
            keys,
            format_func=lambda name: f"{EXOTIC_REGISTRY[name].title}  ({EXOTIC_REGISTRY[name].category})",
            key="book_exotic_key",
            help="Path-dependent products. The book stores the contract as it is the moment you "
            "trade it, so a fresh Asian starts averaging and a lookback starts looking from today.",
        )
        columns[1].number_input(
            "Quantity (+ long, - short)", value=10.0, step=1.0, key="book_exotic_qty"
        )
        spec = EXOTIC_REGISTRY[exotic_key]
        st.caption(spec.description)
        ui.param_controls(spec, spot, "book", columns=3)

    with st.expander("Execution: price and fees", icon=":material/payments:"):
        st.segmented_control(
            "Traded at",
            [MODEL_PRICE, MY_PRICE],
            default=MODEL_PRICE,
            required=True,
            key="book_price_mode",
            help="At the model price a trade is P&L neutral: value simply moves between cash and "
            "the position. Deal away from it to practise buying cheap or selling rich — the "
            "difference is booked as immediate mark-to-model P&L.",
        )
        price_columns = st.columns(2)
        price_columns[0].number_input(
            "My unit price",
            min_value=0.0,
            value=1.0,
            step=0.25,
            key="book_price",
            disabled=st.session_state.get("book_price_mode") != MY_PRICE,
            help="Price per unit of the instrument (per package for a strategy).",
        )
        price_columns[1].number_input(
            "Fee for this ticket",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key="book_fee",
            help="A fixed cost, always paid out of cash. Real desks also pay a bid-ask spread on "
            "every leg, which is why a four-leg condor is expensive to trade.",
        )

    try:
        preview_instrument, preview_quantity, _ = build_from_state()
        build_error = None
    except (KeyError, ValueError) as exc:
        preview_instrument, preview_quantity, build_error = None, 0.0, str(exc)

    if build_error is not None:
        st.warning(f"This contract cannot be built: {build_error}", icon=":material/warning:")
    else:
        unit_price = float(preview_instrument.price(mkt))
        manual_price, ticket_fee = execution_from_state()
        traded_price = unit_price if manual_price is None else manual_price
        cash_flow = -preview_quantity * traded_price - ticket_fee
        edge = preview_quantity * (unit_price - traded_price)

        st.caption(f"Ticket: {preview_quantity:+,.6g} x **{preview_instrument.label}**.")
        preview = [
            ("Model price (per unit)", ui.money(unit_price), "What the library says one unit is worth today."),
            (
                "Cash if you trade now",
                ui.money(cash_flow, decimals=2, signed=True),
                "-(quantity x price) - fees. Negative means you pay.",
            ),
        ]
        if manual_price is not None:
            preview.append(
                (
                    "Edge vs model",
                    ui.money(edge, decimals=2, signed=True),
                    "Immediate mark-to-model P&L of dealing away from the model price. Positive "
                    "means you bought below (or sold above) the model.",
                )
            )
        ui.metric_row(preview)

        if abs(preview_quantity) > 0:
            st.caption("Risk this ticket would add to the book:")
            ui.greek_metrics(
                preview_instrument.greeks(mkt), trader_units=trader_units, quantity=preview_quantity
            )
            ui.units_caption(trader_units)

        observed = preview_instrument.observe(spot, mkt.t)
        if getattr(observed, "knocked", False) and not getattr(preview_instrument, "knocked", False):
            st.warning(
                "At the current spot this barrier is already touched: the book would knock it the "
                "moment the trade is booked. Move the barrier, or move the spot in the sidebar.",
                icon=":material/warning:",
            )

    with st.container(horizontal=True):
        st.button(
            "Add to the book",
            icon=":material/add:",
            type="primary",
            key="book_add",
            on_click=add_position,
            disabled=build_error is not None,
        )

    # --- Positions table --------------------------------------------------- #
    ui.section(
        "Positions",
        "One row per netted line, marked to model in the sidebar's market. Greek columns are "
        "POSITION Greeks — already multiplied by the quantity — so the total row is their sum.",
    )

    if book.is_empty:
        ui.empty_state("No open lines. Add a position above to see the table.")
    else:
        show_all = st.toggle(
            "Show every Greek",
            key="book_all_greeks",
            help="Adds the higher-order Greeks: vanna, volga, charm, speed, colour and zomma.",
        )
        greek_columns = list(GREEK_KEYS[1:]) if show_all else list(HEADLINE_COLUMNS)
        frame = book.positions_frame(mkt, trader_units=trader_units, include_cash=True)
        display = frame[["label", "quantity", "expiry", "unit_price", "value", *greek_columns]]

        column_config = {
            "label": st.column_config.TextColumn("Line", pinned=True),
            "quantity": st.column_config.NumberColumn(
                "Quantity", format="localized", help="Signed: positive is long, negative is short."
            ),
            "expiry": st.column_config.NumberColumn(
                "Expiry (years)",
                format="%.2f",
                help="Absolute expiry. Blank for the stock and for the cash row, which never expire.",
            ),
            "unit_price": st.column_config.NumberColumn(
                "Unit price", format="accounting", help="Model price of ONE unit."
            ),
            "value": st.column_config.NumberColumn(
                "Value", format="accounting", help="Quantity x unit price. The cash row is included, "
                "so the total is the net liquidation value."
            ),
        }
        for name in greek_columns:
            column_config[name] = st.column_config.NumberColumn(
                GREEK_INFO[name]["label"],
                format="localized",
                help=ui.greek_help(name, trader_units=trader_units),
            )
        ui.table(display, column_config=column_config)
        ui.units_caption(trader_units)

        labels = position_labels(book)
        if st.session_state.get("book_close_pick") not in labels:
            st.session_state["book_close_pick"] = labels[0]
        close_columns = st.columns([3, 1], vertical_alignment="bottom")
        close_columns[0].selectbox(
            "Line to close",
            labels,
            key="book_close_pick",
            help="Closing trades the opposite quantity at the model price, using the instrument "
            "AS STORED in the book — which matters for path-dependent products that carry their "
            "own running state.",
        )
        close_columns[1].button(
            "Close this line", icon=":material/close:", key="book_close", on_click=close_line
        )

    ui.explain(
        "How to read the positions table",
        "- **Value** is the mark to model, not what you paid. The difference between the two is "
        "your P&L on that line.\n"
        "- **Greek columns are position Greeks**: a line of 10 calls with a delta of 0.5 each shows "
        "5, not 0.5. They add up, which is the whole point of running a book — you hedge the NET "
        "exposure, not each ticket.\n"
        "- A short line shows negative Greeks: selling a call is short delta, short gamma, short "
        "vega and LONG theta.\n"
        "- Buying two of a contract and selling two of the same contract leaves no line at all: "
        "the book nets, exactly like a real position keeping system.",
    )

    ui.interview_note(
        "That is long realised volatility and short implied volatility — typically long short-dated "
        "options (where gamma lives) against short long-dated ones (where vega lives), a short "
        "calendar. You earn on the gamma by re-hedging when the spot actually moves, and you earn "
        "again if implied volatility falls. What kills you is a quiet market with a bid implied: you "
        "pay theta every day for gamma you never get to monetise, while the short vega leg makes "
        "nothing. Say the view in one line — *realised will beat implied* — then name the risk: the "
        "term structure moving against you, and the fact that long gamma and short vega is a bet on "
        "two different things that can easily go opposite ways.",
        question="You are long 500 gamma and short 2,000 vega. What view is that, and what kills you?",
    )

# --------------------------------------------------------------------------- #
# Risk reports tab
# --------------------------------------------------------------------------- #
with risk_tab:
    if book.is_empty:
        ui.empty_state(
            "Every report here is a full repricing of the book, and an empty book has no risk. "
            "Add a position on the Positions tab first.",
            icon=":material/monitoring:",
        )
    else:
        book_json = book.to_json(indent=None)
        gap_ladder = spot_ladder_frame(
            book_json, mkt.spot, mkt.vol, mkt.rate, mkt.div, (-0.10, 0.0, 0.10), trader_units
        )
        gap_down = float(gap_ladder["pnl"].iloc[0])
        gap_up = float(gap_ladder["pnl"].iloc[2])
        breakeven = book.breakeven_move(mkt, days=1.0)

        ui.metric_row(
            [
                (
                    "P&L if the spot gaps -10%",
                    ui.money(gap_down, decimals=0, signed=True),
                    "Full repricing at a spot 10% lower, everything else unchanged. This is the "
                    "honest number for a large move — delta and gamma are only local.",
                ),
                (
                    "P&L if the spot gaps +10%",
                    ui.money(gap_up, decimals=0, signed=True),
                    "Same repricing, 10% higher. Compare the two: a symmetric book loses about the "
                    "same both ways, a short-put book does not.",
                ),
                (
                    "Daily break-even move",
                    ui.percent(breakeven),  # 'n/a' when gamma and theta share a sign
                    "The spot move over one day at which gamma P&L exactly offsets theta. Long "
                    "gamma needs MORE than this to make money; short gamma needs less. It is 'n/a' "
                    "when gamma and theta have the same sign, because there is then no trade-off.",
                ),
            ]
        )

        report = st.segmented_control(
            "Report",
            REPORTS,
            default=REPORTS[0],
            required=True,
            key="book_report",
            help="Five ways of asking the same question: what happens to this book if the market "
            "moves. Only the selected one is computed.",
        )
        report = report or REPORTS[0]

        if report == "Risk profile":
            controls = st.columns([1, 2])
            span = controls[0].slider(
                "Spot range", min_value=5, max_value=60, value=30, step=5, format="%d%%",
                key="book_profile_range",
                help="How far either side of the spot the x-axis goes.",
            )
            panels = controls[1].pills(
                "Panels",
                ["pnl", "delta", "gamma", "vega", "theta"],
                selection_mode="multi",
                default=["pnl", "delta", "gamma", "vega"],
                format_func=quantity_label,
                key="book_profile_panels",
            )
            if not panels:
                st.warning("Pick at least one panel to plot.", icon=":material/warning:")
            else:
                with st.spinner("Repricing the book across the spot range..."):
                    figure = book_risk_profile(
                        book,
                        mkt,
                        greeks=tuple(panels),
                        spot_range=(-span / 100.0, span / 100.0),
                        trader_units=trader_units,
                        mode=ui.CHART_MODE,
                    )
                ui.chart(figure, key="book_chart_profile")

                ui.explain(
                    "How to read this",
                    "Read the panels together. The **P&L** panel is the picture; **delta** is its "
                    "slope and **gamma** its curvature, so a steep P&L means a big delta and a "
                    "curved P&L means gamma. The dashed vertical line is today's spot.\n\n"
                    "Watch what the Greeks themselves do as you move along the x-axis. A book that "
                    "is delta-neutral today is only flat AT the dashed line: the delta panel shows "
                    "how fast the hedge goes stale. A short-gamma book gets LONGER delta as the "
                    "market falls — exactly when you least want it — which is why short-gamma books "
                    "are hedged mechanically rather than by judgement.",
                )
                ui.takeaways(
                    [
                        f"Delta goes from **{ui.number(float(gap_ladder['delta'].iloc[0]), decimals=1)}** "
                        f"at -10% to **{ui.number(float(gap_ladder['delta'].iloc[2]), decimals=1)}** at "
                        "+10%. Rising with the spot is long gamma; falling is short gamma.",
                        f"A 10% fall reprices the book at **{ui.money(gap_down, decimals=0, signed=True)}**, "
                        f"a 10% rally at **{ui.money(gap_up, decimals=0, signed=True)}**. The asymmetry "
                        "between the two is the shape you are really carrying.",
                        "If a Greek panel is flat at zero, that risk is genuinely absent. If it "
                        "crosses zero inside the range, your exposure changes sign as the market moves.",
                    ]
                )

        elif report == "Spot x vol grid":
            controls = st.columns(3)
            span = controls[0].slider(
                "Spot shocks", min_value=4, max_value=40, value=20, step=2, format="%d%%",
                key="book_grid_span", help="The grid runs from minus to plus this, in 9 steps.",
            )
            vol_span = controls[1].slider(
                "Vol shocks", min_value=2, max_value=25, value=10, step=1, format="%d pts",
                key="book_grid_vol", help="Absolute moves of implied volatility, in 5 steps.",
            )
            horizon = controls[2].slider(
                "Horizon", min_value=0, max_value=90, value=0, step=1, format="%d days",
                key="book_grid_days",
                help="Calendar days that pass before the shock hits. Move it forward to see theta "
                "eat into (or pay for) the shock.",
            )
            spot_shocks = tuple(float(x) for x in np.round(np.linspace(-span / 100.0, span / 100.0, 9), 4))
            vol_shocks = tuple(float(x) for x in np.round(np.linspace(-vol_span / 100.0, vol_span / 100.0, 5), 4))

            with st.spinner("Repricing the book on the grid..."):
                figure = book_scenario_heatmap(
                    book,
                    mkt,
                    spot_shocks,
                    vol_shocks,
                    horizon_days=float(horizon),
                    quantity="pnl",
                    trader_units=trader_units,
                    mode=ui.CHART_MODE,
                )
            ui.chart(figure, key="book_chart_grid")

            grid = scenario_grid_frame(
                book_json, mkt.spot, mkt.vol, mkt.rate, mkt.div, spot_shocks, vol_shocks, float(horizon)
            )
            worst_spot, worst_vol = grid.stack().idxmin()
            worst = float(grid.loc[worst_spot, worst_vol])
            best_spot, best_vol = grid.stack().idxmax()
            best = float(grid.loc[best_spot, best_vol])

            ui.explain(
                "How to read this",
                "Each cell is the book repriced with BOTH shocks applied at once, "
                f"{'today' if horizon == 0 else f'{horizon} calendar days forward'}. Blue is a gain, "
                "red a loss, and the signed number in the cell means you never have to trust the "
                "colour alone.\n\n"
                "Corners are where books die. Bottom-left (spot down, vol up) is the crash corner, "
                "because equity vol jumps when the market falls — that is why the library shocks "
                "spot and vol together in its stress scenarios rather than one at a time. A "
                "long-gamma, long-vega book is blue in every corner and red in the quiet middle; a "
                "short-option book is the mirror image and only makes money if nothing happens.",
            )
            ui.takeaways(
                [
                    f"Worst cell: spot **{worst_spot * 100:+.0f}%** with vol **{worst_vol * 100:+.0f} "
                    f"points**, for **{ui.money(worst, decimals=0, signed=True)}**. That is the "
                    "scenario to have an answer for.",
                    f"Best cell: spot **{best_spot * 100:+.0f}%** with vol **{best_vol * 100:+.0f} "
                    f"points**, for **{ui.money(best, decimals=0, signed=True)}**.",
                    "Move the horizon slider forward: for a short-option book the whole grid drifts "
                    "blue as theta is collected, for a long-option book it drifts red.",
                ]
            )

        elif report == "Greeks by line":
            choices = ["value", *GREEK_KEYS[1:]]
            greek = st.selectbox(
                "Quantity",
                choices,
                index=choices.index("delta"),
                format_func=quantity_label,
                key="book_breakdown_greek",
            )
            figure = book_greeks_breakdown(
                book, mkt, greek=greek, trader_units=trader_units, mode=ui.CHART_MODE
            )
            ui.chart(figure, key="book_chart_breakdown")

            frame = book.positions_frame(mkt, trader_units=trader_units)
            lines = frame.iloc[:-1]
            biggest = lines.loc[lines[greek].abs().idxmax()] if len(lines) else None

            ui.explain(
                "How to read this",
                "One bar per line, coloured by product (calls blue, puts orange, the stock yellow, "
                "everything else violet), and the grey bar at the bottom is the book total.\n\n"
                "This is the chart to look at BEFORE hedging, because it answers a question the "
                "total cannot: is the book flat because it holds nothing, or because a large long "
                "and a large short happen to offset? The second kind of flat is far more fragile — "
                "the two legs can move apart (different strikes, different expiries, different "
                "implied vols) and the netting you were relying on disappears.",
            )
            if biggest is not None:
                ui.takeaways(
                    [
                        f"**{biggest['label']}** carries the most {quantity_label(greek).lower()}: "
                        f"**{ui.number(float(biggest[greek]), decimals=2)}** out of a book total of "
                        f"**{ui.number(float(frame.iloc[-1][greek]), decimals=2)}**.",
                        "Switch the quantity: the line that dominates delta is rarely the line that "
                        "dominates vega. Delta lives in the stock and in deep options, gamma in "
                        "short-dated at-the-money options, vega in long-dated ones.",
                    ]
                )

        elif report == "P&L by horizon":
            controls = st.columns([2, 1])
            horizons = controls[0].pills(
                "Horizons (calendar days)",
                [0, 1, 7, 30, 90, 180],
                selection_mode="multi",
                default=[0, 7, 30, 90],
                format_func=lambda days: "Today" if days == 0 else f"{days} d",
                key="book_horizons",
            )
            span = controls[1].slider(
                "Spot range", min_value=5, max_value=60, value=30, step=5, format="%d%%",
                key="book_horizon_range",
            )
            if not horizons:
                st.warning("Pick at least one horizon.", icon=":material/warning:")
            else:
                with st.spinner("Repricing the book at each horizon..."):
                    figure = book_pnl_by_horizon(
                        book,
                        mkt,
                        days=tuple(float(d) for d in sorted(horizons)),
                        spot_range=(-span / 100.0, span / 100.0),
                        mode=ui.CHART_MODE,
                    )
                ui.chart(figure, key="book_chart_horizon")

                ui.explain(
                    "How to read this",
                    "Each curve is the P&L against spot at one date in the future, with implied "
                    "volatility held constant. Lines run from light (today) to dark (the furthest "
                    "horizon).\n\n"
                    "The **vertical gap between two curves at a given spot is the time decay** "
                    "earned or paid between those two dates. Long-option books sink towards their "
                    "kinked expiry payoff; short-option books rise towards it. Positions that "
                    "expire inside a horizon are worth their settlement value there, which is why "
                    "the far curves develop kinks at the strikes.",
                )
                ui.takeaways(
                    [
                        f"Theta today is **{ui.money(dollars['theta_cash'], decimals=0, signed=True)}** "
                        "a day with nothing else moving. Over 30 days that is roughly "
                        f"**{ui.money(dollars['theta_cash'] * 30, decimals=0, signed=True)}** — but "
                        "only roughly, because theta itself changes as time passes.",
                        "Where the curves cross is where time stops helping and starts hurting (or "
                        "the other way round): that is the spot level your position wants to sit at.",
                    ]
                )

        else:
            frame = stress_frame(book_json, mkt.spot, mkt.vol, mkt.rate, mkt.div)
            figure = stress_test_chart(book, mkt, mode=ui.CHART_MODE)
            ui.chart(figure, key="book_chart_stress")

            table = frame.reset_index()[["scenario", "spot", "vol", "days", "value", "pnl"]]
            ui.table(
                table,
                column_config={
                    "scenario": st.column_config.TextColumn("Scenario", pinned=True),
                    "spot": st.column_config.NumberColumn(
                        "Shocked spot", format="accounting", help="The spot level in that scenario."
                    ),
                    "vol": st.column_config.NumberColumn(
                        "Shocked vol", format="percent", help="Implied volatility in that scenario."
                    ),
                    "days": st.column_config.NumberColumn(
                        "Days", format="%.0f", help="Calendar days that pass before the shock."
                    ),
                    "value": st.column_config.NumberColumn(
                        "Net liq. value", format="accounting", help="Positions plus cash after the shock."
                    ),
                    "pnl": st.column_config.NumberColumn(
                        "P&L", format="accounting", help="Change in value versus today's market. "
                        "Carry on the cash account is not included in the time scenarios."
                    ),
                },
            )

            worst_name = str(frame["pnl"].idxmin())
            worst_pnl = float(frame["pnl"].min())
            ui.explain(
                "What each scenario means",
                "\n".join(f"- {scenario_sentence(name, shocks)}" for name, shocks in DEFAULT_SCENARIOS.items())
                + "\n\nSpot and vol are shocked TOGETHER in the directional scenarios because that "
                "is how equity markets behave: vol jumps in a crash and leaks in a rally. Greeks "
                "answer *what if the market moves a little*; stress tests answer *what if it moves "
                "a lot, in several dimensions at once*. A book that looks flat on every Greek can "
                "still lose a great deal in the crash scenario.",
            )
            ui.takeaways(
                [
                    f"Worst scenario: **{worst_name}**, for "
                    f"**{ui.money(worst_pnl, decimals=0, signed=True)}**.",
                    "Look for ASYMMETRY rather than for the average: a book that makes a little in "
                    "most scenarios and loses a lot in one is short a tail, and that is the trade "
                    "that ends careers.",
                ]
            )

        ui.interview_note(
            "Not from the Greeks alone. Delta times a 10% move plus half gamma times the move "
            "squared is a Taylor expansion around today's spot, and a 10% gap is far outside the "
            "range where that is accurate — it misses how gamma itself changes, and it ignores the "
            "volatility that jumps at the same time. The honest answer is the repriced number: the "
            f"P&L of this book is **{ui.money(gap_down, decimals=0, signed=True)}** on a -10% gap "
            f"and **{ui.money(gap_up, decimals=0, signed=True)}** on a +10% gap, before any vol move. "
            "In an interview, say the ladder number first, then add that you would want the crash "
            "scenario too, because in a real gap down implied volatility is up 10 to 15 points.",
            question="How much does the book lose if the spot gaps 10% overnight?",
        )

# --------------------------------------------------------------------------- #
# Hedging tab
# --------------------------------------------------------------------------- #
with hedging_tab:
    if book.is_empty:
        ui.empty_state(
            "There is nothing to hedge yet. Add a position on the Positions tab first.",
            icon=":material/balance:",
        )
    else:
        ui.section(
            "Delta hedge",
            "The stock has a delta of exactly 1 and no other Greek, so it is the one instrument "
            "that changes your delta without changing anything else.",
        )

        controls = st.columns(2)
        target = controls[0].number_input(
            "Target delta",
            value=0.0,
            step=10.0,
            key="book_hedge_target",
            help="Flat is 0. A desk that wants a small long bias hedges to a positive target "
            "instead of to zero.",
        )
        controls[1].number_input(
            "No-trade band (shares)",
            min_value=0.0,
            value=0.0,
            step=5.0,
            key="book_hedge_band",
            help="Do nothing when the required trade is smaller than this. A band is the practical "
            "answer to transaction costs: hedging less often saves money but leaves more risk.",
        )

        try:
            trade_size = book.delta_hedge_trade_size(mkt, target)
            hedge_error = None
        except ValueError as exc:
            trade_size, hedge_error = 0.0, str(exc)

        if hedge_error is not None:
            st.error(f"The hedge cannot be computed: {hedge_error}", icon=":material/error:")
        else:
            ui.metric_row(
                [
                    (
                        "Book delta",
                        ui.number(float(raw_greeks["delta"]), decimals=2),
                        "Sum of the position deltas: the number of shares the book behaves like.",
                    ),
                    (
                        "Delta cash",
                        ui.money(dollars["delta_cash"], decimals=0, signed=True),
                        "The same exposure in currency, delta x spot.",
                    ),
                    (
                        "Shares to trade",
                        ui.number(trade_size, decimals=2),
                        "Positive means buy, negative means sell. It is simply target minus book "
                        "delta, because the stock's delta is 1.",
                    ),
                ]
            )

            preview_book = book.copy()
            try:
                preview_book.hedge_delta(mkt, target_delta=target)
                after = preview_book.greeks(mkt)
            except ValueError:
                after = raw_greeks
            ui.table(
                greek_comparison(raw_greeks, after, in_trader_units=trader_units),
                column_config={
                    "Greek": st.column_config.TextColumn("Greek", pinned=True),
                    "Now": st.column_config.NumberColumn(format="localized"),
                    "After the hedge": st.column_config.NumberColumn(format="localized"),
                    "Change": st.column_config.NumberColumn(format="localized"),
                },
            )
            st.caption(
                "The stock only moves the delta row. Everything below it is what the delta hedge "
                "leaves you with."
            )
            with st.container(horizontal=True):
                st.button(
                    "Trade the delta hedge",
                    icon=":material/swap_horiz:",
                    type="primary",
                    key="book_do_hedge",
                    on_click=hedge_delta,
                )

        ui.explain(
            "What a delta hedge does and does not do",
            "- It removes your exposure to **small** spot moves, at today's spot, for an instant.\n"
            "- It leaves **gamma**: the hedge goes stale as soon as the spot moves, and you have to "
            "re-trade. Long gamma re-hedges by buying low and selling high (you get paid to fix the "
            "hedge); short gamma re-hedges the wrong way round.\n"
            "- It leaves **vega**: the stock has none, so a change in implied volatility hits the "
            "book unchanged.\n"
            "- It leaves **theta**: the option time value keeps decaying. Gamma and theta are the "
            "same trade seen from two sides, which is why the break-even move on the Risk tab is "
            "the number to compare with what the market actually does.\n"
            "- It also creates **carry**: the shares you are long earn dividends and cost financing.",
        )

        ui.interview_note(
            "No. Delta-neutral means the first derivative is zero at this spot, at this moment — "
            "nothing more. I would still be carrying gamma, so my delta comes straight back as soon "
            "as the market moves; vega, so a change in implied volatility moves my P&L with the spot "
            "sitting still; and theta, which is the daily rent for that gamma. On this book, after "
            "hedging to the target, the numbers left are in the table above. The honest one-line "
            "answer is: my delta is flat, my BOOK is not — I am long or short volatility, and that "
            "is the position I actually have.",
            question="Your delta is flat. Are you flat?",
        )

        ui.section(
            "Neutralise a second Greek",
            "The stock cannot touch gamma or vega, so a second Greek needs an option — and that "
            "option brings its own delta, which is why the two are solved together.",
        )

        second = st.columns(4)
        second[0].selectbox(
            "Greek to neutralise",
            HEDGEABLE,
            format_func=lambda name: GREEK_INFO[name]["label"],
            key="book_hedge_greek",
        )
        second[1].segmented_control(
            "Type", ["call", "put"], default="call", required=True, key="book_hedge_type"
        )
        second[2].number_input(
            "Strike",
            min_value=0.01,
            value=float(round(spot, 2)),
            step=max(0.5, round(spot / 100.0, 2)),
            key="book_hedge_strike",
        )
        second[3].slider(
            "Days to expiry", min_value=1, max_value=1095, value=90, step=1, key="book_hedge_days"
        )
        st.checkbox(
            "Bring delta back to the target with stock at the same time",
            value=True,
            key="book_hedge_with_delta",
            help="Solves both equations at once: the option is sized to kill the chosen Greek, then "
            "the stock mops up the delta of the book AND of the hedge option.",
        )

        chosen_greek = str(st.session_state.get("book_hedge_greek", HEDGEABLE[0]))
        try:
            hedge_option = EuropeanOption(
                str(st.session_state["book_hedge_type"]),
                float(st.session_state["book_hedge_strike"]),
                float(st.session_state["book_hedge_days"]) / 365.0,
            )
            if st.session_state.get("book_hedge_with_delta", True):
                sizes = book.solve_hedge(mkt, [Underlying(), hedge_option], ("delta", chosen_greek))
            else:
                sizes = {hedge_option: book.hedge_size(mkt, chosen_greek, hedge_option)}
            solve_error = None
        except (KeyError, ValueError) as exc:
            sizes, solve_error = {}, str(exc)

        if solve_error is not None:
            st.warning(
                f"This hedge cannot be solved: {solve_error}", icon=":material/warning:"
            )
        else:
            ui.table(
                pd.DataFrame(
                    [
                        {
                            "Instrument": instrument.label,
                            "Quantity to trade": float(size),
                            "Unit price": float(instrument.price(mkt)),
                            "Cash": -float(size) * float(instrument.price(mkt)),
                        }
                        for instrument, size in sizes.items()
                    ]
                ),
                column_config={
                    "Instrument": st.column_config.TextColumn("Instrument", pinned=True),
                    "Quantity to trade": st.column_config.NumberColumn(
                        format="localized", help="Positive means buy, negative means sell."
                    ),
                    "Unit price": st.column_config.NumberColumn(format="accounting"),
                    "Cash": st.column_config.NumberColumn(
                        format="accounting", help="What the hedge costs (negative) or raises (positive)."
                    ),
                },
            )
            with st.container(horizontal=True):
                st.button(
                    f"Trade the {GREEK_INFO[chosen_greek]['label'].lower()} hedge",
                    icon=":material/balance:",
                    key="book_do_greek_hedge",
                    on_click=hedge_second_greek,
                )

        ui.explain(
            "Why two Greeks need two instruments",
            "Hedging is a linear system: each instrument is a column of Greeks, and you need as "
            "many instruments as Greeks you want to pin. The stock is a column with a 1 in the "
            "delta row and zeros everywhere else, so it can never help with gamma or vega — ask it "
            "to and the system is singular, which is exactly the error you get.\n\n"
            "Two options with the SAME expiry do not help either: their gamma and vega are "
            "proportional, so the two columns are not independent. To pin delta, gamma and vega at "
            "once you need the stock plus two options with DIFFERENT expiries — gamma lives in the "
            "short-dated one, vega in the long-dated one. That is the whole reason a desk runs "
            "several expiries.",
        )

# --------------------------------------------------------------------------- #
# Blotter and file tab
# --------------------------------------------------------------------------- #
with file_tab:
    ui.section(
        "Blotter",
        "Every ticket in execution order, with the cash it moved. A package traded as one ticket "
        "stays one line here, even though the book stores its legs separately.",
    )
    blotter = book.blotter_frame()
    if blotter.empty:
        ui.empty_state("No trades yet.")
    else:
        ui.table(
            blotter,
            column_config={
                "t": st.column_config.NumberColumn(
                    "Time", format="%.3f", help="Market time of the trade, in years."
                ),
                "instrument": st.column_config.TextColumn("Instrument", pinned=True),
                "quantity": st.column_config.NumberColumn(
                    "Quantity", format="localized", help="Positive bought, negative sold."
                ),
                "price": st.column_config.NumberColumn("Price", format="accounting"),
                "fees": st.column_config.NumberColumn("Fees", format="accounting"),
                "cash_flow": st.column_config.NumberColumn(
                    "Cash", format="accounting", help="-(quantity x price) - fees."
                ),
                "note": st.column_config.TextColumn("Note"),
            },
        )

    settlements = book.settlements_frame()
    if not settlements.empty:
        ui.section(
            "Settlements",
            "Cash paid by positions that reached expiry, or by a path event such as a barrier "
            "rebate. Settlement converts value into cash; it does not create P&L.",
        )
        ui.table(settlements)

    ui.section("Save, load, start again", "The book is a plain JSON document, so it travels.")

    save_columns = st.columns(2)
    save_columns[0].download_button(
        "Download the book",
        data=book.to_json(),
        file_name="optionlab_book.json",
        mime="application/json",
        icon=":material/download:",
        key="book_download",
        help="Positions, cash, blotter and settlements. Keep one file per trade idea and reload it "
        "before an interview.",
    )
    save_columns[1].file_uploader(
        "Restore a saved book",
        type=["json"],
        key="book_upload",
        help="A file written by the download button above.",
    )
    with st.container(horizontal=True):
        st.button(
            "Replace my book with this file",
            icon=":material/upload:",
            key="book_load",
            on_click=load_uploaded_book,
            disabled=st.session_state.get("book_upload") is None,
        )

    with st.container(border=True):
        st.markdown(":red-badge[:material/restart_alt: Start again]")
        reset_columns = st.columns([1, 2], vertical_alignment="bottom")
        reset_columns[0].number_input(
            "Starting cash",
            min_value=0.0,
            value=100_000.0,
            step=10_000.0,
            key="book_reset_cash",
        )
        reset_columns[1].checkbox(
            "Yes, discard every position and trade",
            key="book_reset_confirm",
            help="A reset also clears the trading simulation, because the simulator runs on this book.",
        )
        st.button(
            "Reset the book",
            icon=":material/delete:",
            key="book_reset",
            on_click=reset_book,
            disabled=not st.session_state.get("book_reset_confirm", False),
        )

    ui.explain(
        "Why the book is worth saving",
        "The book is self-financing: positions plus cash, with every trade paid out of the cash "
        "account. That is what makes `P&L = value - initial cash` meaningful and what lets the "
        "trading simulator take this exact book and step it through a market scenario.\n\n"
        "A saved file keeps the path state of exotic lines too — a barrier that has already "
        "knocked, the running average of an Asian — so a reloaded book is the same risk, not an "
        "approximation of it.",
    )

# --------------------------------------------------------------------------- #
# Always visible, whatever the book holds: how to talk about a book out loud
# --------------------------------------------------------------------------- #
ui.section(
    "Saying it out loud",
    "The two things an interviewer will ask you to do with a book, in the order they ask them.",
)

ui.interview_note(
    "Aggregate first, then name the biggest risk, then say what you would trade. The order "
    "matters: candidates who walk through the positions one at a time never finish, and they "
    "sound like a settlement clerk rather than a trader.\n\n"
    "The script is four sentences. *Net delta is X, so I am long or short the market by X shares.* "
    "*Net gamma is Y, so my delta moves against me (or for me) as the spot moves, and I would "
    "hedge it mechanically rather than by judgement.* *Net vega is Z, which is my real position: "
    "long or short volatility.* *Theta is what I pay or collect for it, per day.* Then finish with "
    "the trade: the shares that flatten the delta, and the option you would use if you also wanted "
    "to flatten the gamma or the vega.\n\n"
    "The headline row at the top of this page is exactly those four numbers, in currency, and the "
    "'Say it out loud' line under it is the same sentence generated from their signs.",
    question="Here are three positions. What are you running, and what would you do about it?",
)

ui.interview_note(
    "Say the number first, then the caveat, and never quote a Greek where a repricing is "
    "available. Delta times the move plus half gamma times the move squared is a Taylor "
    "expansion around today's spot, and a 10% gap is far outside where that holds — it misses "
    "how gamma itself changes, and it ignores the volatility that jumps at the same time. The "
    "Risk reports tab on this page reprices the whole book at the shocked spot, which is the "
    "honest answer.\n\n"
    "The follow-up is always the same: *and if volatility moves too?* In a real gap down, equity "
    "implied volatility is 10 to 15 points higher, which is why the library shocks spot and "
    "volatility together in its stress scenarios rather than one at a time. Quote the spot ladder, "
    "then volunteer the crash scenario before you are asked for it.",
    question="What is your worst case, and how did you work it out?",
)
