"""Strategies: explore, read and compare the 37 predefined option structures.

The page walks through the way a trader actually looks at a package: what it
is and what view it expresses, what it costs and what it can make or lose, what
its risk looks like today, how it compares with an alternative, which legs you
actually trade, and finally how to put it in the book.

Every number comes from :mod:`optionlab.strategies` (``summary``, ``breakevens``,
``max_profit``, ``max_loss``, ``probability_of_profit``, ``legs_table``) and
every figure from :mod:`optionlab.plotting.strategy_plots`, so nothing shown
here is invented.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pandas as pd
import streamlit as st

from optionlab import GREEK_INFO, Market, build_strategy
from optionlab.plotting import strategy_plots
from optionlab.strategies import STRATEGY_REGISTRY, legs_table, summary

from app_lib import state, ui

# --------------------------------------------------------------------------- #
# Constants derived from the registry (the library is the single source of truth)
# --------------------------------------------------------------------------- #
ALL_NAMES: list[str] = list(STRATEGY_REGISTRY)
CATEGORIES: list[str] = list(dict.fromkeys(spec.category for spec in STRATEGY_REGISTRY.values()))
DEFAULT_CATEGORY = "Volatility"
DEFAULT_NAME = "long_straddle"
DEFAULT_COMPARE = "iron_condor"

#: Badge colour per market-view tag (optionlab.strategies.VIEW_TAGS).
VIEW_COLOURS: dict[str, str] = {
    "bullish": "green",
    "bearish": "red",
    "neutral": "gray",
    "long vol": "violet",
    "short vol": "orange",
    "income": "blue",
    "hedge": "yellow",
    "arbitrage": "primary",
}

#: What the comparison chart can be drawn on.
COMPARE_DIMENSIONS: dict[str, str] = {
    "P&L at expiry": "pnl_expiry",
    "P&L today": "pnl_today",
    "Delta": "delta",
    "Gamma": "gamma",
    "Vega": "vega",
    "Theta": "theta",
}

GREEK_CHOICES = ("delta", "gamma", "vega", "theta", "rho")
DEFAULT_GREEKS = ["delta", "gamma", "vega", "theta"]
TOL = 1e-9

DISPLAY_COLUMNS = {
    "leg": "Leg",
    "side": "Side",
    "quantity": "Quantity",
    "option_type": "Type",
    "strike": "Strike",
    "expiry": "Expiry (years)",
    "unit_price": "Unit price",
    "value": "Position value",
    "delta": "Delta",
    "gamma": "Gamma",
    "vega": "Vega",
    "theta": "Theta",
    "rho": "Rho",
}


# --------------------------------------------------------------------------- #
# Cached analytics: arguments are primitives, the instruments are rebuilt inside
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=512)
def strategy_analytics(
    name: str,
    params: tuple[tuple[str, Any], ...],
    market: tuple[float, float, float, float, float],
) -> dict[str, Any]:
    """Everything :func:`optionlab.strategies.summary` knows about one package."""
    spot, vol, rate, div, now = market
    strategy = build_strategy(name, **dict(params))
    return summary(strategy, Market(spot=spot, vol=vol, rate=rate, div=div, t=now))


@st.cache_data(show_spinner=False, max_entries=256)
def legs_frame(
    name: str,
    params: tuple[tuple[str, Any], ...],
    market: tuple[float, float, float, float, float],
    trader: bool,
) -> pd.DataFrame:
    """Leg-by-leg table, ready to display (stock legs labelled, columns renamed)."""
    spot, vol, rate, div, now = market
    strategy = build_strategy(name, **dict(params))
    frame = legs_table(strategy, Market(spot=spot, vol=vol, rate=rate, div=div, t=now), trader)
    frame = frame.drop(columns=["instrument"], errors="ignore")
    if "option_type" in frame:
        frame["option_type"] = [value if isinstance(value, str) else "stock" for value in frame["option_type"]]
    return frame.rename(columns=DISPLAY_COLUMNS)


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def view_badges(tags: tuple[str, ...]) -> str:
    """The library's view tags as coloured badges."""
    return " ".join(f":{VIEW_COLOURS.get(tag, 'gray')}-badge[{tag}]" for tag in tags)


def premium_phrase(data: dict[str, Any], title: str) -> str:
    """"X costs 15.69 to put on" / "X pays you 6.01 up front"."""
    amount = ui.money(abs(float(data["net_premium"])))
    kind = data["premium_type"]
    if kind == "debit":
        return f"{title} costs {amount} to put on (a net debit)"
    if kind == "credit":
        return f"{title} pays you {amount} up front (a net credit)"
    return f"{title} is zero cost"


def nearest_breakeven_move(data: dict[str, Any], spot: float) -> float | None:
    """Distance from today's spot to the closest breakeven, as a fraction."""
    roots = [float(root) for root in data["breakevens"]]
    if not roots or spot <= 0:
        return None
    return min(abs(root - spot) for root in roots) / spot


def worst_case_phrase(value: float, title: str) -> str:
    """"X cannot lose more than 3.99" / "X has no cap on its loss"."""
    if not math.isfinite(value):
        return f"{title} has no cap on its loss at the horizon"
    if value >= 0:
        return f"{title} cannot lose at all — its worst case is still a profit of {ui.money(value)}"
    return f"{title} cannot lose more than {ui.money(abs(value))}"


def vol_word(vega: float) -> str:
    """"long vol" / "short vol" / "vega-neutral", from the sign of vega."""
    return "long vol" if vega > TOL else "short vol" if vega < -TOL else "vega-neutral"


def pick_random_strategy() -> None:
    """Callback: jump to a random structure (category first, so the menu holds it)."""
    name = random.choice(ALL_NAMES)
    st.session_state["strategies_category"] = STRATEGY_REGISTRY[name].category
    st.session_state["strategies_name"] = name


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #
mkt = state.current_market()
spot = state.spot()
trader_units = state.trader_units()
mkt_key = ui.market_key(mkt)
unit_key = "trader_unit" if trader_units else "raw_unit"

ui.page_header(
    "Strategies",
    "Thirty-seven classic structures, all priced in the market you set in the sidebar: "
    "what each one costs, what it can make, what it can lose, and how it compares with the alternative.",
)
st.caption(
    f"Priced at spot {ui.money(spot)}, implied volatility {ui.percent(mkt.vol)}, rate {ui.percent(mkt.rate)} "
    f"and dividend yield {ui.percent(mkt.div)}. Change any of them in the sidebar and every number below follows."
)

ui.how_to_use(
    """
Work down the page: it is the order a trader looks at a package in.

1. **Pick a structure.** Choose the family, then the name — or press *Surprise me* and
   practise explaining a payoff you did not choose.
2. **Set the strikes and the expiry**, then press *Update strategy*. The inputs are
   generated from the library's own specification of that structure, so the defaults are
   where a broker would quote it.
3. **Read the five numbers**: what it costs, what it can make, what it can lose, where the
   P&L turns, and the model's own odds. Then read the Greeks underneath them.
4. **Look at the shape** in the four chart tabs. Only the open tab is computed.
5. **Compare it** with a second structure on the same spot and the same expiry, and finish
   by trading the package into your book.

Everything is priced in the sidebar market, so the whole page reprices when you move the
spot or the volatility.
"""
)

# --------------------------------------------------------------------------- #
# 1. Pick a structure
# --------------------------------------------------------------------------- #
ui.section(
    "Pick a structure",
    "Choose the family first, then the strategy. The description and the view tags below come from the library itself.",
)

category = st.pills(
    "Family",
    CATEGORIES,
    default=DEFAULT_CATEGORY,
    key="strategies_category",
    help="Families group structures that are built the same way. Deselect to search all 37 at once.",
)
names = [key for key, entry in STRATEGY_REGISTRY.items() if category is None or entry.category == category]
if not names:
    ui.empty_state("No strategy in this family.")
    st.stop()

picker = st.columns([3, 1], vertical_alignment="bottom")
name = picker[0].selectbox(
    "Strategy",
    names,
    index=names.index(DEFAULT_NAME) if DEFAULT_NAME in names else 0,
    format_func=lambda key: STRATEGY_REGISTRY[key].title,
    key="strategies_name",
    help="All the structures in the selected family. The list changes when you change family.",
)
picker[1].button(
    "Surprise me",
    icon=":material/casino:",
    on_click=pick_random_strategy,
    key="strategies_surprise",
    width="stretch",
    help="Jump to a random structure. Explaining an unfamiliar payoff out loud is the fastest interview drill there is.",
)

if not name:
    ui.empty_state("Pick a strategy to get started.")
    st.stop()

spec = STRATEGY_REGISTRY[name]
with st.container(border=True):
    st.markdown(f"**{spec.title}** — {spec.category.lower()}")
    st.markdown(view_badges(spec.view))
    st.markdown(spec.description)

# --------------------------------------------------------------------------- #
# 2. Parameters
# --------------------------------------------------------------------------- #
ui.section(
    "Set the strikes and the expiry",
    "The defaults are placed around today's spot, the way a broker would quote the structure. "
    "Change them and press update: every number and chart on the page is rebuilt.",
)

with st.form("strategies_params", border=True):
    params = ui.param_controls(spec, spot, key_prefix="strategies", columns=3)
    st.form_submit_button("Update strategy", icon=":material/refresh:", type="primary")

strike_bits = [
    f"**{param.name.replace('_', ' ')}** {ui.money(params[param.name])} ({ui.moneyness_caption(params[param.name], spot)})"
    for param in spec.params
    if param.kind == "strike"
]
expiry_bits = [
    f"**{param.name.replace('_', ' ')}** in {params[param.name] * 365:,.0f} days ({params[param.name]:.2f} years)"
    for param in spec.params
    if param.kind == "expiry"
]
if strike_bits or expiry_bits:
    st.caption(" · ".join(strike_bits + expiry_bits))

try:
    strategy = build_strategy(spec.name, **params)
    data = strategy_analytics(spec.name, tuple(sorted(params.items())), mkt_key)
except ValueError as error:
    st.warning(
        f"This combination of parameters is not a valid {spec.title.lower()}: {error}. "
        "Adjust the inputs above and press update again.",
        icon=":material/report:",
    )
    st.stop()

# --------------------------------------------------------------------------- #
# 3. The numbers
# --------------------------------------------------------------------------- #
ui.section(
    "What it costs, what it can make, what it can lose",
    f"Everything below is measured at the analysis horizon, {ui.number(data['horizon'] or 0.0, decimals=2)} "
    "years from today"
    + (
        " — the front expiry, with the legs that outlive it valued at today's implied vol. "
        "That last part is a model assumption, and it is the main risk of a calendar."
        if data["is_multi_expiry"]
        else ", when every leg expires."
    ),
)

premium = float(data["net_premium"])
premium_label = {"debit": "Net debit", "credit": "Net credit", "zero cost": "Zero cost"}[data["premium_type"]]
probability = data["probability_of_profit"]
roots = [float(root) for root in data["breakevens"]]
if not roots:
    breakeven_text = "none"
elif len(roots) <= 2:
    breakeven_text = " / ".join(ui.money(root) for root in roots)
else:
    breakeven_text = f"{ui.money(roots[0])} … {ui.money(roots[-1])}"

ui.metric_row(
    [
        (
            premium_label,
            ui.money(abs(premium)),
            "What the whole package costs today, stock legs included. A debit is paid, a credit is received. "
            f"Counting the option legs only, the premium is {ui.money(abs(float(data['options_premium'])))}.",
        ),
        (
            "Max profit",
            ui.extreme(float(data["max_profit"])),
            "Best possible profit at the horizon, per package. 'Unlimited' means the P&L keeps growing "
            "with the spot, so no cap exists. A negative number means the structure cannot make money "
            "at all at these prices — worth knowing before you quote it.",
        ),
        (
            "Max loss",
            ui.extreme(float(data["max_loss"])),
            "Worst possible loss at the horizon, per package. This is the number a risk manager looks at first.",
        ),
        (
            "Breakeven" + ("s" if len(roots) > 1 else ""),
            breakeven_text,
            "Spot levels at the horizon where the P&L changes sign.",
        ),
        (
            "Chance of profit",
            "n/a" if probability is None else ui.percent(float(probability)),
            "Risk-neutral probability that the P&L is positive at the horizon, under the market's flat vol. "
            "It is the model's own odds, not a forecast: sellers of options show a high number precisely "
            "because their rare losses are large. Read it next to the max loss.",
        ),
    ]
)

if roots:
    moves = " and ".join(f"{ui.money(root)} ({(root / spot - 1) * 100:+.1f}% from spot)" for root in roots)
    st.caption(f"The P&L turns at {moves}.")
else:
    st.caption(
        "The P&L never changes sign at the horizon: this package makes (or loses) the same money "
        "whatever the spot does — which is what an arbitrage structure is meant to do."
    )

ui.section("The risk you are carrying today", "The Greeks of the whole package, one row, before anything moves.")
ui.greek_metrics(data["greeks"], trader_units=trader_units)
ui.units_caption(trader_units)

greeks_shown = data["greeks_trader"] if trader_units else data["greeks"]
delta, gamma = float(greeks_shown["delta"]), float(greeks_shown["gamma"])
vega, theta = float(greeks_shown["vega"]), float(greeks_shown["theta"])

notes: list[str] = []
if abs(delta) < 0.01:
    notes.append(
        "**Direction.** Delta is roughly zero at today's spot: the package is direction-neutral here, "
        "so what you own is the shape, not a view on where the stock goes."
    )
else:
    notes.append(
        f"**Direction.** Delta is {ui.greek_value('delta', delta)}: the package moves like being "
        f"{'long' if delta > 0 else 'short'} {abs(delta):.2f} shares of the underlying, per package."
    )
notes.append(
    f"**Convexity.** Gamma is {ui.greek_value('gamma', gamma)} {GREEK_INFO['gamma'][unit_key]}, so you are "
    + (
        "long gamma: the delta grows in your favour as the spot moves, which is what you paid theta for."
        if gamma > TOL
        else "short gamma: the delta moves against you as the spot moves, which is what the theta pays you for."
        if gamma < -TOL
        else "gamma-flat here."
    )
)
notes.append(
    f"**Volatility.** Vega is {ui.greek_value('vega', vega)} {GREEK_INFO['vega'][unit_key]}, so you are "
    + (
        "long volatility: the package gains when implied vol rises."
        if vega > TOL
        else "short volatility: the package gains when implied vol falls."
        if vega < -TOL
        else "vega-neutral: a parallel move in implied vol barely touches it."
    )
)
notes.append(
    f"**Time.** Theta is {ui.greek_value('theta', theta)} {GREEK_INFO['theta'][unit_key]}: "
    + (
        "time works against you — the package bleeds if the spot sits still."
        if theta < -TOL
        else "time works for you — you are paid to wait, as long as nothing moves."
        if theta > TOL
        else "time is roughly neutral here."
    )
)
ui.takeaways(notes, title="What these Greeks are telling you")

ui.interview_note(
    "Say the structure, then the view, then the risk, in that order. For a risk reversal: buy the out-of-the-money "
    "call, sell the out-of-the-money put, strikes solved so the two premiums net to zero — the client keeps the "
    "upside and pays nothing today. If they already hold the stock, the same trade is a collar: the call they sell "
    "pays for the put they buy. The risk is what they sold. Below the put strike they are long the stock "
    "synthetically, so the downside is the same as owning it outright, and the position is short skew: a fall in "
    "the spot usually comes with a bid for puts, which hurts the short leg twice. Build a risk reversal on this "
    "page, move the strikes until the net premium prints near zero, and read the max loss line.",
    question="A client wants upside exposure but does not want to pay a premium. What do you show them, and what is the risk?",
)

# --------------------------------------------------------------------------- #
# 4. The charts
# --------------------------------------------------------------------------- #
ui.section(
    "See the shape",
    "Four views of the same package. Only the open tab is computed, so switching is cheap.",
)

tab_payoff, tab_greeks, tab_decay, tab_vol = st.tabs(
    ["Payoff", "Greeks by leg", "Time decay", "Vol sensitivity"],
    on_change="rerun",
    key="strategies_chart_tab",
)

if tab_payoff.open is not False:
    with tab_payoff:
        with st.container(horizontal=True):
            show_legs = st.toggle("Show each leg", value=True, key="strategies_payoff_legs")
            show_today = st.toggle("Show today's curve", value=True, key="strategies_payoff_today")
            show_midway = st.toggle("Show a half-way date", value=True, key="strategies_payoff_midway")
        try:
            with st.spinner("Pricing the payoff…"):
                figure = strategy_plots.payoff_diagram(
                    strategy,
                    mkt,
                    show_legs=show_legs,
                    show_today=show_today,
                    horizons=(0.5,) if show_midway else (),
                    mode=ui.CHART_MODE,
                )
            ui.chart(figure, key="strategies_payoff")
        except ValueError as error:
            st.warning(f"The payoff diagram could not be drawn: {error}", icon=":material/report:")
        ui.explain(
            "How to read this",
            "- The **bold line** is the P&L at expiry: the contract, the part that is certain.\n"
            "- The **blue lines** are model values before expiry — today, and half way to expiry. The gap "
            "between blue and bold is the time value still to be earned or lost.\n"
            "- **Thin dashed lines** are the individual legs. A strategy is nothing but their sum, so this is "
            "where you see *why* the shape looks the way it does.\n"
            "- The **green and red washes** are profit and loss, and the **diamonds** on the zero line are the "
            "breakevens.\n"
            "- Financing is ignored on purpose: the premium is not capitalised to expiry and dividends on stock "
            "legs are not added. That is the textbook payoff diagram.",
        )
        ui.interview_note(
            "Read the slopes first, then the kinks. Each kink is a strike, so count them. Then look far left and "
            "far right: a flat wing means the risk is capped on that side, a slope of +1 means you are long the "
            "stock there, -1 means short. A tent with flat wings is a butterfly or a condor; a V is a straddle or "
            "a strangle; a flat top with a -1 slope on the left is a covered call. Finally ask where the curve "
            "sits at today's spot: a structure that starts above the zero line was sold for a credit, one that "
            "starts below was bought for a debit.",
            question="Here is a payoff diagram with no labels. Tell me what the structure is.",
        )

if tab_greeks.open is not False:
    with tab_greeks:
        controls = st.columns([3, 2], vertical_alignment="bottom")
        chosen_greeks = controls[0].pills(
            "Greeks to show",
            GREEK_CHOICES,
            selection_mode="multi",
            default=DEFAULT_GREEKS,
            format_func=lambda key: GREEK_INFO[key]["label"],
            key="strategies_greek_pick",
        )
        legs_mode = controls[1].segmented_control(
            "Legs",
            ["overlay", "stack", "none"],
            default="overlay",
            key="strategies_greek_legs",
            help="Overlay draws one dashed line per leg; stack piles the contributions so the total is the "
            "top of the positive pile plus the bottom of the negative one.",
        )
        if not chosen_greeks:
            ui.empty_state("Pick at least one Greek to draw.", icon=":material/touch_app:")
        else:
            try:
                with st.spinner("Computing the Greeks…"):
                    figure = strategy_plots.strategy_greeks_dashboard(
                        strategy,
                        mkt,
                        greeks=tuple(chosen_greeks),
                        legs=legs_mode or "overlay",
                        trader_units=trader_units,
                        mode=ui.CHART_MODE,
                    )
                ui.chart(figure, key="strategies_greeks")
            except ValueError as error:
                st.warning(f"The Greeks dashboard could not be drawn: {error}", icon=":material/report:")
        ui.explain(
            "How to read this",
            "- One panel per Greek, each with its own scale: Greeks live on different orders of magnitude and "
            "must never share an axis.\n"
            "- The **bold line** is the package; the **dashed lines** are the legs. A butterfly's gamma, for "
            "instance, is the long gamma of the two wings minus twice the gamma of the body — you can see it.\n"
            "- The **dashed vertical line** is today's spot. Everything to the left is what happens if the "
            "market sells off before you have done anything about it.\n"
            f"- Units: {GREEK_INFO['vega'][unit_key]} for vega, {GREEK_INFO['theta'][unit_key]} for theta. "
            "The sidebar toggle switches between trader units and raw derivatives.",
        )

if tab_decay.open is not False:
    with tab_decay:
        decay_pnl = st.toggle(
            "Show P&L rather than value",
            value=True,
            key="strategies_decay_pnl",
            help="P&L subtracts today's entry cost, so the curve starts at zero at today's spot.",
        )
        try:
            with st.spinner("Rolling the clock forward…"):
                figure = strategy_plots.strategy_time_decay(strategy, mkt, as_pnl=decay_pnl, mode=ui.CHART_MODE)
            ui.chart(figure, key="strategies_decay")
        except ValueError as error:
            st.warning(f"The decay chart could not be drawn: {error}", icon=":material/report:")
        ui.explain(
            "How to read this",
            "The model value of any option package is a smoothed version of its payoff, and time is what "
            "sharpens it. Light blue is far from expiry, dark blue is close to it, and the bold line is expiry "
            "itself. If the curves **sink** towards the bold line you are paying theta; if they **rise** towards "
            "it you are earning theta. That direction is the whole economics of the trade, and it is the mirror "
            "image of the gamma sign in the previous tab.",
        )

if tab_vol.open is not False:
    with tab_vol:
        vol_pnl = st.toggle(
            "Show P&L rather than value",
            value=True,
            key="strategies_vol_pnl",
            help="P&L is measured against today's entry cost at the market vol.",
        )
        try:
            with st.spinner("Repricing at other vols…"):
                figure = strategy_plots.strategy_vol_sensitivity(strategy, mkt, as_pnl=vol_pnl, mode=ui.CHART_MODE)
            ui.chart(figure, key="strategies_vol")
        except ValueError as error:
            st.warning(f"The vol chart could not be drawn: {error}", icon=":material/report:")
        ui.explain(
            "How to read this",
            "The same package priced today at 50%, 75%, 100%, 125% and 150% of the market's implied vol "
            f"(currently {ui.percent(mkt.vol)}); the thick curve is the market vol itself. **The vertical gap "
            "between two curves at a given spot is the vega there.** A long-vega package lifts when vol rises, "
            "a short-vega one sinks — compare the spacing here with the vega number in the metric row above.",
        )

# --------------------------------------------------------------------------- #
# 5. Compare with another structure
# --------------------------------------------------------------------------- #
ui.section(
    "Compare it with another structure",
    "The second structure is built on the same spot and the same expiry with the library's textbook strikes, "
    "so what you see is the difference between the two shapes, not between two sets of inputs.",
)

compare_controls = st.columns([3, 2], vertical_alignment="bottom")
other_name = compare_controls[0].selectbox(
    "Second strategy",
    ALL_NAMES,
    index=ALL_NAMES.index(DEFAULT_COMPARE),
    format_func=lambda key: f"{STRATEGY_REGISTRY[key].title} · {STRATEGY_REGISTRY[key].category.lower()}",
    key="strategies_compare_name",
)
width = compare_controls[1].slider(
    "Strike width of the second structure",
    min_value=0.25,
    max_value=2.0,
    value=1.0,
    step=0.05,
    key="strategies_compare_width",
    help="Scales how far its strikes sit from the spot. 1.0 is the library's textbook placement; 2.0 pushes "
    "every strike twice as far out of the money; 0.25 pulls them all in around the spot.",
)
dimension = st.segmented_control(
    "Compare on",
    list(COMPARE_DIMENSIONS),
    default="P&L at expiry",
    key="strategies_compare_what",
    help="Both packages drawn on one axis, against the same spot ladder.",
)

other_spec = STRATEGY_REGISTRY[other_name]
horizon = float(data["horizon"]) if data["horizon"] else 1.0
try:
    other_params = other_spec.default_params(spot=spot, expiry=horizon)
    for param in other_spec.params:
        if param.kind == "strike":
            other_params[param.name] = spot + width * (float(other_params[param.name]) - spot)
    other = build_strategy(other_name, **other_params)
    other_data = strategy_analytics(other_name, tuple(sorted(other_params.items())), mkt_key)
except ValueError as error:
    st.warning(
        f"{other_spec.title} cannot be built at this strike width: {error}. Move the width slider back towards 1.0.",
        icon=":material/report:",
    )
else:
    left_title = spec.title
    right_title = other_spec.title if other_spec.title != spec.title else f"{other_spec.title} (2)"

    left_roots = [float(root) for root in data["breakevens"]]
    right_roots = [float(root) for root in other_data["breakevens"]]
    left_greeks = data["greeks_trader"] if trader_units else data["greeks"]
    right_greeks = other_data["greeks_trader"] if trader_units else other_data["greeks"]

    rows = [
        (
            "Net premium (+ paid / - received)",
            ui.money(float(data["net_premium"]), signed=True),
            ui.money(float(other_data["net_premium"]), signed=True),
        ),
        ("Max profit", ui.extreme(float(data["max_profit"])), ui.extreme(float(other_data["max_profit"]))),
        ("Max loss", ui.extreme(float(data["max_loss"])), ui.extreme(float(other_data["max_loss"]))),
        (
            "Breakevens",
            " / ".join(ui.money(root) for root in left_roots) or "none",
            " / ".join(ui.money(root) for root in right_roots) or "none",
        ),
        (
            "Chance of profit",
            "n/a" if data["probability_of_profit"] is None else ui.percent(float(data["probability_of_profit"])),
            "n/a"
            if other_data["probability_of_profit"] is None
            else ui.percent(float(other_data["probability_of_profit"])),
        ),
        (
            "Delta",
            ui.greek_value("delta", float(left_greeks["delta"])),
            ui.greek_value("delta", float(right_greeks["delta"])),
        ),
        (
            "Gamma",
            ui.greek_value("gamma", float(left_greeks["gamma"])),
            ui.greek_value("gamma", float(right_greeks["gamma"])),
        ),
        (
            "Vega",
            ui.greek_value("vega", float(left_greeks["vega"])),
            ui.greek_value("vega", float(right_greeks["vega"])),
        ),
        (
            "Theta",
            ui.greek_value("theta", float(left_greeks["theta"])),
            ui.greek_value("theta", float(right_greeks["theta"])),
        ),
        ("Number of legs", str(len(data["legs"])), str(len(other_data["legs"]))),
    ]
    ui.table(
        pd.DataFrame(rows, columns=["Measure", left_title, right_title]),
        key="strategies_compare_table",
    )
    st.caption(
        f"Both packages are priced at spot {ui.money(spot)} with a horizon of {horizon:.2f} years. "
        f"Vega is {GREEK_INFO['vega'][unit_key]} and theta {GREEK_INFO['theta'][unit_key]}."
    )

    try:
        with st.spinner("Overlaying the two structures…"):
            figure = strategy_plots.compare_strategies(
                [strategy, other],
                mkt,
                what=COMPARE_DIMENSIONS.get(dimension or "P&L at expiry", "pnl_expiry"),
                trader_units=trader_units,
                mode=ui.CHART_MODE,
            )
        ui.chart(figure, key="strategies_compare")
    except ValueError as error:
        st.warning(f"The comparison could not be drawn: {error}", icon=":material/report:")

    ui.explain(
        "How to read this",
        "Both packages on one axis, over the same spot ladder, so the vertical distance between the two "
        "curves at any spot is exactly what you gain or give up by choosing one over the other.\n\n"
        "- On **P&L at expiry**, each curve is drawn at its own horizon, and where a curve is flat the risk "
        "is capped on that side.\n"
        "- On **P&L today**, you are comparing what each package is worth before anything has expired — that "
        "is the mark you would actually see on your screen tomorrow morning.\n"
        "- On a **Greek**, you are comparing the risk you have to manage, not the outcome: two structures "
        "with a similar payoff can carry very different gamma and vega, and that difference is what your "
        "hedging bill will be made of.",
    )

    commentary: list[str] = [
        f"**Cost.** {premium_phrase(data, left_title)}; {premium_phrase(other_data, right_title)}."
    ]
    left_vega, right_vega = float(left_greeks["vega"]), float(right_greeks["vega"])
    commentary.append(
        f"**Volatility.** {left_title} is {vol_word(left_vega)} (vega {ui.greek_value('vega', left_vega)}); "
        f"{right_title} is {vol_word(right_vega)} (vega {ui.greek_value('vega', right_vega)}) "
        f"({GREEK_INFO['vega'][unit_key]})."
    )
    left_theta, right_theta = float(left_greeks["theta"]), float(right_greeks["theta"])
    commentary.append(
        f"**Time.** {left_title} {'pays away' if left_theta < 0 else 'collects'} "
        f"{ui.greek_value('theta', abs(left_theta))}; {right_title} "
        f"{'pays away' if right_theta < 0 else 'collects'} {ui.greek_value('theta', abs(right_theta))} "
        f"({GREEK_INFO['theta'][unit_key]}). Whoever collects the theta is the one who is short the gamma."
    )
    left_loss, right_loss = float(data["max_loss"]), float(other_data["max_loss"])
    risk = (
        f"**Risk.** {worst_case_phrase(left_loss, left_title)}; "
        f"{worst_case_phrase(right_loss, right_title)}."
    )
    if math.isfinite(left_loss) != math.isfinite(right_loss):
        risk += " That difference, not the premium, is usually what decides which of the two gets done."
    elif math.isfinite(left_loss) and math.isfinite(right_loss):
        worse = left_title if left_loss < right_loss else right_title
        risk += f" {worse} is the bigger ticket if it goes wrong."
    commentary.append(risk)
    left_move, right_move = nearest_breakeven_move(data, spot), nearest_breakeven_move(other_data, spot)
    if left_move is not None and right_move is not None:
        patient = left_title if left_move > right_move else right_title
        commentary.append(
            f"**How far the spot has to go.** The nearest point where the P&L changes sign is {left_move:.1%} away "
            f"for {left_title} and {right_move:.1%} away for {right_title}: {patient} is the one that needs the "
            "market to do more before anything happens."
        )
    ui.takeaways(commentary, title="What the numbers say")

ui.interview_note(
    "Both are short volatility, short gamma and long theta, and both are sold because you think realised vol will "
    "come in below implied. The difference is the tail. The straddle keeps the whole credit but its loss is "
    "unlimited on both sides, so it only works if you can delta hedge it and you are allowed to carry an open "
    "tail. The condor buys both wings back: the credit is smaller, but the worst case is a known number — the "
    "wider wing width minus the credit — which is what makes it acceptable under a risk limit, in a retail or "
    "private-bank account, or when margin is expensive. You also prefer the condor when the smile is steep, "
    "because the wings you have to buy are cheap relative to the body you are selling. Put numbers on it with the "
    "table above: compare the two credits, the two maximum losses, and the chance of profit.",
    question="When would you sell an iron condor rather than a short straddle?",
)

# --------------------------------------------------------------------------- #
# 6. The legs
# --------------------------------------------------------------------------- #
ui.section(
    "The legs you actually trade",
    "One row per elementary leg, priced in the same market. This is the ticket a broker would read back to you.",
)
try:
    legs = legs_frame(spec.name, tuple(sorted(params.items())), mkt_key, trader_units)
except ValueError as error:
    st.warning(f"The legs could not be priced: {error}", icon=":material/report:")
else:
    ui.table(
        legs,
        key="strategies_legs",
        column_config={
            "Quantity": st.column_config.NumberColumn(format="%.2f", help="Signed: positive is long, negative is short."),
            "Strike": st.column_config.NumberColumn(format="%.2f"),
            "Expiry (years)": st.column_config.NumberColumn(format="%.2f"),
            "Unit price": st.column_config.NumberColumn(format="%.4f", help="Model price of one unit of that leg."),
            "Position value": st.column_config.NumberColumn(format="%.2f", help="Unit price times the signed quantity."),
            "Delta": st.column_config.NumberColumn(format="%.4f"),
            "Gamma": st.column_config.NumberColumn(format="%.4f"),
            "Vega": st.column_config.NumberColumn(format="%.4f"),
            "Theta": st.column_config.NumberColumn(format="%.4f"),
            "Rho": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption(
        "Position value and the Greeks are already multiplied by the signed quantity, so every column adds up to "
        "the package total shown above. "
        + ("Trader units." if trader_units else "Raw derivatives.")
    )

# --------------------------------------------------------------------------- #
# 7. Trade it
# --------------------------------------------------------------------------- #
ui.section(
    "Trade it into your book",
    "Sends the package to your book at the model price, leg by leg, so it nets against whatever you already hold.",
)

book = state.get_book()
with st.container(border=True):
    trade_controls = st.columns([1, 1, 2], vertical_alignment="bottom")
    side = trade_controls[0].segmented_control(
        "Side", ["Buy", "Sell"], default="Buy", key="strategies_trade_side"
    )
    quantity = trade_controls[1].number_input(
        "Packages",
        min_value=1.0,
        max_value=10_000.0,
        value=10.0,
        step=1.0,
        key="strategies_trade_qty",
        help="Number of packages. One package is the structure exactly as priced above.",
    )
    signed_quantity = float(quantity) * (-1.0 if side == "Sell" else 1.0)
    cash_move = -signed_quantity * premium
    trade_controls[2].markdown(
        f"**{'Sell' if signed_quantity < 0 else 'Buy'} {abs(signed_quantity):g} × {spec.title}** at "
        f"{ui.money(premium, signed=True)} per package — cash moves by {ui.money(cash_move, signed=True)}."
    )

    if st.button("Add to my book", icon=":material/add:", type="primary", key="strategies_trade_add"):
        try:
            book.trade(strategy, signed_quantity, mkt)
            state.set_book(book)
        except ValueError as error:
            st.error(f"That trade was rejected: {error}", icon=":material/block:")
        else:
            st.toast(f"{abs(signed_quantity):g} × {spec.title} added to your book.", icon=":material/check:")
            st.success(
                f"Done: {'sold' if signed_quantity < 0 else 'bought'} {abs(signed_quantity):g} × {spec.title} at "
                f"{ui.money(premium, signed=True)} per package. Your book now holds {len(book.positions)} lines "
                f"and {ui.money(book.cash)} of cash.",
                icon=":material/check_circle:",
            )

    st.caption(
        (
            f"Your book is empty and holds {ui.money(book.cash)} of cash. "
            if not book.positions
            else f"Your book holds {len(book.positions)} lines and {ui.money(book.cash)} of cash. "
        )
        + "Trading at the model price leaves the book's value unchanged — only the risk changes."
    )
    st.page_link("app_pages/book.py", label="Open my book", icon=":material/inventory_2:")
