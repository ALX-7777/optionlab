"""Start here — the landing page of OptionLab.

Answers three questions in about fifteen seconds: what this app is, how an
option desk's view of risk fits together, and which page to open next.

Everything shown here is priced in the market set in the sidebar, so the page
reacts the moment the spot or the volatility moves. Nothing on it is expensive:
one analytic snapshot of the at-the-money call and put, one three-panel figure
and one reference table, all cached on the market inputs.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from app_lib import state, ui
from optionlab import GREEK_INFO, EuropeanOption, Market
from optionlab.plotting import profiles

# --------------------------------------------------------------------------- #
# Page-level constants
# --------------------------------------------------------------------------- #
TENORS: dict[str, float] = {
    "1 month": 1.0 / 12.0,
    "3 months": 0.25,
    "6 months": 0.5,
    "1 year": 1.0,
}
DEFAULT_TENOR = "3 months"

LEARN_PAGES = [
    {
        "page": "app_pages/greeks.py",
        "title": "Greeks explorer",
        "icon": ":material/show_chart:",
        "what": "One option at a time: price, delta, gamma, vega and theta against spot, "
        "volatility and time to expiry.",
        "question": "What happens to your delta if the stock drops ten percent overnight?",
    },
    {
        "page": "app_pages/strategies.py",
        "title": "Strategies",
        "icon": ":material/account_tree:",
        "what": "Thirty-seven standard structures — spreads, condors, risk reversals — "
        "with their payoff and their risk leg by leg.",
        "question": "Build me a trade that is long volatility but costs almost nothing.",
    },
    {
        "page": "app_pages/exotics.py",
        "title": "Exotics",
        "icon": ":material/diamond:",
        "what": "Digitals, barriers, Asians and lookbacks, each shown against the vanilla "
        "option a trader would compare it with.",
        "question": "Why can a knock-out call have a delta larger than one?",
    },
]

PRACTICE_PAGES = [
    {
        "page": "app_pages/book.py",
        "title": "My book",
        "icon": ":material/inventory_2:",
        "what": "Trade the instruments you have built into one book and see the risk that "
        "actually matters: the total.",
        "question": "You are long five hundred gamma. What do you do into the close?",
    },
    {
        "page": "app_pages/simulator.py",
        "title": "Trading simulator",
        "icon": ":material/speed:",
        "what": "Run the market forward day by day, hedge on a rule, and watch where the "
        "profit and loss actually came from.",
        "question": "You delta-hedge a long call daily. When do you make money?",
    },
    {
        "page": "app_pages/interview.py",
        "title": "Interview drills",
        "icon": ":material/psychology:",
        "what": "Timed questions on pricing, Greeks and market intuition, marked against "
        "the library's own numbers.",
        "question": "A stock is at 100, vol is 20. Price me the one-year at-the-money call.",
    },
]

MENTAL_MODEL = [
    (
        ":material/public:",
        "1. The market",
        "Spot, volatility, rate and dividend. Set once in the sidebar; every page uses it.",
    ),
    (
        ":material/receipt_long:",
        "2. The instrument",
        "A call, a spread, a barrier. Black-Scholes turns the market into one number: its price.",
    ),
    (
        ":material/function:",
        "3. The Greeks",
        "Derivatives of that price. They say how it moves when the market moves.",
    ),
    (
        ":material/inventory_2:",
        "4. The book",
        "Positions add up, and so do their Greeks. The desk hedges the total, never one trade.",
    ),
    (
        ":material/speed:",
        "5. The simulator",
        "Move the world forward a day at a time and those Greeks become realised profit and loss.",
    ),
]

DRILLS = [
    {
        "claim": "Say what gamma costs you",
        "detail": "Long gamma means re-hedging buys low and sells high. Theta is the rent you "
        "pay for it. Be able to quote both numbers for the same option, in the same breath.",
        "page": "app_pages/greeks.py",
        "label": "Greeks explorer",
        "icon": ":material/show_chart:",
    },
    {
        "claim": "Know the profit and loss of a delta-hedged option",
        "detail": "Over one step it is roughly half gamma times the squared move, minus theta. "
        "Realised volatility above implied pays; below it, you bleed.",
        "page": "app_pages/simulator.py",
        "label": "Trading simulator",
        "icon": ":material/speed:",
    },
    {
        "claim": "Explain why a barrier option's delta can exceed one",
        "detail": "Near the barrier the value jumps as the spot moves, so the hedge ratio blows "
        "up. That discontinuity is why barriers are hard to risk-manage, not just to price.",
        "page": "app_pages/exotics.py",
        "label": "Exotics",
        "icon": ":material/diamond:",
    },
    {
        "claim": "Quote the risk of a book, not of a trade",
        "detail": "An interviewer will hand you three positions and ask what you are running. "
        "Aggregate the Greeks, name the biggest one, and say how you would flatten it.",
        "page": "app_pages/book.py",
        "label": "My book",
        "icon": ":material/inventory_2:",
    },
]


# --------------------------------------------------------------------------- #
# Cached computations (primitive arguments, primitive results)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=64)
def atm_snapshot(
    spot: float, vol: float, rate: float, div: float, expiry: float
) -> dict[str, float]:
    """Price, Greeks and forward of the at-the-money call and put.

    Arguments are the plain market numbers so the cache key stays hashable; the
    market and the two options are rebuilt inside. Raw (not trader) Greeks are
    returned, flattened into one dict of floats, because that is what
    :func:`ui.greek_metrics` expects.
    """
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    call = EuropeanOption("call", spot, expiry)
    put = EuropeanOption("put", spot, expiry)
    out: dict[str, float] = {}
    for prefix, option in (("call_", call), ("put_", put)):
        for name, value in option.greeks(mkt).items():
            out[prefix + name] = float(value)
    out["forward"] = float(mkt.forward(expiry))
    out["discount"] = float(mkt.discount_factor(expiry))
    return out


@st.cache_data(show_spinner=False)
def greek_reference(trader_units: bool) -> pd.DataFrame:
    """The library's own description of every Greek, as a table.

    Only the first sentence of each description is kept so the table stays a
    reference card; the full text is on the tooltip of every Greek metric.
    """
    unit_key = "trader_unit" if trader_units else "raw_unit"
    rows = [
        {
            "Greek": info.get("label", name.capitalize()),
            "Symbol": info.get("symbol", ""),
            "Quoted in": info.get(unit_key, ""),
            "In one sentence": info.get("description", "").split(". ")[0].rstrip(".") + ".",
        }
        for name, info in GREEK_INFO.items()
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# The shared market
# --------------------------------------------------------------------------- #
mkt = state.current_market()
spot = state.spot()
trader_units = state.trader_units()

ui.page_header(
    "OptionLab",
    "Understand option risk, then practise trading it.",
)

st.markdown(
    "A small options desk you can run from a browser. Set the world in the sidebar, "
    "build an option or a strategy, watch its Greeks, put it in a book, then move the "
    "market forward and see what the position actually earns. Every number on every "
    "page comes from the same Black-Scholes library, so nothing here is hand-waved."
)

ui.how_to_use(
    """
Three things to know before you start, and then you can ignore this box.

1. **The sidebar is the world.** Spot, implied volatility, interest rate and dividend
   yield are set once, on the left, and every page prices in that same market. Change the
   volatility and this page, the Greeks explorer and your book all move together.
2. **The Greeks toggle changes the units, not the risk.** *Trader units* quotes vega per
   volatility point, theta per calendar day and rho per 1% — what a desk says out loud.
   Switch it off for the raw mathematical derivatives a textbook uses.
3. **Every page has the same furniture.** A box like this one at the top, a *How to read
   this* box under every chart, *What to notice* where there is something specific to see,
   and *Interview angle* boxes with a real question and an answer built from the numbers
   on your screen.

Your book and your simulation are shared across pages, so a strategy you build on the
Strategies page is waiting for you on the My book page.
"""
)


# --------------------------------------------------------------------------- #
# 1. The market right now
# --------------------------------------------------------------------------- #
ui.section(
    "The market right now",
    "Set in the sidebar on the left. Change it and this page — and every other — reprices.",
)

ui.metric_row(
    [
        ("Spot", ui.money(spot), "Price of the underlying today. Strikes on every page default to it."),
        (
            "Implied volatility",
            ui.percent(mkt.vol, decimals=1),
            "Annualised. Twenty percent is a one-standard-deviation move of about 1.25% a day.",
        ),
        (
            "Interest rate",
            ui.percent(mkt.rate),
            "Continuously compounded. It lifts the forward and is the carry cost of a hedge.",
        ),
        (
            "Dividend yield",
            ui.percent(mkt.div),
            "Continuous. It lowers the forward, so it makes calls cheaper and puts dearer.",
        ),
    ]
)


# --------------------------------------------------------------------------- #
# 2. The at-the-money option today
# --------------------------------------------------------------------------- #
ui.section(
    "The at-the-money option today",
    f"A call and a put, both struck at the spot of {ui.money(spot)}. Choose how long they run.",
)

tenor_choice = st.segmented_control(
    "Maturity",
    list(TENORS),
    default=DEFAULT_TENOR,
    key="home_tenor",
    help="Time to expiry of the two options below. A longer maturity carries more vega and "
    "less gamma.",
)
tenor = tenor_choice or DEFAULT_TENOR
if tenor_choice is None:
    st.caption(f"No maturity selected, so the default of {DEFAULT_TENOR} is shown.")

expiry = TENORS[tenor]
strike = float(spot)

snapshot: dict[str, float] | None = None
if spot <= 0:
    st.warning(
        "The spot has to be above zero before an option can be priced. Raise it in the sidebar.",
        icon=":material/warning:",
    )
else:
    try:
        snapshot = atm_snapshot(spot, float(mkt.vol), float(mkt.rate), float(mkt.div), expiry)
    except (ValueError, ZeroDivisionError) as exc:
        st.error(
            f"This market cannot be priced: {exc}. Reset it with the button in the sidebar.",
            icon=":material/error:",
        )

if snapshot is not None:
    call_greeks = {k[5:]: v for k, v in snapshot.items() if k.startswith("call_")}
    put_greeks = {k[4:]: v for k, v in snapshot.items() if k.startswith("put_")}

    st.markdown(f":blue-badge[:material/trending_up: Call struck at {strike:g}]")
    ui.greek_metrics(call_greeks, trader_units=trader_units)

    st.markdown(f":orange-badge[:material/trending_down: Put struck at {strike:g}]")
    ui.greek_metrics(put_greeks, trader_units=trader_units)

    ui.units_caption(trader_units)
    st.caption(
        f"Forward for {tenor}: {ui.money(snapshot['forward'])} — the spot carried at the rate "
        f"and leaked by the dividend. Cash paid at expiry is discounted by "
        f"{ui.number(snapshot['discount'], decimals=4)}."
    )

    fig = profiles.call_put_comparison(
        strike,
        expiry,
        mkt,
        greeks=("price", "delta", "gamma"),
        ncols=3,
        trader_units=trader_units,
        mode=ui.CHART_MODE,
    )
    ui.chart(fig, key="home_call_put")

    ui.explain(
        "How to read this",
        "Three panels, all with the spot on the horizontal axis and everything else frozen "
        "at today's market. Blue is the call, orange is the put, and the dot marks where the "
        "spot is right now.\n\n"
        "- **Price** is what you pay. The call rises with the spot, the put falls.\n"
        "- **Delta** is the slope of the price panel. Read it as the number of shares to trade "
        "to be hedged, and roughly as the chance of finishing in the money.\n"
        "- **Gamma** is the curvature of the price panel, so it is the slope of the delta panel. "
        "It peaks at the strike and dies away from it.\n\n"
        "A derivative of a derivative is hard to picture in words; here you can simply look at "
        "the panel to its left.",
        icon=":material/help:",
    )

    call_price = call_greeks["price"]
    put_price = put_greeks["price"]
    parity = snapshot["discount"] * (snapshot["forward"] - strike)
    delta_gap = call_greeks["delta"] - put_greeks["delta"]

    ui.takeaways(
        [
            f"The call is worth {ui.money(call_price)} and the put {ui.money(put_price)}, so "
            f"call minus put is {ui.money(call_price - put_price, signed=True)}. The discounted "
            f"distance from the forward to the strike is the same number, "
            f"{ui.money(parity, signed=True)} — not approximately, exactly. That is put-call "
            "parity, and it holds at any volatility, which is why a desk quotes one side and "
            "derives the other.",
            f"Call delta is {ui.greek_value('delta', call_greeks['delta'])} and put delta "
            f"{ui.greek_value('delta', put_greeks['delta'])}. They differ by "
            f"{ui.number(delta_gap, decimals=4)}, which is the dividend discount factor over "
            "the life of the option: long a call and short a put is simply a forward.",
            f"Gamma is {ui.greek_value('gamma', call_greeks['gamma'])} for both, and so is "
            "vega. The difference between a call and a put is a forward, and a forward has no "
            "optionality at all — so all the convexity sits in whichever one you own.",
        ]
    )

    dearer = "call" if call_price >= put_price else "put"
    ui.interview_note(
        "Because the option is struck on the spot but priced off the forward, and the two are "
        f"not the same thing. With the rate at {ui.percent(mkt.rate)} and the dividend at "
        f"{ui.percent(mkt.div)}, the forward for {tenor} is {ui.money(snapshot['forward'])} "
        f"against a strike of {strike:g}. Discount that "
        f"{ui.money(snapshot['forward'] - strike, decimals=3, signed=True)} gap back to today "
        f"and you get {ui.money(parity, decimals=3, signed=True)}, the entire difference between the two "
        f"prices — so the {dearer} is the dearer one here. It says nothing about direction: "
        "swap the rate and the dividend in the sidebar and the answer swaps with them.",
        question="Spot equals strike, so why are the call and the put not worth the same?",
    )


# --------------------------------------------------------------------------- #
# 3. Where to go
# --------------------------------------------------------------------------- #
ui.section(
    "Where to go",
    "Three pages to learn the mechanics, three to practise them under pressure.",
)

st.markdown(":blue-badge[:material/school: Learn the mechanics]")
for column, destination in zip(st.columns(3), LEARN_PAGES):
    with column.container(border=True, height="stretch"):
        st.markdown(f"{destination['icon']} **{destination['title']}**")
        st.caption(destination["what"])
        st.markdown(f'Prepares you for *"{destination["question"]}"*')
        st.page_link(
            destination["page"],
            label=f"Open {destination['title'].lower()}",
            icon=":material/arrow_forward:",
        )

st.markdown(":violet-badge[:material/fitness_center: Practise under pressure]")
for column, destination in zip(st.columns(3), PRACTICE_PAGES):
    with column.container(border=True, height="stretch"):
        st.markdown(f"{destination['icon']} **{destination['title']}**")
        st.caption(destination["what"])
        st.markdown(f'Prepares you for *"{destination["question"]}"*')
        st.page_link(
            destination["page"],
            label=f"Open {destination['title'].lower()}",
            icon=":material/arrow_forward:",
        )


# --------------------------------------------------------------------------- #
# 4. The mental model
# --------------------------------------------------------------------------- #
ui.section(
    "How it all fits together",
    "The same chain a desk uses when it talks about risk. Each link is a page of this app.",
)

for column, (icon, name, body) in zip(st.columns(5, gap="small"), MENTAL_MODEL):
    with column.container(border=True, height="stretch"):
        st.markdown(f"{icon} **{name}**")
        st.caption(body)

ui.interview_note(
    "I would start with the market, because everything else is a function of it: spot, "
    "volatility, rate, dividend. An instrument turns that market into a single price. The "
    "Greeks are the derivatives of that price, so they tell me how it moves — delta for the "
    "spot, gamma for how fast delta itself moves, vega for volatility, theta for time. "
    "Positions aggregate, so what I actually run is the book's Greeks, not any one trade's. "
    "And the Greeks are only a forecast: the test is what the position earns when the market "
    "really moves and I have to re-hedge at real prices. That last step is where the gamma "
    "against theta trade-off stops being algebra and starts being profit and loss.",
    question="Walk me through how an option desk sees risk.",
)


# --------------------------------------------------------------------------- #
# 5. The Greeks in one screen
# --------------------------------------------------------------------------- #
ui.section(
    "The Greeks in one screen",
    "Straight from the library's own definitions, in the units the sidebar toggle selects.",
)

scope_choice = st.segmented_control(
    "Show",
    ["Headline five", "All twelve"],
    default="Headline five",
    key="home_greek_scope",
    help="The headline five are what a trader quotes without being asked. The rest are "
    "second-order and come up when an interviewer wants to push you.",
)
scope = scope_choice or "Headline five"

reference = greek_reference(trader_units)
if scope == "Headline five":
    headline_labels = [GREEK_INFO[name]["label"] for name in ui.HEADLINE_GREEKS]
    reference = reference[reference["Greek"].isin(headline_labels)]

ui.table(
    reference,
    height=min(56 * len(reference) + 44, 560),
    column_config={
        "Greek": st.column_config.TextColumn("Greek", width=140),
        "Symbol": st.column_config.TextColumn(
            "Symbol", width=140, help="The derivative the Greek actually is."
        ),
        "Quoted in": st.column_config.TextColumn(
            "Quoted in", width=250, help="Follows the trader-units toggle in the sidebar."
        ),
        "In one sentence": st.column_config.TextColumn("In one sentence", width="large"),
    },
    row_height=56,
)
st.caption(
    "Price is not a Greek, but it belongs on the list: every other row describes how that one "
    "number moves. Hover any Greek metric on any page for the library's full description."
)


# --------------------------------------------------------------------------- #
# 6. Before the interview
# --------------------------------------------------------------------------- #
ui.section(
    "Before the interview",
    "Four things you should be able to say out loud, and the page that trains each one.",
)

for row_start in (0, 2):
    for column, drill in zip(st.columns(2), DRILLS[row_start : row_start + 2]):
        with column.container(border=True, height="stretch"):
            st.markdown(f":material/check_circle: **{drill['claim']}**")
            st.caption(drill["detail"])
            st.page_link(drill["page"], label=drill["label"], icon=drill["icon"])

if snapshot is not None:
    rule_of_thumb = 0.4 * spot * float(mkt.vol) * math.sqrt(expiry)
    ui.interview_note(
        "Use the desk shortcut: an at-the-money option is worth roughly 0.4 x spot x volatility "
        f"x the square root of the time. Here that is 0.4 x {ui.money(spot)} x "
        f"{ui.percent(mkt.vol, decimals=1)} x the square root of {expiry:.2f} years, so about "
        f"{ui.money(rule_of_thumb)}. Black-Scholes says {ui.money(call_price)}, a gap of "
        f"{ui.money(abs(call_price - rule_of_thumb))}. Say the shortcut out loud as a shortcut, "
        "then name what it drops: it prices the option off the spot rather than off the forward, "
        f"which sits at {ui.money(snapshot['forward'])} here, and it ignores the higher-order "
        "terms that start to bite once the volatility or the maturity gets large. Push the "
        "sidebar to 150% volatility and watch the two answers separate.",
        question=f"Spot {spot:g}, volatility {ui.percent(mkt.vol, decimals=1)}. Price the "
        f"at-the-money call with {tenor} to run, in your head.",
    )
else:
    ui.interview_note(
        "Reach for the desk shortcut: an at-the-money option is worth roughly 0.4 x spot x "
        "volatility x the square root of the time. Say it as an approximation, then name what "
        "it leaves out — it prices the option off the spot, while the model prices it off the "
        "forward. Set a valid market in the sidebar and this box will work the numbers through.",
        question="Price the at-the-money call in your head.",
    )

st.page_link(
    "app_pages/greeks.py",
    label="Start with the Greeks explorer",
    icon=":material/play_arrow:",
    width="stretch",
)
