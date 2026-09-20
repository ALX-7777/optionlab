"""Greeks explorer — one option, seen from every angle.

The core teaching page of OptionLab. You build a single European option in the
control row at the top; everything below is that same contract, priced in the
market set in the sidebar, shown from seven angles:

* **Against spot** — six Greeks side by side, then one of them in detail.
* **Through time** — optionality being squeezed into the strike as expiry
  approaches, and the at-the-money blow-up of gamma and theta.
* **Against volatility** — where vega lives, and why it is a maturity story.
* **Surface** — the two previous views merged into one picture.
* **Compare** — call against put, strike against strike, maturity against
  maturity, with the numbers underneath the chart.
* **P&L intuition** — what the Greeks are for: explaining a P&L, and the rent
  (theta) paid for convexity (gamma).
* **Implied vol** — a price in, a volatility out.

Every figure comes from :mod:`optionlab.plotting.profiles` and every number
from :mod:`optionlab.black_scholes`; nothing here is invented. The option is
priced in closed form and vectorised over each grid, and only the open tab is
computed, so changing the strike or the expiry redraws in a fraction of a
second.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from app_lib import state, ui
from optionlab import GREEK_INFO, EuropeanOption, Position, black_scholes as bs, to_trader_units
from optionlab.plotting import profiles

# --------------------------------------------------------------------------- #
# Page-level constants
# --------------------------------------------------------------------------- #
VIEWS = (
    "Against spot",
    "Through time",
    "Against volatility",
    "Surface",
    "Compare",
    "P&L intuition",
    "Implied vol",
)

#: Greeks offered in the "zoom in on one" controls, in teaching order.
SPOT_GREEKS = ("price", "delta", "gamma", "vega", "theta", "rho", "vanna", "volga")
TIME_GREEKS = ("price", "delta", "gamma", "vega", "theta", "charm", "color")
VOL_GREEKS = ("price", "delta", "gamma", "vega", "theta", "vanna", "volga")
SURFACE_GREEKS = ("price", "delta", "gamma", "vega", "theta")

#: Panels of the "all Greeks" dashboard and of the comparison charts.
DASHBOARD_GREEKS = ("price", "delta", "gamma", "vega", "theta", "rho")
COMPARE_GREEKS = ("price", "delta", "gamma", "vega", "theta", "rho")

#: Columns of every numeric table on the page, and how many decimals each gets.
TABLE_GREEKS = ("price", "delta", "gamma", "vega", "theta")
TABLE_FORMATS = {"price": "%.3f", "delta": "%.3f", "gamma": "%.4f", "vega": "%.3f", "theta": "%.3f"}

#: Maturity ladder, in days, used by the comparison and the vega story.
TENORS: dict[str, int] = {
    "1 week": 7,
    "1 month": 30,
    "3 months": 91,
    "6 months": 182,
    "1 year": 365,
    "2 years": 730,
}

#: Time-to-expiry ladder of the decay table, in years, longest first.
DECAY_TAUS = (1.0, 0.5, 0.25, 1.0 / 12.0, 1.0 / 52.0, 1.0 / 365.0)
DECAY_LABELS = ("1 year", "6 months", "3 months", "1 month", "1 week", "1 day")

#: Strikes offered by the strike ladder, as a fraction of the spot.
MONEYNESS = (0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20)

ONE_DAY = 1.0 / 365.0


# --------------------------------------------------------------------------- #
# Long-form copy (kept at column 0 so the markdown never indents into a code block)
# --------------------------------------------------------------------------- #
EXPLAIN_DASHBOARD = """
Six panels, one option, the same horizontal axis: the spot the underlying could
be trading at. Every other market input is frozen, so each panel answers a
*what if the stock were there, right now*.

- **Price** is the curve everything else is a slope of. It sits above the
  payoff by the time value.
- **Delta** is the slope of the price panel; **gamma** is the slope of the
  delta panel. A derivative of a derivative is easier to see than to define.
- **Gamma, vega and theta all peak in the same place**, around the strike.
  They are one property — optionality — seen from three angles: convexity,
  sensitivity to future moves, and the rent paid for both.
- The dot marks today's market; the dashed vertical line is the strike.
"""

EXPLAIN_PROFILE = """
The same what-if as the dashboard, one Greek at a time and with room to read it.

Hover anywhere on the curve to get the exact value. The dot is where the market
is now, so the distance from the dot to the peak tells you how much the risk
would change on a move you consider plausible today.
"""

EXPLAIN_EVOLUTION_TAU = """
Each line is the **same option** at a different time to expiry: light lines are
far from expiry, dark lines are close to it. Time runs from light to dark.

This is the single most useful picture in options. Watch how a lazy S-shaped
delta hardens into a step, how a low gamma hill turns into a spike pinned on
the strike, and how vega melts away everywhere. Nothing about the contract has
changed — only the time left for the spot to travel.
"""

EXPLAIN_VS_TIME = """
Now the horizontal axis is the calendar, from today to expiry, and each line
freezes the spot at a different level: below the strike, at it, and above it.

The three lines tell three different stories. Away from the strike the Greek
dies out as expiry approaches, because the option's fate is already decided. At
the strike it blows up, because the last day still decides everything. The
dashed vertical line is the expiry.
"""

EXPLAIN_DECAY_TABLE = """
The same at-the-money option (strike set equal to today's spot) priced at six
maturities, so the shape above becomes numbers you can quote.

Gamma and theta both grow like one over the square root of the time left, so
they roughly double every time the remaining life is quartered. Vega does the
opposite: it shrinks like the square root of time. The last column is the move
the spot has to make in a day for gamma to pay for one day of theta.
"""

EXPLAIN_VOL_PROFILE = """
The horizontal axis is now the volatility the option is priced at, with the
spot frozen at today's level. Today's market is the dot.

Price rises with volatility everywhere (that is vega, and it is why the implied
volatility solver in the last tab always finds exactly one answer). Vega itself
is roughly flat at the money and hump-shaped away from it — that curvature is
volga, and it is what makes out-of-the-money options interesting to volatility
traders.
"""

EXPLAIN_VOL_EVOLUTION = """
The profile against spot, redrawn at several volatility levels (light = low
volatility, dark = high).

Compare it with the same picture through time: lowering the volatility looks
almost exactly like removing time. That is not a coincidence — a Black-Scholes
option only ever sees the total standard deviation left, volatility times the
square root of the time to expiry, so halving one is the same as quartering the
other.
"""

EXPLAIN_VEGA_LADDER = """
The same strike at several maturities, overlaid. Click a legend entry to hide
that line.

Long-dated options carry far more vega and far less gamma; short-dated options
are the opposite. That single trade-off is why a desk expresses a *view on the
level of volatility* with long maturities and a *view on movement* with short
ones — and why the two books are hedged with different instruments.
"""

EXPLAIN_SURFACE = """
The profile and its evolution, merged: spot runs along one axis, the second
market variable along the other, and the Greek is the height or the colour.

Drag to rotate the 3-D view, or switch to the heatmap when you want to read a
value precisely — the heatmap also marks the strike. Signed Greeks (delta,
theta) use a red-blue scale centred on zero; one-signed Greeks (gamma, vega)
use a single ramp. The marker is today's market.
"""

EXPLAIN_COMPARE = """
One panel per Greek, one line per contract, all on the same axes and the same
scale. Colour and dash pattern both identify a contract, so two curves that lie
exactly on top of each other are still distinguishable.

Click a legend entry to hide a contract; double-click to isolate it. The table
underneath is the same comparison at today's spot, in numbers.
"""

EXPLAIN_PNL = """
Profit and loss against spot, assuming you paid today's model value: the flat
line at zero is your entry, the lowest curve is today, and the kinked line is
the payoff at expiry.

The vertical gap between today's curve and the payoff **is** the time value —
what theta takes away, day after day, fastest near the strike. The diamonds
mark the break-even spot at expiry. Volatility and rates are frozen, and the
cost of financing the premium is ignored.
"""

EXPLAIN_TAYLOR = """
How a desk explains its P&L every evening. The solid line is the truth: reprice
the option at the new spot. The dotted line is what delta alone predicts, and
the dashed line adds the gamma term.

Delta alone is a tangent — right for small moves, and always too pessimistic
for a long option because it ignores the curvature. Adding one half gamma times
the squared move bends it into a parabola that hugs the truth much further out.
What is left over is the higher orders, and it grows fast near expiry, where the
Greeks themselves move quickly.
"""

EXPLAIN_GAMMA_THETA = """
The same position, but **delta-hedged**: the delta P&L has been removed, so
what remains is the part a trader actually owns.

A long option makes money on any large move in either direction (gamma) and
loses a fixed amount if nothing happens (theta). The two cancel at the
break-even moves marked on the chart, which land close to one standard
deviation of the implied distribution. A long-gamma position is therefore a bet
that realised movement beats implied movement; a short one is the opposite bet,
with the profit and the loss swapped.
"""

EXPLAIN_IV = """
Black-Scholes maps a volatility to a price. Implied volatility runs that map
backwards: it is the volatility at which the model reproduces the price the
market is actually showing.

It is not a forecast. It is a quoting convention that makes premiums
comparable: a cheap one-month option and an expensive one-year option say
nothing to each other until both are expressed as a volatility. The solver here
is Newton's method started at the Manaster-Koehler point and safeguarded by a
bisection bracket, so it cannot diverge.
"""

VIEW_GUIDE = """
Seven ways of looking at the same contract. Only the open one is computed, so
switching is instant.

- **Against spot** — the six Greeks side by side today, then one of them full
  size. Start here.
- **Through time** — the same profile redrawn as expiry approaches, plus the
  numbers behind the at-the-money blow-up of gamma and theta.
- **Against volatility** — what changes when the market re-prices risk, and why
  vega is really a maturity story.
- **Surface** — spot and a second variable at once, as a 3-D surface or a
  heatmap.
- **Compare** — call against put, strike against strike, maturity against
  maturity, chart and table side by side.
- **P&L intuition** — the money view: what you would book, how a desk explains
  it, and the move at which gamma pays for theta.
- **Implied vol** — type a market price, get the volatility it implies.
"""

EXPLAIN_IV_CURVE = """
The price of this option as a function of the volatility used to price it, with
the spot frozen at today's level.

The curve only ever rises — vega is positive for every vanilla option — which
is exactly why the implied volatility is unique. The solver walks along this
curve until the height matches the price you typed. The slope at that point is
the vega: how many currency units one volatility point is worth.
"""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def greek_label(name: str) -> str:
    """The library's display name for a Greek (``"gamma"`` -> ``"Gamma"``)."""
    return GREEK_INFO.get(name, {}).get("label", name.capitalize())


def breakeven_move(gamma: float, theta: float, dt: float) -> float:
    """Spot move over ``dt`` at which gamma exactly pays for theta.

    Solves ``0.5 * gamma * move**2 + theta * dt = 0`` with RAW Greeks, the same
    equation :func:`optionlab.plotting.profiles.gamma_theta_tradeoff` marks on
    its chart. Returns ``nan`` when gamma and theta share a sign (no crossing).
    """
    if not (math.isfinite(gamma) and math.isfinite(theta)) or gamma * theta >= 0 or gamma == 0:
        return float("nan")
    return math.sqrt(-2.0 * theta * dt / gamma)


# --------------------------------------------------------------------------- #
# Cached computations — primitive arguments in, DataFrames and floats out
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=128)
def greek_table(
    labels: tuple[str, ...],
    strikes: tuple[float, ...],
    taus: tuple[float, ...],
    option_types: tuple[str, ...],
    spot: float,
    vol: float,
    rate: float,
    div: float,
    trader_units: bool,
) -> pd.DataFrame:
    """One row of price and Greeks per contract, in the requested units.

    Everything is computed with :func:`optionlab.black_scholes.greeks`; the
    column headers are the library's own axis labels, so the table and the
    charts always quote the same units. The last column is the break-even daily
    move (see :func:`breakeven_move`).
    """
    rows = []
    for label, strike, tau, option_type in zip(labels, strikes, taus, option_types):
        raw = {k: float(v) for k, v in bs.greeks(spot, strike, tau, vol, rate, div, option_type).items()}
        shown = to_trader_units(raw) if trader_units else raw
        row = {"Contract": label}
        for name in TABLE_GREEKS:
            row[profiles.greek_axis_label(name, trader_units)] = shown[name]
        row["Break-even daily move"] = breakeven_move(raw["gamma"], raw["theta"], ONE_DAY)
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=64)
def decay_table(
    spot: float, vol: float, rate: float, div: float, option_type: str, trader_units: bool
) -> pd.DataFrame:
    """The at-the-money option (strike = spot) along the standard tenor ladder."""
    return greek_table(
        DECAY_LABELS,
        tuple(spot for _ in DECAY_TAUS),
        DECAY_TAUS,
        tuple(option_type for _ in DECAY_TAUS),
        spot,
        vol,
        rate,
        div,
        trader_units,
    )


@st.cache_data(show_spinner=False, max_entries=128)
def probability_in_the_money(
    spot: float, strike: float, tau: float, vol: float, rate: float, div: float, option_type: str
) -> float:
    """Risk-neutral chance of finishing in the money, read off the dual delta.

    The library's ``dual_delta`` is the discounted probability with a sign that
    depends on the option type, so undiscounting its absolute value gives
    ``N(d2)`` for a call and ``N(-d2)`` for a put.
    """
    dual = float(bs.dual_delta(spot, strike, tau, vol, rate, div, option_type))
    return abs(dual) * math.exp(rate * max(tau, 0.0))


@st.cache_data(show_spinner=False, max_entries=256)
def solve_implied_vol(
    price: float, spot: float, strike: float, tau: float, rate: float, div: float, option_type: str
) -> float:
    """Implied volatility, or ``nan`` when no volatility reproduces the price."""
    return float(bs.implied_vol(price, spot, strike, tau, rate, div, option_type))


@st.cache_data(show_spinner=False, max_entries=128)
def price_bounds(
    spot: float, strike: float, tau: float, rate: float, div: float, option_type: str
) -> tuple[float, float]:
    """Lowest and highest price the model can produce for this contract.

    The floor is the value at a volatility of essentially zero (the discounted
    intrinsic value); the ceiling is the value at an extreme volatility. A quote
    outside that band has no implied volatility because it is an arbitrage.
    """
    floor = float(bs.price(spot, strike, tau, 1e-9, rate, div, option_type))
    ceiling = float(bs.price(spot, strike, tau, 30.0, rate, div, option_type))
    return floor, ceiling


def table_config(trader_units: bool) -> dict[str, object]:
    """Number formats and tooltips for the columns :func:`greek_table` builds."""
    config: dict[str, object] = {
        profiles.greek_axis_label(name, trader_units): st.column_config.NumberColumn(
            format=TABLE_FORMATS[name], help=ui.greek_help(name, trader_units=trader_units)
        )
        for name in TABLE_GREEKS
    }
    config["Break-even daily move"] = st.column_config.NumberColumn(
        format="%.2f",
        help="How far the spot must travel in one day for gamma to pay for one day of theta. "
        "Below that, a delta-hedged long option loses money.",
    )
    return config


# --------------------------------------------------------------------------- #
# Callbacks (the only place a widget-bound key is written to)
# --------------------------------------------------------------------------- #
def set_strike_at_the_money() -> None:
    """Move the strike onto the current spot."""
    st.session_state.greeks_strike = round(state.spot(), 2)


def use_model_price() -> None:
    """Copy the model value of the current option into the market-price input."""
    try:
        market = state.current_market()
        contract = EuropeanOption(
            str(st.session_state.greeks_type or "call"),
            float(st.session_state.greeks_strike),
            float(market.t) + int(st.session_state.greeks_days) / 365.0,
        )
        st.session_state.greeks_market_price = round(float(contract.price(market)), 2)
    except (ValueError, KeyError, TypeError):  # a half-built page: leave the input alone
        pass


# --------------------------------------------------------------------------- #
# The market and the option under study
# --------------------------------------------------------------------------- #
mkt = state.current_market()
spot = state.spot()
trader_units = state.trader_units()

ui.page_header(
    "Greeks explorer",
    "One option, seen from every angle: how its value and its Greeks move with spot, time and "
    "volatility — and what that means for the P&L you would actually book.",
)

ui.how_to_use(
    """
Build one option in the control row below — type, strike, days to expiry, position size —
and every chart on the page is that same contract.

- The **headline row** is what a trader reads out first: price and the five Greeks, in the
  units the sidebar toggle selects, already scaled by your position size.
- The **seven views** underneath are seven questions about the same option. Only the open
  tab is computed, so switching between them is instant. Start with *Against spot*.
- A **negative position size** makes you short: every Greek flips, which is the fastest way
  to see what being short gamma looks like.
- The market itself is not on this page. Spot, volatility, rate and dividend come from the
  sidebar, so the contract is always priced in the same world as your book.
"""
)

ui.section(
    "The option under study",
    "Everything on this page is this one contract, priced in the market set in the sidebar.",
)

with st.container(border=True):
    kind_col, strike_col, expiry_col, size_col = st.columns([1.0, 1.2, 1.6, 1.0])

    option_type = kind_col.segmented_control(
        "Option type",
        ["call", "put"],
        default="call",
        required=True,
        key="greeks_type",
        help="A call is the right to buy at the strike, a put the right to sell. "
        "They share the same gamma and the same vega; only the direction differs.",
    ) or "call"

    strike = strike_col.number_input(
        "Strike",
        min_value=0.01,
        value=round(spot, 2),
        step=max(0.5, round(spot / 100.0, 2)),
        key="greeks_strike",
        help="The level at which the option can be exercised. Everything interesting "
        "(gamma, vega, theta) happens around it.",
    )
    strike_col.caption(ui.moneyness_caption(strike, spot, option_type))
    strike_col.button(
        "At the money",
        icon=":material/center_focus_strong:",
        key="greeks_atm",
        on_click=set_strike_at_the_money,
        help="Put the strike back on the spot.",
    )

    days = expiry_col.slider(
        "Days to expiry",
        min_value=1,
        max_value=730,
        value=30,
        step=1,
        key="greeks_days",
        help="Calendar days until the option expires. Drag it towards 1 and watch gamma "
        "and theta take over the charts below.",
    )
    expiry_col.caption(f"{days / 30.44:.1f} months, or {days / 365.0:.2f} years")

    quantity = size_col.number_input(
        "Position size",
        value=1.0,
        step=1.0,
        format="%.0f",
        key="greeks_qty",
        help="How many options you hold. Negative means short: every Greek flips sign, so "
        "-1 is the quickest way to see what being short gamma looks like.",
    )
    size_col.caption(
        "Short: every Greek is flipped" if quantity < 0 else "Long: you own the optionality"
    )

if quantity == 0:
    st.warning(
        "A position size of zero has no risk at all, so there would be nothing to draw. "
        "The charts below show one option.",
        icon=":material/warning:",
    )
    quantity = 1.0

expiry = float(mkt.t) + days / 365.0
tau = days / 365.0

try:
    option = EuropeanOption(option_type, float(strike), expiry)
except ValueError as error:  # pragma: no cover - the widgets already constrain the inputs
    st.error(f"That option cannot be built: {error}", icon=":material/error:")
    st.stop()

subject = option if quantity == 1.0 else Position(option, float(quantity))
raw_greeks = {name: float(value) for name, value in option.greeks(mkt).items()}
position_greeks = {name: value * quantity for name, value in raw_greeks.items()}

# --------------------------------------------------------------------------- #
# Headline: price, Greeks and the context a trader would quote with them
# --------------------------------------------------------------------------- #
ui.section(
    f"{subject.label} — today's risk",
    "The numbers a trader would read out before touching anything else.",
)

ui.greek_metrics(position_greeks, trader_units=trader_units)
ui.units_caption(trader_units)

prob_itm = probability_in_the_money(
    spot, float(strike), tau, float(mkt.vol), float(mkt.rate), float(mkt.div), option_type
)
daily_breakeven = breakeven_move(raw_greeks["gamma"], raw_greeks["theta"], ONE_DAY)
implied_daily_move = float(mkt.vol) * spot * math.sqrt(ONE_DAY)
forward = float(mkt.forward(expiry))

ui.metric_row(
    [
        (
            "Time to expiry",
            f"{days} days",
            f"{tau:.4f} years. The model only ever sees the time left, never the calendar.",
        ),
        (
            "Forward",
            ui.money(forward),
            "Price for delivery at expiry: spot grown at the rate and leaked by the dividend "
            "yield. Moneyness is really measured against this, not against the spot.",
        ),
        (
            "Chance of finishing in the money",
            ui.percent(prob_itm),
            "Risk-neutral probability, undiscounted from the library's dual delta. It is what "
            "the option market implies, not a forecast of where the stock is going.",
        ),
        (
            "Break-even daily move",
            ui.money(daily_breakeven),  # prints 'n/a' when gamma and theta share a sign
            "How far the spot must move in one day for gamma to pay for one day of theta. "
            f"One implied standard deviation over the same day is {ui.money(implied_daily_move)}.",
        ),
    ]
)

ui.explain(
    "What each of these Greeks means",
    "\n".join(
        f"- **{greek_label(name)}** "
        f"*({GREEK_INFO[name]['trader_unit' if trader_units else 'raw_unit']})* — "
        f"{GREEK_INFO[name]['description']}"
        for name in ui.HEADLINE_GREEKS
    ),
    icon=":material/menu_book:",
)

# --------------------------------------------------------------------------- #
# The seven views. Only the open tab is computed.
# --------------------------------------------------------------------------- #
ui.explain("What is in each view", VIEW_GUIDE, icon=":material/map:")

spot_tab, time_tab, vol_tab, surface_tab, compare_tab, pnl_tab, iv_tab = st.tabs(
    VIEWS, key="greeks_view", on_change="rerun"
)

# --------------------------------------------------------------------------- #
# View 1 — against spot
# --------------------------------------------------------------------------- #
if spot_tab.open:
    with spot_tab:
        ui.section(
            "All six Greeks against the spot",
            "One panel per Greek, same horizontal axis, each on its own vertical scale.",
        )
        ui.chart(
            profiles.greek_dashboard(
                subject,
                mkt,
                DASHBOARD_GREEKS,
                trader_units=trader_units,
                mode=ui.CHART_MODE,
            ),
            key="greeks_spot_dashboard",
        )
        ui.explain("How to read this", EXPLAIN_DASHBOARD)
        ui.takeaways(
            [
                "Delta is the **slope** of the price panel and gamma is the slope of the delta "
                "panel — read them left to right and the definitions stop being abstract.",
                "Gamma, vega and the most negative theta all sit **around the strike**. Owning "
                "convexity, owning volatility and paying rent are the same thing.",
                "Rho is usually the flattest panel: at short maturities interest rates are a "
                "rounding error next to spot and volatility, and only start to matter at the "
                "long end.",
            ]
        )

        ui.section("Zoom in on one Greek")
        spot_greek = (
            st.segmented_control(
                "Greek",
                SPOT_GREEKS,
                default="delta",
                required=True,
                format_func=greek_label,
                key="greeks_spot_greek",
                help="Pick the panel you want to see full size.",
            )
            or "delta"
        )
        st.caption(GREEK_INFO[spot_greek]["description"])
        ui.chart(
            profiles.greek_profile(
                subject, mkt, spot_greek, x="spot", trader_units=trader_units, mode=ui.CHART_MODE
            ),
            key="greeks_spot_profile",
        )
        ui.explain("How to read this", EXPLAIN_PROFILE)

        one_percent = 0.01 * spot
        ui.interview_note(
            "Gamma is the second derivative of the option value with respect to the spot: how "
            "fast delta itself moves. Long gamma means the position re-hedges in your favour — "
            "delta grows as the market rallies and shrinks as it falls, so every hedge sells "
            "high and buys low. You pay for that with theta.\n\n"
            "Short gamma is the mirror image, and it is dangerous for three reasons. The loss "
            "is **convex**: it grows with the square of the move while the premium you "
            "collected is fixed and known. Every re-hedge is at a **worse** price, so the "
            "bleeding is realised, not just marked. And gamma **explodes near the strike into "
            "expiry**, so a small short-dated position can become a large risk overnight "
            "without the spot doing anything unusual — exactly when liquidity is thinnest.\n\n"
            f"On the numbers in front of you: gamma is "
            f"{ui.greek_value('gamma', position_greeks['gamma'])} per 1 of spot, so a "
            f"{ui.money(one_percent)} move (1% of spot) changes your delta by about "
            f"{ui.number(position_greeks['gamma'] * one_percent, decimals=3)}. That is how much "
            "stock you would have to trade to stay flat.",
            question="What is gamma, and why is being short it dangerous?",
        )

        atm_model_price = float(EuropeanOption(option_type, spot, expiry).price(mkt))
        rule_of_thumb = 0.4 * spot * float(mkt.vol) * math.sqrt(tau)
        ui.interview_note(
            "Use the at-the-money approximation: an at-the-money option is worth roughly "
            "**0.4 times spot times volatility times the square root of the time to expiry**, in "
            "years. It comes from the Black-Scholes formula with d1 and d2 near zero, where the "
            "price collapses to about 0.8 times the spot times the standard normal density at "
            "zero times the total standard deviation.\n\n"
            f"With the market in the sidebar: 0.4 x {spot:g} x {ui.percent(float(mkt.vol))} x "
            f"the square root of {tau:.4f} years gives {ui.money(rule_of_thumb)}. The model's "
            f"own answer for the at-the-money {option_type} at this maturity is "
            f"{ui.money(atm_model_price)}.\n\n"
            "Two things the interviewer is checking. That you can do it in your head in a few "
            "seconds, and that you know **where it breaks**: it is an at-the-money rule, it "
            "ignores rates and dividends, and it degrades once the strike is far from the "
            "forward — exactly the cases where you should say so rather than quote a number.",
            question="The stock is at 100, volatility is 20, one month to expiry. Roughly what "
            "is the at-the-money call worth?",
        )

# --------------------------------------------------------------------------- #
# View 2 — through time
# --------------------------------------------------------------------------- #
if time_tab.open:
    with time_tab:
        time_greek = (
            st.segmented_control(
                "Greek",
                TIME_GREEKS,
                default="gamma",
                required=True,
                format_func=greek_label,
                key="greeks_time_greek",
                help="Gamma and theta tell the decay story best; try charm to see delta drift "
                "with no spot move at all.",
            )
            or "gamma"
        )
        st.caption(GREEK_INFO[time_greek]["description"])

        ui.section(
            f"{greek_label(time_greek)} against spot, as expiry approaches",
            "The same contract, redrawn at six times to expiry.",
        )
        try:
            ui.chart(
                profiles.greek_evolution(
                    subject,
                    mkt,
                    time_greek,
                    x="spot",
                    vary="tau",
                    trader_units=trader_units,
                    mode=ui.CHART_MODE,
                ),
                key="greeks_time_evolution",
            )
            ui.explain("How to read this", EXPLAIN_EVOLUTION_TAU)
        except ValueError as error:
            st.warning(f"That picture cannot be drawn here: {error}", icon=":material/warning:")

        ui.section(
            f"{greek_label(time_greek)} along the calendar",
            "Three spot levels held fixed while the clock runs down to expiry.",
        )
        try:
            ui.chart(
                profiles.greek_vs_time(
                    subject, mkt, time_greek, x="t", trader_units=trader_units, mode=ui.CHART_MODE
                ),
                key="greeks_time_vs_time",
            )
            ui.explain("How to read this", EXPLAIN_VS_TIME)
        except ValueError as error:
            st.warning(
                "This option is too close to expiry to draw against the calendar. "
                f"Add a few days above. ({error})",
                icon=":material/warning:",
            )

        ui.section(
            "The at-the-money blow-up, in numbers",
            "Strike set equal to today's spot, priced at six maturities.",
        )
        decay = decay_table(
            spot, float(mkt.vol), float(mkt.rate), float(mkt.div), option_type, trader_units
        )
        ui.table(decay, column_config=table_config(trader_units))
        ui.explain("How to read this", EXPLAIN_DECAY_TABLE)

        gamma_column = profiles.greek_axis_label("gamma", trader_units)
        theta_column = profiles.greek_axis_label("theta", trader_units)
        vega_column = profiles.greek_axis_label("vega", trader_units)
        gamma_1y = float(decay[gamma_column].iloc[0])
        gamma_1d = float(decay[gamma_column].iloc[-1])
        theta_1m = float(decay[theta_column].iloc[3])
        theta_1w = float(decay[theta_column].iloc[4])
        vega_1m = float(decay[vega_column].iloc[3])
        vega_1y = float(decay[vega_column].iloc[0])
        gamma_ratio = f"{gamma_1d / gamma_1y:.0f} times" if gamma_1y > 0 else "many times"

        ui.takeaways(
            [
                f"At the money, gamma goes from {ui.greek_value('gamma', gamma_1y)} with a year "
                f"to run to {ui.greek_value('gamma', gamma_1d)} on the last day — about "
                f"{gamma_ratio} larger for the very same contract.",
                f"Theta follows it: {ui.greek_value('theta', theta_1m)} a day at one month "
                f"against {ui.greek_value('theta', theta_1w)} a day at one week. The last "
                "fortnight of an option costs more than the first two months.",
                f"Vega moves the other way, from {ui.greek_value('vega', vega_1y)} at a year to "
                f"{ui.greek_value('vega', vega_1m)} at a month. Short-dated options are a bet "
                "on movement; long-dated ones are a bet on the level of volatility.",
                "Away from the strike every line dies instead of exploding — the option's fate "
                "is already decided, so there is nothing left to be sensitive to.",
            ]
        )

        ui.interview_note(
            "You lose money, and you lose it faster every day.\n\n"
            "A straddle is delta-neutral by construction, so a flat spot earns you nothing from "
            "re-hedging: the gamma you paid for never gets used. Meanwhile theta takes the time "
            "value out of both legs, and at the money it accelerates like one over the square "
            f"root of the time left. On the ladder above, the at-the-money option bleeds "
            f"{ui.greek_value('theta', theta_1m)} a day at one month and "
            f"{ui.greek_value('theta', theta_1w)} a day at one week — per leg.\n\n"
            "Vega usually compounds the problem: a market that stops moving tends to see its "
            "implied volatility marked lower, so the position loses on the vol mark as well as "
            "on the clock.\n\n"
            "The honest one-line answer: a long straddle is a bet that **realised** volatility "
            "beats the **implied** volatility you paid. Flat spot means realised is near zero, "
            "so you pay the rent and get nothing back.",
            question="You are long a one-month at-the-money straddle and the stock does not "
            "move. What happens?",
        )

# --------------------------------------------------------------------------- #
# View 3 — against volatility
# --------------------------------------------------------------------------- #
if vol_tab.open:
    with vol_tab:
        vol_greek = (
            st.segmented_control(
                "Greek",
                VOL_GREEKS,
                default="vega",
                required=True,
                format_func=greek_label,
                key="greeks_vol_greek",
                help="Start with price to see why the implied volatility is unique, then look "
                "at vega itself.",
            )
            or "vega"
        )
        st.caption(GREEK_INFO[vol_greek]["description"])

        ui.section(
            f"{greek_label(vol_greek)} against the volatility input",
            "Spot frozen at today's level; only the volatility the option is priced at moves.",
        )
        ui.chart(
            profiles.greek_profile(
                subject, mkt, vol_greek, x="vol", trader_units=trader_units, mode=ui.CHART_MODE
            ),
            key="greeks_vol_profile",
        )
        ui.explain("How to read this", EXPLAIN_VOL_PROFILE)

        ui.section(
            f"{greek_label(vol_greek)} against spot, at several volatility levels",
            "The profile you already know, redrawn for a calm market and a nervous one.",
        )
        ui.chart(
            profiles.greek_evolution(
                subject,
                mkt,
                vol_greek,
                x="spot",
                vary="vol",
                trader_units=trader_units,
                mode=ui.CHART_MODE,
            ),
            key="greeks_vol_evolution",
        )
        ui.explain("How to read this", EXPLAIN_VOL_EVOLUTION)

        ui.section(
            "Vega is a maturity story",
            f"The same strike ({strike:g}) at four maturities.",
        )
        vega_tenors = ("1 week", "1 month", "3 months", "1 year")
        ladder = {
            name: EuropeanOption(option_type, float(strike), float(mkt.t) + TENORS[name] / 365.0)
            for name in vega_tenors
        }
        ui.chart(
            profiles.compare_instruments(
                ladder,
                mkt,
                ("vega", "gamma"),
                x="spot",
                ncols=2,
                trader_units=trader_units,
                mode=ui.CHART_MODE,
                title=f"Vega and gamma by maturity: {option_type} struck at {strike:g}",
            ),
            key="greeks_vol_maturity",
        )
        ui.explain("How to read this", EXPLAIN_VEGA_LADDER)

        ladder_table = greek_table(
            vega_tenors,
            tuple(float(strike) for _ in vega_tenors),
            tuple(TENORS[name] / 365.0 for name in vega_tenors),
            tuple(option_type for _ in vega_tenors),
            spot,
            float(mkt.vol),
            float(mkt.rate),
            float(mkt.div),
            trader_units,
        )
        ui.table(ladder_table, column_config=table_config(trader_units))

        vega_column = profiles.greek_axis_label("vega", trader_units)
        gamma_column = profiles.greek_axis_label("gamma", trader_units)
        vega_1w = float(ladder_table[vega_column].iloc[0])
        vega_1y = float(ladder_table[vega_column].iloc[-1])
        gamma_1w = float(ladder_table[gamma_column].iloc[0])
        gamma_1y = float(ladder_table[gamma_column].iloc[-1])

        ui.takeaways(
            [
                f"Vega grows from {ui.greek_value('vega', vega_1w)} at one week to "
                f"{ui.greek_value('vega', vega_1y)} at one year. At the money that ratio is "
                "close to the square root of the ratio of the maturities, because the model "
                "only ever sees volatility times the square root of the time left.",
                f"Gamma goes the other way: {ui.greek_value('gamma', gamma_1w)} at one week "
                f"against {ui.greek_value('gamma', gamma_1y)} at one year.",
                "Vega peaks at the money and fades on both sides, so a book that is long "
                "wings and short the body is short vega even when it is long options.",
            ]
        )

        ui.interview_note(
            "Vega grows with the square root of the time to expiry. For an at-the-money option "
            "it is roughly the spot times the standard normal density at d1 times the square "
            "root of tau, so a one-year option has about three and a half times the vega of a "
            "one-month one, not twelve times.\n\n"
            "Two consequences an interviewer is listening for. First, **vega and gamma pull in "
            "opposite directions along the curve**: the front end is where you own movement, "
            "the back end is where you own the level of volatility. Second, vega decays slowly "
            "and smoothly, unlike gamma and theta, which blow up at the end — so a long-dated "
            "vega position is a position you can hold, while a short-dated gamma position is one "
            "you have to manage.\n\n"
            f"On this ladder: {ui.greek_value('vega', vega_1w)} at one week against "
            f"{ui.greek_value('vega', vega_1y)} at one year, for the same strike.",
            question="How does vega change with maturity?",
        )

# --------------------------------------------------------------------------- #
# View 4 — the surface
# --------------------------------------------------------------------------- #
if surface_tab.open:
    with surface_tab:
        control_row = st.container(horizontal=True, vertical_alignment="bottom")
        surface_greek = (
            control_row.segmented_control(
                "Greek",
                SURFACE_GREEKS,
                default="gamma",
                required=True,
                format_func=greek_label,
                key="greeks_surface_greek",
            )
            or "gamma"
        )
        second_axis = (
            control_row.segmented_control(
                "Second axis",
                ["Time to expiry", "Volatility"],
                default="Time to expiry",
                required=True,
                key="greeks_surface_axis",
                help="Spot always runs along the first axis; choose what the second one shows.",
            )
            or "Time to expiry"
        )
        surface_kind = (
            control_row.segmented_control(
                "Style",
                ["3-D surface", "Heatmap"],
                default="3-D surface",
                required=True,
                key="greeks_surface_kind",
                help="The surface shows the shape; the heatmap is easier to read precisely and "
                "marks the strike.",
            )
            or "3-D surface"
        )

        y_axis = "tau" if second_axis == "Time to expiry" else "vol"
        kind = "surface" if surface_kind == "3-D surface" else "heatmap"

        ui.section(
            f"{greek_label(surface_greek)} over spot and {second_axis.lower()}",
            "Two market variables at once, so you never have to hold one picture in your head "
            "while looking at another.",
        )
        try:
            ui.chart(
                profiles.greek_surface(
                    subject,
                    mkt,
                    surface_greek,
                    x="spot",
                    y=y_axis,
                    kind=kind,
                    trader_units=trader_units,
                    mode=ui.CHART_MODE,
                ),
                key="greeks_surface",
            )
            ui.explain("How to read this", EXPLAIN_SURFACE)
            ui.takeaways(
                [
                    "The gamma surface over spot and time is a ridge that starts wide and flat "
                    "and narrows into a wall at the strike — that wall is the whole risk of "
                    "short-dated options.",
                    "The vega surface over spot and volatility is almost flat in the volatility "
                    "direction: vega barely cares what level volatility is at, which is why "
                    "volga is small for options near the money.",
                    "A theta surface is entirely one colour. Long options only ever pay rent; "
                    "the question is how much, and where.",
                ]
            )
        except ValueError as error:
            st.warning(
                f"That surface cannot be drawn for this option: {error}", icon=":material/warning:"
            )

# --------------------------------------------------------------------------- #
# View 5 — compare
# --------------------------------------------------------------------------- #
if compare_tab.open:
    with compare_tab:
        ui.section(
            "Put two contracts on the same axes",
            "The fastest way to answer 'what does this change?' is to draw both.",
        )
        compare_mode = (
            st.segmented_control(
                "What to compare",
                ["Call vs put", "Strike ladder", "Maturity ladder"],
                default="Call vs put",
                required=True,
                key="greeks_compare_mode",
            )
            or "Call vs put"
        )
        chosen_greeks = st.pills(
            "Greeks to show",
            COMPARE_GREEKS,
            selection_mode="multi",
            default=["delta", "gamma", "vega", "theta"],
            format_func=greek_label,
            key="greeks_compare_greeks",
            help="One panel per Greek, sharing a legend.",
        )

        labels: tuple[str, ...] = ()
        strikes: tuple[float, ...] = ()
        taus: tuple[float, ...] = ()
        types: tuple[str, ...] = ()
        instruments: dict[str, EuropeanOption] = {}

        if compare_mode == "Strike ladder":
            chosen_moneyness = st.pills(
                "Strikes, as a percentage of the spot",
                MONEYNESS,
                selection_mode="multi",
                default=[0.90, 1.00, 1.10],
                format_func=lambda m: f"{m:.0%}",
                key="greeks_compare_strikes",
                help=f"Struck off today's spot of {spot:g}, all with {days} days to expiry.",
            )
            for moneyness in sorted(chosen_moneyness or []):
                level = round(spot * moneyness, 2)
                name = f"K {level:g} ({moneyness:.0%})"
                labels += (name,)
                strikes += (level,)
                taus += (tau,)
                types += (option_type,)
                instruments[name] = EuropeanOption(option_type, level, expiry)
        elif compare_mode == "Maturity ladder":
            chosen_tenors = st.pills(
                "Maturities",
                tuple(TENORS),
                selection_mode="multi",
                default=["1 month", "3 months", "1 year"],
                key="greeks_compare_tenors",
                help=f"All struck at {strike:g}, the strike you set above.",
            )
            for name in sorted(chosen_tenors or [], key=lambda label: TENORS[label]):
                tenor_days = TENORS[name]
                labels += (name,)
                strikes += (float(strike),)
                taus += (tenor_days / 365.0,)
                types += (option_type,)
                instruments[name] = EuropeanOption(
                    option_type, float(strike), float(mkt.t) + tenor_days / 365.0
                )
        else:
            for name, side in ((f"Call {strike:g}", "call"), (f"Put {strike:g}", "put")):
                labels += (name,)
                strikes += (float(strike),)
                taus += (tau,)
                types += (side,)
                instruments[name] = EuropeanOption(side, float(strike), expiry)

        if not chosen_greeks:
            ui.empty_state("Pick at least one Greek to compare.", icon=":material/touch_app:")
        elif not instruments:
            ui.empty_state("Pick at least one contract to compare.", icon=":material/touch_app:")
        else:
            try:
                if compare_mode == "Call vs put":
                    figure = profiles.call_put_comparison(
                        float(strike),
                        expiry,
                        mkt,
                        tuple(chosen_greeks),
                        ncols=2,
                        trader_units=trader_units,
                        mode=ui.CHART_MODE,
                    )
                else:
                    figure = profiles.compare_instruments(
                        instruments,
                        mkt,
                        tuple(chosen_greeks),
                        x="spot",
                        ncols=2,
                        trader_units=trader_units,
                        mode=ui.CHART_MODE,
                        title=f"{compare_mode}: {option_type}s",
                    )
                ui.chart(figure, key="greeks_compare_chart")
                ui.explain("How to read this", EXPLAIN_COMPARE)
            except ValueError as error:
                st.warning(f"That comparison cannot be drawn: {error}", icon=":material/warning:")

            ui.section("The same comparison at today's spot")
            ui.table(
                greek_table(
                    labels,
                    strikes,
                    taus,
                    types,
                    spot,
                    float(mkt.vol),
                    float(mkt.rate),
                    float(mkt.div),
                    trader_units,
                ),
                column_config=table_config(trader_units),
            )

        if compare_mode == "Call vs put":
            ui.takeaways(
                [
                    "Gamma, vega, volga and vanna are **identical** for the call and the put. "
                    "Their difference is a forward, and a forward has no optionality at all.",
                    "The deltas differ by exactly the forward's delta: call delta minus put "
                    "delta equals the discounted dividend factor, near 1 for a short maturity.",
                    "Theta and rho do differ, because the two contracts finance differently: "
                    "the call is long the forward, the put is short it.",
                ]
            )
            ui.interview_note(
                "Because put-call parity says the call minus the put is a forward, and a "
                "forward has no optionality. Its value is linear in the spot and does not "
                "depend on volatility at all, so every Greek that measures curvature or "
                "volatility sensitivity — gamma, vega, vanna, volga — must be the same for "
                "the two options.\n\n"
                "What *does* differ is everything the forward itself carries: delta differs by "
                "the discounted dividend factor, rho differs by the discounted strike, and "
                "theta differs by the carry. The practical version of this on a desk: a call "
                "and a put at the same strike are the **same volatility position**. Which one "
                "you trade is a funding, borrow and margin decision, not a risk decision — you "
                "can always turn one into the other with a delta hedge.",
                question="Your call and your put at the same strike show identical gamma and "
                "vega. Is that a bug?",
            )
        elif compare_mode == "Strike ladder":
            ui.takeaways(
                [
                    "Each strike owns its own neighbourhood: the gamma peak sits on the strike, "
                    "so a ladder of strikes is a ladder of local risk.",
                    "Out-of-the-money options have less vega but more volga, which is why they "
                    "are the instrument of choice for a view on the volatility of volatility.",
                    "Delta is a rough read on the chance of finishing in the money, so the "
                    "strike ladder is also a probability ladder.",
                ]
            )
        else:
            ui.takeaways(
                [
                    "Short maturities give a tall narrow gamma peak; long maturities give a low "
                    "wide one. The area under the curve is the optionality, and it moves from "
                    "concentrated to spread out.",
                    "Vega does the opposite, growing with the square root of the maturity.",
                    "A calendar spread — short the front, long the back — is exactly this "
                    "picture: short gamma, long vega.",
                ]
            )

# --------------------------------------------------------------------------- #
# View 6 — P&L intuition
# --------------------------------------------------------------------------- #
if pnl_tab.open:
    with pnl_tab:
        ui.section(
            "What the position is actually worth",
            "P&L against spot, today and at every point between now and expiry.",
        )
        ui.chart(
            profiles.pnl_profile(subject, mkt, mode=ui.CHART_MODE),
            key="greeks_pnl_profile",
        )
        ui.explain("How to read this", EXPLAIN_PNL)

        with st.form("greeks_scenario", border=True):
            st.markdown("**A scenario to explain**")
            st.caption(
                "Both charts below assume the same period passes and the same volatility shock "
                "happens during it."
            )
            vol_col, period_col = st.columns(2)
            vol_shock_points = vol_col.slider(
                "Volatility change over the period (vol points)",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                key="greeks_dvol",
                help="+1 means implied volatility rises by one point, for example from 20% to "
                "21%.",
            )
            period_days = period_col.slider(
                "Length of the period (days)",
                min_value=1,
                max_value=30,
                value=1,
                step=1,
                key="greeks_period",
                help="One day is the classic overnight P&L explain; a fortnight shows how "
                "quickly the Taylor approximation stops working.",
            )
            st.form_submit_button("Apply scenario", icon=":material/play_arrow:", type="primary")

        vol_shock = vol_shock_points / 100.0
        if float(mkt.vol) + vol_shock < 0:
            vol_shock = -float(mkt.vol)
            st.warning(
                "Volatility cannot go below zero, so the shock was capped at "
                f"{ui.percent(-float(mkt.vol))}.",
                icon=":material/warning:",
            )
        period = min(period_days, days)
        if period < period_days:
            st.caption(
                f"The option only has {days} days left, so the period was shortened to {period}."
            )
        period_years = period / 365.0

        ui.section(
            "Explaining the P&L of a spot move",
            "Delta alone, delta plus gamma, and the truth.",
        )
        try:
            ui.chart(
                profiles.taylor_pnl_explain(
                    subject,
                    mkt,
                    dvol=vol_shock,
                    dt=period_years,
                    trader_units=trader_units,
                    mode=ui.CHART_MODE,
                ),
                key="greeks_pnl_taylor",
            )
            ui.explain("How to read this", EXPLAIN_TAYLOR)
        except ValueError as error:
            st.warning(f"That scenario cannot be priced: {error}", icon=":material/warning:")

        ui.section(
            f"Gamma against theta over {period} day{'s' if period != 1 else ''}",
            "The delta-hedged P&L: what you own once the direction is hedged away.",
        )
        try:
            ui.chart(
                profiles.gamma_theta_tradeoff(
                    subject,
                    mkt,
                    dt=period_years,
                    trader_units=trader_units,
                    mode=ui.CHART_MODE,
                ),
                key="greeks_pnl_gamma_theta",
            )
            ui.explain("How to read this", EXPLAIN_GAMMA_THETA)
        except ValueError as error:
            st.warning(f"That trade-off cannot be drawn: {error}", icon=":material/warning:")

        period_breakeven = breakeven_move(
            raw_greeks["gamma"] * quantity, raw_greeks["theta"] * quantity, period_years
        )
        implied_period_move = float(mkt.vol) * spot * math.sqrt(period_years)
        rent = position_greeks["theta"] * period_years

        ui.metric_row(
            [
                (
                    "Break-even move over the period",
                    ui.money(period_breakeven),  # 'n/a' when there is no crossing
                    "Solves half gamma times the move squared plus theta times the period "
                    "equals zero, with the position's own Greeks.",
                ),
                (
                    "Implied move over the period",
                    ui.money(implied_period_move),
                    "One standard deviation at the sidebar volatility: spot times volatility "
                    "times the square root of the period in years.",
                ),
                (
                    "Time value paid over the period",
                    ui.money(rent, signed=True),
                    "Theta times the length of the period. Negative means it costs you; "
                    "positive means you collect it.",
                ),
            ]
        )

        ui.takeaways(
            [
                "The break-even move and the implied move are close to each other by design — "
                "that is what it means for an option to be priced at a volatility.",
                "So 'is this option cheap?' is never about the premium. It is about whether you "
                "think the underlying will move more or less than the implied move above.",
                "Beyond about one standard deviation the delta-plus-gamma line starts to drift "
                "away from the truth. That gap is why desks reprice rather than rely on Greeks "
                "for large shocks.",
            ]
        )

        ui.interview_note(
            "Once delta is hedged, the P&L over a short period is close to one half gamma times "
            "the squared spot move, plus theta times the time elapsed. The first term is always "
            "positive when you are long gamma, whichever way the market goes; the second is a "
            "fixed cost.\n\n"
            "So a delta-hedged long option makes money when the underlying **realises** more "
            "movement than the price implied. Concretely, you need daily moves bigger than "
            f"{ui.money(daily_breakeven) if math.isfinite(daily_breakeven) else 'the break-even'}"
            " on this position; one implied standard deviation over a day is "
            f"{ui.money(implied_daily_move)}. Below that you are paying rent for convexity you "
            "never use.\n\n"
            "Two refinements worth adding out loud: the profit depends on the **path**, not "
            "just the final level, because gamma is collected move by move as you re-hedge; and "
            "hedging more often reduces the variance of the outcome but not its expectation, "
            "while adding transaction costs.",
            question="You are long a call and you delta-hedge it. When do you make money?",
        )

# --------------------------------------------------------------------------- #
# View 7 — implied volatility
# --------------------------------------------------------------------------- #
if iv_tab.open:
    with iv_tab:
        ui.section(
            "From a price to a volatility",
            "Type the price you see quoted; the solver returns the volatility that reproduces "
            "it in this market.",
        )

        model_price = float(option.price(mkt))
        input_col, result_col = st.columns([1.0, 2.0])

        market_price = input_col.number_input(
            "Market price of the option",
            min_value=0.0,
            value=round(model_price, 2),
            step=0.05,
            format="%.2f",
            key="greeks_market_price",
            help="The premium quoted for one option, in the same currency as the spot.",
        )
        input_col.button(
            "Use the model price",
            icon=":material/restart_alt:",
            key="greeks_reset_price",
            on_click=use_model_price,
            help=f"Today's model value is {ui.money(model_price)}.",
        )

        implied = solve_implied_vol(
            float(market_price),
            spot,
            float(strike),
            tau,
            float(mkt.rate),
            float(mkt.div),
            option_type,
        )

        with result_col:
            if math.isnan(implied):
                floor, ceiling = price_bounds(
                    spot, float(strike), tau, float(mkt.rate), float(mkt.div), option_type
                )
                st.error(
                    f"No volatility reproduces {ui.money(float(market_price))}. In this market "
                    f"the model can only produce prices between {ui.money(floor)} (the value at "
                    f"a volatility of essentially zero) and {ui.money(ceiling)} (the value at an "
                    "extreme volatility). A quote outside that band is an arbitrage, not a "
                    "volatility.",
                    icon=":material/error:",
                )
            else:
                difference_points = (implied - float(mkt.vol)) * 100.0
                # Always per vol point here, whatever the sidebar toggle says: the
                # question "what is one point worth?" only has one sensible unit.
                vega_points = float(to_trader_units(raw_greeks)["vega"])
                ui.metric_row(
                    [
                        (
                            "Implied volatility",
                            ui.percent(implied),
                            "The volatility at which this model returns exactly the price you "
                            "typed.",
                        ),
                        (
                            "Sidebar volatility",
                            ui.percent(float(mkt.vol)),
                            "The volatility every other page is priced at.",
                        ),
                        (
                            "Difference",
                            f"{difference_points:+.2f} pts",
                            "Positive means the market is charging more volatility than your "
                            "own mark — the option looks dear on your curve.",
                        ),
                        (
                            "Value of one vol point",
                            ui.money(abs(vega_points)),
                            "Vega per volatility point: how much the premium moves for a one "
                            "point change in implied volatility.",
                        ),
                    ]
                )
                if abs(difference_points) < 0.005:
                    st.caption(
                        "That is the sidebar volatility itself: the model reproduces its own "
                        "price, which is a useful sanity check on the solver."
                    )
                elif difference_points > 0:
                    st.caption(
                        f"The market is charging {difference_points:.2f} volatility points more "
                        f"than your mark. Selling this option at {ui.money(float(market_price))} "
                        f"collects about {ui.money(abs(vega_points) * difference_points)} of "
                        "edge on the vega, before any hedging cost."
                    )
                else:
                    st.caption(
                        f"The market is charging {abs(difference_points):.2f} volatility points "
                        "less than your mark, so on your own curve the option looks cheap by "
                        f"about {ui.money(abs(vega_points) * abs(difference_points))}."
                    )

        ui.explain("What implied volatility is (and is not)", EXPLAIN_IV, expanded=False)

        ui.section(
            "Why the answer is unique",
            "The model price of this option as the volatility input is turned up.",
        )
        ui.chart(
            profiles.greek_profile(
                option, mkt, "price", x="vol", trader_units=trader_units, mode=ui.CHART_MODE
            ),
            key="greeks_iv_curve",
        )
        ui.explain("How to read this", EXPLAIN_IV_CURVE)

        ui.takeaways(
            [
                "The curve is strictly increasing, so there is one volatility per price. That "
                "is the whole reason the market can quote in volatility at all.",
                "It is also quite flat at the extremes: far out of the money, a tiny change in "
                "price moves the implied volatility a long way, which is why wing quotes look "
                "noisy.",
                "The floor of the curve is the discounted intrinsic value. A price below it "
                "would be an arbitrage, and the solver returns nothing rather than a number.",
            ]
        )

        ui.interview_note(
            "It means the option market is paying more for protection and convexity than your "
            "own estimate of future movement justifies — so on your own view, selling "
            "volatility has positive expected value.\n\n"
            "But the interviewer is testing whether you stop there. Three caveats belong in the "
            "answer. Implied volatility is a **risk-neutral** price, not a forecast: the gap "
            "usually contains a genuine risk premium, because the seller is short a convex, "
            "hard-to-hedge payoff. The gap is also **not a free option**: capturing it means "
            "delta-hedging through the period, paying transaction costs and surviving the path. "
            "And the comparison only holds if you are honest about which realised volatility "
            "you mean — the one measured over the same horizon, on the same underlying, at the "
            "same sampling frequency.\n\n"
            "A clean answer ends with the trade: sell the option, delta-hedge it, and your P&L "
            "is the gamma-weighted difference between realised and implied variance.",
            question="The implied volatility is well above your estimate of realised volatility. "
            "What does that tell you, and what would you do?",
        )
