"""Exotics: nine products, each measured against the vanilla it is built from.

The page follows the order a desk looks at an exotic in:

1. what the contract is, in the library's own words;
2. what it costs against its vanilla benchmark, Greek by Greek;
3. the one picture that explains where the difference comes from;
4. a Monte Carlo check of the closed form, with its standard error;
5. a ticket to put the trade in the book.

Everything is priced in the market set in the sidebar, so an exotic, a strategy
and the book are always looked at in the same world.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pandas as pd
import streamlit as st

from optionlab import EXOTIC_REGISTRY, GREEK_INFO, Market, build_exotic, mc_price, to_trader_units
from optionlab.exotics import mc_price_brownian_bridge, mc_price_with_control_variate
from optionlab.plotting import exotic_plots, profiles

from app_lib import state, ui

# --------------------------------------------------------------------------- #
# Constants and small helpers (pure: no Streamlit output)
# --------------------------------------------------------------------------- #
CATEGORIES = ("Digital", "Barrier", "Asian", "Lookback")

CATEGORY_BLURB = {
    "Digital": "All-or-nothing bets on where the spot finishes. No hockey stick: a step.",
    "Barrier": "Vanillas switched on or off the first time the spot touches a level.",
    "Asian": "Paid on the average spot over a window instead of the last print.",
    "Lookback": "Paid on the best price of the whole life: trading with hindsight.",
}

#: Greeks compared with the vanilla in the table.
COMPARED = ("price", "delta", "gamma", "vega", "theta", "rho")

#: Monitoring frequencies offered to the Monte Carlo check.
MONITORING = {"Daily": 252, "Weekly": 52, "Monthly": 12}

VIEWS = ("Against the vanilla", "Inside the product", "Model check")


def _items(mapping: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """A dict of primitives as a hashable, cache-friendly key."""
    return tuple(sorted(mapping.items()))


def _build(name: str, params: dict[str, Any], extra: dict[str, Any]):
    """Build a registry product, then apply the path state (knocked, averages...)."""
    instrument = build_exotic(name, **params)
    return dataclasses.replace(instrument, **extra) if extra else instrument


def _breached(instrument, spot: float) -> bool:
    """True when the spot already sits at or beyond a barrier."""
    return spot >= instrument.barrier if instrument.is_up else spot <= instrument.barrier


def _ratio_sentence(share: float) -> str:
    return f"{share:.0%} of the vanilla" if math.isfinite(share) else "not comparable"


def _delta_at(product, mkt: Market, spot_level: float, days_left: float | None = None) -> float:
    """Delta of a product at a given spot, optionally close to expiry.

    Delta is the same number in raw and trader units, so no conversion is needed.
    """
    market = mkt.bumped(spot=float(spot_level))
    if days_left is not None:
        market = market.bumped(t=float(product.expiry) - days_left / 365.0)
    return float(product.greeks(market)["delta"])


@st.cache_data(max_entries=64, show_spinner=False)
def _monte_carlo(
    name: str,
    params_key: tuple[tuple[str, Any], ...],
    extra_key: tuple[tuple[str, Any], ...],
    market_key: tuple[float, float, float, float, float],
    n_paths: int,
    n_steps: int,
    seed: int,
    estimator: str,
) -> dict[str, float]:
    """Monte Carlo price and standard error. Arguments are primitives, so this caches."""
    spot, vol, rate, div, now = market_key
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div, t=now)
    instrument = _build(name, dict(params_key), dict(extra_key))
    if estimator == "bridge":
        price, stderr = mc_price_brownian_bridge(
            instrument, mkt, n_paths=n_paths, n_steps=n_steps, seed=seed
        )
    elif estimator == "control":
        price, stderr = mc_price_with_control_variate(
            instrument, mkt, n_paths=n_paths, n_steps=n_steps, seed=seed
        )
    else:
        price, stderr = mc_price(instrument, mkt, n_paths=n_paths, n_steps=n_steps, seed=seed)
    return {"price": float(price), "stderr": float(stderr)}


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #
mkt = state.current_market()
spot = state.spot()
trader_units = state.trader_units()

ui.page_header(
    "Exotics",
    "Nine products that are not vanilla — and exactly what each one costs "
    "against the vanilla it is built from.",
)

ui.how_to_use(
    """
The page is one walk through any of the nine products, in five numbered steps.

1. **Pick a product** — four families, nine contracts, straight from the library's registry.
2. **Set the contract** — the inputs come from the product's own specification. Underneath
   them, *Path state* is what the contract has already seen: a barrier that has been
   touched, an Asian that is part way through its averaging window, a lookback that has
   already recorded an extreme.
3. **What it costs** — the exotic, its vanilla twin with the same strike and expiry, and
   the difference Greek by Greek. The ratio between the two prices is the commercial story.
4. **See it** — three views: against the vanilla, inside the mechanism, or the closed form
   against a Monte Carlo. Only the selected one is computed.
5. **Put it in your book** — the ticket goes to the same book the other pages use.
"""
)

ui.explain(
    "What an exotic is, and why the vanilla is always the benchmark",
    """
An exotic option changes **one** thing about a vanilla: what the payoff looks at.
A digital looks at whether the spot finished in the money rather than by how much;
a barrier looks at whether a level was ever touched; an Asian looks at the average
rather than the last print; a lookback looks at the best price of the whole life.

Every price on this page is therefore quoted twice: once for the exotic and once for
the vanilla with the same type, strike and expiry. The ratio between the two is the
whole commercial story — *"this knock-out costs 13% of the vanilla, because you give
up every path that trades through the barrier"*. The Greeks tell you the other half:
what the desk has to hedge once it has sold the thing.

All prices are per unit of the contract, in the market set in the sidebar.
""",
    icon=":material/school:",
)

# --------------------------------------------------------------------------- #
# 1. Pick a product
# --------------------------------------------------------------------------- #
ui.section("1. Pick a product", "Four families, nine contracts, all from the library's registry.")

with st.container(border=True):
    family = st.segmented_control(
        "Family",
        CATEGORIES,
        default="Barrier",
        required=True,
        key="exotics_family",
        help="Digitals bet on where the spot ends; barriers, Asians and lookbacks "
        "all depend on the path the spot takes to get there.",
    )
    family = family or "Barrier"
    st.caption(CATEGORY_BLURB[family])

    names = [key for key, candidate in EXOTIC_REGISTRY.items() if candidate.category == family]
    name = st.selectbox(
        "Product",
        names,
        format_func=lambda key: EXOTIC_REGISTRY[key].title,
        key=f"exotics_product_{family.lower()}",
        help="Each entry carries its own description, parameters and defaults.",
    )
    spec = EXOTIC_REGISTRY[name]

with st.container(border=True):
    st.markdown(f":gray-badge[:material/description: {spec.category}] **{spec.title}**")
    st.markdown(spec.description)

# --------------------------------------------------------------------------- #
# 2. Set the contract
# --------------------------------------------------------------------------- #
ui.section(
    "2. Set the contract",
    "The inputs and their defaults come from the product's own specification.",
)

extra: dict[str, Any] = {}

with st.container(border=True):
    params = ui.param_controls(spec, spot, "exotics", columns=3)

    if "strike" in params:
        st.caption(
            f"Strike {float(params['strike']):g} against a spot of {spot:g}: "
            f"{ui.moneyness_caption(float(params['strike']), spot, params.get('option_type'))}."
        )

    st.markdown(":gray-badge[:material/history: Path state]")
    if family == "Digital":
        st.caption(
            "A digital has no path state: only where the spot finishes matters, so nothing "
            "can be fixed before expiry."
        )
    elif family == "Barrier":
        knocked = st.toggle(
            "The barrier has already been touched",
            key="exotics_knocked",
            help="Switch this on to see what the contract becomes after the event: a "
            "knock-out is worth zero (its rebate was paid at the hit), a knock-in has "
            "become the plain vanilla.",
        )
        extra["knocked"] = bool(knocked)
    elif family == "Asian":
        if float(params.get("avg_start", 0.0)) > 0:
            st.caption(
                "This contract is forward-starting: the averaging window has not begun, "
                "so nothing is fixed yet."
            )
        else:
            seasoning_cols = st.columns(2)
            seasoned_days = seasoning_cols[0].slider(
                "Days already averaged",
                min_value=0,
                max_value=730,
                value=0,
                key="exotics_asian_seasoned",
                help="How long the contract has been running. 0 means it starts averaging "
                "today; 90 means a quarter of the window is already fixed.",
            )
            average_pct = seasoning_cols[1].slider(
                "Average so far, as a share of today's spot",
                min_value=70,
                max_value=130,
                value=100,
                format="%d%%",
                key="exotics_asian_average",
                help="The average already printed over the observed stretch. It is a fixed "
                "number now: no future spot move can change it.",
            )
            if seasoned_days > 0:
                running_average = round(spot * average_pct / 100.0, 4)
                params["avg_start"] = -seasoned_days / 365.0
                extra["running_average"] = running_average
                extra["observed_until"] = 0.0
                extra["last_spot"] = float(spot)
                st.caption(
                    f"The window runs from {seasoned_days} days ago to expiry, and its first "
                    f"stretch is locked at {ui.money(running_average)}."
                )
            else:
                st.caption("Fresh contract: the whole window is still in front of you.")
    else:  # Lookback
        uses_min = (spec.name == "floating_lookback") == (params["option_type"] == "call")
        if uses_min:
            low_pct = st.slider(
                "Lowest spot seen so far, as a share of today's spot",
                min_value=50,
                max_value=100,
                value=100,
                format="%d%%",
                key="exotics_lookback_min",
                help="100% means the contract is fresh (the running minimum is today's spot). "
                "Move it down to see a contract that has already watched the market fall.",
            )
            extra["running_min"] = round(spot * low_pct / 100.0, 4)
            st.caption(f"Running minimum: {ui.money(extra['running_min'])}.")
        else:
            high_pct = st.slider(
                "Highest spot seen so far, as a share of today's spot",
                min_value=100,
                max_value=150,
                value=100,
                format="%d%%",
                key="exotics_lookback_max",
                help="100% means the contract is fresh (the running maximum is today's spot). "
                "Move it up to see a contract that has already watched a rally.",
            )
            extra["running_max"] = round(spot * high_pct / 100.0, 4)
            st.caption(f"Running maximum: {ui.money(extra['running_max'])}.")

try:
    instrument = _build(name, params, extra)
except ValueError as exc:
    st.error(
        f"That contract does not exist: {exc}. Adjust the inputs above and it will price again.",
        icon=":material/error:",
    )
    st.stop()

try:
    vanilla = exotic_plots.vanilla_benchmark(instrument, mkt)
except ValueError as exc:  # pragma: no cover - every registry product has a benchmark
    st.error(f"No vanilla benchmark for this contract: {exc}", icon=":material/error:")
    st.stop()

if family == "Barrier" and not extra.get("knocked") and _breached(instrument, spot):
    side = "at or above" if instrument.is_up else "at or below"
    st.warning(
        f"The spot ({spot:g}) is already {side} the barrier ({instrument.barrier:g}), so the "
        f"contract has already knocked "
        f"{'in and is simply the vanilla' if instrument.is_knock_in else 'out'}. "
        "Move the barrier to the other side of the spot to price a live contract.",
        icon=":material/warning:",
    )

# --------------------------------------------------------------------------- #
# 3. Price and Greeks against the vanilla
# --------------------------------------------------------------------------- #
price = float(instrument.price(mkt))
vanilla_price = float(vanilla.price(mkt))
share = price / vanilla_price if vanilla_price > 0 else float("nan")

ui.section(
    "3. What it costs, and what you are paid for",
    "The exotic, its vanilla twin, and the difference between them.",
)

extra_metric: tuple[str, str, str]
headline: str

if family == "Digital":
    probability = float(instrument.probability_itm(mkt))
    extra_metric = (
        "Probability in the money",
        ui.percent(probability, decimals=1),
        "Risk-neutral probability of finishing in the money, N(omega d2). It is the price "
        "of the digital paying 1, before discounting — not a real-world probability.",
    )
    if instrument.is_cash:
        headline = (
            f"The digital is worth {ui.money(price)}: the {ui.money(instrument.payout)} payout "
            f"times the {probability:.1%} risk-neutral chance of finishing in the money, "
            f"discounted. The vanilla with the same strike costs {ui.money(vanilla_price)} "
            f"because it pays *how far* in the money you finish, not just *whether*."
        )
    else:
        headline = (
            f"The asset-or-nothing delivers the share itself, so it is worth "
            f"{ui.money(price)} — much more than the {ui.money(vanilla_price)} vanilla. "
            f"The identity to remember: asset digital = vanilla + strike x cash digital."
        )
elif family == "Barrier":
    flipped = "up-and-in" if instrument.barrier_type == "up-and-out" else (
        "up-and-out" if instrument.barrier_type == "up-and-in" else (
            "down-and-in" if instrument.barrier_type == "down-and-out" else "down-and-out"
        )
    )
    twin = dataclasses.replace(instrument, barrier_type=flipped, rebate=0.0)
    twin_price = float(twin.price(mkt))
    extra_metric = (
        "The other half",
        ui.money(twin_price),
        f"The matching {flipped} option with no rebate. In-out parity: knock-in plus "
        "knock-out is the vanilla, because exactly one of the two is alive at expiry.",
    )
    touch_side = "at or above" if instrument.is_up else "at or below"
    headline = (
        f"This {instrument.barrier_type} option is {_ratio_sentence(share)}. Its mirror image, "
        f"the {flipped}, is worth {ui.money(twin_price)}; together they cost "
        f"{ui.money(price + twin_price)}, which is the vanilla's {ui.money(vanilla_price)} — "
        f"in-out parity. What separates them is only whether the spot ever trades "
        f"{touch_side} {instrument.barrier:g}."
    )
elif family == "Asian":
    effective_vol = float(instrument.effective_vol(mkt))
    market_vol = float(mkt.vol)
    remaining = float(instrument.remaining_weight(mkt))
    extra_metric = (
        "Volatility of the average",
        ui.percent(effective_vol, decimals=1),
        "Black volatility of the still-random part of the average. An average moves less "
        "than its last point, so this sits well below the market's volatility.",
    )
    ratio = effective_vol / market_vol if market_vol > 0 else float("nan")
    headline = (
        f"The Asian is {_ratio_sentence(share)}, and the reason is one number: the still-random "
        f"average carries {ui.percent(effective_vol, decimals=1)} of volatility against the "
        f"market's {ui.percent(market_vol, decimals=1)} — a ratio of {ratio:.2f}, the "
        f"1 / sqrt(3) = 0.58 of a continuous average."
    )
    if remaining < 0.999:
        headline += (
            f" On top of that, only {remaining:.0%} of the payoff is still exposed to it: the "
            f"other {1 - remaining:.0%} of the window is already printed and cannot move again, "
            "which is why the price and every Greek are smaller than for a fresh contract."
        )
else:  # Lookback
    extra_metric = (
        "Cost of hindsight",
        ui.money(price - vanilla_price, signed=True),
        "What you pay on top of the vanilla for being struck at the best price of the whole "
        "life instead of at today's level.",
    )
    multiple = price / vanilla_price if vanilla_price > 0 else float("nan")
    multiple_text = f"{multiple:.2f} times as much" if math.isfinite(multiple) else "far more"
    headline = (
        f"The lookback costs {ui.money(price)} against the vanilla's {ui.money(vanilla_price)}: "
        f"{multiple_text}. Hindsight is never free, and a fresh floating-strike lookback is "
        f"famously worth about twice the at-the-money option."
    )

ui.metric_row(
    [
        (
            spec.title,
            ui.money(price),
            f"Model value today of {instrument.label}, per unit of the contract.",
        ),
        (
            "Vanilla benchmark",
            ui.money(vanilla_price),
            f"{vanilla.label}: the plain European option this product is measured against.",
        ),
        (
            "Exotic / vanilla",
            ui.percent(share) if math.isfinite(share) else "n/a",
            "Above 100% the exotic is dearer than the vanilla (lookbacks); below it, you are "
            "being paid to give something up (barriers, Asians).",
        ),
        extra_metric,
    ]
)
st.markdown(headline)

ui.section("Greeks, side by side", "What the desk has to hedge once the trade is done.")

raw_exotic = instrument.greeks(mkt)
raw_vanilla = vanilla.greeks(mkt)
shown_exotic = to_trader_units(dict(raw_exotic)) if trader_units else dict(raw_exotic)
shown_vanilla = to_trader_units(dict(raw_vanilla)) if trader_units else dict(raw_vanilla)

ui.greek_metrics(raw_exotic, trader_units=trader_units)

comparison = pd.DataFrame(
    {
        "Greek": [GREEK_INFO[key]["label"] for key in COMPARED],
        "Unit": [
            GREEK_INFO[key]["trader_unit" if trader_units else "raw_unit"] for key in COMPARED
        ],
        spec.title: [ui.greek_value(key, float(shown_exotic[key])) for key in COMPARED],
        "Vanilla": [ui.greek_value(key, float(shown_vanilla[key])) for key in COMPARED],
        "Difference": [
            ui.greek_value(key, float(shown_exotic[key]) - float(shown_vanilla[key]))
            for key in COMPARED
        ],
    }
)
ui.table(comparison)
ui.units_caption(trader_units)

ui.explain(
    "How to read the difference column",
    """
The difference column is the risk the exotic *adds* to (or removes from) the vanilla —
which is exactly the position a desk ends up with when it sells the exotic and hedges
it with the vanilla.

- A **negative delta difference** on a knock-out call means the exotic is less long the
  stock than the vanilla, and close to a reverse barrier the exotic's own delta can be
  negative: owning a call while being short the market.
- A **negative gamma or vega difference** is the warning sign. The vanilla is long both;
  anything that takes them away is a product whose value falls when the market moves or
  when volatility rises. That is why barriers near the knock-out and digitals near the
  strike are hard to hedge, and why they are cheap.
- A **smaller everything** (the Asian) means a quieter book: the averaging damps the
  Greeks, and it keeps damping them as the window fills in.
""",
)

if family == "Digital":
    ui.interview_note(
        f"Almost, and the 'almost' is the answer. A cash digital is the discounted "
        f"risk-neutral probability of finishing in the money: this one prices "
        f"{ui.money(price)} on a {ui.money(instrument.payout)} payout, which is a "
        f"{float(instrument.probability_itm(mkt)):.1%} chance once you undo the discounting. "
        "Two caveats worth saying out loud: it is risk-neutral, not real-world — the stock is "
        "assumed to drift at the risk-free rate minus the dividend, not at its own expected "
        "return; and it is the probability implied by *this* volatility, so quoting a digital "
        "is quoting a view on the whole distribution, not just on the level.",
        question="Is the price of a digital call the probability that the stock finishes above "
        "the strike?",
    )
elif family == "Barrier":
    ui.interview_note(
        f"Because the barrier kills the option exactly where it is worth most. This contract "
        f"is {_ratio_sentence(share)}: you have sold away every path that trades through "
        f"{instrument.barrier:g}. For a reverse knock-out (barrier in the money) the option "
        "carries large intrinsic value just before the touch and nothing just after, so as "
        "the spot approaches the barrier delta turns negative — a rally now destroys your "
        "option — gamma is large and negative, and you are short volatility. The 'Inside the "
        "product' view puts numbers on that.",
        question="Why is a reverse knock-out call cheaper than the vanilla, and what happens "
        "to its delta near the barrier?",
    )
elif family == "Asian":
    ui.interview_note(
        f"Because you are buying an option on an average, and an average moves less than its "
        f"last point. For a fresh continuous average the effective volatility is about "
        f"vol / sqrt(3) = 58% of the spot's; here the library computes "
        f"{ui.percent(float(instrument.effective_vol(mkt)), decimals=1)} against a market "
        f"volatility of {ui.percent(float(mkt.vol), decimals=1)}, and the option prices "
        f"{_ratio_sentence(share)}. The second half of the answer is who buys it: a commodity "
        "or FX hedger whose real exposure *is* an average price, plus the fact that an average "
        "is much harder to squeeze on the fixing date than a single close.",
        question="Why is an Asian option cheaper than the equivalent vanilla?",
    )
else:
    ui.interview_note(
        f"Because you are paid on the best level of the whole life, not the last one: the "
        f"option can never give its gains back. Here that costs "
        f"{ui.money(price - vanilla_price)} on top of the {ui.money(vanilla_price)} vanilla. "
        "The honest follow-up is that nobody can hedge it statically with vanillas: the "
        "payoff depends on the running extreme, so the hedge has to be rebalanced along the "
        "path, and the more often the market prints a new extreme the more expensive that "
        "rebalancing turns out to be.",
        question="A client asks for a lookback call. Why does it cost roughly twice the "
        "at-the-money vanilla?",
    )

# --------------------------------------------------------------------------- #
# 4. The pictures
# --------------------------------------------------------------------------- #
ui.section("4. See it", "Three ways to look at the same contract.")

view = st.segmented_control(
    "View",
    VIEWS,
    default=VIEWS[0],
    required=True,
    key="exotics_view",
    help="Against the vanilla: price and Greeks overlaid. Inside the product: the mechanism "
    "that makes it different. Model check: the closed form against a simulation.",
)
view = view or VIEWS[0]

# --- 4a. Exotic vs vanilla ------------------------------------------------- #
if view == VIEWS[0]:
    panels = st.pills(
        "Panels",
        ["price", "delta", "gamma", "vega", "theta"],
        selection_mode="multi",
        default=["price", "delta", "gamma", "vega"],
        format_func=lambda key: GREEK_INFO[key]["label"],
        key="exotics_panels",
        help="Price plus any Greeks you want to compare. Gamma and vega are where exotics "
        "stop looking like vanillas.",
    )
    chosen = tuple(key for key in ("price", "delta", "gamma", "vega", "theta") if key in panels)
    if not chosen:
        chosen = ("price", "delta", "gamma", "vega")
        st.caption("Nothing selected, so the default four panels are shown.")

    try:
        figure = exotic_plots.exotic_vs_vanilla(
            instrument,
            mkt,
            greeks=chosen,
            trader_units=trader_units,
            mode=ui.CHART_MODE,
        )
        ui.chart(figure, key="exotics_vs_vanilla")
    except ValueError as exc:
        st.warning(
            f"This contract cannot be drawn against its vanilla: {exc}",
            icon=":material/warning:",
        )

    ui.explain(
        "How to read this",
        """
Each panel plots the same quantity for two instruments against today's spot: the exotic
(solid, blue for a call and orange for a put) and its vanilla benchmark (dashed grey).
The vertical lines mark the strike and, where there is one, the barrier.

Read it from left to right as "what happens if the spot is here instead of where it is
now". Where the two lines sit on top of each other the exotic behaves like the vanilla;
where they separate, that gap *is* the product. The Greek panels are the important ones:
a price difference is a commercial fact, a Greek difference is a hedging problem.
""",
    )

    if family == "Digital":
        tenor_days = max(1.0, float(instrument.expiry - mkt.t) * 365.0)
        late_days = min(5.0, max(0.5, tenor_days / 2.0))
        delta_now = _delta_at(instrument, mkt, instrument.strike)
        delta_late = _delta_at(instrument, mkt, instrument.strike, days_left=late_days)
        ui.takeaways(
            [
                "The price panel is a smoothed step, not a hockey stick: the digital is bounded "
                f"by its {ui.money(instrument.payout)} payout however far in the money it ends.",
                f"Delta is a bump centred on the strike, not a ramp. With the spot exactly at "
                f"{instrument.strike:g} it is {ui.number(delta_now, decimals=2)} today and "
                f"{ui.number(delta_late, decimals=2)} with {late_days:g} days left: it grows "
                "without limit as expiry approaches, and that spike is the pin risk.",
                "Gamma and vega change sign at the strike: out of the money the digital wants "
                "movement, in the money it wants the market to stop moving and settle.",
            ]
        )
    elif family == "Barrier":
        if instrument.knocked:
            ui.takeaways(
                [
                    "The barrier has been touched, so there is no optionality left to draw: "
                    + (
                        "the knock-in has simply become the vanilla and the two lines sit on "
                        "top of each other."
                        if instrument.is_knock_in
                        else "the knock-out line is flat at zero while the vanilla keeps its "
                        "shape. Its rebate, if any, was paid at the hit."
                    ),
                    "This is the whole life cycle of a barrier in one picture: before the touch "
                    "it is a delicate hedging problem, after it there is nothing left to hedge.",
                    "Switch the path-state toggle off to go back to the live contract.",
                ]
            )
        else:
            near_spot = instrument.barrier * (0.99 if instrument.is_up else 1.01)
            near_greeks = instrument.greeks(mkt.bumped(spot=near_spot))
            near_shown = to_trader_units(dict(near_greeks)) if trader_units else dict(near_greeks)
            ui.takeaways(
                [
                    f"With the spot 1% inside the barrier ({near_spot:.2f}) this contract's "
                    f"delta is {ui.number(float(near_shown['delta']))} and its gamma "
                    f"{ui.greek_value('gamma', float(near_shown['gamma']))}"
                    + (
                        " — a negative delta on a call: a rally now destroys the option, so you "
                        "hedge it by buying stock as the market falls."
                        if float(near_shown["delta"]) < 0
                        else " — the barrier bends the whole profile towards it."
                    ),
                    f"At the barrier the price line is pulled to "
                    f"{ui.money(float(instrument.rebate))} (the rebate) instead of following "
                    "the vanilla: that discontinuity is what the whole discount pays for.",
                    "Far from the barrier the two lines merge. A barrier option only differs "
                    "from the vanilla where the touch is plausible.",
                ]
            )
    elif family == "Asian":
        ui.takeaways(
            [
                "Same shapes as the vanilla, scaled down: averaging does not change the "
                "direction of any Greek, it damps all of them.",
                f"{float(instrument.remaining_weight(mkt)):.0%} of the averaging window is still "
                "random. As that number falls the whole set of curves flattens towards a "
                "straight line, because the payoff is turning into a known number.",
                "This is the opposite of a vanilla near expiry, whose gamma explodes. An Asian "
                "gets easier to hedge as it ages, which is exactly why hedgers like it.",
            ]
        )
    else:
        ui.takeaways(
            [
                "The lookback's price line sits above the vanilla everywhere and never touches "
                "zero: with hindsight the option is always worth something.",
                "There is a kink where the spot meets the running extreme. On one side the "
                "extreme is dragged along by the spot and the price is almost linear; on the "
                "other the extreme is frozen and the option behaves like a normal call.",
                "That kink is where all the gamma lives. Away from the extreme the lookback "
                "drifts towards a forward, which is why its delta flattens out.",
            ]
        )

# --- 4b. Inside the product ------------------------------------------------ #
elif view == VIEWS[1]:
    if family == "Digital":
        st.caption(
            "A desk never hedges the step itself. It books the digital as a tight vanilla "
            "spread and lives with the difference."
        )
        hedge_cols = st.columns(3)
        placement = hedge_cols[0].segmented_control(
            "Spread placement",
            ["centered", "conservative"],
            default="centered",
            required=True,
            key="exotics_digital_placement",
            help="Centered straddles the strike and is the most accurate. Conservative puts "
            "the whole ramp on the out-of-the-money side, so the spread pays at least as much "
            "as the digital everywhere: that is what a seller books.",
        )
        width_pct = hedge_cols[1].slider(
            "Tightest spread, as a share of the strike",
            min_value=1,
            max_value=25,
            value=5,
            format="%d%%",
            key="exotics_digital_width",
            help="The narrower the spread, the closer it prices to the digital — and the more "
            "options you have to trade to build it.",
        )
        pin_request = hedge_cols[2].slider(
            "Days to expiry for the pin-risk check",
            min_value=1,
            max_value=90,
            value=5,
            key="exotics_digital_pin",
            help="Where the trouble is: the deltas below are measured at this point in the "
            "life of the trade, with the spot sitting exactly on the strike.",
        )
        width = max(0.01, instrument.strike * width_pct / 100.0)
        tenor_days = max(1.0, float(instrument.expiry - mkt.t) * 365.0)
        pin_days = min(float(pin_request), max(0.5, tenor_days / 2.0))

        spread_price, spread_delta, digital_delta, units = price, 0.0, 0.0, 0.0
        try:
            spread = instrument.replicating_call_spread(width, placement)
            spread_price = float(spread.price(mkt))
            units = max(abs(float(leg.quantity)) for leg in spread.flatten())
            spread_delta = _delta_at(spread, mkt, instrument.strike, days_left=pin_days)
            digital_delta = _delta_at(instrument, mkt, instrument.strike, days_left=pin_days)
            ui.metric_row(
                [
                    ("Digital", ui.money(price), "The model price of the step payoff."),
                    (
                        f"{placement.capitalize()} spread",
                        ui.money(spread_price),
                        f"Price of the replicating vanilla spread, {width:g} wide.",
                    ),
                    (
                        "Cost of the hedge",
                        ui.money(spread_price - price, decimals=4, signed=True),
                        "What the replication costs on top of the theoretical price. A centered "
                        "spread is accurate to the square of the width; a conservative one is "
                        "deliberately dearer, so the hedge always pays at least what was sold.",
                    ),
                    (
                        "Options in the hedge",
                        ui.number(units, decimals=1),
                        (
                            "Vanilla options per unit of digital: payout / width."
                            if instrument.is_cash
                            else "Vanilla options per unit of digital: payout x strike / width "
                            "for the spread, plus one option at the strike (an asset digital is "
                            "a vanilla plus strike cash digitals)."
                        )
                        + " Halve the width and this doubles: that is the cost of a tighter "
                        "replication.",
                    ),
                ]
            )
            st.caption(
                f"Pin risk, measured {pin_days:g} days before expiry with the spot exactly on "
                f"the strike ({instrument.strike:g}):"
            )
            ui.metric_row(
                [
                    (
                        "Delta of the digital",
                        ui.number(digital_delta, decimals=2),
                        "Shares to hold per unit of digital. It grows without limit as expiry "
                        "approaches: there is no finite hedge for a step.",
                    ),
                    (
                        "Delta of the spread",
                        ui.number(spread_delta, decimals=2),
                        "The same number for the hedge a desk actually trades. It cannot exceed "
                        "payout / width, whatever the market does.",
                    ),
                ]
            )
            figure = exotic_plots.digital_replication_figure(
                instrument,
                mkt,
                widths=(0.20 * instrument.strike, 0.10 * instrument.strike, width),
                placement=placement,
                trader_units=trader_units,
                mode=ui.CHART_MODE,
            )
            ui.chart(figure, key="exotics_replication")
        except ValueError as exc:
            st.warning(f"That spread cannot be built: {exc}", icon=":material/warning:")

        ui.explain(
            "How to read this",
            """
Three panels: the payoff at expiry, the price today and the delta today. The bold dark
line is the digital; the light-to-dark lines are vanilla spreads of decreasing width,
each rescaled to pay the same amount (`payout / width` of them).

The spread is the finite-difference version of the digital: a digital is exactly
`-dC/dK`, the derivative of the call price in the strike, and a spread is that
derivative approximated over a finite gap. Tightening the gap makes the payoff ramp
steeper, so it converges to the step — and multiplies the number of options you hold.
""",
        )
        ui.takeaways(
            [
                f"On price the replication is easy: the {width:g}-wide {placement} spread costs "
                f"{ui.money(spread_price)} against the digital's {ui.money(price)}, a difference "
                f"of {ui.money(spread_price - price, decimals=4, signed=True)}.",
                f"On risk it is not. {pin_days:g} days before expiry, at the strike, the digital "
                f"asks for {ui.number(digital_delta, decimals=1)} of stock and the spread for "
                f"{ui.number(spread_delta, decimals=1)} — but the spread's delta is bounded by "
                f"{ui.number(units, decimals=1)} whatever happens, and the digital's is not.",
                "The gap between the two payoff lines is the gap risk. A seller who books the "
                "conservative spread is over-hedged everywhere: never short the step, at the "
                "price of a slightly worse quote.",
            ]
        )
        ui.interview_note(
            f"You do not hedge the step, you replicate it with a vanilla spread: buy options "
            f"just below the strike and sell as many just above, enough of them that the ramp "
            f"pays what the step pays. Here that is {ui.number(units, decimals=1)} options per "
            f"digital for a {width:g}-wide {placement} spread, which prices "
            f"{ui.money(spread_price)} against a theoretical {ui.money(price)}. "
            f"Then say the two things that matter. First, you quote the "
            "conservative spread, so the hedge pays at least what you sold and you are never "
            "short the step — the extra premium is the real price of the gap risk. Second, the "
            f"width is a risk decision, not a pricing one: {pin_days:g} days out the digital's "
            f"delta at the strike is already {ui.number(digital_delta, decimals=1)} and rising, "
            "so a tighter spread replicates better and leaves you a bigger position to manage "
            "into the fixing.",
            question="How would you hedge a digital option?",
        )

    elif family == "Barrier":
        st.caption(
            "A barrier option is a bet on the path, not on where the spot ends. Two paths "
            "finishing at the same level can pay completely different amounts."
        )
        path_cols = st.columns(2)
        drawn = path_cols[0].slider(
            "Sample paths",
            min_value=20,
            max_value=200,
            value=60,
            step=10,
            key="exotics_paths_n",
            help="Risk-neutral simulations from today to expiry, watched at 252 dates.",
        )
        path_seed = path_cols[1].number_input(
            "Seed",
            min_value=0,
            max_value=9_999,
            value=7,
            step=1,
            key="exotics_paths_seed",
            help="Change it to draw a different sample. Everything else stays the same.",
        )
        try:
            figure = exotic_plots.barrier_paths_figure(
                instrument,
                mkt,
                n_paths=int(drawn),
                seed=int(path_seed),
                n_steps=252,
                mode=ui.CHART_MODE,
            )
            ui.chart(figure, key="exotics_paths")
        except ValueError as exc:
            st.warning(f"These paths cannot be drawn: {exc}", icon=":material/warning:")

        ui.explain(
            "How to read this",
            """
Every thin line is one simulated future for the spot, from today to expiry. Blue paths
are the ones on which the option is alive at expiry, grey the ones on which it is dead,
and the orange dots mark the first touch of the barrier. The dashed line is the barrier,
the dotted one the strike; the subtitle reports the share of paths that touched.

For a knock-**out** the touching paths are the dead ones. For a knock-**in** they are the
only ones that pay. Either way the share of touching paths is what the discount to the
vanilla (knock-out) or the price itself (knock-in) is made of — and it is a share under
the *risk-neutral* measure, at the volatility you typed in the sidebar. Raise that
volatility and the barrier gets touched far more often.
""",
        )
        ui.takeaways(
            [
                f"The option is {_ratio_sentence(share)} "
                + (
                    "because only the blue paths ever activate it; the grey ones expire having "
                    "never touched the barrier and pay nothing but the rebate."
                    if instrument.is_knock_in
                    else "precisely because of the grey paths: they are the value you sold."
                ),
                "Look at where the dots cluster in time. A touch early in the life is much more "
                "damaging than one near expiry for a knock-out, because there is still a lot of "
                "optionality left to destroy.",
                "The barrier here is watched at 252 dates, not continuously. A contract watched "
                "on daily closes is touched less often than the closed form assumes — the "
                "'Model check' view puts a number on that gap.",
            ]
        )
        ui.interview_note(
            "Say the structure first: knock-in plus knock-out with the same strike, barrier and "
            f"expiry is the vanilla, because exactly one of the two is alive at expiry. So the "
            f"{instrument.barrier_type} at {ui.money(price)} and its mirror image add up to the "
            f"vanilla's {ui.money(vanilla_price)}. Then the practical consequence: you can "
            "always price one from the other, and a desk that is long one and short the other "
            "with the same barrier is just long a vanilla — no barrier risk at all. The risk "
            "only appears when the two legs have different barriers or one carries a rebate.",
            question="What is in-out parity, and how would you use it on a trading floor?",
        )

    elif family == "Asian":
        st.caption(
            "One simulated future, with the running average the option actually settles on."
        )
        average_seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=9_999,
            value=5,
            step=1,
            key="exotics_asian_seed",
            help="Change it to draw a different future. The contract does not change.",
        )
        ui.metric_row(
            [
                (
                    "Still random",
                    ui.percent(float(instrument.remaining_weight(mkt)), decimals=0),
                    "Share of the averaging window that has not been fixed yet. It is the "
                    "number that scales the Greeks down as the contract ages.",
                ),
                (
                    "Volatility of the average",
                    ui.percent(float(instrument.effective_vol(mkt)), decimals=1),
                    "Black volatility of the still-random part of the average, annualised to "
                    "expiry.",
                ),
                (
                    "Market volatility",
                    ui.percent(float(mkt.vol), decimals=1),
                    "What the sidebar says the spot itself does. The average always moves less.",
                ),
                (
                    "Ratio",
                    ui.number(
                        float(instrument.effective_vol(mkt)) / float(mkt.vol)
                        if float(mkt.vol) > 0
                        else float("nan"),
                        decimals=2,
                    ),
                    "1 / sqrt(3) = 0.58 for an average running from today to expiry. Seasoning "
                    "does not change this ratio — it changes how much of the payoff is still "
                    "exposed to it, which is the 'still random' number.",
                ),
            ]
        )
        try:
            figure = exotic_plots.asian_averaging_figure(
                instrument,
                mkt,
                seed=int(average_seed),
                n_steps=252,
                mode=ui.CHART_MODE,
            )
            ui.chart(figure, key="exotics_averaging")
        except ValueError as exc:
            st.warning(f"This path cannot be drawn: {exc}", icon=":material/warning:")

        ui.explain(
            "How to read this",
            """
The thin blue line is one simulated path of the spot; the thick orange line is the running
average built observation by observation, exactly as a book would record it. The
horizontal line is the strike, and the subtitle compares what the Asian and the vanilla
would pay on this particular path.

Watch two things. First, the average always lags: it cannot jump, because every new
observation is one of many. Second, it *calms down* as the window fills in — by the last
few weeks a large move in the spot barely moves the average at all. Draw a few seeds and
you will see the spot finish far from the average more often than not.
""",
        )
        ui.takeaways(
            [
                "The average is a smoothed, delayed version of the spot. That is the entire "
                "product: less volatility in the settlement, so a lower premium.",
                f"Its volatility here is "
                f"{ui.percent(float(instrument.effective_vol(mkt)), decimals=1)} against the "
                f"market's {ui.percent(float(mkt.vol), decimals=1)}, and the option costs "
                f"{_ratio_sentence(share)}.",
                "The flattening at the right-hand side is the Greeks fading. Near expiry an "
                "Asian is almost a fixed cash flow, while a vanilla is at its most dangerous.",
            ]
        )
        ui.interview_note(
            "Start with the exposure: an airline buys fuel every day, so its real cost is an "
            "average price, not a single close. A strip of vanillas would over-hedge the "
            "settlement risk and cost more; the Asian matches the exposure exactly and prices "
            f"{_ratio_sentence(share)} because the average carries only "
            f"{ui.percent(float(instrument.effective_vol(mkt)), decimals=1)} of volatility. "
            "Add the market-integrity point if you want to sound like you have sat on a desk: "
            "an average over many fixings is far harder to manipulate than a single expiry "
            "print, which matters a lot in illiquid commodities.",
            question="A client hedges their monthly fuel bill. Why would you sell them an "
            "Asian rather than a strip of vanillas?",
        )

    else:  # Lookback
        st.caption(
            "A lookback is priced off its running extreme. Move that extreme and you move the "
            "whole risk profile — this is what seasoning does to the contract."
        )
        uses_min = instrument.is_floating == instrument.is_call
        field = "running_min" if uses_min else "running_max"
        current_level = float(getattr(instrument, field) or spot)
        further = current_level * (0.9 if uses_min else 1.1)
        try:
            aged = dataclasses.replace(instrument, **{field: round(further, 4)})
            series = {
                f"Today's contract (extreme {current_level:.2f})": instrument,
                f"If the extreme were {further:.2f}": aged,
                f"Vanilla benchmark ({vanilla.label})": vanilla,
            }
            figure = profiles.compare_instruments(
                series,
                mkt,
                greek=("price", "delta", "gamma"),
                trader_units=trader_units,
                title="Lookback: the running extreme is the contract",
                mode=ui.CHART_MODE,
            )
            ui.chart(figure, key="exotics_lookback_extremes")
            aged_price = float(aged.price(mkt))
        except ValueError as exc:
            st.warning(f"That comparison cannot be drawn: {exc}", icon=":material/warning:")
            aged_price = price

        ui.metric_row(
            [
                (
                    "Today's contract",
                    ui.money(price),
                    f"Running extreme at {current_level:.2f}.",
                ),
                (
                    f"Extreme at {further:.2f}",
                    ui.money(aged_price),
                    "The same contract after the market has printed a new extreme 10% away. "
                    "Nothing else changed.",
                ),
                (
                    "Value of that print",
                    ui.money(aged_price - price, signed=True),
                    "A lookback banks every new extreme and can never give it back.",
                ),
                (
                    "Vanilla benchmark",
                    ui.money(vanilla_price),
                    f"{vanilla.label}, for scale.",
                ),
            ]
        )
        ui.explain(
            "How to read this",
            """
Three instruments, three panels: price, delta and gamma against today's spot. The two
lookbacks differ only in the extreme already recorded, and the vanilla is there for scale.

The interesting point on each curve is where the spot meets the running extreme. To one
side, every new spot print *sets* a new extreme, so the payoff moves one-for-one with the
spot and the price line is nearly straight; to the other side the extreme is locked, and
the contract behaves like an ordinary option struck there. The kink between the two
regimes is where the gamma concentrates — and it is also why the delta of a fresh
floating lookback is close to one.
""",
        )
        ui.takeaways(
            [
                f"Moving the recorded extreme from {current_level:.2f} to {further:.2f} is "
                f"worth {ui.money(aged_price - price, signed=True)} without the spot moving at "
                "all: the past is part of this contract's value.",
                "The gamma panel is concentrated around the extreme, not around a strike. That "
                "is the practical difference for a hedger: the danger point moves with the "
                "market instead of staying where it was written.",
                f"Against the vanilla's {ui.money(vanilla_price)}, hindsight costs "
                f"{ui.money(price - vanilla_price)} here — which is the honest answer to "
                "'why would anyone sell this?'.",
            ]
        )
        ui.interview_note(
            "No — and being clear about why is the point. A static vanilla portfolio pays on "
            "the terminal spot; a lookback pays on the extreme of the path, so two paths ending "
            "at the same level can owe very different amounts. You can only hedge it "
            "dynamically, and the hedge has to be rolled every time the market prints a new "
            "extreme, which is exactly when it is most expensive to trade. That path-dependence "
            f"is what the {ui.money(price - vanilla_price)} premium over the vanilla is really "
            "paying for, and it is why lookbacks are rare outside structured notes.",
            question="Could you replicate a lookback with a portfolio of vanillas?",
        )

# --- 4c. Model check ------------------------------------------------------- #
else:
    st.caption(
        "Closed forms are fast but they assume things. A simulation prices the contract from "
        "its definition, so it is the referee — at the cost of statistical noise."
    )

    path_dependent = family != "Digital"
    estimator = "plain"
    with st.form("exotics_mc_form", border=True):
        mc_cols = st.columns(3)
        n_paths = mc_cols[0].select_slider(
            "Paths",
            options=[2_000, 5_000, 10_000, 20_000, 50_000],
            value=20_000,
            format_func=lambda value: f"{value // 1000}k",
            key="exotics_mc_paths",
            help="The standard error falls like 1 / sqrt(paths): four times the paths for half "
            "the noise, which is why brute force runs out of road quickly (50k daily-monitored "
            "paths already hold half a gigabyte of spot in memory). Antithetic variates are "
            "used throughout, and the control variates below buy far more precision per second.",
        )
        monitoring = mc_cols[1].segmented_control(
            "Monitoring",
            list(MONITORING),
            default="Daily",
            required=True,
            disabled=not path_dependent,
            key="exotics_mc_monitoring",
            help="How often the simulation looks at the spot. It matters a great deal for "
            "barriers, Asians and lookbacks — and not at all for a digital, which only reads "
            "the last print, so a single step is used there and the control is switched off.",
        )
        mc_seed = mc_cols[2].number_input(
            "Seed",
            min_value=0,
            max_value=9_999,
            value=11,
            step=1,
            key="exotics_mc_seed",
            help="Re-run with a different seed to see the estimate move inside its error bar.",
        )
        if family == "Asian":
            if st.checkbox(
                "Use the geometric control variate",
                value=True,
                key="exotics_mc_control",
                help="The geometric average of a path has an exact price and is almost "
                "perfectly correlated with the arithmetic one, so simulating the difference "
                "instead of the level cuts the standard error by a factor of 20 to 50.",
            ):
                estimator = "control"
        elif family == "Lookback":
            if st.checkbox(
                "Sample the extremes between dates (Brownian bridge)",
                value=True,
                key="exotics_mc_bridge",
                help="A simulation that only looks at grid dates misses the excursions in "
                "between and systematically under-prices a lookback. The bridge draws the "
                "extreme of each step from its exact law, which removes that bias.",
            ):
                estimator = "bridge"
        st.form_submit_button(
            "Run the simulation",
            icon=":material/play_arrow:",
            type="primary",
            key="exotics_mc_run",
        )

    tau = max(float(instrument.expiry - mkt.t), 1.0 / 365.0)
    n_steps = max(1, int(round(tau * MONITORING[monitoring or "Daily"]))) if path_dependent else 1

    with st.spinner("Simulating the paths..."):
        result = _monte_carlo(
            name,
            _items(params),
            _items(extra),
            ui.market_key(mkt),
            int(n_paths),
            int(n_steps),
            int(mc_seed),
            estimator,
        )
    mc_value, stderr = result["price"], result["stderr"]

    comparator, comparator_label, monitoring_note = price, "Closed form", ""
    if family == "Barrier" and not instrument.knocked and not _breached(instrument, spot):
        try:
            adjusted = instrument.discrete_barrier_adjusted(float(mkt.vol), dt=tau / n_steps)
            comparator = float(adjusted.price(mkt))
            comparator_label = "Closed form, monitoring-adjusted"
            monitoring_note = (
                f"The contract's own closed form assumes the barrier is watched *continuously* "
                f"and prices {ui.money(price)}. The simulation only looks at {n_steps} dates, so "
                f"it misses excursions between them and touches the barrier less often. The "
                f"Broadie-Glasserman-Kou correction moves the barrier from "
                f"{instrument.barrier:g} to {adjusted.barrier:.2f} to represent that monitoring, "
                f"and the corrected price is {ui.money(comparator)} — that is the number the "
                "simulation should reproduce."
            )
        except ValueError:
            comparator, comparator_label = price, "Closed form"
    elif family == "Lookback" and estimator != "bridge":
        monitoring_note = (
            f"This simulation only reads the extreme at {n_steps} dates, so it misses the "
            "excursions in between and will sit *below* the continuous closed form. That gap is "
            "a discretisation bias, not noise: more paths will not close it. Switch the "
            "Brownian bridge on to remove it."
        )

    gap = mc_value - comparator
    sigmas = abs(gap) / stderr if stderr > 0 else float("inf")

    ui.metric_row(
        [
            (
                comparator_label,
                ui.money(comparator),
                "The analytic price the simulation is being checked against.",
            ),
            (
                "Monte Carlo",
                ui.money(mc_value),
                f"{int(n_paths):,} antithetic paths, "
                + (
                    f"{n_steps} monitoring dates"
                    if path_dependent
                    else "terminal spot only (the payoff reads nothing else)"
                )
                + f", seed {int(mc_seed)}.",
            ),
            (
                "Standard error",
                ui.money(stderr, decimals=4),
                "One standard deviation of the estimate itself. The true value is inside "
                "roughly plus or minus two of these with 95% confidence.",
            ),
            (
                "Gap",
                f"{ui.money(gap, decimals=4, signed=True)}"
                + (f" ({sigmas:.1f} s.e.)" if math.isfinite(sigmas) else ""),
                "Monte Carlo minus closed form, and how many standard errors that is. Under "
                "three is agreement; a stable gap of ten is a modelling difference.",
            ),
        ]
    )

    if stderr <= 0:
        st.info(
            "There is no sampling noise to report: every antithetic pair produced exactly the "
            "same discounted payoff. That happens when the contract is already dead, or when "
            "the mirrored draws cancel each other exactly — a digital struck at the forward is "
            "the classic case, since one path of every pair finishes in the money and the other "
            f"does not. The gap to the closed form is "
            f"{ui.money(gap, decimals=4, signed=True)}.",
            icon=":material/info:",
        )
    elif sigmas <= 3:
        st.success(
            f"Agreement: the simulation lands {sigmas:.1f} standard errors from the closed form, "
            f"i.e. {ui.money(mc_value)} ± {ui.money(stderr, decimals=4)} against "
            f"{ui.money(comparator)}. Two independent implementations of the same contract "
            "produce the same number.",
            icon=":material/check_circle:",
        )
    else:
        st.warning(
            f"The simulation sits {sigmas:.1f} standard errors from {comparator_label.lower()} "
            f"({ui.money(mc_value)} against {ui.money(comparator)}). With this many paths that is "
            "too big to be luck: it is a difference between what the two methods are pricing — "
            "usually the monitoring frequency, or an approximation inside the closed form.",
            icon=":material/warning:",
        )

    if monitoring_note:
        st.markdown(monitoring_note)

    if family == "Asian":
        relative = abs(gap) / comparator if comparator > 0 else float("nan")
        asian_note = (
            "The closed form for an arithmetic Asian is itself an approximation: the average of "
            "lognormals is not lognormal, so the library matches its first two moments instead. "
            "The simulation makes no such assumption, which is exactly why it is worth running."
        )
        if estimator == "control":
            asian_note += (
                f" The control variate makes the estimate so precise that the approximation "
                f"itself becomes visible: the gap of {ui.money(abs(gap), decimals=4)} is "
                f"{relative:.2%} of the price, and it will not shrink as you add paths — it is "
                "a modelling difference, not noise. Switch the control variate off and the same "
                "gap disappears inside a standard error twenty times larger."
            )
        st.markdown(asian_note)

    ui.explain(
        "What the standard error actually is",
        """
Monte Carlo prices an option by averaging the discounted payoff over simulated futures.
Being an average of a finite sample, it is itself random: run it again with another seed
and it moves. The **standard error** is the standard deviation of that average —
`sample standard deviation / sqrt(number of samples)`. So:

- the estimate is worth quoting as `price ± standard error`, not as a single number;
- it shrinks like `1 / sqrt(paths)`: **four times** the paths for **half** the noise, which
  is why simulation is expensive when you need precision;
- antithetic variates (every random draw is used with its mirror image) and control
  variates (simulate the difference to something you can price exactly) buy precision far
  more cheaply than brute force. On an Asian, compare the standard error at 2k paths with
  the control variate against 50k paths without it.

A gap of one or two standard errors means the two methods agree. A gap that stays the
same size as you add paths is a **bias**, not noise — the usual culprit being that the
simulation watches the path at a finite number of dates while the closed form assumes it
is watched continuously.
""",
    )
    ui.interview_note(
        f"Here the closed form gives {ui.money(comparator)} and {int(n_paths):,} simulated paths "
        f"give {ui.money(mc_value)} ± {ui.money(stderr, decimals=4)} — "
        f"{'agreement' if math.isfinite(sigmas) and sigmas <= 3 else 'a gap worth explaining'}. "
        "The answer an interviewer wants has three parts: closed forms are instant and exact "
        "*under their assumptions*, simulation is slow and noisy but prices the contract from "
        "its definition, and the noise is quantified — quote the standard error and you have "
        "said how much you trust the number. Then the punchline: if the two disagree by far "
        "more than the standard error, do not add paths, look for the assumption that differs.",
        question="You have a closed form and a Monte Carlo for the same exotic. Which do you "
        "trust, and how would you know if one of them is wrong?",
    )

# --------------------------------------------------------------------------- #
# 5. Trade it
# --------------------------------------------------------------------------- #
ui.section("5. Put it in your book", "Add the contract to the book you carry across the app.")

with st.container(border=True):
    ticket_cols = st.columns([1, 1, 2], vertical_alignment="bottom")
    side = ticket_cols[0].segmented_control(
        "Side",
        ["Buy", "Sell"],
        default="Buy",
        required=True,
        key="exotics_side",
        help="Selling an exotic is the more realistic exercise: it is what a desk does, and "
        "the Greeks you have to hedge flip sign.",
    )
    quantity = ticket_cols[1].number_input(
        "Quantity",
        min_value=1.0,
        value=10.0,
        step=1.0,
        key="exotics_quantity",
        help="Number of contracts. The book stores the trade at today's model price, so it "
        "starts flat and then lives with the market.",
    )
    signed = float(quantity) * (1.0 if (side or "Buy") == "Buy" else -1.0)
    ticket_cols[2].markdown(
        f"**Ticket:** {'buy' if signed > 0 else 'sell'} {abs(signed):g} x {instrument.label} "
        f"at {ui.money(price)} — cash {ui.money(-signed * price, signed=True)}."
    )

    if st.button("Add to my book", icon=":material/add:", type="primary", key="exotics_add"):
        try:
            state.get_book().trade(instrument, signed, mkt)
            st.success(
                f"Done: {'bought' if signed > 0 else 'sold'} {abs(signed):g} x "
                f"{instrument.label} at {ui.money(price)}.",
                icon=":material/check_circle:",
            )
        except ValueError as exc:
            st.error(f"That trade was refused: {exc}", icon=":material/error:")

    st.page_link(
        "app_pages/book.py",
        label="Open my book to see the risk",
        icon=":material/inventory_2:",
    )
    st.caption(
        "Path-dependent products start their life at the trade: the book stores the contract "
        "as it is today, so a lookback begins watching and an Asian begins averaging from the "
        "spot you traded at."
    )

position_delta = signed * float(shown_exotic["delta"])
position_gamma = signed * float(shown_exotic["gamma"])
position_vega = signed * float(shown_exotic["vega"])
ui.interview_note(
    f"Be concrete, and start with the delta. This ticket carries "
    f"{ui.number(position_delta, decimals=2)} of delta, so the first trade is to "
    f"{'sell' if position_delta > 0 else 'buy'} {abs(position_delta):,.2f} shares and go home "
    f"flat in the spot. Then say what that does not cover: the position is "
    f"{ui.greek_value('gamma', position_gamma)} of gamma and "
    f"{ui.greek_value('vega', position_vega)} of vega, so it bleeds (or earns) every time you "
    "re-hedge and it moves with implied volatility, neither of which a stock hedge touches. "
    "Finish with the honest sentence, which is what separates a good answer from a textbook "
    "one: that delta is a model's delta, and for this product it is the number that moves "
    "fastest exactly when the market is hardest to trade — at the barrier, at the strike on "
    "expiry day, or on the fixing.",
    question="You have just sold this to a client. What is your first hedge, and what does it "
    "not protect you from?",
)
