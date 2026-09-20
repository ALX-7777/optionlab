"""Trading simulator: run the book through a market you do not control.

The page is in three acts. **Set up** a scenario -- how long, how volatile, which
implied-vol dynamics, which hedging rule. **Walk through it** one step at a time and
watch the value, the risk and the hedge react. **Read the P&L explain**: the same
Taylor expansion on the start-of-day Greeks that a desk publishes every morning,
which says how much of the day came from gamma, how much theta cost, and what the
Greeks failed to account for.

The last section is a laboratory of its own: one option, thousands of simulated
paths, several rebalancing frequencies. That is where the two classical results
live -- the hedging error shrinks like ``1 / sqrt(N)``, and the average P&L is the
realised-versus-implied variance term, not a reward for trading more.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from optionlab import EuropeanOption, Market, TransactionCosts
from optionlab.plotting import sim_plots
from optionlab.simulator import (
    GREEK_TERMS,
    DeltaBandHedge,
    DeltaHedgeAtVol,
    DeltaHedgeEveryN,
    NoHedge,
    TradingSimulator,
    attribution_totals,
    delta_hedging_experiment,
    gamma_scalping_summary,
    gbm_scenario,
    hedging_experiment_summary,
)
from optionlab.strategies import STRATEGY_REGISTRY

from app_lib import state, ui

DAYS_PER_YEAR = 365.0

#: One-click positions offered when the book is empty: registry key, size, tenor.
STARTERS: dict[str, dict] = {
    "Long 10 straddles": {
        "key": "long_straddle",
        "quantity": 10.0,
        "days": 60,
        "watch": "Long gamma, long vega, and a theta bill every step. Watch whether the "
        "moves pay the rent.",
    },
    "Short 10 iron condors": {
        "key": "iron_condor",
        "quantity": 10.0,
        "days": 45,
        "watch": "The mirror image: you collect theta and are short gamma and vega, so a "
        "quiet market pays you and a gap hurts.",
    },
    "Long 20 calls": {
        "key": "long_call",
        "quantity": 20.0,
        "days": 90,
        "watch": "A directional position. Without a hedge the delta term dominates "
        "everything else; switch the hedge on and gamma against theta is all that is left.",
    },
}

#: Hedging rules offered on the page, with the one line that matters for each.
POLICIES: dict[str, str] = {
    "No hedge": "The book keeps whatever delta it has, so the P&L is dominated by the "
    "direction of the spot. Run it once to see how large the delta term is next to gamma "
    "and theta.",
    "Delta hedge every N steps": "Bring the delta back to zero on a calendar. Hedging more "
    "often does not change the expected P&L, it shrinks its dispersion -- and it pays more fees.",
    "Delta band": "Trade only when the delta drifts outside a band around zero. A band is the "
    "practical answer to transaction costs: the wider it is, the fewer the trades and the more "
    "residual delta risk you carry.",
    "Delta hedge at your own vol": "Compute the hedge delta at the volatility you believe in "
    "rather than the one the market quotes. Hedging at the realised vol locks in a known total "
    "profit with a noisy path; hedging at the implied vol gives a smooth daily P&L whose total "
    "depends on where the spot spends its time.",
}

VOL_MODELS: dict[str, str] = {
    "constant": "Constant implied vol",
    "mean_reverting": "Mean-reverting implied vol",
    "spot_correlated": "Implied vol correlated with the spot",
}

VOL_MODEL_NOTES: dict[str, str] = {
    "constant": "The options stay marked at the sidebar vol for the whole run, so the vega term "
    "is exactly zero and only gamma against theta is left. The cleanest place to start.",
    "mean_reverting": "The log of the implied vol is pulled back towards its starting level, "
    "independently of the spot. Your vega P&L becomes noise around zero.",
    "spot_correlated": "The equity leverage effect: implied vol jumps when the market falls and "
    "leaks when it rallies. A long-vega book now has a hidden long-put profile.",
}

#: One note per legend entry of the attribution charts, in the same order and with
#: the same names, so the box below a chart reads as its key.
TERM_NOTES: dict[str, str] = {
    "Delta": "`Delta x dS` -- the directional bet. A delta-hedged book should show almost nothing "
    "here; a large number means you were carrying a view, on purpose or not.",
    "Gamma": "`0.5 x Gamma x dS^2` -- always positive for a long-option book, whichever way the "
    "spot went. This is what you bought when you paid the premium.",
    "Theta": "`Theta x dt` -- the rent on that gamma, paid every step whether the spot moves or not.",
    "Vega": "`Vega x dvol` -- the re-marking of the options when the implied vol level moves. "
    "Nothing to do with how much the spot actually moved.",
    "Vanna + volga": "`Vanna x dS x dvol` and `0.5 x Volga x dvol^2` -- the cross terms. They "
    "matter when spot and vol move together, which is exactly what happens in a sell-off.",
    "Rates and carry": "`Rho x dr` plus the carry: interest on the cash balance and dividends on "
    "the shares held. Financing the delta hedge lives here.",
    "Fees and trading edge": "Transaction costs paid, plus the edge on any trade done away from "
    "the model price. Trading at the model price never creates P&L, so a hedge shows up here "
    "only through what it costs.",
    "Unexplained": "Everything the start-of-step Greeks could not account for: a big gap, a long "
    "step, an expiry settling. A flat unexplained line means the Greeks told the whole story.",
}

st.session_state.setdefault("simulator_experiment_params", None)
st.session_state.setdefault("simulator_starter_error", "")


# --------------------------------------------------------------------------- #
# Cached work
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=24)
def run_hedging_experiment(
    option_type: str,
    strike: float,
    expiry_years: float,
    quantity: float,
    spot: float,
    implied_vol: float,
    rate: float,
    div: float,
    realized_vol: float,
    hedge_vol: float,
    n_paths: int,
    frequencies: tuple[int, ...],
    seed: int,
    stock_bps: float,
) -> tuple[pd.DataFrame, dict]:
    """Buy or sell one option, delta hedge it over many paths, at several frequencies.

    Everything is rebuilt from primitives inside so the cache key is stable. The
    frame's ``attrs`` are returned separately because a cached copy may drop them.
    """
    mkt0 = Market(spot=spot, vol=implied_vol, rate=rate, div=div)
    option = EuropeanOption(option_type, strike, expiry_years)
    costs = TransactionCosts(stock_bps=stock_bps) if stock_bps > 0 else None
    frame = delta_hedging_experiment(
        option,
        mkt0,
        n_paths=int(n_paths),
        rebalance_steps_list=frequencies,
        realized_vol=realized_vol,
        seed=int(seed),
        quantity=quantity,
        hedge_vol=hedge_vol,
        costs=costs,
    )
    return frame, dict(frame.attrs)


# --------------------------------------------------------------------------- #
# Callbacks: the only place the simulator is mutated
# --------------------------------------------------------------------------- #
def add_starter_position() -> None:
    """Open the chosen starter position in the shared book."""
    st.session_state["simulator_starter_error"] = ""
    choice = st.session_state.get("simulator_starter") or next(iter(STARTERS))
    recipe = STARTERS.get(choice)
    if recipe is None:
        return
    market = state.current_market()
    spec = STRATEGY_REGISTRY[recipe["key"]]
    try:
        strategy = spec.build_default(
            spot=float(market.spot), expiry=recipe["days"] / DAYS_PER_YEAR, t=float(market.t)
        )
        book = state.get_book()
        book.trade(strategy, recipe["quantity"], market, note="simulator starter")
    except ValueError as exc:
        st.session_state["simulator_starter_error"] = str(exc)
        return
    state.set_book(book)


def advance(n: int | None) -> None:
    """Run ``n`` steps of the simulation (all the remaining ones when ``None``)."""
    sim = state.get_sim()
    if sim is None or sim.is_finished:
        return
    sim.run(n)


def reset_simulation() -> None:
    """Rewind to step 0 with the book as it was when the simulation started."""
    sim = state.get_sim()
    if sim is not None:
        sim.reset()


def hedge_now() -> None:
    """Bring the book delta back to zero by hand, at the current market."""
    sim = state.get_sim()
    if sim is None or sim.is_finished:
        return
    sim.book.hedge_delta(sim.current_market, note="manual delta hedge")


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def signed(value: float, decimals: int = 2) -> str:
    """A signed number for a metric delta, e.g. ``+12.40``."""
    return ui.money(float(value), decimals=decimals, signed=True)


def share(part: float, whole: float) -> str:
    """``part`` as a percentage of ``whole``, or ``n/a`` when there is no scale."""
    return f"{100.0 * part / whole:.0f}%" if whole > 1e-12 else "n/a"


def earned(value: float, gain: str = "made", loss: str = "lost") -> str:
    """"made 19.26" or "paid 10.60": a P&L term written the way a trader says it."""
    return f"{gain if value >= 0 else loss} {ui.money(abs(value))}"


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #
ui.page_header(
    "Trading simulator",
    "Take your book into a market you do not control: hedge it step by step, then read "
    "the P&L explain line by line.",
)

mkt = state.current_market()
spot = state.spot()
trader_units = state.trader_units()
sim = state.get_sim()

ui.how_to_use(
    """
The page is in three acts, and it runs on the book you built on the My book page.

1. **Set up** — the position you will carry, the hedging rule the simulator applies for
   you, and the market you will face: how long, how volatile, how the implied volatility
   moves, how much a hedge costs. Nothing is computed until you press start. If your book
   is empty, take one of the starter positions.
2. **Walk through it** — *Next day*, *Next 5 days* or *Run to the end*. After each step,
   read where you stand, then open one of the four analysis views: where the P&L came
   from, gamma against theta, the full dashboard, or the raw log.
3. **The hedging laboratory** at the bottom is independent of all that: one option,
   thousands of paths, several rebalancing frequencies. It answers *does hedging more
   often make me more money?* — and it always uses the sidebar market, so you can run it
   before you have a book at all.

The market path is drawn when you press start and never changes, so the same scenario can
be replayed with a different hedging rule and the comparison is honest.
"""
)

ui.explain(
    "How one step of the simulation works",
    "The market path is drawn **before** the run and never changes, so the same scenario can "
    "be replayed with and without a hedge and the comparison is honest. Each step then does "
    "the same five things:\n\n"
    "1. snapshot the Greeks and the P&L at the start of the step;\n"
    "2. move the market to the next date (a new spot **and** a new implied vol);\n"
    "3. accrue the carry, show the new spot to path-dependent products, cash-settle whatever "
    "expired;\n"
    "4. measure the actual P&L and attribute it to those start-of-step Greeks;\n"
    "5. apply the hedging rule at the new market and log the row.\n\n"
    "Two volatilities live in a scenario and they do different jobs. The **realised** vol is "
    "how much the spot actually moves, so it drives the gamma P&L against the theta bill. The "
    "**implied** vol is the level the options are marked at, so its moves drive the vega P&L. "
    "Keeping them apart is the whole point of the exercise.",
    icon=":material/conveyor_belt:",
)

# --------------------------------------------------------------------------- #
# Act 1 -- set up a simulation
# --------------------------------------------------------------------------- #
if sim is None:
    book = state.get_book()
    positions = book.positions
    live = [p for p in positions if not p.is_expired(mkt)]
    expiries = [p.instrument.expiry for p in live if p.instrument.expiry is not None]

    ui.section(
        "1. The position you will carry",
        "The simulation starts from your book exactly as it stands on the My book page.",
    )

    if not positions:
        ui.empty_state(
            "Your book is empty, so there is nothing to simulate yet. Open a position on the "
            "Strategies page, or take one of these starters.",
            icon=":material/inventory_2:",
        )
        choice = st.segmented_control(
            "Starter position",
            list(STARTERS),
            default=next(iter(STARTERS)),
            key="simulator_starter",
            help="A ready-made position to practise on. It is opened in your real book, so you "
            "will also find it on the My book page.",
        )
        recipe = STARTERS.get(choice or next(iter(STARTERS)), STARTERS[next(iter(STARTERS))])
        spec = STRATEGY_REGISTRY[recipe["key"]]
        with st.container(border=True):
            tags = " ".join(f":blue-badge[{tag}]" for tag in spec.view)
            st.markdown(
                f"**{spec.title}** — {recipe['quantity']:g} lots, {recipe['days']} days to "
                f"expiry, struck around the current spot. {tags}"
            )
            st.caption(spec.description)
            st.markdown(f":material/visibility: {recipe['watch']}")
            st.button(
                "Open this position in my book",
                icon=":material/add_shopping_cart:",
                type="primary",
                on_click=add_starter_position,
                key="simulator_add_starter",
            )
        if st.session_state.get("simulator_starter_error"):
            st.error(
                f"That position could not be opened: {st.session_state['simulator_starter_error']}",
                icon=":material/error:",
            )
    elif not live:
        st.warning(
            "Every line in your book has already expired at the current market time, so there "
            "is nothing left to simulate. Open a new position on the Strategies page.",
            icon=":material/event_busy:",
        )
    else:
        frame = book.positions_frame(mkt, trader_units=trader_units)
        greeks = book.greeks(mkt)
        ui.greek_metrics(greeks, trader_units=trader_units)
        ui.units_caption(trader_units)
        ui.table(
            frame[["label", "quantity", "expiry", "unit_price", "value", "delta", "gamma", "vega", "theta"]],
            column_config={
                "label": st.column_config.TextColumn("Line"),
                "quantity": st.column_config.NumberColumn("Quantity", format="%.2f"),
                "expiry": st.column_config.NumberColumn("Expiry (years)", format="%.3f"),
                "unit_price": st.column_config.NumberColumn("Unit price", format="%.2f"),
                "value": st.column_config.NumberColumn("Value", format="%.2f"),
                "delta": st.column_config.NumberColumn("Delta", format="%.2f"),
                "gamma": st.column_config.NumberColumn("Gamma", format="%.3f"),
                "vega": st.column_config.NumberColumn("Vega", format="%.2f"),
                "theta": st.column_config.NumberColumn("Theta", format="%.2f"),
            },
        )
        st.caption(
            f"Cash in the book: {ui.money(book.cash)} of an initial "
            f"{ui.money(book.initial_cash)}."
        )

    # ---------------- the hedging rule (outside the form so its parameter can follow it)
    ui.section("2. How you will hedge", "The rule the simulator applies at every step, for you.")
    policy_name = st.segmented_control(
        "Hedging rule",
        list(POLICIES),
        default="Delta hedge every N steps",
        key="simulator_policy",
        help="A hedging policy is a rule that looks at the book and the market and decides "
        "which shares to trade. It is applied after the market has moved, at the new spot.",
    )
    policy_name = policy_name or "Delta hedge every N steps"
    st.caption(POLICIES[policy_name])

    # ---------------- the market scenario
    ui.section(
        "3. The market you will face",
        "Nothing is computed until you press start, so set everything you want first.",
    )
    with st.form("simulator_setup", border=True):
        c1, c2, c3 = st.columns(3)
        horizon_days = c1.slider(
            "Horizon (calendar days)",
            min_value=5,
            max_value=250,
            value=30,
            step=5,
            key="simulator_horizon",
            help="How far the simulation runs. Time passes for your options too: theta is paid "
            "over exactly this period.",
        )
        n_steps = c2.slider(
            "Number of steps",
            min_value=5,
            max_value=250,
            value=30,
            step=5,
            key="simulator_steps",
            help="How often you get to look at the market and hedge. 30 steps over 30 days is "
            "one look a day; fewer steps means bigger moves between hedges.",
        )
        realized_pct = c3.slider(
            "Realised volatility",
            min_value=1.0,
            max_value=100.0,
            value=30.0,
            step=0.5,
            format="%.1f%%",
            key="simulator_realized",
            help="The hidden truth: how much the spot really moves. Above the sidebar implied "
            "vol, a long-gamma book should win; below it, the option seller wins.",
        )

        c4, c5, c6 = st.columns(3)
        vol_model = c4.selectbox(
            "Implied-vol dynamics",
            list(VOL_MODELS),
            format_func=lambda name: VOL_MODELS[name],
            key="simulator_vol_model",
            help="How the level the options are marked at moves. It drives the vega P&L and "
            "nothing else.",
        )
        seed = c5.number_input(
            "Random seed",
            min_value=0,
            max_value=100_000,
            value=9,
            step=1,
            key="simulator_seed",
            help="Same seed, same path. Change it to see another draw of the same market; keep "
            "it to compare two hedging rules on exactly the same path.",
        )
        stock_bps = c6.slider(
            "Cost of trading the stock (bp)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5,
            key="simulator_costs",
            help="Paid on every hedge trade, in basis points of the traded notional. This is "
            "what makes hedging more often stop being free.",
        )

        hedge_every, band_width, hedge_vol_pct = 1, 0.05, 30.0
        if policy_name == "Delta hedge every N steps":
            hedge_every = st.slider(
                "Rebalance every N steps",
                min_value=1,
                max_value=20,
                value=1,
                key="simulator_hedge_every",
                help="1 means hedge at every step. Larger values leave the delta to drift "
                "between trades.",
            )
        elif policy_name == "Delta band":
            band_width = st.slider(
                "Band half-width (delta per option held)",
                min_value=0.01,
                max_value=0.50,
                value=0.05,
                step=0.01,
                key="simulator_band",
                help="0.05 on a book of 100 options re-hedges once the book is off by more than "
                "5 shares: five deltas per contract.",
            )
        elif policy_name == "Delta hedge at your own vol":
            cv1, cv2 = st.columns(2)
            hedge_vol_pct = cv1.slider(
                "Volatility used for the hedge delta",
                min_value=1.0,
                max_value=100.0,
                value=30.0,
                step=0.5,
                format="%.1f%%",
                key="simulator_hedge_vol",
                help="Set it to the realised vol to lock in a known total profit with a noisy "
                "path; set it to the implied vol for a smooth path with a path-dependent total.",
            )
            hedge_every = cv2.slider(
                "Rebalance every N steps",
                min_value=1,
                max_value=20,
                value=1,
                key="simulator_hedge_every_vol",
            )

        with st.expander("Advanced: drift, jumps and vol of vol", icon=":material/tune:"):
            a1, a2 = st.columns(2)
            drift_pct = a1.slider(
                "Expected drift of the spot (% a year)",
                min_value=-30.0,
                max_value=30.0,
                value=0.0,
                step=1.0,
                key="simulator_drift",
                help="A delta-hedged book barely notices the drift: that is the whole point of "
                "the Black-Scholes argument. An unhedged one notices nothing else.",
            )
            vol_of_vol_pct = a2.slider(
                "Volatility of the implied vol (% a year)",
                min_value=0.0,
                max_value=200.0,
                value=80.0,
                step=5.0,
                key="simulator_vol_of_vol",
                help="Only used by the two stochastic implied-vol models. 80% means the level of "
                "the implied vol itself moves about 80% a year in relative terms.",
            )
            a3, a4, a5 = st.columns(3)
            jump_intensity = a3.slider(
                "Jumps per year",
                min_value=0.0,
                max_value=12.0,
                value=0.0,
                step=0.5,
                key="simulator_jump_intensity",
                help="0 is a pure diffusion. Jumps are what delta hedging cannot handle: the "
                "spot gaps straight through your hedge.",
            )
            jump_mean_pct = a4.slider(
                "Average jump size (%)",
                min_value=-20.0,
                max_value=10.0,
                value=-5.0,
                step=1.0,
                key="simulator_jump_mean",
                help="In log-spot terms. Negative is the equity convention: gaps are downward.",
            )
            rho_spot_vol = a5.slider(
                "Spot / implied-vol correlation",
                min_value=-1.0,
                max_value=1.0,
                value=-0.7,
                step=0.05,
                key="simulator_rho",
                help="Used by the spot-correlated model only. Negative is the equity leverage "
                "effect: vol up when the market falls.",
            )
            zero_cash = st.toggle(
                "Ignore the interest on idle cash",
                value=True,
                key="simulator_zero_cash",
                help="Your book holds cash that earns the risk-free rate, and over a month that "
                "interest can dwarf the option P&L. With this on, the simulation starts from zero "
                "cash, so the carry term shows only the financing of your hedge -- exactly the "
                "term in the replication argument. Your P&L is unchanged either way.",
            )

        start = st.form_submit_button(
            "Start the simulation",
            icon=":material/play_arrow:",
            type="primary",
            disabled=not live,
            key="simulator_start",
        )

    st.caption(VOL_MODEL_NOTES[vol_model])
    if live and expiries:
        first_expiry_days = (min(expiries) - mkt.t) * DAYS_PER_YEAR
        st.caption(
            f"The first expiry in your book is in {first_expiry_days:.0f} days. "
            "A horizon longer than that settles those lines mid-run."
        )

    if start and live:
        try:
            sim_book = book.copy("Simulation book")
            if zero_cash:
                sim_book.deposit(-sim_book.cash)
            scenario = gbm_scenario(
                mkt,
                horizon=horizon_days / DAYS_PER_YEAR,
                n_steps=int(n_steps),
                realized_vol=realized_pct / 100.0,
                drift=drift_pct / 100.0,
                implied_vol_model=vol_model,
                vol_of_vol=vol_of_vol_pct / 100.0,
                rho_spot_vol=rho_spot_vol,
                jump_intensity=jump_intensity,
                jump_mean=jump_mean_pct / 100.0,
                seed=int(seed),
            )
            if policy_name == "No hedge":
                policy = NoHedge()
            elif policy_name == "Delta band":
                policy = DeltaBandHedge(band_width, relative=True)
            elif policy_name == "Delta hedge at your own vol":
                policy = DeltaHedgeAtVol(hedge_vol_pct / 100.0, n_steps=int(hedge_every))
            else:
                policy = DeltaHedgeEveryN(int(hedge_every))
            costs = TransactionCosts(stock_bps=stock_bps)
            new_sim = TradingSimulator(sim_book, scenario, policy, costs)
        except ValueError as exc:
            st.error(f"The simulation could not be built: {exc}", icon=":material/error:")
        else:
            state.set_sim(
                new_sim,
                {
                    "horizon_days": float(horizon_days),
                    "n_steps": int(n_steps),
                    "realized_vol": realized_pct / 100.0,
                    "implied_vol": float(mkt.vol),
                    "vol_model": vol_model,
                    "policy": policy.label,
                    "stock_bps": float(stock_bps),
                    "zero_cash": bool(zero_cash),
                    "jump_intensity": float(jump_intensity),
                },
            )
            st.rerun()

# --------------------------------------------------------------------------- #
# Act 2 -- walk through the scenario
# --------------------------------------------------------------------------- #
else:
    settings = state.sim_settings()
    scenario = sim.scenario
    total_steps = scenario.n_steps
    days_per_step = scenario.horizon * DAYS_PER_YEAR / max(total_steps, 1)
    one_day = abs(days_per_step - 1.0) < 0.05
    history = sim.history
    row = sim.last_row
    previous = history.iloc[-2] if len(history) > 1 else None

    ui.section("The run", scenario.name)
    ui.metric_row(
        [
            (
                "Horizon",
                f"{scenario.horizon * DAYS_PER_YEAR:.0f} days in {total_steps} steps",
                "One step is how often you get to look at the market and hedge.",
            ),
            (
                "Realised vs implied vol",
                f"{ui.percent(settings.get('realized_vol', float('nan')), decimals=1)} vs "
                f"{ui.percent(settings.get('implied_vol', float('nan')), decimals=1)}",
                "What the spot will really do over the run, against the implied vol you started "
                "from. That gap is the edge a delta-hedged book is trying to harvest.",
            ),
            ("Hedging rule", sim.policy.label, "Applied at every step, after the market moved."),
            (
                "Stock cost",
                f"{settings.get('stock_bps', 0.0):.1f} bp",
                "Paid on every hedge trade, in basis points of the traded notional.",
            ),
        ]
    )

    st.caption(
        "The run carries its own market path, drawn when you pressed start, so the sidebar no "
        "longer moves it. Start a new setup to simulate from today's sidebar market."
    )

    progress = sim.step_index / total_steps if total_steps else 1.0
    st.progress(
        progress,
        text=f"Step {sim.step_index} of {total_steps} "
        f"({sim.step_index * days_per_step:.0f} of {scenario.horizon * DAYS_PER_YEAR:.0f} days)",
    )

    greeks_now = sim.book.greeks(sim.current_market)
    with st.container(horizontal=True):
        st.button(
            "Next day" if one_day else "Next step",
            icon=":material/skip_next:",
            type="primary",
            on_click=advance,
            args=(1,),
            disabled=sim.is_finished,
            key="simulator_next",
            help="Move the market on by one step, hedge, and log the row.",
        )
        st.button(
            "Next 5 days" if one_day else "Next 5 steps",
            icon=":material/fast_forward:",
            on_click=advance,
            args=(5,),
            disabled=sim.is_finished,
            key="simulator_next5",
        )
        st.button(
            "Run to the end",
            icon=":material/play_circle:",
            on_click=advance,
            args=(None,),
            disabled=sim.is_finished,
            key="simulator_run_all",
        )
        st.button(
            "Hedge delta now",
            icon=":material/balance:",
            on_click=hedge_now,
            disabled=sim.is_finished or abs(greeks_now["delta"]) < 1e-9,
            key="simulator_hedge_now",
            help="Trade the shares that bring the book delta back to zero, by hand, right now. "
            "The effect shows up in the next step's delta term.",
        )
        st.button(
            "Reset",
            icon=":material/restart_alt:",
            on_click=reset_simulation,
            key="simulator_reset",
            help="Back to step 0 with the book as it was, on the same market path.",
        )
        st.button(
            "New setup",
            icon=":material/tune:",
            on_click=state.clear_sim,
            key="simulator_clear",
            help="Drop this simulation and choose a new scenario.",
        )

    if sim.is_finished:
        st.success(
            f"The scenario is finished: {total_steps} steps, "
            f"{ui.money(row['pnl_cum'], signed=True)} of P&L. Reset to replay the same path, or "
            "start a new setup to change the market.",
            icon=":material/flag:",
        )

    # ---------------- live panel
    ui.section("Where you stand", "Everything below is the book at the current step.")
    ui.metric_row(
        [
            (
                "Spot",
                ui.money(row["spot"]),
                "The level of the underlying at this step.",
                None if previous is None else signed(row["spot"] - previous["spot"]),
            ),
            (
                "Implied volatility",
                ui.percent(row["implied_vol"], decimals=1),
                "The level the options are marked at. Its moves are the vega P&L.",
                None
                if previous is None
                else f"{100.0 * (row['implied_vol'] - previous['implied_vol']):+.2f} pts",
            ),
            (
                "Net liquidation value",
                ui.money(row["value"]),
                "Positions marked to model plus cash: what you would walk away with if you "
                "closed everything now."
                + (
                    " This run started from zero cash, so it is the value of what you hold "
                    "rather than a bank balance; the P&L next to it is the number to watch."
                    if settings.get("zero_cash")
                    else ""
                ),
                signed(row["pnl_step"]),
            ),
            (
                "P&L since the start",
                ui.money(row["pnl_cum"], signed=True),
                "Fees included. The small number underneath is the P&L of the latest step.",
                signed(row["pnl_step"]),
            ),
            (
                "Shares held (the hedge)",
                ui.number(row["stock_position"], decimals=2),
                "Net shares in the book after the policy traded. The small number underneath "
                "is what it traded at this step.",
                signed(row["hedge_shares"]),
            ),
        ]
    )

    # The book's own price is already shown as the net liquidation value above, and a
    # hedged book's "price" mixes in the short stock, so only the risk is shown here.
    ui.greek_metrics(
        greeks_now,
        keys=("delta", "gamma", "vega", "theta", "rho"),
        trader_units=trader_units,
    )
    ui.units_caption(trader_units)

    breakeven = sim.book.breakeven_move(sim.current_market, days=days_per_step)
    if row["hedge_trades"]:
        lines = [
            f"The market moved the book delta to {ui.number(row['delta_before_hedge'], decimals=2)}"
            f" shares; the policy then traded {ui.number(row['hedge_shares'], decimals=2)} shares "
            f"and left it at {ui.number(row['delta'], decimals=2)}."
        ]
    else:
        lines = [
            f"The book is carrying {ui.number(row['delta'], decimals=2)} shares of delta and the "
            "policy did not trade at this step."
        ]
    if breakeven is not None and math.isfinite(breakeven):
        lines.append(
            f"Over the next step the spot has to move more than "
            f"{ui.percent(breakeven, decimals=2)} for gamma to pay the theta bill."
        )
    else:
        lines.append(
            "There is no gamma-against-theta trade-off right now: the book's gamma and theta "
            "have the same sign, so time and movement are both working the same way on you."
        )
    if row["n_settled"]:
        lines.append(
            f"{int(row['n_settled'])} line(s) settled at this step for "
            f"{ui.money(row['settlement_cash'], signed=True)} of cash."
        )
    st.caption(" ".join(lines))

    if sim.step_index == 0:
        ui.empty_state(
            "Take a step to start building the P&L explain. Everything below fills in as the "
            "market moves.",
            icon=":material/play_arrow:",
        )
    else:
        # ---------------- analysis views
        ui.section("Read the run", "Four ways of looking at the same rows.")
        view = st.segmented_control(
            "Analysis",
            ["P&L explain", "Gamma vs theta", "Dashboard", "History"],
            default="P&L explain",
            required=True,
            key="simulator_view",
        )

        if view == "Gamma vs theta":
            scalp = gamma_scalping_summary(history)
            ui.metric_row(
                [
                    ("Gamma P&L", ui.money(scalp["gamma_pnl"], signed=True), TERM_NOTES["Gamma"]),
                    ("Theta P&L", ui.money(scalp["theta_pnl"], signed=True), TERM_NOTES["Theta"]),
                    (
                        "Gamma + theta",
                        ui.money(scalp["gamma_plus_theta"], signed=True),
                        "The gamma-scalping result: what the moves earned net of the rent.",
                    ),
                    (
                        "Realised vs mean implied",
                        f"{ui.percent(scalp['realized_vol'], decimals=1)} vs "
                        f"{ui.percent(scalp['mean_implied_vol'], decimals=1)}",
                        "Realised vol of the path walked so far, against the time-average implied "
                        "vol the book was marked at.",
                    ),
                ]
            )
            ui.chart(
                sim_plots.gamma_theta_chart(history, mode=ui.CHART_MODE), key="simulator_gamma_theta"
            )
            ui.explain(
                "How to read this",
                "The top panel is the race: cumulative gamma climbing in steps (it only moves on "
                "the days the spot does), cumulative theta bleeding steadily, and the thick line "
                "their sum -- the verdict.\n\n"
                "The bottom panel says why. Each bar is the absolute spot move of one step; the "
                "dashed line is the breakeven move, `implied vol x sqrt(dt)`. For a delta-hedged "
                "vanilla book that is exactly where `0.5 x Gamma x dS^2 + Theta x dt = 0`, so "
                "bars above the line are steps long gamma won and bars below are steps the option "
                "seller won. Realised vol above implied simply means more tall bars than the line "
                "can pay for.",
                icon=":material/query_stats:",
            )
            short_gamma = scalp["theta_pnl"] > 0
            won = scalp["gamma_plus_theta"] > 0
            if short_gamma:
                verdict = (
                    f"You collected {ui.money(scalp['theta_pnl'])} of theta and gave back "
                    f"{ui.money(abs(scalp['gamma_pnl']))} on the moves: the market was "
                    + (
                        "quiet enough to let you keep the premium."
                        if won
                        else "too lively, and the moves cost more than the premium was worth."
                    )
                )
            else:
                verdict = (
                    f"Gamma earned {ui.money(scalp['gamma_pnl'])} against a theta bill of "
                    f"{ui.money(abs(scalp['theta_pnl']))}: the moves "
                    + ("more than paid the rent." if won else "did not pay the rent.")
                )
            ui.takeaways(
                [
                    verdict,
                    f"The path realised {ui.percent(scalp['realized_vol'], decimals=1)} against a "
                    f"mean implied of {ui.percent(scalp['mean_implied_vol'], decimals=1)}, a vol "
                    f"edge of {100 * scalp['vol_edge']:+.1f} points. The book is "
                    f"{'short' if short_gamma else 'long'} gamma, so gamma plus theta should "
                    f"carry the sign {'opposite to' if short_gamma else 'of'} that edge.",
                    f"Everything outside gamma and theta -- delta, vega, carry, fees -- came to "
                    f"{ui.money(scalp['delta_pnl'] + scalp['vega_pnl'] + scalp['other_pnl'], signed=True)}.",
                ]
            )
            ui.interview_note(
                "Gamma scalping is being long options and delta hedging them: every time the spot "
                "moves you re-hedge, which mechanically sells rallies and buys dips, and the "
                "round trips are the gamma P&L. It loses money whenever realised volatility comes "
                "in below the implied vol you paid -- a quiet, drifting market -- because the "
                "theta bill arrives every single day while the scalps do not. It also loses when "
                "the spot moves far from your strikes, since gamma dies there while the premium "
                "is already spent, and in a gapping market, where the move happens in one jump "
                "your hedge never gets to trade through.",
                question="What is gamma scalping, and when does it lose money?",
            )

        elif view == "Dashboard":
            ui.chart(
                sim_plots.simulation_dashboard(history, x="step", mode=ui.CHART_MODE),
                key="simulator_dashboard",
            )
            ui.explain(
                "How to read this",
                "Read it top to bottom. Row 1 is the market you were dealt: the spot path and the "
                "implied-vol path. Row 2 is what it did to you: the cumulative P&L and the share "
                "position the hedge had to carry. Row 3 shows the delta before and after the "
                "policy traded -- the gap between the two lines is the work the hedge did -- next "
                "to the dollar gamma. Row 4 is the risk you are carrying in currency: vega cash "
                "per vol point and theta cash per calendar day.\n\n"
                "The shape to look for: gamma and theta both swell as an at-the-money option nears "
                "expiry, and both die if the spot walks away from the strike.",
                icon=":material/dashboard:",
            )

        elif view == "History":
            display = pd.DataFrame(
                {
                    "Step": history.index,
                    "Day": (history["t"] - history["t"].iloc[0]) * DAYS_PER_YEAR,
                    "Spot": history["spot"],
                    "Implied vol (%)": history["implied_vol"] * 100.0,
                    "Step P&L": history["pnl_step"],
                    "Cumulative P&L": history["pnl_cum"],
                    "Gamma P&L": history["pnl_gamma"],
                    "Theta P&L": history["pnl_theta"],
                    "Vega P&L": history["pnl_vega"],
                    "Delta before hedge": history["delta_before_hedge"],
                    "Shares traded": history["hedge_shares"],
                    "Shares held": history["stock_position"],
                    "Fees": history["fees_paid"],
                }
            )
            rows = st.slider(
                "Rows to show (most recent first)",
                min_value=5,
                max_value=max(int(len(display)), 6),
                value=min(20, max(int(len(display)), 6)),
                key="simulator_history_rows",
                help="The table is the raw log of the run: one row per step, every number "
                "already used by the charts above.",
            )
            ui.table(
                display.tail(int(rows)).iloc[::-1],
                column_config={
                    "Step": st.column_config.NumberColumn(format="%d"),
                    "Day": st.column_config.NumberColumn(format="%.1f"),
                    "Spot": st.column_config.NumberColumn(format="%.2f"),
                    "Implied vol (%)": st.column_config.NumberColumn(format="%.2f"),
                    "Step P&L": st.column_config.NumberColumn(format="%+.2f"),
                    "Cumulative P&L": st.column_config.NumberColumn(format="%+.2f"),
                    "Gamma P&L": st.column_config.NumberColumn(format="%+.2f"),
                    "Theta P&L": st.column_config.NumberColumn(format="%+.2f"),
                    "Vega P&L": st.column_config.NumberColumn(format="%+.2f"),
                    "Delta before hedge": st.column_config.NumberColumn(format="%.2f"),
                    "Shares traded": st.column_config.NumberColumn(format="%+.2f"),
                    "Shares held": st.column_config.NumberColumn(format="%.2f"),
                    "Fees": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            st.download_button(
                "Download the full history (CSV)",
                data=history.to_csv().encode("utf-8"),
                file_name="optionlab_simulation_history.csv",
                mime="text/csv",
                icon=":material/download:",
                key="simulator_download_history",
                help=f"All {history.shape[1]} columns, including every attribution term and its "
                "running total.",
            )
            st.caption(
                "Each row satisfies the desk identity: the attribution terms of the step sum "
                "exactly to the step P&L."
            )

        else:  # P&L explain
            totals = attribution_totals(history)
            scalp = gamma_scalping_summary(history)
            cumulative = not st.toggle(
                "Show each step instead of the running total",
                value=False,
                key="simulator_attrib_steps",
                help="Running totals answer where the money came from; step bars answer which "
                "days it came from.",
            )
            ui.chart(
                sim_plots.pnl_attribution_chart(
                    history, cumulative=cumulative, x="step", mode=ui.CHART_MODE
                ),
                key="simulator_attribution",
            )
            ui.chart(
                sim_plots.pnl_attribution_waterfall(history, mode=ui.CHART_MODE),
                key="simulator_waterfall",
            )

            greek_scale = max(abs(totals[term]) for term in GREEK_TERMS)
            vol_start = float(history["implied_vol"].iloc[0])
            vol_end = float(history["implied_vol"].iloc[-1])
            n_trades = int(history["hedge_trades"].sum())
            vol_line = (
                "The implied vol never moved, so the vega term is exactly zero. "
                if abs(vol_end - vol_start) < 1e-12
                else f"Implied vol went from {ui.percent(vol_start, decimals=1)} to "
                f"{ui.percent(vol_end, decimals=1)}, which is the whole of that vega number. "
            )
            hedge_line = (
                "You never hedged, so there were no trading fees. "
                if n_trades == 0
                else f"The hedge cost {ui.money(-totals['fees'])} in fees over {n_trades} trades. "
            )
            ui.takeaways(
                [
                    f"Total P&L {ui.money(totals['actual'], signed=True)} over "
                    f"{sim.step_index} steps. You {earned(totals['gamma'])} from gamma, "
                    f"{earned(totals['theta'], 'collected', 'paid')} in theta and "
                    f"{earned(totals['vega'])} on vega.",
                    vol_line
                    + f"The spot realised {ui.percent(scalp['realized_vol'], decimals=1)}, which "
                    "is what the gamma number is made of.",
                    f"Direction contributed {ui.money(totals['delta'], signed=True)} through "
                    f"delta. " + hedge_line + "Carry, the interest and dividends on what you "
                    f"hold, came to {ui.money(totals['carry'], signed=True)}.",
                    f"Unexplained {ui.money(totals['unexplained'], signed=True)}, or "
                    f"{share(abs(totals['unexplained']), greek_scale)} of the largest Greek term. "
                    "Small means the start-of-step Greeks told the whole story.",
                ],
                title="What the numbers say",
            )
            ui.explain(
                "What each term means",
                "The P&L explain is a Taylor expansion of the step on the Greeks you held at the "
                "**start** of it. Because the Greeks are raw derivatives -- theta per year, vega "
                "per 1.00 of vol -- each product is directly a currency amount. The names below "
                "are the legend of the charts above.\n\n"
                + "\n".join(f"- **{name}** — {note}" for name, note in TERM_NOTES.items())
                + "\n\nThe terms are defined so that they always sum to the actual P&L, which is "
                "exactly the check a desk runs on its P&L explain every morning.",
                icon=":material/functions:",
            )
            ui.explain(
                "How to read the two charts",
                "The **cumulative chart** draws one line per term as running totals, plus the "
                "thick actual P&L line. Lines rather than a stack, because the terms have "
                "opposite signs: a long-gamma book earns gamma and pays theta, and stacking them "
                "would hide exactly the tug-of-war you came to see. Terms that stay below 1% of "
                "the largest one are not drawn.\n\n"
                "The **waterfall** is the same run as one total per term, ending on the actual "
                "P&L. Bars up are gains, bars down are losses. It answers 'where did the money "
                "come from?' in a single picture -- and it is the format a risk report uses.",
                icon=":material/stacked_line_chart:",
            )

# --------------------------------------------------------------------------- #
# Act 3 -- the hedging laboratory (independent of the running simulation)
# --------------------------------------------------------------------------- #
ui.section(
    "The hedging laboratory",
    "One option, thousands of simulated paths, several rebalancing frequencies. The two "
    "classical results of discrete hedging, in two charts.",
)

ui.explain(
    "What this experiment does",
    "You buy (or sell) one option at the implied vol of the sidebar and delta hedge it all the "
    "way to expiry, over and over, on thousands of simulated spot paths. The **same paths** are "
    "then hedged again at each rebalancing frequency you pick, so the results differ only by how "
    "often the hedge was adjusted -- nothing else changes.\n\n"
    "Two questions get answered at once. How *precise* is discrete hedging (the width of the "
    "final P&L distribution), and how *profitable* is it (where that distribution sits). The "
    "first depends on how often you trade; the second does not.\n\n"
    "It is independent of the simulation above: it always uses the current sidebar market, so "
    "you can run it before you have a book.",
    icon=":material/science:",
)

with st.form("simulator_experiment", border=True):
    e1, e2, e3, e4 = st.columns(4)
    exp_type = e1.selectbox(
        "Option",
        ["call", "put"],
        key="simulator_exp_type",
        help="What you buy or sell at the sidebar implied vol, then hedge all the way to expiry.",
    )
    exp_strike = e2.number_input(
        "Strike",
        min_value=0.01,
        value=float(round(spot, 2)),
        step=max(0.5, round(spot / 100.0, 2)),
        key="simulator_exp_strike",
        help="At the money is where gamma and theta are largest, so it is where the experiment "
        "is most visible.",
    )
    exp_days = e3.slider(
        "Days to expiry",
        min_value=7,
        max_value=365,
        value=90,
        step=1,
        key="simulator_exp_days",
        help="The hedge runs from today to expiry, whatever the rebalancing frequency.",
    )
    exp_quantity = e4.number_input(
        "Quantity",
        value=10.0,
        step=1.0,
        key="simulator_exp_quantity",
        help="Positive means you buy the option (long gamma: the hedge sells rallies and buys "
        "dips). Negative is the option seller's version of the same experiment.",
    )

    e5, e6, e7 = st.columns(3)
    exp_realized_pct = e5.slider(
        "Realised volatility",
        min_value=1.0,
        max_value=100.0,
        value=30.0,
        step=0.5,
        format="%.1f%%",
        key="simulator_exp_realized",
        help="Set it equal to the sidebar implied vol to see pure hedging noise; set it above to "
        "see the buyer get paid for the vol that realises.",
    )
    exp_hedge_at = e6.selectbox(
        "Delta computed at",
        ["the implied vol", "the realised vol"],
        key="simulator_exp_hedge_at",
        help="Hedging at the implied vol gives a smooth daily P&L whose total depends on the "
        "path; hedging at the realised vol gives a known total with a noisy path.",
    )
    exp_paths = e7.slider(
        "Number of paths",
        min_value=200,
        max_value=3000,
        value=1000,
        step=100,
        key="simulator_exp_paths",
        help="More paths mean smaller error bars on the statistics, and a slower run.",
    )

    e8, e9 = st.columns(2)
    exp_frequencies = e8.multiselect(
        "Rebalancing frequencies (number of hedges until expiry)",
        [4, 13, 26, 52, 126, 252],
        default=[4, 13, 52, 126],
        key="simulator_exp_frequencies",
        help="The same paths are hedged at each of these frequencies, so the columns differ only "
        "by how often the hedge was adjusted. Pick at least two.",
    )
    exp_bps = e9.slider(
        "Cost of trading the stock (bp)",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.5,
        key="simulator_exp_bps",
        help="Set it above zero to see the frequent hedger start paying for the precision.",
    )
    run_experiment = st.form_submit_button(
        "Run the experiment", icon=":material/science:", key="simulator_run_experiment"
    )

if run_experiment:
    # Only the form choices are stored; the market comes from the sidebar at render
    # time, so changing the spot or the implied vol updates the experiment.
    st.session_state["simulator_experiment_params"] = {
        "option_type": exp_type,
        "strike": float(exp_strike),
        "expiry_years": float(exp_days) / DAYS_PER_YEAR,
        "quantity": float(exp_quantity),
        "realized_vol": exp_realized_pct / 100.0,
        "hedge_at": exp_hedge_at,
        "n_paths": int(exp_paths),
        "frequencies": tuple(sorted(int(n) for n in exp_frequencies)),
        "stock_bps": float(exp_bps),
    }

choices = st.session_state.get("simulator_experiment_params")
params = None
if choices is not None:
    params = {
        "option_type": choices["option_type"],
        "strike": choices["strike"],
        "expiry_years": choices["expiry_years"],
        "quantity": choices["quantity"],
        "spot": float(mkt.spot),
        "implied_vol": float(mkt.vol),
        "rate": float(mkt.rate),
        "div": float(mkt.div),
        "realized_vol": choices["realized_vol"],
        "hedge_vol": (
            choices["realized_vol"] if choices["hedge_at"] == "the realised vol" else float(mkt.vol)
        ),
        "n_paths": choices["n_paths"],
        "frequencies": choices["frequencies"],
        "seed": 11,
        "stock_bps": choices["stock_bps"],
    }

if params is None:
    ui.empty_state(
        "Choose an option and press 'Run the experiment'. Nothing is simulated until you do.",
        icon=":material/science:",
    )
elif len(params["frequencies"]) < 2:
    st.warning(
        "Pick at least two rebalancing frequencies: the whole point is to compare them.",
        icon=":material/rule:",
    )
elif params["quantity"] == 0:
    st.warning("A quantity of zero has nothing to hedge. Buy or sell at least one option.",
               icon=":material/rule:")
else:
    try:
        with st.spinner("Simulating the paths and hedging every one of them..."):
            experiment, attrs = run_hedging_experiment(**params)
        experiment.attrs.update(attrs)
        summary = hedging_experiment_summary(experiment)
    except ValueError as exc:
        st.error(f"The experiment could not be run: {exc}", icon=":material/error:")
    else:
        frequencies = list(summary.index)
        first, last = summary.loc[frequencies[0]], summary.loc[frequencies[-1]]
        ui.metric_row(
            [
                (
                    "Premium traded",
                    ui.money(abs(params["quantity"]) * attrs["premium"]),
                    "What the option cost at the implied vol, for the quantity chosen.",
                ),
                (
                    f"Mean P&L at {frequencies[-1]} hedges",
                    ui.money(last["mean"], signed=True),
                    "Average final P&L over all the simulated paths.",
                ),
                (
                    "Standard error of that mean",
                    ui.money(last["stderr"]),
                    "How precisely the simulation knows the mean. The mean is only meaningful "
                    "next to this number.",
                ),
                (
                    "Error shrinks by",
                    f"{first['std'] / last['std']:.1f}x"
                    if last["std"] > 0
                    else "n/a",
                    f"Ratio of the standard deviation at {frequencies[0]} hedges to the one at "
                    f"{frequencies[-1]} hedges.",
                ),
            ]
        )
        ui.chart(
            sim_plots.hedging_error_histogram(experiment, mode=ui.CHART_MODE),
            key="simulator_experiment_histogram",
        )
        ui.explain(
            "How to read this",
            "One panel per rebalancing frequency, all sharing the same bins and the same "
            "horizontal axis, so the distributions can be compared by eye. Two things move "
            "independently: the **width**, which is hedging error and shrinks as you trade more "
            "often, and the **centre**, which is set by realised versus implied vol and barely "
            "moves at all. Each panel title states its mean and standard deviation.",
            icon=":material/bar_chart:",
        )
        ui.chart(
            sim_plots.hedging_error_vs_frequency(experiment, mode=ui.CHART_MODE),
            key="simulator_experiment_frequency",
        )
        ui.explain(
            "How to read this",
            "The dashed reference has slope -1/2 on a log-log scale: `std(N) = std(N0) x "
            "sqrt(N0 / N)`. When realised vol equals implied vol the dots sit on it, so halving "
            "the hedging error costs four times as many trades. When the two vols differ the dots "
            "flatten out: part of the dispersion is no longer hedging noise but the "
            "path-dependence of the gamma P&L itself, and no amount of rebalancing removes it.",
            icon=":material/trending_down:",
        )

        display = summary.reset_index().rename(
            columns={
                "rebalance_steps": "Hedges",
                "mean": "Mean P&L",
                "stderr": "Std error of the mean",
                "std": "Std of P&L",
                "q05": "5th percentile",
                "q95": "95th percentile",
                "std_pct_premium": "Std (% of premium)",
                "std_x_sqrt_n": "Std x sqrt(N)",
                "theory_mean": "Textbook gamma formula",
            }
        )
        ui.table(
            display,
            column_config={
                "Hedges": st.column_config.NumberColumn(format="%d"),
                "Mean P&L": st.column_config.NumberColumn(format="%+.2f"),
                "Std error of the mean": st.column_config.NumberColumn(format="%.2f"),
                "Std of P&L": st.column_config.NumberColumn(format="%.2f"),
                "5th percentile": st.column_config.NumberColumn(format="%+.2f"),
                "95th percentile": st.column_config.NumberColumn(format="%+.2f"),
                "Std (% of premium)": st.column_config.NumberColumn(format="%.1f"),
                "Std x sqrt(N)": st.column_config.NumberColumn(format="%.1f"),
                "Textbook gamma formula": st.column_config.NumberColumn(format="%+.2f"),
            },
        )

        vol_gap = params["realized_vol"] - params["implied_vol"]
        flat = abs(vol_gap) < 1e-9
        points = [
            f"`Std x sqrt(N)` goes from {first['std_x_sqrt_n']:,.1f} at {frequencies[0]} hedges to "
            f"{last['std_x_sqrt_n']:,.1f} at {frequencies[-1]}. "
            + (
                "Flat is the `1 / sqrt(N)` law: four times as many trades halve the hedging error."
                if flat
                else "It climbs because realised vol is away from implied: only the hedging-noise "
                "part of the dispersion obeys `1 / sqrt(N)`, and the rest is the path-dependence "
                "of the gamma P&L itself, which no amount of rebalancing removes."
            ),
            f"The mean P&L moves from {ui.money(first['mean'], signed=True)} to "
            f"{ui.money(last['mean'], signed=True)} across those frequencies, while the standard "
            f"error of the last one is {ui.money(last['stderr'])}. Hedging harder buys precision, "
            "not profit.",
        ]
        if flat:
            points.append(
                "Realised vol equals the implied vol you paid, so the mean sits near zero: "
                "discrete hedging replicates the option on average and the rest is noise."
            )
        else:
            points.append(
                f"You traded {ui.percent(params['implied_vol'], decimals=1)} implied against "
                f"{ui.percent(params['realized_vol'], decimals=1)} realised, a gap of "
                f"{100 * vol_gap:+.1f} points, and the "
                f"{'buyer' if params['quantity'] > 0 else 'seller'} of the option is the one paid "
                f"for it. The textbook gamma formula predicts "
                f"{ui.money(last['theory_mean'], signed=True)} against a simulated "
                f"{ui.money(last['mean'], signed=True)}."
            )
        if params["stock_bps"] > 0:
            points.append(
                f"With {params['stock_bps']:.1f} bp of cost on every stock trade, the mean falls "
                f"from {ui.money(first['mean'], signed=True)} at {frequencies[0]} hedges to "
                f"{ui.money(last['mean'], signed=True)} at {frequencies[-1]}: precision is no "
                "longer free."
            )
        ui.takeaways(points, title="What the experiment shows")

ui.section(
    "Interview angles", "Two questions a desk actually asks, and the answer this page proves."
)

ui.interview_note(
    "Roughly the premium you overpaid, and you can size it. A delta-hedged option earns "
    "`0.5 x Gamma x S^2 x (realised^2 - implied^2)` per unit of time, so if realised comes in "
    "below the implied you paid, every one of those daily terms is negative: gamma never earns "
    "back the theta you are billed. You lose about the variance gap times the dollar gamma you "
    "carry, day after day, and the bleed is worst when the spot sits on the strike, because that "
    "is where gamma and theta are both largest. Run the laboratory above with realised below "
    "implied and the whole distribution shifts below zero, at every hedging frequency.",
    question="You delta-hedge a long call daily and realised vol comes in below implied. "
    "What is your P&L?",
)

ui.interview_note(
    "Because hedging more often buys you precision, not expected profit. The expected P&L of a "
    "delta-hedged option is set by realised against implied volatility -- it is the same whether "
    "you hedge four times or two hundred. What more hedging does is shrink the dispersion of the "
    "outcome, and only like `1 / sqrt(N)`: four times the trades for half the error. Meanwhile "
    "the costs grow linearly with the number of trades. So there is an optimum, and in practice a "
    "desk hedges on a band rather than a clock: trade when the market has actually moved, not "
    "when the bell rings.",
    question="Why does hedging more often not always make you more money?",
)
