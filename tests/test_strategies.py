"""Tests for optionlab.strategies and optionlab.plotting.strategy_plots."""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy.special import ndtr

from optionlab import black_scholes as bs
from optionlab import strategies as st
from optionlab.instruments import (
    GREEK_KEYS,
    CompositeInstrument,
    EuropeanOption,
    Instrument,
    Position,
    Underlying,
    instrument_from_dict,
)
from optionlab.market import Market
from optionlab.monte_carlo import mc_price
from optionlab.plotting import strategy_plots as sp
from optionlab.plotting.theme import PALETTE

T = 1.0
SPOTS = np.array([70.0, 90.0, 100.0, 110.0, 130.0])
MKT = Market(spot=100.0, vol=0.2, rate=0.02, div=0.01)
FLAT = Market(spot=100.0, vol=0.2)  # zero rate and dividend: clean hand-computed numbers

# (factory name, kwargs, expected payoff at SPOTS) -- all computed by hand.
PAYOFF_CASES = [
    ("long_call", dict(strike=100), [0, 0, 0, 10, 30]),
    ("long_put", dict(strike=100), [30, 10, 0, 0, 0]),
    ("short_call", dict(strike=100), [0, 0, 0, -10, -30]),
    ("short_put", dict(strike=100), [-30, -10, 0, 0, 0]),
    ("covered_call", dict(strike=110), [70, 90, 100, 110, 110]),
    ("protective_put", dict(strike=90), [90, 90, 100, 110, 130]),
    ("collar", dict(put_strike=90, call_strike=110), [90, 90, 100, 110, 110]),
    ("bull_call_spread", dict(low_strike=90, high_strike=110), [0, 0, 10, 20, 20]),
    ("bear_put_spread", dict(low_strike=90, high_strike=110), [20, 20, 10, 0, 0]),
    ("bull_put_spread", dict(low_strike=90, high_strike=110), [-20, -20, -10, 0, 0]),
    ("bear_call_spread", dict(low_strike=90, high_strike=110), [0, 0, -10, -20, -20]),
    ("long_straddle", dict(strike=100), [30, 10, 0, 10, 30]),
    ("short_straddle", dict(strike=100), [-30, -10, 0, -10, -30]),
    ("long_strangle", dict(put_strike=90, call_strike=110), [20, 0, 0, 0, 20]),
    ("short_strangle", dict(put_strike=90, call_strike=110), [-20, 0, 0, 0, -20]),
    ("strip", dict(strike=100), [60, 20, 0, 10, 30]),
    ("strap", dict(strike=100), [30, 10, 0, 20, 60]),
    ("long_call_butterfly", dict(low_strike=90, mid_strike=100, high_strike=110), [0, 0, 10, 0, 0]),
    ("long_put_butterfly", dict(low_strike=90, mid_strike=100, high_strike=110), [0, 0, 10, 0, 0]),
    ("iron_butterfly", dict(low_strike=90, mid_strike=100, high_strike=110), [-10, -10, 0, -10, -10]),
    (
        "long_call_condor",
        dict(strike_1=80, strike_2=90, strike_3=110, strike_4=120),
        [0, 10, 10, 10, 0],
    ),
    (
        "iron_condor",
        dict(strike_1=80, strike_2=90, strike_3=110, strike_4=120),
        [-10, 0, 0, 0, -10],
    ),
    ("call_ratio_spread", dict(low_strike=100, high_strike=110), [0, 0, 0, 10, -10]),
    ("put_ratio_spread", dict(low_strike=90, high_strike=100), [-10, 10, 0, 0, 0]),
    ("call_backspread", dict(low_strike=100, high_strike=110), [0, 0, 0, -10, 10]),
    ("put_backspread", dict(low_strike=90, high_strike=100), [10, -10, 0, 0, 0]),
    ("risk_reversal", dict(put_strike=90, call_strike=110), [-20, 0, 0, 0, 20]),
    ("seagull", dict(put_strike=90, call_strike=100, cap_strike=110), [-20, 0, 0, 10, 10]),
    ("jade_lizard", dict(put_strike=90, call_strike=110, wing_strike=120), [-20, 0, 0, 0, -10]),
    ("synthetic_long", dict(strike=100), [-30, -10, 0, 10, 30]),
    ("synthetic_short", dict(strike=100), [30, 10, 0, -10, -30]),
    ("conversion", dict(strike=100), [100, 100, 100, 100, 100]),
    ("reversal", dict(strike=100), [-100, -100, -100, -100, -100]),
    ("box_spread", dict(low_strike=90, high_strike=110), [20, 20, 20, 20, 20]),
]
TIME_SPREADS = {"calendar_spread", "diagonal_spread", "double_calendar"}


def bs_price(kind: str, strike: float, tau: float, mkt: Market = MKT, spot=None) -> float:
    spot = mkt.spot if spot is None else spot
    return bs.price(spot, strike, tau, mkt.vol, mkt.rate, mkt.div, kind)


# ---------------------------------------------------------------------- #
# Payoffs
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("name, kwargs, expected", PAYOFF_CASES, ids=[c[0] for c in PAYOFF_CASES])
def test_payoff_at_hand_computed_points(name, kwargs, expected):
    strategy = getattr(st, name)(expiry=T, **kwargs)
    assert strategy.payoff(SPOTS) == pytest.approx(expected)
    assert st.payoff_at_expiry(strategy, SPOTS) == pytest.approx(expected)
    assert strategy.key == name
    # settlement value at expiry is the payoff too
    assert strategy.price(FLAT.bumped(spot=SPOTS, t=T)) == pytest.approx(expected)


def test_every_registered_strategy_has_a_payoff_test():
    assert {c[0] for c in PAYOFF_CASES} | TIME_SPREADS == set(st.STRATEGY_REGISTRY)


def test_quantity_and_ratio_scale_the_legs():
    triple = st.bull_call_spread(90, 110, T, quantity=3)
    assert triple.payoff(SPOTS) == pytest.approx([0, 0, 30, 60, 60])
    assert triple.price(MKT) == pytest.approx(3 * st.bull_call_spread(90, 110, T).price(MKT))
    assert "(x3)" in triple.name
    one_by_three = st.call_ratio_spread(100, 110, T, ratio=3)
    assert one_by_three.payoff(np.array([130.0])) == pytest.approx([30 - 3 * 20])
    assert [leg.quantity for leg in st.put_backspread(90, 100, T, ratio=2.5, quantity=2).legs] == [-2, 5]


# ---------------------------------------------------------------------- #
# Parity identities and decompositions
# ---------------------------------------------------------------------- #
def test_synthetic_long_is_a_forward():
    expected = 100 * math.exp(-0.01 * T) - 95 * math.exp(-0.02 * T)
    assert st.synthetic_long(95, T).price(MKT) == pytest.approx(expected, abs=1e-10)
    assert st.synthetic_short(95, T).price(MKT) == pytest.approx(-expected, abs=1e-10)
    greeks = st.synthetic_long(95, T).greeks(MKT)
    assert greeks["delta"] == pytest.approx(math.exp(-0.01 * T))
    assert greeks["gamma"] == pytest.approx(0.0, abs=1e-12)
    assert greeks["vega"] == pytest.approx(0.0, abs=1e-10)


def test_box_spread_is_a_zero_coupon_bond():
    box = st.box_spread(90, 110, T)
    assert box.price(MKT) == pytest.approx(20 * math.exp(-0.02 * T), abs=1e-10)
    greeks = box.greeks(MKT)
    for name in ("delta", "gamma", "vega", "vanna", "volga"):
        assert greeks[name] == pytest.approx(0.0, abs=1e-10)
    assert greeks["rho"] == pytest.approx(-T * 20 * math.exp(-0.02 * T))


def test_conversion_and_reversal_lock_in_the_strike():
    # stock + put - call = K e^{-rT} + S (1 - e^{-qT}): no arbitrage profit left
    fair = 100 * math.exp(-0.02 * T) + 100 * (1 - math.exp(-0.01 * T))
    conversion = st.conversion(100, T)
    assert conversion.price(MKT) - fair == pytest.approx(0.0, abs=1e-10)
    assert st.reversal(100, T).price(MKT) + conversion.price(MKT) == pytest.approx(0.0, abs=1e-12)
    assert conversion.greeks(MKT)["gamma"] == pytest.approx(0.0, abs=1e-12)
    no_div = MKT.bumped(div=0.0)
    assert conversion.price(no_div) == pytest.approx(100 * math.exp(-0.02 * T), abs=1e-10)
    assert conversion.greeks(no_div)["delta"] == pytest.approx(0.0, abs=1e-12)


def test_butterfly_is_the_sum_of_two_vertical_spreads():
    fly = st.long_call_butterfly(90, 100, 110, T)
    bull, bear = st.bull_call_spread(90, 100, T), st.bear_call_spread(100, 110, T)
    grid = np.linspace(60, 140, 33)
    ladder = MKT.bumped(spot=grid)
    assert fly.price(ladder) == pytest.approx(bull.price(ladder) + bear.price(ladder), abs=1e-12)
    assert fly.payoff(grid) == pytest.approx(bull.payoff(grid) + bear.payoff(grid))
    # and the call and put butterflies are the same thing (put-call parity)
    assert fly.price(ladder) == pytest.approx(st.long_put_butterfly(90, 100, 110, T).price(ladder), abs=1e-10)


def test_iron_structures_are_shifted_long_structures():
    ladder = MKT.bumped(spot=np.linspace(60, 140, 17))
    iron_fly, fly = st.iron_butterfly(90, 100, 110, T), st.long_call_butterfly(90, 100, 110, T)
    assert fly.price(ladder) - iron_fly.price(ladder) == pytest.approx(10 * math.exp(-0.02 * T), abs=1e-10)
    iron_condor, condor = st.iron_condor(80, 90, 110, 120, T), st.long_call_condor(80, 90, 110, 120, T)
    assert condor.payoff(SPOTS) - iron_condor.payoff(SPOTS) == pytest.approx([10] * 5)


def test_covered_call_equals_short_put_plus_bond():
    ladder = MKT.bumped(spot=np.linspace(60, 140, 17), div=0.0)
    lhs = st.covered_call(105, T).price(ladder)
    rhs = st.short_put(105, T).price(ladder) + 105 * math.exp(-0.02 * T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


# ---------------------------------------------------------------------- #
# Greeks
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(st.STRATEGY_REGISTRY))
def test_greeks_are_quantity_weighted_sums_of_leg_greeks(name):
    strategy = st.STRATEGY_REGISTRY[name].build_default(100.0, 0.5, quantity=2)
    total = strategy.greeks(MKT)
    assert tuple(total) == GREEK_KEYS
    for key in GREEK_KEYS:
        expected = sum(leg.quantity * leg.instrument.greeks(MKT)[key] for leg in strategy.legs)
        assert total[key] == pytest.approx(expected, abs=1e-12)
    assert total["price"] == pytest.approx(strategy.price(MKT), abs=1e-12)


def test_greeks_are_vectorised_over_the_market():
    ladder = MKT.bumped(spot=np.linspace(80, 120, 9))
    greeks = st.iron_condor(85, 95, 105, 115, T).greeks(ladder)
    assert all(np.shape(greeks[key]) == (9,) for key in GREEK_KEYS)


def test_straddle_delta_is_small_at_the_money_forward():
    forward = MKT.forward(T)
    delta = st.long_straddle(forward, T).greeks(MKT)["delta"]
    assert 0 < delta < 0.1  # 2 N(sigma sqrt(T) / 2) - 1, about 0.08
    delta_neutral_strike = forward * math.exp(0.5 * 0.2**2 * T)  # d1 = 0
    assert st.long_straddle(delta_neutral_strike, T).greeks(MKT)["delta"] == pytest.approx(0.0, abs=1e-12)


def test_signs_of_the_classic_exposures():
    long_vol, short_vol = st.long_straddle(100, T).greeks(MKT), st.iron_condor(85, 95, 105, 115, T).greeks(MKT)
    assert long_vol["gamma"] > 0 and long_vol["vega"] > 0 and long_vol["theta"] < 0
    assert short_vol["gamma"] < 0 and short_vol["vega"] < 0 and short_vol["theta"] > 0
    calendar = st.calendar_spread(100, 0.25, 1.0).greeks(MKT)
    assert calendar["vega"] > 0 and calendar["gamma"] < 0 and calendar["theta"] > 0
    assert st.strip(100, T).greeks(MKT)["delta"] < 0 < st.strap(100, T).greeks(MKT)["delta"]


# ---------------------------------------------------------------------- #
# Premium, breakevens, extremes
# ---------------------------------------------------------------------- #
def test_net_premium_sign_convention_and_stock_legs():
    assert st.net_premium(st.long_straddle(100, T), MKT) > 0  # debit
    assert st.net_premium(st.iron_condor(85, 95, 105, 115, T), MKT) < 0  # credit
    call = bs_price("call", 105, T)
    covered = st.covered_call(105, T)
    assert st.net_premium(covered, MKT) == pytest.approx(100 - call)
    assert st.net_premium(covered, MKT, include_underlying=False) == pytest.approx(-call)
    assert st.net_premium(Position(covered, -2), MKT) == pytest.approx(-2 * (100 - call))


def test_pnl_is_payoff_minus_undiscounted_premium():
    spread = st.bull_call_spread(95, 105, T)
    debit = st.net_premium(spread, MKT)
    assert st.pnl_at_expiry(spread, SPOTS, MKT) == pytest.approx(spread.payoff(SPOTS) - debit)
    assert st.pnl_at_expiry(spread, SPOTS, entry_cost=4.0) == pytest.approx(spread.payoff(SPOTS) - 4.0)
    with pytest.raises(ValueError, match="Market or an explicit entry_cost"):
        st.pnl_at_expiry(spread, SPOTS)


def test_straddle_breakevens_are_strike_plus_minus_premium():
    straddle = st.long_straddle(100, T)
    premium = st.net_premium(straddle, MKT)
    assert st.breakevens(straddle, MKT) == pytest.approx([100 - premium, 100 + premium], abs=1e-9)
    assert st.breakevens(st.short_straddle(100, T), MKT) == pytest.approx(
        [100 - premium, 100 + premium], abs=1e-9
    )


def test_vertical_spread_extremes_and_breakevens():
    debit = st.net_premium(st.bull_call_spread(95, 105, T), MKT)
    bull_call = st.bull_call_spread(95, 105, T)
    assert st.max_profit(bull_call, MKT) == pytest.approx(10 - debit)
    assert st.max_loss(bull_call, MKT) == pytest.approx(-debit)
    assert st.breakevens(bull_call, MKT) == pytest.approx([95 + debit], abs=1e-9)

    bear_put = st.bear_put_spread(95, 105, T)
    put_debit = st.net_premium(bear_put, MKT)
    assert st.max_profit(bear_put, MKT) == pytest.approx(10 - put_debit)
    assert st.max_loss(bear_put, MKT) == pytest.approx(-put_debit)
    assert st.breakevens(bear_put, MKT) == pytest.approx([105 - put_debit], abs=1e-9)

    bull_put = st.bull_put_spread(95, 105, T)
    credit = -st.net_premium(bull_put, MKT)
    assert credit == pytest.approx(put_debit)
    assert st.max_profit(bull_put, MKT) == pytest.approx(credit)
    assert st.max_loss(bull_put, MKT) == pytest.approx(-(10 - credit))
    assert st.breakevens(bull_put, MKT) == pytest.approx([105 - credit], abs=1e-9)

    bear_call = st.bear_call_spread(95, 105, T)
    assert st.max_profit(bear_call, MKT) == pytest.approx(debit)
    assert st.max_loss(bear_call, MKT) == pytest.approx(-(10 - debit))
    assert st.breakevens(bear_call, MKT) == pytest.approx([95 + debit], abs=1e-9)


def test_butterfly_and_condor_extremes():
    fly = st.long_call_butterfly(90, 100, 110, T)
    debit = st.net_premium(fly, MKT)
    assert 0 < debit < 10
    assert st.max_profit(fly, MKT) == pytest.approx(10 - debit)
    assert st.max_loss(fly, MKT) == pytest.approx(-debit)
    assert st.breakevens(fly, MKT) == pytest.approx([90 + debit, 110 - debit], abs=1e-9)

    iron_fly = st.iron_butterfly(90, 100, 110, T)
    credit = -st.net_premium(iron_fly, MKT)
    assert st.max_profit(iron_fly, MKT) == pytest.approx(credit)
    assert st.max_loss(iron_fly, MKT) == pytest.approx(credit - 10)

    condor = st.long_call_condor(85, 95, 105, 115, T)
    condor_debit = st.net_premium(condor, MKT)
    assert st.max_profit(condor, MKT) == pytest.approx(10 - condor_debit)
    assert st.max_loss(condor, MKT) == pytest.approx(-condor_debit)

    iron_condor = st.iron_condor(85, 95, 105, 115, T)
    ic_credit = -st.net_premium(iron_condor, MKT)
    assert st.max_profit(iron_condor, MKT) == pytest.approx(ic_credit)
    assert st.max_loss(iron_condor, MKT) == pytest.approx(ic_credit - 10)
    assert st.breakevens(iron_condor, MKT) == pytest.approx([95 - ic_credit, 105 + ic_credit], abs=1e-9)

    broken_wing = st.long_call_butterfly(90, 100, 115, T)  # payoff stays at -5 above 115
    cost = st.net_premium(broken_wing, MKT)
    assert st.max_loss(broken_wing, MKT) == pytest.approx(-5 - cost)


@pytest.mark.parametrize(
    "strategy, best, worst",
    [
        (st.long_call(100, T), math.inf, None),
        (st.short_call(100, T), None, -math.inf),
        (st.long_straddle(100, T), math.inf, None),
        (st.short_straddle(100, T), None, -math.inf),
        (st.short_strangle(90, 110, T), None, -math.inf),
        (st.strap(100, T), math.inf, None),
        (st.call_ratio_spread(100, 110, T), None, -math.inf),
        (st.call_backspread(100, 110, T), math.inf, None),
        (st.risk_reversal(90, 110, T), math.inf, None),
        (st.synthetic_long(100, T), math.inf, None),
        (st.synthetic_short(100, T), None, -math.inf),
        (st.protective_put(95, T), math.inf, None),
    ],
    ids=lambda value: getattr(value, "key", None),
)
def test_unbounded_profit_or_loss_is_detected_from_the_slope(strategy, best, worst):
    if best is not None:
        assert st.max_profit(strategy, MKT) == best
        assert math.isfinite(st.max_loss(strategy, MKT))
    if worst is not None:
        assert st.max_loss(strategy, MKT) == worst
        assert math.isfinite(st.max_profit(strategy, MKT))


def test_bounded_extremes_reached_at_zero_spot():
    premium = bs_price("put", 100, T)
    assert st.max_profit(st.long_put(100, T), MKT) == pytest.approx(100 - premium)
    assert st.max_loss(st.short_put(100, T), MKT) == pytest.approx(-(100 - premium))
    assert st.max_profit(st.short_put(100, T), MKT) == pytest.approx(premium)

    ratio = st.put_ratio_spread(90, 100, T)
    cost = st.net_premium(ratio, MKT)
    assert st.max_profit(ratio, MKT) == pytest.approx(10 - cost)
    assert st.max_loss(ratio, MKT) == pytest.approx(-80 - cost)  # 100 - 2 * 90 at S = 0
    back = st.put_backspread(90, 100, T)
    assert st.max_profit(back, MKT) == pytest.approx(80 - st.net_premium(back, MKT))

    covered = st.covered_call(105, T)
    entry = st.net_premium(covered, MKT)
    assert st.max_profit(covered, MKT) == pytest.approx(105 - entry)
    assert st.max_loss(covered, MKT) == pytest.approx(-entry)
    collar = st.collar(95, 105, T)
    entry = st.net_premium(collar, MKT)
    assert st.max_profit(collar, MKT) == pytest.approx(105 - entry)
    assert st.max_loss(collar, MKT) == pytest.approx(95 - entry)


def test_box_spread_has_no_breakeven_and_a_locked_pnl():
    box = st.box_spread(95, 105, T)
    locked = 10 - 10 * math.exp(-0.02 * T)
    assert st.breakevens(box, MKT) == []
    assert st.max_profit(box, MKT) == pytest.approx(locked)
    assert st.max_loss(box, MKT) == pytest.approx(locked)
    assert st.probability_of_profit(box, MKT) == 1.0


def test_breakeven_beyond_the_search_grid_is_found_from_the_slope():
    # +1 C100 -1.001 C102: above 102 the P&L falls by 0.001 per point -> root far away
    custom = st.Strategy("skinny ratio", (EuropeanOption("call", 100, T) * 1, EuropeanOption("call", 102, T) * -1.001))
    cost = st.net_premium(custom, MKT)
    roots = st.breakevens(custom, MKT)
    far_root = (102 * 1.001 - 100 - cost) / 0.001
    assert far_root > 3 * 102
    assert roots == pytest.approx([100 + cost, far_root], rel=1e-9)
    assert st.max_loss(custom, MKT) == -math.inf


@dataclass(frozen=True)
class LogContract(Instrument):
    """Pays ln(S_T): minus infinity at S = 0, which the analytics must survive."""

    expiry: float

    def price(self, mkt):
        return np.log(mkt.spot)

    def payoff(self, spot_T):
        return np.log(np.asarray(spot_T, dtype=float))


def test_analytics_survive_a_payoff_that_blows_up_at_zero_spot():
    contract = LogContract(T)
    assert st.breakevens(contract, entry_cost=math.log(100.0)) == pytest.approx([100.0])
    assert st.max_profit(contract, entry_cost=math.log(100.0)) == math.inf
    assert math.isfinite(st.max_loss(contract, entry_cost=math.log(100.0)))


def test_flat_zero_stretch_returns_its_ends():
    reversal = st.risk_reversal(90, 110, T)
    assert st.breakevens(reversal, entry_cost=0.0) == pytest.approx([90, 110])
    # zero below the low strike down to S = 0: only the interior end is a breakeven
    assert st.breakevens(st.bull_call_spread(95, 105, T), entry_cost=0.0) == pytest.approx([95])


def test_probability_of_profit_matches_the_lognormal_formula():
    call = st.long_call(100, T)
    breakeven = 100 + st.net_premium(call, MKT)
    _, d2 = bs.d1_d2(100.0, breakeven, T, 0.2, 0.02, 0.01)
    expected = float(ndtr(d2))
    assert st.probability_of_profit(call, MKT) == pytest.approx(expected, abs=1e-9)
    assert st.probability_of_profit(st.short_call(100, T), MKT) == pytest.approx(1 - expected, abs=1e-9)
    long_pop = st.probability_of_profit(st.long_straddle(100, T), MKT)
    short_pop = st.probability_of_profit(st.short_straddle(100, T), MKT)
    assert long_pop + short_pop == pytest.approx(1.0)
    assert short_pop > 0.5  # the seller wins more often... and loses bigger
    zero_vol = MKT.bumped(vol=0.0)
    assert st.probability_of_profit(st.long_call(90, T), zero_vol) in (0.0, 1.0)
    with pytest.raises(ValueError, match="expiry"):
        st.probability_of_profit(Underlying(), MKT)


# ---------------------------------------------------------------------- #
# Calendars: analysed at the front expiry
# ---------------------------------------------------------------------- #
def test_calendar_value_at_front_expiry_prices_the_back_leg():
    calendar = st.calendar_spread(100, 0.25, 1.0)
    assert calendar.is_multi_expiry and not st.long_call(100, T).is_multi_expiry
    assert st.analysis_horizon(calendar) == 0.25
    with pytest.raises(ValueError, match="calendar"):
        st.payoff_at_expiry(calendar, SPOTS)
    expected = bs_price("call", 100, 0.75, spot=SPOTS) - np.maximum(SPOTS - 100, 0)
    assert st.payoff_at_expiry(calendar, SPOTS, MKT) == pytest.approx(expected, abs=1e-12)

    put_calendar = st.calendar_spread(100, 0.25, 1.0, option_type="PUT")
    expected = bs_price("put", 100, 0.75, spot=SPOTS) - np.maximum(100 - SPOTS, 0)
    assert st.payoff_at_expiry(put_calendar, SPOTS, MKT) == pytest.approx(expected, abs=1e-12)


def test_calendar_tent_breakevens_and_extremes():
    calendar = st.calendar_spread(100, 0.25, 1.0)
    debit = st.net_premium(calendar, FLAT)
    assert debit > 0
    low, high = st.breakevens(calendar, FLAT)
    assert low < 100 < high
    assert st.pnl_at_expiry(calendar, np.array([low, high]), FLAT) == pytest.approx([0, 0], abs=1e-8)
    # with zero rates the tent peaks exactly at the strike, and the worst case is the debit
    peak = bs.price(100.0, 100.0, 0.75, 0.2) - debit
    assert st.max_profit(calendar, FLAT) == pytest.approx(peak, abs=1e-8)
    assert st.max_loss(calendar, FLAT) == pytest.approx(-debit, abs=1e-8)
    # a dividend yield makes the deep in-the-money side drift down forever (documented)
    assert st.max_loss(calendar, FLAT.bumped(div=0.03)) == -math.inf


def test_diagonal_and_double_calendar_build_the_expected_legs():
    diagonal = st.diagonal_spread(105, 95, 0.25, 1.0)
    assert [(leg.instrument.strike, leg.expiry, leg.quantity) for leg in diagonal.legs] == [
        (105, 0.25, -1),
        (95, 1.0, 1),
    ]
    double = st.double_calendar(95, 105, 0.25, 1.0)
    assert [leg.instrument.option_type for leg in double.legs] == ["put", "put", "call", "call"]
    assert [leg.quantity for leg in double.legs] == [-1, 1, -1, 1]
    single_sum = st.calendar_spread(95, 0.25, 1.0, "put").price(MKT) + st.calendar_spread(105, 0.25, 1.0).price(MKT)
    assert double.price(MKT) == pytest.approx(single_sum)
    assert len(st.breakevens(double, FLAT)) == 2


def test_horizon_moves_to_the_next_expiry_once_the_front_leg_is_gone():
    calendar = st.calendar_spread(100, 0.25, 1.0)
    assert st.analysis_horizon(calendar, MKT) == 0.25
    assert st.analysis_horizon(calendar, MKT.bumped(t=0.25)) == 1.0
    assert st.analysis_horizon(calendar, MKT.bumped(t=2.0)) == 1.0
    assert st.analysis_horizon(Underlying(), MKT) is None
    assert st.summary(calendar, MKT.bumped(t=0.5))["horizon"] == 1.0


# ---------------------------------------------------------------------- #
# Registry
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(st.STRATEGY_REGISTRY))
def test_every_spec_builds_round_trips_and_prices(name):
    spec = st.STRATEGY_REGISTRY[name]
    strategy = spec.build_default(spot=123.4, expiry=0.75)
    assert isinstance(strategy, st.Strategy)
    assert strategy.key == name and strategy.view == spec.view
    assert strategy.description == spec.description and len(spec.description) > 80

    clone = instrument_from_dict(json.loads(json.dumps(strategy.to_dict())))
    assert clone == strategy and isinstance(clone, st.Strategy) and hash(clone) == hash(strategy)

    mkt = Market(spot=123.4, vol=0.25, rate=0.03, div=0.01)
    greeks = strategy.greeks(mkt)
    assert all(math.isfinite(greeks[key]) for key in GREEK_KEYS)
    report = st.summary(strategy, mkt)
    assert math.isfinite(report["net_premium"])
    assert not math.isnan(report["max_profit"]) and not math.isnan(report["max_loss"])
    assert report["max_loss"] <= report["max_profit"]
    assert all(math.isfinite(b) and b >= 0 for b in report["breakevens"])


@pytest.mark.parametrize("name", list(st.STRATEGY_REGISTRY))
def test_schema_matches_the_factory_signature(name):
    spec = st.STRATEGY_REGISTRY[name]
    assert spec.name == name and spec.factory is getattr(st, name)
    assert list(spec.schema) == list(inspect.signature(spec.factory).parameters)
    assert all(p.kind in st.PARAM_KINDS for p in spec.params)
    assert set(spec.view) <= set(st.VIEW_TAGS) and spec.view
    assert spec.category and spec.title
    strikes = [p for p in spec.params if p.kind == "strike"]
    assert strikes and all(0.5 < p.default < 1.5 for p in strikes)


def test_default_params_are_relative_to_spot_and_clock():
    spec = st.get_strategy_spec("calendar_spread")
    params = spec.default_params(spot=200.0, expiry=1.5, t=1.0)
    assert params == {
        "strike": 200.0,
        "near_expiry": 1.5,
        "far_expiry": 2.0,
        "option_type": "call",
        "quantity": 1.0,
    }
    collar = st.STRATEGY_REGISTRY["collar"].build_default(200.0, 0.5, call_strike=230.0)
    assert [getattr(leg.instrument, "strike", None) for leg in collar.legs] == [None, 190.0, 230.0]
    with pytest.raises(ValueError, match="unknown parameter"):
        spec.build_default(100.0, 1.0, strikee=90)
    with pytest.raises(ValueError, match="later than"):
        spec.default_params(100.0, expiry=1.0, t=1.0)
    with pytest.raises(ValueError, match="spot"):
        spec.default_params(-1.0)


def test_list_and_build_by_name():
    names = st.list_strategies()
    assert names == list(st.STRATEGY_REGISTRY) and len(names) >= 35
    assert "iron_condor" in st.list_strategies(category="butterfly & condor")
    assert set(st.list_strategies(view="long vol")) >= {"long_straddle", "calendar_spread", "call_backspread"}
    assert "long_straddle" not in st.list_strategies(view="short vol")
    built = st.build_strategy("Bull Call-Spread", low_strike=95, high_strike=105, expiry=T)
    assert built == st.bull_call_spread(95, 105, T)
    with pytest.raises(ValueError, match="Unknown strategy"):
        st.build_strategy("bull_cal_spread", low_strike=95, high_strike=105, expiry=T)
    with pytest.raises(ValueError, match="unknown parameter"):
        st.build_strategy("long_call", strike=100, expiry=T, colour="red")
    with pytest.raises(ValueError, match=r"missing parameter\(s\) \['expiry'\]"):
        st.build_strategy("long_call", strike=100)


# ---------------------------------------------------------------------- #
# Validation
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "build",
    [
        lambda: st.bull_call_spread(105, 95, T),
        lambda: st.bear_put_spread(100, 100, T),
        lambda: st.collar(105, 95, T),
        lambda: st.long_strangle(110, 90, T),
        lambda: st.long_call_butterfly(90, 110, 100, T),
        lambda: st.iron_butterfly(100, 100, 110, T),
        lambda: st.iron_condor(85, 105, 95, 115, T),
        lambda: st.long_call_condor(85, 95, 105, 105, T),
        lambda: st.risk_reversal(110, 90, T),
        lambda: st.seagull(90, 110, 100, T),
        lambda: st.jade_lizard(90, 120, 110, T),
        lambda: st.box_spread(110, 90, T),
        lambda: st.double_calendar(105, 95, 0.25, 1.0),
    ],
)
def test_strike_ordering_is_validated(build):
    with pytest.raises(ValueError, match="strictly increasing"):
        build()


def test_other_validation_errors():
    with pytest.raises(ValueError, match="earlier than"):
        st.calendar_spread(100, 1.0, 0.25)
    with pytest.raises(ValueError, match="earlier than"):
        st.diagonal_spread(105, 95, 0.5, 0.5)
    with pytest.raises(ValueError, match="option_type|call|put"):
        st.calendar_spread(100, 0.25, 1.0, option_type="straddle")
    with pytest.raises(ValueError, match="ratio"):
        st.call_ratio_spread(100, 110, T, ratio=1.0)
    with pytest.raises(ValueError, match="quantity"):
        st.long_call(100, T, quantity=0)
    with pytest.raises(ValueError, match="quantity"):
        st.long_call(100, T, quantity=-1)
    with pytest.raises(ValueError, match="strike"):
        st.long_put(-5, T)
    with pytest.raises(ValueError, match="strike"):
        st.long_straddle("abc", T)
    with pytest.raises(ValueError, match="scalar Market"):
        st.net_premium(st.long_call(100, T), MKT.bumped(spot=np.array([90.0, 100.0])))
    with pytest.raises(ValueError, match="Instrument or a Position"):
        st.breakevens("long_call", MKT)


# ---------------------------------------------------------------------- #
# Strategy as an Instrument
# ---------------------------------------------------------------------- #
def test_strategy_is_a_frozen_hashable_composite():
    condor = st.iron_condor(85, 95, 105, 115, T)
    assert isinstance(condor, CompositeInstrument)
    assert condor.label == condor.name == "Iron condor 85/95/105/115"
    assert {condor: 1}[st.iron_condor(85, 95, 105, 115, T)] == 1
    with pytest.raises(AttributeError):
        condor.name = "other"
    assert condor.observe(101.0, 0.1) is condor
    doubled = condor.scaled(2)
    assert isinstance(doubled, st.Strategy) and doubled.view == condor.view
    assert doubled.price(MKT) == pytest.approx(2 * condor.price(MKT))
    book_line = Position(condor, -3)
    assert book_line.price(MKT) == pytest.approx(-3 * condor.price(MKT))
    nested = CompositeInstrument("hedged", (condor * 1, Underlying() * 0.5))
    assert len(nested.flatten()) == 5


def test_custom_strategy_metadata_and_validation():
    custom = st.Strategy("my trade", (EuropeanOption("call", 100, T),), description="test", view="bullish")
    assert custom.view == ("bullish",) and custom.key == ""
    assert instrument_from_dict(custom.to_dict()) == custom
    with pytest.raises(ValueError, match="view"):
        st.Strategy("bad", (EuropeanOption("call", 100, T),), view=(1, 2))
    with pytest.raises(ValueError, match="strings"):
        st.Strategy("bad", (EuropeanOption("call", 100, T),), description=None)


def test_monte_carlo_agrees_with_the_closed_form():
    spread = st.bull_call_spread(95, 105, 0.5)
    price, stderr = mc_price(spread, MKT, n_paths=40_000, n_steps=1, seed=7)
    assert abs(price - spread.price(MKT)) < 4 * stderr + 1e-3


# ---------------------------------------------------------------------- #
# Summary and tables
# ---------------------------------------------------------------------- #
def test_summary_contents():
    condor = st.iron_condor(85, 95, 105, 115, T)
    report = st.summary(condor, MKT)
    assert report["name"] == condor.name and report["key"] == "iron_condor"
    assert report["premium_type"] == "credit" and report["net_premium"] < 0
    assert report["options_premium"] == pytest.approx(report["net_premium"])
    assert report["horizon"] == T and report["is_multi_expiry"] is False
    assert len(report["breakevens"]) == 2
    assert 0 < report["probability_of_profit"] < 1
    assert tuple(report["greeks"]) == GREEK_KEYS
    assert report["greeks_trader"]["vega"] == pytest.approx(report["greeks"]["vega"] / 100)
    assert report["greeks_trader"]["theta"] == pytest.approx(report["greeks"]["theta"] / 365)
    assert len(report["legs"]) == 4
    assert sum(leg["value"] for leg in report["legs"]) == pytest.approx(report["net_premium"])
    assert sum(leg["delta"] for leg in report["legs"]) == pytest.approx(report["greeks"]["delta"])
    json.dumps(report)  # JSON-friendly (inf allowed by the json module)

    assert st.summary(st.long_straddle(100, T), MKT)["premium_type"] == "debit"
    calendar = st.summary(st.calendar_spread(100, 0.25, 1.0), MKT)
    assert calendar["is_multi_expiry"] is True and calendar["horizon"] == 0.25
    stock = st.summary(Underlying(), MKT)
    assert stock["horizon"] is None and stock["probability_of_profit"] is None
    assert stock["max_profit"] == math.inf and stock["max_loss"] == pytest.approx(-100)
    assert stock["breakevens"] == pytest.approx([100])


def test_legs_table():
    covered = st.covered_call(105, T, quantity=2)
    table = st.legs_table(covered, MKT)
    assert isinstance(table, pd.DataFrame) and len(table) == 2
    assert list(table["side"]) == ["long", "short"] and list(table["quantity"]) == [2, -2]
    assert list(table["instrument"]) == ["Underlying", "EuropeanOption"]
    assert table["value"].sum() == pytest.approx(covered.price(MKT))
    assert table["vega"].sum() == pytest.approx(covered.greeks(MKT)["vega"])
    trader = st.legs_table(covered, MKT, trader_units=True)
    assert trader["vega"].sum() == pytest.approx(covered.greeks(MKT)["vega"] / 100)
    assert trader["theta"].sum() == pytest.approx(covered.greeks(MKT)["theta"] / 365)
    assert trader["delta"].sum() == pytest.approx(table["delta"].sum())


def test_spot_grid_and_reference_levels():
    condor = st.iron_condor(85, 95, 105, 115, T)
    assert st.reference_levels(condor) == [85, 95, 105, 115]
    assert st.reference_levels(condor, MKT) == [85, 95, 100, 105, 115]
    grid = st.spot_grid(condor, MKT, n_points=50)
    assert np.all(np.diff(grid) > 0) and grid[0] > 0
    assert set([85.0, 95.0, 100.0, 105.0, 115.0]) <= set(grid.tolist())
    assert grid[0] < 85 * 0.76 and grid[-1] > 115 * 1.24
    custom = st.spot_grid(condor, MKT, n_points=11, spot_range=(90, 110))
    assert custom[0] == 90 and custom[-1] == 110 and 85.0 not in custom
    with pytest.raises(ValueError, match="spot_range"):
        st.spot_grid(condor, MKT, spot_range=(110, 90))
    with pytest.raises(ValueError, match="n_points"):
        st.spot_grid(condor, MKT, n_points=1)


# ---------------------------------------------------------------------- #
# Figures
# ---------------------------------------------------------------------- #
def names(fig: go.Figure) -> list[str]:
    return [trace.name for trace in fig.data]


def test_payoff_diagram_traces_and_annotations():
    condor = st.iron_condor(85, 95, 105, 115, T)
    fig = sp.payoff_diagram(condor, MKT)
    assert isinstance(fig, go.Figure)
    assert names(fig) == [
        "Profit zone",
        "Loss zone",
        "+1 x P 85 T=1.00",
        "-1 x P 95 T=1.00",
        "-1 x C 105 T=1.00",
        "+1 x C 115 T=1.00",
        "In 182d",
        "Today",
        "At expiry",
        "Breakeven",
    ]
    by_name = {trace.name: trace for trace in fig.data}
    x = np.asarray(by_name["At expiry"].x)
    assert np.asarray(by_name["At expiry"].y) == pytest.approx(st.pnl_at_expiry(condor, x, MKT))
    assert by_name["At expiry"].line.width == 3
    leg_sum = sum(np.asarray(fig.data[i].y) for i in range(2, 6))
    assert leg_sum == pytest.approx(np.asarray(by_name["At expiry"].y), abs=1e-10)
    assert list(by_name["Breakeven"].x) == pytest.approx(st.breakevens(condor, MKT))
    assert set(by_name["Breakeven"].x) <= set(x)  # shading corners are exact
    assert np.asarray(by_name["Profit zone"].y).min() == 0 and np.asarray(by_name["Loss zone"].y).max() == 0
    assert len({trace.line.dash for trace in fig.data[2:6]}) == 4  # legs differ by dash, not colour alone

    assert "Iron condor" in fig.layout.title.text and "at expiry" in fig.layout.title.text
    subtitle = fig.layout.title.subtitle.text
    assert "Net credit" in subtitle and "Max profit" in subtitle and "Breakeven" in subtitle
    notes = [a.text for a in fig.layout.annotations]
    assert any(n.startswith("spot") for n in notes)
    assert any(n.startswith("max profit") for n in notes) and any(n.startswith("max loss") for n in notes)
    assert fig.layout.hovermode == "x unified"
    # labels sit on the free side of the curve; the y axis is focused on the package, not the legs
    assert list(by_name["Breakeven"].textposition) == ["top left", "top right"]
    low, high = fig.layout.yaxis.range
    best, worst = st.max_profit(condor, MKT), st.max_loss(condor, MKT)
    assert worst - 10.5 < low < worst and best < high < best + 10.5  # at most one span (10) beyond
    assert min(np.min(t.y) for t in fig.data[2:6]) < low  # single legs are clipped context
    fig.to_json()


def test_payoff_diagram_options_and_unbounded_strategies():
    call = st.long_call(100, T)
    fig = sp.payoff_diagram(call, MKT, show_legs=False, show_today=False, horizons=())
    assert names(fig) == ["Profit zone", "Loss zone", "At expiry", "Breakeven"]
    assert "Max profit unlimited" in fig.layout.title.subtitle.text
    notes = [a.text for a in fig.layout.annotations]
    assert not any(n.startswith("max profit") for n in notes)
    assert any(n.startswith("max loss") for n in notes)

    low, high = fig.layout.yaxis.range  # no legs drawn: the package with a little padding
    y = np.asarray(fig.data[2].y)
    assert low == pytest.approx(y.min() - 0.12 * np.ptp(y)) and high == pytest.approx(y.max() + 0.12 * np.ptp(y))

    fig = sp.payoff_diagram(call, MKT, horizons=(0.25, 0.75), spot_range=(80, 120), n_points=41)
    assert names(fig).count("Today") == 1 and sum(n.startswith("In ") for n in names(fig)) == 2
    x = np.asarray(fig.data[-2].x)
    assert x.min() == 80 and x.max() == 120

    at_expiry = sp.payoff_diagram(call, MKT.bumped(t=T))
    assert "Today" not in names(at_expiry)
    with pytest.raises(ValueError, match="fractions"):
        sp.payoff_diagram(call, MKT, horizons=(1.5,))
    with pytest.raises(ValueError, match="scalar Market"):
        sp.payoff_diagram(call, MKT.bumped(vol=np.array([0.1, 0.2])))
    with pytest.raises(ValueError, match="Instrument"):
        sp.payoff_diagram("long_call", MKT)


def test_payoff_diagram_of_a_calendar_uses_the_front_expiry():
    calendar = st.calendar_spread(100, 0.25, 1.0)
    fig = sp.payoff_diagram(calendar, MKT)
    assert "front expiry T=0.25" in fig.layout.title.text and "20% vol" in fig.layout.title.text
    by_name = {trace.name: trace for trace in fig.data}
    x = np.asarray(by_name["At expiry"].x)
    assert np.asarray(by_name["At expiry"].y) == pytest.approx(st.pnl_at_expiry(calendar, x, MKT))
    legs = [t for t in fig.data if t.name in ("-1 x C 100 T=0.25", "+1 x C 100 T=1.00")]
    assert len(legs) == 2
    assert sum(np.asarray(t.y) for t in legs) == pytest.approx(np.asarray(by_name["At expiry"].y), abs=1e-10)


@pytest.mark.parametrize("legs, per_panel", [("overlay", 4 + 1), ("stack", 2 * 4 + 1), ("none", 1)])
def test_greeks_dashboard_layouts(legs, per_panel):
    condor = st.iron_condor(85, 95, 105, 115, T)
    fig = sp.strategy_greeks_dashboard(condor, MKT, legs=legs)
    assert len(fig.data) == 4 * per_panel
    totals = [trace for trace in fig.data if trace.name == "Total"]
    assert len(totals) == 4 and sum(bool(t.showlegend) for t in totals) == 1
    x = np.asarray(totals[2].x)
    vega = condor.greeks(MKT.bumped(spot=x))["vega"] / 100  # third panel, trader units
    assert np.asarray(totals[2].y) == pytest.approx(vega)
    titles = [a.text for a in fig.layout.annotations]
    assert "Vega - per 1 vol point (0.01)" in titles and "Theta - per calendar day (1/365 year)" in titles
    assert "trader units" in fig.layout.title.text and "spot 100.00" in fig.layout.title.subtitle.text
    assert len(fig.layout.shapes) == 4  # one spot line per panel
    fig.to_json()


def test_greeks_dashboard_leg_contributions_add_up():
    fly = st.long_call_butterfly(90, 100, 110, T)
    fig = sp.strategy_greeks_dashboard(fly, MKT, greeks=("gamma",), trader_units=False)
    assert "raw units" in fig.layout.title.text
    *legs, total = fig.data
    assert len(legs) == 3
    assert sum(np.asarray(t.y) for t in legs) == pytest.approx(np.asarray(total.y), abs=1e-12)

    stacked = sp.strategy_greeks_dashboard(fly, MKT, greeks=("gamma",), legs="stack")
    parts = [t for t in stacked.data if t.name != "Total"]
    assert sum(np.asarray(t.y) for t in parts) == pytest.approx(np.asarray(stacked.data[-1].y), abs=1e-12)
    assert {t.stackgroup for t in parts} == {"pos-0", "neg-0"}

    with pytest.raises(ValueError, match="unknown Greek"):
        sp.strategy_greeks_dashboard(fly, MKT, greeks=("delta", "lambda"))
    with pytest.raises(ValueError, match="legs must be"):
        sp.strategy_greeks_dashboard(fly, MKT, legs="pile")


def test_more_than_six_legs_are_folded_not_recoloured():
    legs = tuple(EuropeanOption("call", 80 + 5 * i, T) * (1 if i % 2 else -1) for i in range(8))
    ladder = st.Strategy("eight legs", legs)
    fig = sp.payoff_diagram(ladder, MKT, show_today=False, horizons=())
    leg_traces = fig.data[2:-2]
    assert len(leg_traces) == 6 and leg_traces[-1].name == "Other legs (3)"
    colors = [t.line.color for t in leg_traces]
    assert len(set(colors)) == 6 and set(colors[:5]) <= set(PALETTE)
    assert sum(np.asarray(t.y) for t in leg_traces) == pytest.approx(np.asarray(fig.data[-2].y), abs=1e-10)


def test_time_decay_collapses_onto_the_payoff():
    fly = st.long_call_butterfly(90, 100, 110, T)
    fig = sp.strategy_time_decay(fly, MKT, taus=(1.0, 0.25, 1e-6), as_pnl=False)
    assert names(fig) == ["1.00y to expiry", "91d to expiry", "0d to expiry", "At expiry"]
    # one second before expiry only the at-the-money kink still has time value (~0.016)
    assert np.asarray(fig.data[2].y) == pytest.approx(np.asarray(fig.data[3].y), abs=0.05)
    assert np.asarray(fig.data[3].y) == pytest.approx(fly.payoff(np.asarray(fig.data[3].x)))

    default = sp.strategy_time_decay(fly, MKT)
    assert len(default.data) == 5 and default.layout.yaxis.title.text == "Profit / loss"
    assert len({t.line.color for t in default.data}) == 5  # ordered ramp + ink
    with pytest.raises(ValueError, match="positive"):
        sp.strategy_time_decay(fly, MKT, taus=(0.5, 0.0))
    with pytest.raises(ValueError, match="already at expiry"):
        sp.strategy_time_decay(fly, MKT.bumped(t=T))
    with pytest.raises(ValueError, match="expiry"):
        sp.strategy_time_decay(Underlying(), MKT)


def test_vol_sensitivity_orders_long_vega_curves():
    straddle = st.long_straddle(100, T)
    fig = sp.strategy_vol_sensitivity(straddle, MKT)
    assert names(fig) == ["vol 10.0%", "vol 15.0%", "vol 20.0% (market)", "vol 25.0%", "vol 30.0%", "At expiry"]
    curves = np.array([t.y for t in fig.data[:5]])
    assert np.all(np.diff(curves, axis=0) > 0)  # long vega: more vol, more value, everywhere
    assert fig.data[2].line.width == 3
    index = int(np.argmin(np.abs(np.asarray(fig.data[2].x) - 100.0)))
    assert fig.data[2].y[index] == pytest.approx(0.0, abs=1e-12)  # P&L is zero at today's spot and vol

    custom = sp.strategy_vol_sensitivity(straddle, MKT, vols=(0.3, 0.1), as_pnl=False)
    assert names(custom) == ["vol 10.0%", "vol 30.0%", "At expiry"]
    assert custom.layout.yaxis.title.text == "Value"
    with pytest.raises(ValueError, match="non-negative"):
        sp.strategy_vol_sensitivity(straddle, MKT, vols=(-0.1, 0.2))


def test_compare_strategies():
    candidates = [st.long_call(100, T), st.bull_call_spread(100, 110, T), st.calendar_spread(100, 0.25, T)]
    fig = sp.compare_strategies(candidates, MKT)
    assert names(fig) == [s.label for s in candidates]
    assert [t.line.color for t in fig.data] == PALETTE[:3]
    x = np.asarray(fig.data[1].x)
    assert {100.0, 110.0} <= set(x.tolist())
    assert np.asarray(fig.data[1].y) == pytest.approx(st.pnl_at_expiry(candidates[1], x, MKT))
    assert np.asarray(fig.data[2].y) == pytest.approx(st.pnl_at_expiry(candidates[2], x, MKT))

    vega = sp.compare_strategies(candidates, MKT, what="vega", trader_units=False)
    assert np.asarray(vega.data[0].y) == pytest.approx(candidates[0].greeks(MKT.bumped(spot=x))["vega"])
    assert "Vega" in vega.layout.yaxis.title.text
    today = sp.compare_strategies(candidates[:1] * 2, MKT, what="pnl_today")
    assert names(today) == ["Long call 100 #1", "Long call 100 #2"]
    assert sp.compare_strategies(candidates, MKT, what="price").layout.yaxis.title.text == "Value"

    with pytest.raises(ValueError, match="what must be"):
        sp.compare_strategies(candidates, MKT, what="pnl")
    with pytest.raises(ValueError, match="between 1 and 8"):
        sp.compare_strategies([st.long_call(90 + i, T) for i in range(9)], MKT)
    with pytest.raises(ValueError, match="between 1 and 8"):
        sp.compare_strategies([], MKT)


@pytest.mark.parametrize("name", list(st.STRATEGY_REGISTRY))
def test_every_strategy_draws_in_every_mode(name):
    strategy = st.STRATEGY_REGISTRY[name].build_default(100.0, 0.5)
    for mode in ("auto", "light", "dark"):
        assert isinstance(sp.payoff_diagram(strategy, MKT, mode=mode), go.Figure)
    assert len(sp.strategy_greeks_dashboard(strategy, MKT, greeks=("delta", "vega")).data) > 0
    sp.strategy_time_decay(strategy, MKT).to_json()
    sp.strategy_vol_sensitivity(strategy, MKT).to_json()
