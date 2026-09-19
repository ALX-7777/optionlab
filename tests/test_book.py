"""Tests for optionlab.book and optionlab.plotting.book_plots."""

import json
import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from optionlab import black_scholes as bs
from optionlab.book import (
    DEFAULT_SCENARIOS,
    DEFAULT_SPOT_SHOCKS,
    DEFAULT_VOL_SHOCKS,
    MIN_VOL,
    Book,
    Settlement,
    Trade,
    TransactionCosts,
    shocked_market,
)
from optionlab.instruments import (
    GREEK_KEYS,
    CompositeInstrument,
    EuropeanOption,
    Instrument,
    Position,
    Underlying,
    register_instrument,
)
from optionlab.market import Market
from optionlab.plotting import book_plots

pytestmark = pytest.mark.filterwarnings("error")


# ---------------------------------------------------------------------- #
# Dummy instruments (unique names so they never clash in the registry)
# ---------------------------------------------------------------------- #
@register_instrument
@dataclass(frozen=True)
class BookTestKnockOutCall(Instrument):
    """Toy path-dependent product: a call that dies once spot >= barrier is observed."""

    strike: float
    expiry: float
    barrier: float
    hit: bool = False

    def price(self, mkt):
        alive = 0.0 if self.hit else 1.0
        return alive * bs.price(mkt.spot, self.strike, self.expiry - mkt.t, mkt.vol, mkt.rate, mkt.div)

    def payoff(self, spot_T):
        alive = 0.0 if self.hit else 1.0
        return alive * np.maximum(np.asarray(spot_T, dtype=float) - self.strike, 0.0)

    def observe(self, spot, t):
        if self.hit or spot < self.barrier:
            return self
        return replace(self, hit=True)

    @property
    def label(self):
        return f"KO call {self.strike:g}/{self.barrier:g}"


@register_instrument
@dataclass(frozen=True)
class BookTestScalarOnlyCall(Instrument):
    """A vanilla call whose pricer REFUSES array markets (forces the loop fallback)."""

    strike: float
    expiry: float

    def price(self, mkt):
        if not mkt.is_scalar:
            raise ValueError("scalar markets only")
        tau = self.expiry - mkt.t
        if tau <= 0:
            return max(mkt.spot - self.strike, 0.0)
        return bs.price(mkt.spot, self.strike, tau, mkt.vol, mkt.rate, mkt.div)

    def payoff(self, spot_T):
        return np.maximum(np.asarray(spot_T, dtype=float) - self.strike, 0.0)


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #
@pytest.fixture
def mkt():
    return Market(spot=100.0, vol=0.2, rate=0.03, div=0.01)


@pytest.fixture
def call():
    return EuropeanOption("call", 100.0, 1.0)


@pytest.fixture
def put():
    return EuropeanOption("put", 95.0, 0.5)


@pytest.fixture
def book(mkt, call, put):
    """Long 10 calls, short 5 puts, long 3 shares, 1000 of starting cash."""
    b = Book("test book", cash=1_000.0)
    b.trade(call, 10, mkt)
    b.trade(put, -5, mkt)
    b.trade(Underlying(), 3, mkt)
    return b


# ---------------------------------------------------------------------- #
# Cash accounting and trading
# ---------------------------------------------------------------------- #
def test_new_book_is_empty_and_worth_its_cash(mkt):
    b = Book(cash=250.0)
    assert b.is_empty and len(b) == 0 and b.positions == ()
    assert b.value(mkt) == 250.0
    assert b.pnl(mkt) == 0.0
    assert b.positions_value(mkt) == 0.0
    assert all(v == 0.0 for v in b.greeks(mkt).values())
    assert list(b.greeks(mkt)) == list(GREEK_KEYS)


def test_trade_at_model_price_moves_cash_not_value(mkt, call):
    b = Book(cash=1_000.0)
    trade = b.trade(call, 10, mkt, note="open")
    model = call.price(mkt)
    assert isinstance(trade, Trade)
    assert trade.price == pytest.approx(model)
    assert trade.t == mkt.t and trade.note == "open" and trade.fees == 0.0
    assert trade.cash_flow == pytest.approx(-10 * model)
    assert b.cash == pytest.approx(1_000.0 - 10 * model)
    assert b.positions_value(mkt) == pytest.approx(10 * model)
    assert b.value(mkt) == pytest.approx(1_000.0)
    assert b.pnl(mkt) == pytest.approx(0.0, abs=1e-10)


def test_buy_then_sell_same_price_costs_only_the_fees(mkt, call):
    b = Book(cash=500.0)
    b.trade(call, 4, mkt, price=9.0, fee=1.5)
    b.trade(call, -4, mkt, price=9.0, fee=1.5)
    assert b.cash == pytest.approx(500.0 - 3.0)
    assert b.is_empty
    assert b.pnl(mkt) == pytest.approx(-3.0)
    assert len(b.trades) == 2


def test_user_price_creates_immediate_mark_to_model_pnl(mkt, call):
    b = Book()
    model = call.price(mkt)
    b.trade(call, 2, mkt, price=model - 0.5)  # bought 0.5 cheap
    assert b.pnl(mkt) == pytest.approx(1.0)


def test_short_sale_credits_cash(mkt, put):
    b = Book()
    b.trade(put, -5, mkt)
    assert b.cash == pytest.approx(5 * put.price(mkt))
    assert b.quantity(put) == -5.0
    assert b.value(mkt) == pytest.approx(0.0, abs=1e-12)


def test_netting_and_removal_of_flat_lines(mkt, call, put):
    b = Book()
    b.trade(call, 3, mkt)
    b.trade(call, 2, mkt)
    assert len(b) == 1 and b.quantity(call) == 5.0
    b.trade(EuropeanOption("CALL", 100, 1), -5, mkt)  # equal instrument, different spelling
    assert call not in b and b.is_empty
    b.trade(put, 0.1, mkt)
    b.trade(put, 0.2, mkt)
    b.trade(put, -0.3, mkt)  # 0.1 + 0.2 - 0.3 != 0 in floating point
    assert b.is_empty


def test_trade_accepts_a_position(mkt, call):
    b = Book()
    b.trade(Position(call, -2), 3, mkt)
    assert b.quantity(call) == -6.0


def test_trade_validation(mkt, call):
    b = Book()
    with pytest.raises(ValueError, match="non-zero"):
        b.trade(call, 0, mkt)
    with pytest.raises(ValueError, match="finite"):
        b.trade(call, math.nan, mkt)
    with pytest.raises(ValueError, match="fee"):
        b.trade(call, 1, mkt, fee=-1.0)
    with pytest.raises(ValueError, match="Instrument"):
        b.trade("call", 1, mkt)
    with pytest.raises(ValueError, match="scalar"):
        b.trade(call, 1, Market(spot=np.array([99.0, 101.0]), vol=0.2))
    with pytest.raises(ValueError, match="expired"):
        b.trade(call, 1, mkt.bumped(t=1.0))
    with pytest.raises(ValueError, match="TransactionCosts"):
        Book(costs={"stock_bps": 1.0})
    with pytest.raises(ValueError, match="finite"):
        Book(cash=math.inf)
    assert b.is_empty and b.trades == [] and b.cash == 0.0


def test_close_position_liquidate_and_deposit(book, mkt, call):
    value_before = book.value(mkt)
    trade = book.close_position(call, mkt)
    assert trade.quantity == -10.0 and call not in book
    with pytest.raises(ValueError, match="no open position"):
        book.close_position(call, mkt)
    trades = book.liquidate(mkt)
    assert len(trades) == 2 and book.is_empty
    assert book.cash == pytest.approx(value_before)
    book.deposit(100.0)
    assert book.cash == pytest.approx(value_before + 100.0)
    assert book.pnl(mkt) == pytest.approx(value_before - 1_000.0)


# ---------------------------------------------------------------------- #
# Transaction costs
# ---------------------------------------------------------------------- #
def test_transaction_costs_model(mkt, call):
    costs = TransactionCosts(stock_bps=10.0, option_pct=0.01, option_vol_spread=0.005)
    stock_cost = costs.cost(Underlying(), -50, mkt)
    assert stock_cost == pytest.approx(50 * 100.0 * 10 / 1e4)
    option_cost = costs.cost(call, 10, mkt)
    expected = 10 * (0.01 * call.price(mkt) + 0.005 * call.greeks(mkt)["vega"])
    assert option_cost == pytest.approx(expected)
    # a package pays one spread per leg, whatever the sign of the leg
    spread = CompositeInstrument("spread", (Position(call, 1), Position(EuropeanOption("call", 110, 1.0), -1)))
    legs_cost = costs.cost(call, 2, mkt) + costs.cost(EuropeanOption("call", 110, 1.0), 2, mkt)
    assert costs.cost(spread, 2, mkt) == pytest.approx(legs_cost)
    assert costs.cost(call, 10, mkt, price=20.0) == pytest.approx(
        10 * (0.01 * 20.0 + 0.005 * call.greeks(mkt)["vega"])
    )
    with pytest.raises(ValueError, match="non-negative"):
        TransactionCosts(stock_bps=-1.0)
    assert TransactionCosts.from_dict(costs.to_dict()) == costs
    with pytest.raises(ValueError, match="unknown"):
        TransactionCosts.from_dict({"bps": 1.0})


def test_costs_are_charged_on_every_trade(mkt, call):
    costs = TransactionCosts(stock_bps=5.0, option_pct=0.02)
    b = Book(cash=0.0, costs=costs)
    t1 = b.trade(call, 10, mkt, fee=1.0)
    assert t1.fees == pytest.approx(1.0 + 0.02 * 10 * call.price(mkt))
    t2 = b.hedge_delta(mkt)
    assert t2.fees == pytest.approx(abs(t2.quantity) * 100.0 * 5e-4)
    assert b.pnl(mkt) == pytest.approx(-(t1.fees + t2.fees))


# ---------------------------------------------------------------------- #
# Composites
# ---------------------------------------------------------------------- #
def test_composite_is_flattened_by_default(mkt, call, put):
    straddle = CompositeInstrument("straddle", (Position(call, 1), Position(put, 1)))
    nested = CompositeInstrument("2x straddle + call", (Position(straddle, 2), Position(call, 1)))
    b = Book()
    trade = b.trade(nested, 3, mkt)
    assert trade.instrument == nested  # ONE ticket on the blotter
    assert trade.price == pytest.approx(nested.price(mkt))
    assert b.quantity(call) == 9.0 and b.quantity(put) == 6.0 and len(b) == 2
    assert b.value(mkt) == pytest.approx(0.0, abs=1e-10)
    b.trade(call, -9, mkt)  # legs net against plain option trades
    assert call not in b


def test_composite_can_stay_one_line(mkt, call, put):
    straddle = CompositeInstrument("straddle", (Position(call, 1), Position(put, 1)))
    b = Book()
    b.trade(straddle, 2, mkt, flatten=False)
    assert len(b) == 1 and b.quantity(straddle) == 2.0
    flat = Book()
    flat.trade(straddle, 2, mkt)
    for key, value in b.greeks(mkt).items():
        assert value == pytest.approx(flat.greeks(mkt)[key])


# ---------------------------------------------------------------------- #
# Valuation and Greeks
# ---------------------------------------------------------------------- #
def test_aggregated_greeks_are_quantity_weighted_sums(book, mkt, call, put):
    total = book.greeks(mkt)
    assert list(total) == list(GREEK_KEYS)
    cg, pg, ug = call.greeks(mkt), put.greeks(mkt), Underlying().greeks(mkt)
    for key in GREEK_KEYS:
        assert total[key] == pytest.approx(10 * cg[key] - 5 * pg[key] + 3 * ug[key])
        assert isinstance(total[key], float)
    assert total["price"] == pytest.approx(book.positions_value(mkt))


def test_valuation_is_vectorised(book, mkt, call, put):
    spots = np.linspace(60.0, 140.0, 9)
    arr = mkt.bumped(spot=spots)
    values = book.value(arr)
    assert values.shape == spots.shape
    for s, v in zip(spots, values):
        assert v == pytest.approx(book.value(mkt.bumped(spot=float(s))))
    greeks = book.greeks(arr)
    assert greeks["gamma"].shape == spots.shape
    assert Book(cash=5.0).value(arr).shape == spots.shape


def test_dollar_greeks_definitions(book, mkt):
    raw = book.greeks(mkt)
    dollar = book.dollar_greeks(mkt)
    assert dollar["delta_cash"] == pytest.approx(raw["delta"] * 100.0)
    assert dollar["gamma_cash"] == pytest.approx(raw["gamma"] * 100.0**2 / 100.0)
    assert dollar["vega_cash"] == pytest.approx(raw["vega"] / 100.0)
    assert dollar["theta_cash"] == pytest.approx(raw["theta"] / 365.0)
    assert dollar["rho_cash"] == pytest.approx(raw["rho"] / 100.0)
    # delta_cash / 100 is the P&L of a +1% move, up to the gamma term gamma_cash / 200
    bumped = book.value(mkt.bumped(spot=101.0)) - book.value(mkt)
    assert bumped == pytest.approx(dollar["delta_cash"] / 100 + dollar["gamma_cash"] / 200, rel=2e-3)


def test_breakeven_move_is_implied_daily_vol_for_hedged_vanilla():
    flat = Market(spot=100.0, vol=0.25)  # zero rates: theta = -0.5 * vol^2 * S^2 * gamma
    b = Book()
    b.trade(EuropeanOption("call", 105.0, 0.5), 10, flat)
    b.hedge_delta(flat)
    assert b.breakeven_move(flat) == pytest.approx(0.25 * math.sqrt(1 / 365))
    assert b.breakeven_move(flat, days=4) == pytest.approx(0.25 * math.sqrt(4 / 365))
    assert math.isnan(Book().breakeven_move(flat))  # no gamma, no trade-off


def test_positions_frame(book, mkt, call):
    frame = book.positions_frame(mkt)
    assert frame.attrs["units"] == "trader"
    assert list(frame["label"]) == [call.label, "P 95 T=0.50", "Underlying", "TOTAL"]
    assert list(frame.columns[:5]) == ["label", "quantity", "expiry", "unit_price", "value"]
    assert list(frame.columns[5:]) == list(GREEK_KEYS[1:])
    trader_total = bs.to_trader_units(book.greeks(mkt))
    total = frame.iloc[-1]
    assert total["value"] == pytest.approx(book.positions_value(mkt))
    for key in GREEK_KEYS[1:]:
        assert total[key] == pytest.approx(trader_total[key])
        assert frame[key].iloc[:-1].sum() == pytest.approx(total[key])
    first = frame.iloc[0]
    assert first["quantity"] == 10.0 and first["expiry"] == 1.0
    assert first["unit_price"] == pytest.approx(call.price(mkt))
    assert first["vega"] == pytest.approx(10 * call.greeks(mkt)["vega"] / 100)
    assert math.isnan(frame.iloc[2]["expiry"])

    raw = book.positions_frame(mkt, trader_units=False, include_cash=True)
    assert raw.attrs["units"] == "raw"
    assert list(raw["label"])[-2:] == ["CASH", "TOTAL"]
    assert raw.iloc[-1]["value"] == pytest.approx(book.value(mkt))
    assert raw.iloc[0]["vega"] == pytest.approx(10 * call.greeks(mkt)["vega"])

    empty = Book().positions_frame(mkt)
    assert list(empty["label"]) == ["TOTAL"] and empty.iloc[0]["delta"] == 0.0


def test_snapshot(book, mkt):
    snap = book.snapshot(mkt)
    assert snap["value"] == pytest.approx(book.value(mkt))
    assert snap["pnl"] == pytest.approx(book.pnl(mkt))
    assert snap["cash"] == book.cash and snap["spot"] == 100.0 and snap["t"] == 0.0
    assert snap["delta"] == pytest.approx(book.greeks(mkt)["delta"])
    assert all(isinstance(v, float) for v in snap.values())


def test_blotter_and_settlement_frames(book, mkt):
    blotter = book.blotter_frame()
    assert list(blotter.columns) == ["t", "instrument", "quantity", "price", "fees", "cash_flow", "note"]
    assert len(blotter) == 3
    assert blotter["cash_flow"].sum() == pytest.approx(book.cash - 1_000.0)
    assert list(Book().blotter_frame().columns) == list(blotter.columns)
    assert Book().settlements_frame().empty


# ---------------------------------------------------------------------- #
# Risk: ladders, grids, stress tests
# ---------------------------------------------------------------------- #
def test_shocked_market(mkt):
    shocked = shocked_market(mkt, spot=-0.1, vol=0.05, rate=0.01, days=73)
    assert shocked.spot == pytest.approx(90.0)
    assert shocked.vol == pytest.approx(0.25)
    assert shocked.rate == pytest.approx(0.04)
    assert shocked.t == pytest.approx(0.2)
    assert shocked.div == mkt.div
    assert shocked_market(mkt, vol=-0.5).vol == MIN_VOL  # floored
    assert shocked_market(Market(100.0, 0.0)).vol == 0.0  # no shock: untouched
    assert shocked_market(mkt) == mkt
    with pytest.raises(ValueError, match="-1"):
        shocked_market(mkt, spot=-1.0)
    with pytest.raises(ValueError, match="forward"):
        shocked_market(mkt, days=-1)


def test_spot_ladder_matches_brute_force(book, mkt):
    ladder = book.spot_ladder(mkt, trader_units=False)
    assert list(ladder.index) == list(DEFAULT_SPOT_SHOCKS)
    assert ladder.index.name == "spot_shock" and ladder.attrs["units"] == "raw"
    base = book.value(mkt)
    for shock, row in ladder.iterrows():
        bumped = mkt.bumped(spot=100.0 * (1 + shock))
        assert row["spot"] == pytest.approx(bumped.spot)
        assert row["value"] == pytest.approx(book.value(bumped))
        assert row["pnl"] == pytest.approx(book.value(bumped) - base, abs=1e-10)
        greeks = book.greeks(bumped)
        for key in GREEK_KEYS[1:]:
            assert row[key] == pytest.approx(greeks[key], abs=1e-12)
    assert ladder.loc[0.0, "pnl"] == 0.0

    trader = book.spot_ladder(mkt, shocks=(-0.1, 0.1))
    assert trader.attrs["units"] == "trader"
    assert trader["vega"].to_numpy() == pytest.approx(ladder.loc[[-0.1, 0.1], "vega"].to_numpy() / 100)
    with pytest.raises(ValueError, match="-1"):
        book.spot_ladder(mkt, shocks=(-1.0, 0.0))
    with pytest.raises(ValueError, match="1-D"):
        book.spot_ladder(mkt, shocks=())


def test_scenario_grid_matches_brute_force(book, mkt):
    spot_shocks, vol_shocks = (-0.2, 0.0, 0.1), (-0.05, 0.0, 0.1)
    grid = book.scenario_grid(mkt, spot_shocks, vol_shocks, horizon_days=30)
    assert grid.shape == (3, 3)
    assert grid.index.name == "spot_shock" and grid.columns.name == "vol_shock"
    base = book.positions_value(mkt)
    for ds in spot_shocks:
        for dv in vol_shocks:
            scenario = mkt.bumped(spot=100.0 * (1 + ds), vol=0.2 + dv, t=30 / 365)
            assert grid.loc[ds, dv] == pytest.approx(book.positions_value(scenario) - base, abs=1e-10)

    today = book.scenario_grid(mkt)
    assert today.shape == (len(DEFAULT_SPOT_SHOCKS), len(DEFAULT_VOL_SHOCKS))
    assert today.loc[0.0, 0.0] == 0.0
    values = book.scenario_grid(mkt, spot_shocks, vol_shocks, quantity="value")
    assert values.loc[0.0, 0.0] == pytest.approx(book.value(mkt))
    vega = book.scenario_grid(mkt, spot_shocks, vol_shocks, quantity="vega")
    assert vega.loc[0.1, 0.1] == pytest.approx(
        book.greeks(mkt.bumped(spot=110.0, vol=0.3))["vega"] / 100
    )
    with pytest.raises(ValueError, match="quantity"):
        book.scenario_grid(mkt, quantity="banana")


def test_stress_tests(book, mkt):
    frame = book.stress_tests(mkt)
    assert list(frame.index) == list(DEFAULT_SCENARIOS)
    base = book.value(mkt)
    for name, shocks in DEFAULT_SCENARIOS.items():
        scenario = shocked_market(mkt, **shocks)
        assert frame.loc[name, "pnl"] == pytest.approx(book.value(scenario) - base, abs=1e-10)
        assert frame.loc[name, "value"] == pytest.approx(book.value(scenario))
        assert frame.loc[name, "spot"] == pytest.approx(scenario.spot)
    custom = book.stress_tests(mkt, {"mine": {"spot": -0.3, "vol": 0.2, "rate": -0.01, "days": 10}})
    expected = book.value(mkt.bumped(spot=70.0, vol=0.4, rate=0.02, t=10 / 365)) - base
    assert custom.loc["mine", "pnl"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="unknown key"):
        book.stress_tests(mkt, {"bad": {"spot_shock": -0.1}})
    with pytest.raises(ValueError, match="at least one"):
        book.stress_tests(mkt, {})


def test_loop_fallback_for_non_vectorised_instruments(mkt):
    """An instrument that cannot price arrays gives the same ladder as its vectorised twin."""
    slow, fast = Book(), Book()
    slow.trade(BookTestScalarOnlyCall(100.0, 1.0), 7, mkt)
    fast.trade(EuropeanOption("call", 100.0, 1.0), 7, mkt)
    shocks = (-0.1, 0.0, 0.1)
    slow_ladder = slow.spot_ladder(mkt, shocks, trader_units=False)
    fast_ladder = fast.spot_ladder(mkt, shocks, trader_units=False)
    assert slow_ladder["pnl"].to_numpy() == pytest.approx(fast_ladder["pnl"].to_numpy())
    # numerical (bump) Greeks of the slow twin agree with the analytic ones
    assert slow_ladder["delta"].to_numpy() == pytest.approx(fast_ladder["delta"].to_numpy(), rel=1e-5)
    assert slow_ladder["gamma"].to_numpy() == pytest.approx(fast_ladder["gamma"].to_numpy(), rel=1e-4)
    slow_grid = slow.scenario_grid(mkt, shocks, (-0.05, 0.05), horizon_days=10)
    fast_grid = fast.scenario_grid(mkt, shocks, (-0.05, 0.05), horizon_days=10)
    pd.testing.assert_frame_equal(slow_grid, fast_grid)
    assert slow.stress_tests(mkt)["pnl"].to_numpy() == pytest.approx(fast.stress_tests(mkt)["pnl"].to_numpy())


# ---------------------------------------------------------------------- #
# Hedging
# ---------------------------------------------------------------------- #
def test_delta_hedge(book, mkt):
    delta = book.greeks(mkt)["delta"]
    assert book.delta_hedge_trade_size(mkt) == pytest.approx(-delta)
    assert book.delta_hedge_trade_size(mkt, target_delta=2.0) == pytest.approx(2.0 - delta)
    value_before = book.value(mkt)
    gamma_before = book.greeks(mkt)["gamma"]
    trade = book.hedge_delta(mkt)
    assert trade.instrument == Underlying() and trade.note == "delta hedge"
    assert book.greeks(mkt)["delta"] == pytest.approx(0.0, abs=1e-12)
    assert book.greeks(mkt)["gamma"] == pytest.approx(gamma_before)  # stock adds no gamma
    assert book.value(mkt) == pytest.approx(value_before)
    assert book.hedge_delta(mkt) is None  # already flat
    assert book.hedge_delta(mkt, target_delta=0.5, min_trade=1.0) is None  # inside the no-trade band
    book.hedge_delta(mkt, target_delta=0.5)
    assert book.greeks(mkt)["delta"] == pytest.approx(0.5)


def test_single_greek_hedge(book, mkt):
    hedge_option = EuropeanOption("put", 100.0, 1.5)
    size = book.hedge_size(mkt, "vega", hedge_option)
    assert size == pytest.approx(-book.greeks(mkt)["vega"] / hedge_option.greeks(mkt)["vega"])
    trade = book.neutralise(mkt, "vega", hedge_option)
    assert trade.quantity == pytest.approx(size) and trade.note == "vega hedge"
    assert book.greeks(mkt)["vega"] == pytest.approx(0.0, abs=1e-9)
    assert book.neutralise(mkt, "vega", hedge_option) is None
    book.neutralise(mkt, "gamma", hedge_option, target=0.25)
    assert book.greeks(mkt)["gamma"] == pytest.approx(0.25)


def test_delta_gamma_and_delta_gamma_vega_solvers(book, mkt):
    short_option = EuropeanOption("call", 105.0, 0.25)
    long_option = EuropeanOption("put", 95.0, 2.0)

    sizes = book.solve_hedge(mkt, [Underlying(), short_option])  # default: delta & gamma -> 0
    assert set(sizes) == {Underlying(), short_option}
    assert book.greeks(mkt)["gamma"] != 0  # nothing executed yet
    trial = book.copy()
    for inst, qty in sizes.items():
        trial.trade(inst, qty, mkt)
    assert trial.greeks(mkt)["delta"] == pytest.approx(0.0, abs=1e-10)
    assert trial.greeks(mkt)["gamma"] == pytest.approx(0.0, abs=1e-12)

    value_before = book.value(mkt)
    trades = book.hedge(
        mkt, [Underlying(), short_option, long_option], {"delta": 1.0, "gamma": 0.0, "vega": 0.0}
    )
    assert len(trades) == 3
    greeks = book.greeks(mkt)
    assert greeks["delta"] == pytest.approx(1.0, abs=1e-9)
    assert greeks["gamma"] == pytest.approx(0.0, abs=1e-11)
    assert greeks["vega"] == pytest.approx(0.0, abs=1e-8)
    assert book.value(mkt) == pytest.approx(value_before)


def test_solve_hedge_validation(book, mkt, call):
    with pytest.raises(ValueError, match="do not span"):
        book.solve_hedge(mkt, [Underlying()], ["gamma"])
    same_expiry = [EuropeanOption("call", 90.0, 1.0), EuropeanOption("put", 110.0, 1.0)]
    with pytest.raises(ValueError, match="do not span"):
        book.solve_hedge(mkt, same_expiry, ["gamma", "vega"])  # proportional in Black-Scholes
    with pytest.raises(ValueError, match="as many"):
        book.solve_hedge(mkt, [Underlying()], ["delta", "gamma"])
    with pytest.raises(ValueError, match="unknown Greek"):
        book.solve_hedge(mkt, [call], ["price"])
    with pytest.raises(ValueError, match="distinct"):
        book.solve_hedge(mkt, [call, call], ["delta", "gamma"])
    with pytest.raises(ValueError, match="expired"):
        book.solve_hedge(mkt.bumped(t=1.0), [call], ["delta"])


# ---------------------------------------------------------------------- #
# Life cycle
# ---------------------------------------------------------------------- #
def test_settle_expired_itm_call_pays_intrinsic(mkt, call, put):
    b = Book(cash=100.0)
    b.trade(call, 10, mkt)
    b.trade(put, -5, mkt)  # expires at 0.5
    b.trade(EuropeanOption("call", 130.0, 1.0), -2, mkt)  # will expire worthless
    assert b.settle_expired(mkt) == []

    just_before = Market(spot=112.0, vol=0.2, rate=0.03, div=0.01, t=1.0 - 1e-9)
    at_expiry = just_before.bumped(t=1.0)
    nlv_before = b.value(just_before)
    cash_before = b.cash
    records = b.settle_expired(at_expiry)

    assert b.is_empty
    assert [r.instrument for r in records] == [call, put, EuropeanOption("call", 130.0, 1.0)]
    assert all(isinstance(r, Settlement) for r in records)
    assert records[0].unit_value == pytest.approx(12.0) and records[0].cash_flow == pytest.approx(120.0)
    assert records[1].unit_value == 0.0 and records[2].cash_flow == 0.0
    assert records[0].spot == 112.0 and records[0].t == 1.0
    assert b.cash == pytest.approx(cash_before + 120.0)
    assert b.value(at_expiry) == pytest.approx(nlv_before, abs=1e-5)  # NLV continuous through expiry
    assert b.settlements == records
    assert b.settle_expired(at_expiry) == []
    frame = b.settlements_frame()
    assert frame["cash_flow"].sum() == pytest.approx(120.0)


def test_short_itm_put_pays_out_at_expiry(mkt, put):
    b = Book()
    b.trade(put, -5, mkt)
    records = b.settle_expired(Market(spot=90.0, vol=0.2, t=0.5))
    assert records[0].cash_flow == pytest.approx(-25.0)
    assert b.cash == pytest.approx(5 * put.price(mkt) - 25.0)


def test_calendar_legs_expire_independently(mkt):
    near, far = EuropeanOption("call", 100.0, 0.25), EuropeanOption("call", 100.0, 1.0)
    calendar = CompositeInstrument("calendar", (Position(near, -1), Position(far, 1)))
    at_first_expiry = Market(spot=108.0, vol=0.2, rate=0.03, div=0.01, t=0.25)
    for flatten in (True, False):
        b = Book()
        b.trade(calendar, 2, mkt, flatten=flatten)
        nlv = b.value(at_first_expiry)
        records = b.settle_expired(at_first_expiry)
        assert len(records) == 1 and records[0].instrument == near
        assert records[0].cash_flow == pytest.approx(-2 * 8.0)
        assert b.positions == (Position(far, 2.0),)  # the package line was split
        assert b.value(at_first_expiry) == pytest.approx(nlv)


def test_accrue_conventions(call):
    carry_mkt = Market(spot=50.0, vol=0.2, rate=0.04, div=0.02)
    b = Book(cash=1_000.0)
    b.trade(Underlying(), 10, carry_mkt)  # cash: 500
    covered = CompositeInstrument("covered call", (Position(Underlying(), 1), Position(call, -1)))
    b.trade(covered, 4, carry_mkt, flatten=False)  # shares inside a package count too
    assert b.underlying_quantity() == 14.0
    cash = b.cash
    out = b.accrue(carry_mkt, dt=0.5)
    assert out["interest"] == pytest.approx(cash * (math.exp(0.04 * 0.5) - 1))
    assert out["dividends"] == pytest.approx(14 * 50.0 * 0.02 * 0.5)
    assert b.cash == pytest.approx(cash + out["interest"] + out["dividends"])
    assert b.interest_accrued == out["interest"] and b.dividends_accrued == out["dividends"]

    short = Book(cash=-200.0)  # borrowed cash and short shares both PAY carry
    short.trade(Underlying(), -10, carry_mkt, price=0.0)
    paid = short.accrue(carry_mkt, dt=1.0)
    assert paid["interest"] == pytest.approx(-200.0 * (math.exp(0.04) - 1)) and paid["interest"] < 0
    assert paid["dividends"] == pytest.approx(-10 * 50.0 * 0.02)
    assert short.accrue(carry_mkt, dt=0.0) == {"interest": 0.0, "dividends": 0.0}
    with pytest.raises(ValueError, match="non-negative"):
        short.accrue(carry_mkt, dt=-0.1)


def test_observe_replaces_path_dependent_instruments(mkt):
    fresh = BookTestKnockOutCall(100.0, 1.0, barrier=120.0)
    dead = replace(fresh, hit=True)
    vanilla = EuropeanOption("call", 100.0, 1.0)
    b = Book()
    b.trade(fresh, 3, mkt)
    b.trade(dead, 2, mkt, price=0.0)
    b.trade(vanilla, 1, mkt)

    assert b.observe(mkt.bumped(spot=110.0, t=0.1)) == {}  # below the barrier: nothing changes
    assert b.quantity(fresh) == 3.0

    up = mkt.bumped(spot=125.0, t=0.2)
    value_alive = b.value(up)
    replaced = b.observe(up)
    assert replaced == {fresh: dead}
    assert fresh not in b
    assert b.quantity(dead) == 5.0  # quantity preserved and netted with the existing line
    assert b.quantity(vanilla) == 1.0
    assert [p.instrument for p in b.positions] == [dead, vanilla]
    assert b.value(up) == pytest.approx(value_alive - 3 * fresh.price(up))  # the knock-out is a real loss

    with pytest.raises(ValueError, match="scalar"):
        b.observe(mkt.bumped(spot=np.array([100.0, 130.0])))


def test_observe_nets_to_zero_and_skips_past_expiries(mkt):
    fresh = BookTestKnockOutCall(100.0, 1.0, barrier=120.0)
    dead = replace(fresh, hit=True)
    b = Book()
    b.trade(fresh, 2, mkt)
    b.trade(dead, -2, mkt, price=0.0)
    b.observe(mkt.bumped(spot=130.0, t=0.5))
    assert b.is_empty  # +2 and -2 of the same (now dead) instrument

    late = Book()
    late.trade(fresh, 1, mkt)
    assert late.observe(mkt.bumped(spot=130.0, t=1.5)) == {}  # expiry already passed
    assert late.quantity(fresh) == 1.0
    assert late.observe(mkt.bumped(spot=130.0, t=1.0)) == {fresh: dead}  # AT expiry still counts


def test_simulated_life_cycle_delta_hedge_replicates_option():
    """observe -> settle -> accrue loop: a hedged short call ends near zero P&L."""
    rng = np.random.default_rng(7)
    vol, rate, div, n_steps, expiry = 0.2, 0.03, 0.01, 2_000, 0.25
    dt = expiry / n_steps
    option = EuropeanOption("call", 100.0, expiry)
    mkt = Market(spot=100.0, vol=vol, rate=rate, div=div)
    b = Book(cash=0.0)
    b.trade(option, -1, mkt)
    b.hedge_delta(mkt)
    for step in range(1, n_steps + 1):
        b.accrue(mkt, dt)
        growth = (rate - div - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * rng.standard_normal()
        mkt = mkt.bumped(spot=mkt.spot * math.exp(growth), t=step * dt)
        b.observe(mkt)
        b.settle_expired(mkt)
        if option in b:
            b.hedge_delta(mkt)
    assert option not in b and len(b.settlements) == 1
    assert abs(b.pnl(mkt)) < 0.15  # premium was about 4.2


# ---------------------------------------------------------------------- #
# Persistence
# ---------------------------------------------------------------------- #
def _full_book(mkt):
    b = Book("persisted", cash=500.0, costs=TransactionCosts(stock_bps=2.0, option_pct=0.01))
    straddle = CompositeInstrument(
        "straddle", (Position(EuropeanOption("call", 100, 1.0), 1), Position(EuropeanOption("put", 100, 1.0), 1))
    )
    b.trade(straddle, 2, mkt, note="open")
    b.trade(straddle, -1, mkt, flatten=False, note="package line")
    b.trade(BookTestKnockOutCall(100.0, 0.75, 130.0), 4, mkt, fee=0.5)
    b.trade(EuropeanOption("put", 90.0, 0.1), 3, mkt)
    b.hedge_delta(mkt)
    b.accrue(mkt, 0.1)
    b.settle_expired(mkt.bumped(spot=85.0, t=0.1))
    return b


def test_json_round_trip_preserves_everything(mkt, tmp_path):
    b = _full_book(mkt)
    later = mkt.bumped(spot=104.0, vol=0.23, t=0.1)
    text = b.to_json()
    assert json.loads(text)["name"] == "persisted"
    clone = Book.from_json(text)

    assert clone.name == b.name and clone.costs == b.costs
    assert clone.cash == b.cash and clone.initial_cash == b.initial_cash
    assert clone.interest_accrued == b.interest_accrued
    assert clone.dividends_accrued == b.dividends_accrued
    assert clone.positions == b.positions
    assert clone.trades == b.trades and clone.settlements == b.settlements
    assert clone.value(later) == b.value(later)
    assert clone.greeks(later) == b.greeks(later)
    assert clone.to_dict() == b.to_dict()

    path = tmp_path / "book.json"
    assert b.to_json(path) == text
    assert Book.from_json(path).to_dict() == b.to_dict()
    assert Book.from_json(str(path)).to_dict() == b.to_dict()
    assert Book.from_dict(Book().to_dict()).to_dict() == Book().to_dict()


def test_from_dict_validation():
    with pytest.raises(ValueError, match="unknown key"):
        Book.from_dict({"name": "x", "money": 1.0})
    with pytest.raises(ValueError, match="mapping"):
        Book.from_dict([1, 2])
    with pytest.raises(ValueError, match="valid book JSON"):
        Book.from_json("{not json")
    with pytest.raises(ValueError, match="Unknown instrument type"):
        Book.from_dict({"positions": [{"type": "Position", "instrument": {"type": "Nope"}, "quantity": 1}]})


def test_copy_is_independent(book, mkt, call):
    clone = book.copy(name="what-if")
    assert clone.name == "what-if" and clone.to_dict()["positions"] == book.to_dict()["positions"]
    clone.trade(call, -10, mkt)
    clone.accrue(mkt, 1.0)
    assert book.quantity(call) == 10.0 and len(book.trades) == 3
    assert book.interest_accrued == 0.0
    assert call not in clone and len(clone.trades) == 4


# ---------------------------------------------------------------------- #
# Figures
# ---------------------------------------------------------------------- #
@pytest.fixture
def chart_book(book, mkt):
    book.trade(BookTestKnockOutCall(100.0, 1.0, 140.0), 2, mkt)  # "other" category, numerical Greeks
    return book


def _assert_valid(fig):
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    json.loads(fig.to_json())  # serialisable, as Streamlit needs
    for trace in fig.data:
        numbers = getattr(trace, {"scatter": "y", "bar": "x", "heatmap": "z"}[trace.type])
        assert np.all(np.isfinite(np.asarray(numbers, dtype=float)))


def test_book_risk_profile(chart_book, mkt):
    fig = book_plots.book_risk_profile(chart_book, mkt, n_points=41)
    _assert_valid(fig)
    lines = [t for t in fig.data if t.hoverinfo != "skip"]
    assert [t.name for t in lines] == ["P&L", "Delta", "Gamma", "Vega", "Theta"]
    ladder = chart_book.spot_ladder(mkt, np.linspace(-0.3, 0.3, 41))
    assert np.allclose(lines[0].y, ladder["pnl"])
    assert np.allclose(lines[3].y, ladder["vega"])  # trader units by default
    assert "per 1 vol point" in fig.layout.annotations[3].text
    assert fig.layout.hovermode == "x unified"
    single = book_plots.book_risk_profile(chart_book, mkt, greeks=("gamma",), trader_units=False, n_points=11)
    assert len(single.data) == 1
    with pytest.raises(ValueError, match="unknown quantity"):
        book_plots.book_risk_profile(chart_book, mkt, greeks=("pnl", "banana"))
    with pytest.raises(ValueError, match="spot_range"):
        book_plots.book_risk_profile(chart_book, mkt, spot_range=(0.2, -0.2))


def test_book_scenario_heatmap(chart_book, mkt):
    fig = book_plots.book_scenario_heatmap(chart_book, mkt, horizon_days=14)
    _assert_valid(fig)
    heat = fig.data[0]
    grid = chart_book.scenario_grid(mkt, horizon_days=14)
    assert np.allclose(np.asarray(heat.z), grid.to_numpy().T)
    assert heat.zmid == 0 and heat.zmin == -heat.zmax  # diverging scale centred on zero
    assert len(heat.x) == len(DEFAULT_SPOT_SHOCKS) and len(heat.y) == len(DEFAULT_VOL_SHOCKS)
    assert heat.text[0][0].startswith(("+", "-"))  # signed annotations
    assert "14 days forward" in fig.layout.title.text
    plain = book_plots.book_scenario_heatmap(chart_book, mkt, quantity="delta", annotate=False)
    assert plain.data[0].text is None
    _assert_valid(book_plots.book_scenario_heatmap(Book(), mkt))  # empty book: all zeros


def test_book_greeks_breakdown(chart_book, mkt):
    fig = book_plots.book_greeks_breakdown(chart_book, mkt, "vega")
    _assert_valid(fig)
    assert [t.name for t in fig.data] == ["Calls", "Puts", "Underlying", "Other", "Book total"]
    frame = chart_book.positions_frame(mkt)
    plotted = {label: x for t in fig.data for label, x in zip(t.y, t.x)}
    assert plotted == pytest.approx(dict(zip(frame["label"], frame["vega"])))
    assert list(fig.layout.yaxis.categoryarray) == list(frame["label"])
    _assert_valid(book_plots.book_greeks_breakdown(chart_book, mkt, "value", trader_units=False))
    _assert_valid(book_plots.book_greeks_breakdown(Book(), mkt))
    with pytest.raises(ValueError, match="book-level"):
        book_plots.book_greeks_breakdown(chart_book, mkt, "pnl")


def test_book_pnl_by_horizon(chart_book, mkt):
    fig = book_plots.book_pnl_by_horizon(chart_book, mkt, days=(30, 0, 7), n_points=31)
    _assert_valid(fig)
    assert [t.name for t in fig.data] == ["today", "+7 d", "+30 d"]
    shocks = np.linspace(-0.3, 0.3, 31)
    expected = chart_book.positions_value(shocked_market(mkt, spot=shocks, days=30)) - chart_book.positions_value(mkt)
    assert np.allclose(fig.data[2].y, expected)
    assert len({t.line.color for t in fig.data}) == 3
    _assert_valid(book_plots.book_pnl_by_horizon(chart_book, mkt, days=(0,), mode="dark", n_points=11))
    with pytest.raises(ValueError, match="non-negative"):
        book_plots.book_pnl_by_horizon(chart_book, mkt, days=(-1, 0))


def test_stress_test_chart(chart_book, mkt):
    fig = book_plots.stress_test_chart(chart_book, mkt)
    _assert_valid(fig)
    bar = fig.data[0]
    frame = chart_book.stress_tests(mkt)
    assert list(bar.y) == list(frame.index)
    assert np.allclose(bar.x, frame["pnl"])
    assert all(text.startswith(("+", "-")) or text == "0" for text in bar.text)
    custom = book_plots.stress_test_chart(chart_book, mkt, {"down": {"spot": -0.1}, "up": {"spot": 0.1}})
    assert list(custom.data[0].y) == ["down", "up"]


def test_charts_need_a_scalar_market(chart_book, mkt):
    arr = mkt.bumped(spot=np.array([90.0, 110.0]))
    for chart in (
        book_plots.book_risk_profile,
        book_plots.book_scenario_heatmap,
        book_plots.book_greeks_breakdown,
        book_plots.book_pnl_by_horizon,
        book_plots.stress_test_chart,
    ):
        with pytest.raises(ValueError, match="scalar"):
            chart(chart_book, arr)
