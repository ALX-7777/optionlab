"""Tests for optionlab.simulator, optionlab.plotting.sim_plots and the trading-game example."""

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from optionlab.book import Book, TransactionCosts
from optionlab.instruments import CompositeInstrument, EuropeanOption, Instrument, Position, Underlying
from optionlab.market import Market
from optionlab.plotting import sim_plots
from optionlab.simulator import (
    ATTRIBUTION_TERMS,
    GREEK_TERMS,
    VOL_CAP,
    VOL_FLOOR,
    DeltaBandHedge,
    DeltaHedgeAtVol,
    DeltaHedgeEveryN,
    HedgingPolicy,
    MarketScenario,
    NoHedge,
    TradingSimulator,
    _generic_hedging_pnl,
    _vectorised_hedging_pnl,
    attribution_totals,
    delta_hedging_experiment,
    explain_book_pnl,
    explain_pnl,
    gamma_scalping_summary,
    gbm_scenario,
    hedging_experiment_summary,
    scenario_from_arrays,
)
from optionlab.monte_carlo import simulate_gbm_paths

TERM_COLUMNS = [f"pnl_{term}" for term in ATTRIBUTION_TERMS]


@dataclass(frozen=True)
class MaxTracker(Instrument):
    """Toy path-dependent product paying the running maximum of the spot at expiry."""

    expiry: float
    running_max: float = 0.0

    def price(self, mkt: Market):
        return np.maximum(self.running_max, mkt.spot) + 0.0

    def payoff(self, spot_T):
        return np.maximum(self.running_max, np.asarray(spot_T, dtype=float))

    def observe(self, spot: float, t: float) -> "MaxTracker":
        return self if spot <= self.running_max else replace(self, running_max=float(spot))


@pytest.fixture
def mkt0() -> Market:
    return Market(spot=100.0, vol=0.2, rate=0.02, div=0.01)


@pytest.fixture
def call() -> EuropeanOption:
    return EuropeanOption("call", 100.0, 0.25)


def _long_call_sim(mkt0, call, policy=None, seed=1, realized_vol=0.3, costs=None, **kwargs):
    book = Book("test", cash=1_000.0)
    book.trade(call, 100, mkt0)
    scenario = gbm_scenario(mkt0, horizon=0.25, n_steps=63, realized_vol=realized_vol, seed=seed, **kwargs)
    return TradingSimulator(book, scenario, policy, costs=costs)


# ---------------------------------------------------------------------- #
# Scenarios
# ---------------------------------------------------------------------- #
class TestScenarios:
    def test_shapes_and_first_market(self, mkt0):
        mkt = mkt0.bumped(t=0.5)
        scenario = gbm_scenario(mkt, horizon=0.25, n_steps=63, seed=0)
        assert scenario.n_steps == 63 and len(scenario) == 64
        assert scenario.times.shape == scenario.spot.shape == scenario.vol.shape == (64,)
        assert scenario.market_at(0) == mkt
        assert scenario.times[-1] == pytest.approx(0.75)
        assert scenario.horizon == pytest.approx(0.25)
        assert scenario.dt(0) == pytest.approx(0.25 / 63)
        last = scenario.market_at(63)
        assert last.is_scalar and last.rate == mkt.rate and last.div == mkt.div
        assert len(list(scenario.markets())) == 64
        assert list(scenario.to_frame().columns) == ["t", "spot", "implied_vol"]

    def test_seed_reproducibility(self, mkt0):
        a = gbm_scenario(mkt0, 1.0, 50, seed=42, implied_vol_model="spot_correlated", jump_intensity=3.0)
        b = gbm_scenario(mkt0, 1.0, 50, seed=42, implied_vol_model="spot_correlated", jump_intensity=3.0)
        c = gbm_scenario(mkt0, 1.0, 50, seed=43, implied_vol_model="spot_correlated", jump_intensity=3.0)
        np.testing.assert_array_equal(a.spot, b.spot)
        np.testing.assert_array_equal(a.vol, b.vol)
        assert not np.allclose(a.spot, c.spot)

    def test_same_seed_same_spot_across_vol_models(self, mkt0):
        paths = [gbm_scenario(mkt0, 1.0, 100, seed=7, implied_vol_model=m).spot
                 for m in ("constant", "mean_reverting", "spot_correlated")]
        np.testing.assert_array_equal(paths[0], paths[1])
        np.testing.assert_array_equal(paths[0], paths[2])

    def test_constant_model_and_matches_core_gbm(self, mkt0):
        scenario = gbm_scenario(mkt0, 1.0, 252, seed=5)
        assert np.all(scenario.vol == mkt0.vol)
        _, paths = simulate_gbm_paths(100.0, 0.2, 0.02, 0.01, 1.0, 252, n_paths=1, seed=5, antithetic=False)
        np.testing.assert_allclose(scenario.spot, paths[0], rtol=1e-12)

    def test_realized_vol_and_drift(self, mkt0):
        scenario = gbm_scenario(mkt0, 4.0, 8000, realized_vol=0.35, seed=3)
        assert scenario.realized_vol() == pytest.approx(0.35, rel=0.05)
        assert scenario.log_returns().shape == (8000,)
        deterministic = gbm_scenario(mkt0, 2.0, 10, realized_vol=0.0, drift=0.07, seed=1)
        assert deterministic.spot[-1] == pytest.approx(100.0 * math.exp(0.14))

    def test_vol_floor_and_cap(self):
        low = gbm_scenario(Market(100.0, 0.02), 2.0, 500, implied_vol_model="mean_reverting",
                           vol_of_vol=3.0, kappa=0.0, seed=11)
        assert low.vol[1:].min() >= VOL_FLOOR - 1e-15 and low.vol.max() <= VOL_CAP + 1e-12
        assert np.isclose(low.vol[1:].min(), VOL_FLOOR) or np.isclose(low.vol.max(), VOL_CAP)
        assert low.vol[0] == 0.02

    @pytest.mark.parametrize("rho", [-0.7, 0.7])
    def test_spot_vol_correlation_sign(self, mkt0, rho):
        scenario = gbm_scenario(mkt0, 4.0, 4000, implied_vol_model="spot_correlated",
                                rho_spot_vol=rho, seed=2)
        corr = np.corrcoef(scenario.log_returns(), np.diff(np.log(scenario.vol)))[0, 1]
        assert corr * np.sign(rho) > 0.5

    def test_mean_reverting_is_uncorrelated_and_reverts(self, mkt0):
        scenario = gbm_scenario(mkt0, 4.0, 4000, implied_vol_model="mean_reverting", seed=2)
        corr = np.corrcoef(scenario.log_returns(), np.diff(np.log(scenario.vol)))[0, 1]
        assert abs(corr) < 0.1
        pulled = gbm_scenario(mkt0, 5.0, 1000, implied_vol_model="mean_reverting", vol_of_vol=0.0,
                              kappa=5.0, long_run_vol=0.4, seed=2)
        assert np.all(np.diff(pulled.vol) > 0) and pulled.vol[-1] == pytest.approx(0.4, rel=1e-6)

    def test_jumps(self, mkt0):
        calm = gbm_scenario(mkt0, 1.0, 252, seed=9)
        jumpy = gbm_scenario(mkt0, 1.0, 252, seed=9, jump_intensity=10.0, jump_mean=-0.08, jump_std=0.02)
        assert np.abs(calm.log_returns()).max() < 0.06 < np.abs(jumpy.log_returns()).max()
        crash = gbm_scenario(mkt0, 1.0, 252, seed=9, jump_intensity=10.0, jump_mean=-0.08,
                             jump_std=0.02, implied_vol_model="spot_correlated")
        worst = int(np.argmin(crash.log_returns()))
        assert crash.vol[worst + 1] > crash.vol[worst]  # vol spikes on the crash day

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"horizon": 0.0}, {"n_steps": 0}, {"n_steps": 2.5}, {"realized_vol": -0.1},
            {"implied_vol_model": "heston"}, {"rho_spot_vol": 1.5}, {"jump_intensity": -1.0},
            {"vol_of_vol": -1.0}, {"kappa": -1.0}, {"jump_std": -0.1},
        ],
    )
    def test_gbm_scenario_validation(self, mkt0, kwargs):
        params = {"horizon": 1.0, "n_steps": 10, **kwargs}
        with pytest.raises(ValueError):
            gbm_scenario(mkt0, **params)

    def test_gbm_scenario_needs_scalar_market(self):
        with pytest.raises(ValueError):
            gbm_scenario(Market(np.array([100.0, 101.0]), 0.2), 1.0, 10)
        with pytest.raises(ValueError):
            gbm_scenario(Market(100.0, 0.0), 1.0, 10, implied_vol_model="mean_reverting")

    def test_scenario_from_arrays(self):
        scenario = scenario_from_arrays([100, 101, 99.5], 0.25, rate=0.01)
        np.testing.assert_allclose(scenario.times, [0.0, 1 / 252, 2 / 252])
        np.testing.assert_array_equal(scenario.vol, [0.25, 0.25, 0.25])
        assert scenario.market_at(2) == Market(99.5, 0.25, 0.01, 0.0, 2 / 252)
        custom = scenario_from_arrays([100, 101], [0.2, 0.3], times=[1.0, 1.5])
        assert custom.market_at(1).t == 1.5 and custom.market_at(1).vol == 0.3
        with pytest.raises(ValueError):
            custom.spot[0] = 1.0  # read-only

    @pytest.mark.parametrize(
        "args",
        [
            dict(spot=[100.0], vol=0.2),
            dict(spot=[100.0, -1.0], vol=0.2),
            dict(spot=[100.0, 101.0], vol=[0.2, 0.2, 0.2]),
            dict(spot=[100.0, 101.0], vol=-0.2),
            dict(spot=[100.0, 101.0], vol=0.2, times=[1.0, 1.0]),
            dict(spot=[100.0, np.nan], vol=0.2),
            dict(spot=[100.0, 101.0], vol=0.2, dt=0.0),
        ],
    )
    def test_scenario_validation(self, args):
        with pytest.raises(ValueError):
            scenario_from_arrays(**args)

    def test_index_errors_and_json_round_trip(self, mkt0):
        scenario = gbm_scenario(mkt0, 0.1, 5, seed=1, implied_vol_model="mean_reverting")
        for bad in (-1, 6):
            with pytest.raises(IndexError):
                scenario.market_at(bad)
        with pytest.raises(IndexError):
            scenario.dt(5)
        clone = MarketScenario.from_dict(json.loads(json.dumps(scenario.to_dict())))
        np.testing.assert_array_equal(clone.spot, scenario.spot)
        np.testing.assert_array_equal(clone.vol, scenario.vol)
        assert clone.name == scenario.name and clone.rate == scenario.rate
        with pytest.raises(ValueError):
            MarketScenario.from_dict({"times": [0, 1], "spot": [1, 2]})
        with pytest.raises(ValueError):
            MarketScenario.from_dict({**scenario.to_dict(), "oops": 1})


# ---------------------------------------------------------------------- #
# Attribution
# ---------------------------------------------------------------------- #
class TestExplainPnl:
    def test_definitions(self):
        greeks = {"delta": 2.0, "gamma": 0.5, "theta": -10.0, "vega": 30.0, "vanna": 4.0,
                  "volga": 8.0, "rho": 50.0}
        start = Market(100.0, 0.20, rate=0.01, t=0.0)
        end = Market(102.0, 0.23, rate=0.015, t=0.01)
        terms = explain_pnl(greeks, start, end)
        assert tuple(terms) == GREEK_TERMS
        assert terms["delta"] == pytest.approx(2.0 * 2.0)
        assert terms["gamma"] == pytest.approx(0.5 * 0.5 * 4.0)
        assert terms["theta"] == pytest.approx(-10.0 * 0.01)
        assert terms["vega"] == pytest.approx(30.0 * 0.03)
        assert terms["vanna"] == pytest.approx(4.0 * 2.0 * 0.03)
        assert terms["volga"] == pytest.approx(0.5 * 8.0 * 0.03**2)
        assert terms["rho"] == pytest.approx(50.0 * 0.005)
        assert explain_pnl({"delta": 1.0}, start, end)["gamma"] == 0.0  # missing Greeks count as 0

    def test_book_level_identity_and_signs(self):
        greeks = {"delta": 1.0, "gamma": 0.1, "theta": -5.0, "vega": 10.0}
        start, end = Market(100.0, 0.2), Market(101.0, 0.21, t=1 / 252)
        terms = explain_book_pnl(greeks, start, end, actual_pnl=1.5, carry=0.2, fees=0.3, trading=0.1)
        assert tuple(terms) == ATTRIBUTION_TERMS
        assert sum(terms.values()) == pytest.approx(1.5, abs=1e-12)
        assert terms["fees"] == -0.3 and terms["carry"] == 0.2 and terms["trading"] == 0.1

    def test_small_step_on_vanilla_is_well_explained(self, call):
        start = Market(100.0, 0.2, rate=0.02, div=0.01)
        end = Market(100.6, 0.203, rate=0.02, div=0.01, t=1 / 252)
        actual = call.price(end) - call.price(start)
        terms = explain_book_pnl(call.greeks(start), start, end, actual)
        assert abs(terms["unexplained"]) < 0.02 * (abs(terms["delta"]) + abs(terms["gamma"]))
        # and the explanation improves with the square of the step or better
        smaller = Market(100.06, 0.2003, rate=0.02, div=0.01, t=0.1 / 252)
        small_terms = explain_book_pnl(call.greeks(start), start, smaller,
                                       call.price(smaller) - call.price(start))
        assert abs(small_terms["unexplained"]) < abs(terms["unexplained"]) / 50

    def test_vectorised(self, call):
        start = Market(100.0, 0.2)
        end = Market(np.array([99.0, 100.0, 101.0]), 0.2, t=0.01)
        terms = explain_pnl(call.greeks(start), start, end)
        assert terms["delta"].shape == (3,) and terms["gamma"][1] == 0.0
        assert terms["gamma"][0] == pytest.approx(terms["gamma"][2])


# ---------------------------------------------------------------------- #
# Policies
# ---------------------------------------------------------------------- #
class TestPolicies:
    def _book(self, mkt, call):
        book = Book()
        book.trade(call, 100, mkt)
        return book

    def test_no_hedge(self, mkt0, call):
        book = self._book(mkt0, call)
        assert NoHedge().rebalance(book, mkt0, 0) == []
        assert len(book.trades) == 1 and isinstance(NoHedge(), HedgingPolicy)

    def test_every_n(self, mkt0, call):
        book = self._book(mkt0, call)
        policy = DeltaHedgeEveryN(5)
        assert policy.rebalance(book, mkt0, 3) == []
        trades = policy.rebalance(book, mkt0, 5)
        assert len(trades) == 1 and isinstance(trades[0].instrument, Underlying)
        assert book.greeks(mkt0)["delta"] == pytest.approx(0.0, abs=1e-10)
        assert policy.rebalance(book, mkt0, 10) == []  # already flat
        shifted = DeltaHedgeEveryN(1, target_delta=25.0)
        shifted.rebalance(book, mkt0, 0)
        assert book.greeks(mkt0)["delta"] == pytest.approx(25.0)

    def test_band(self, mkt0, call):
        book = self._book(mkt0, call)
        delta = book.greeks(mkt0)["delta"]
        assert DeltaBandHedge(band=delta + 1.0).rebalance(book, mkt0, 1) == []
        assert DeltaBandHedge(band=0.9, relative=True).rebalance(book, mkt0, 1) == []  # 90 shares
        trades = DeltaBandHedge(band=0.3, relative=True).rebalance(book, mkt0, 1)  # 30 shares
        assert len(trades) == 1 and trades[0].quantity == pytest.approx(-delta)
        assert book.greeks(mkt0)["delta"] == pytest.approx(0.0, abs=1e-10)

    def test_relative_band_counts_options_inside_packages(self, mkt0, call):
        book = Book()
        straddle = CompositeInstrument("straddle", (call, EuropeanOption("put", 100.0, 0.25)))
        book.trade(straddle, 10, mkt0, flatten=False)
        book.trade(Underlying(), 3.0, mkt0)
        # 20 options -> band of 20 * 0.5 = 10 shares, the book delta is about 4
        assert DeltaBandHedge(0.5, relative=True).rebalance(book, mkt0, 1) == []
        assert len(DeltaBandHedge(0.05, relative=True).rebalance(book, mkt0, 1)) == 1

    def test_hedge_at_vol(self, mkt0, call):
        book = self._book(mkt0, EuropeanOption("call", 110.0, 0.25))
        policy = DeltaHedgeAtVol(hedge_vol=0.4, n_steps=2)
        assert policy.rebalance(book, mkt0, 1) == []
        trades = policy.rebalance(book, mkt0, 2)
        assert len(trades) == 1 and trades[0].price == mkt0.spot
        assert book.greeks(mkt0.bumped(vol=0.4))["delta"] == pytest.approx(0.0, abs=1e-10)
        assert abs(book.greeks(mkt0)["delta"]) > 1.0

    @pytest.mark.parametrize(
        "factory",
        [
            lambda: DeltaHedgeEveryN(0), lambda: DeltaHedgeEveryN(1.5), lambda: DeltaHedgeEveryN(1, min_trade=-1),
            lambda: DeltaBandHedge(-0.1), lambda: DeltaHedgeAtVol(0.0), lambda: DeltaHedgeAtVol(0.2, n_steps=0),
            lambda: DeltaBandHedge(float("nan")),
        ],
    )
    def test_validation(self, factory):
        with pytest.raises(ValueError):
            factory()

    def test_labels(self):
        for policy in (NoHedge(), DeltaHedgeEveryN(), DeltaHedgeEveryN(5), DeltaBandHedge(2.0),
                       DeltaBandHedge(0.05, relative=True), DeltaHedgeAtVol(0.3)):
            assert isinstance(policy.label, str) and policy.label


# ---------------------------------------------------------------------- #
# Simulator
# ---------------------------------------------------------------------- #
class TestTradingSimulator:
    @pytest.mark.parametrize("policy", [None, DeltaHedgeEveryN(1), DeltaBandHedge(5.0), DeltaHedgeAtVol(0.3)])
    def test_attribution_identity(self, mkt0, call, policy):
        sim = _long_call_sim(mkt0, call, policy, implied_vol_model="spot_correlated",
                             costs=TransactionCosts(stock_bps=2.0))
        history = sim.run()
        assert len(history) == 64 and sim.is_finished and history.index.name == "step"
        np.testing.assert_allclose(history[TERM_COLUMNS].sum(axis=1), history["pnl_step"], atol=1e-9)
        np.testing.assert_allclose(history["pnl_step"].cumsum(), history["pnl_cum"], atol=1e-9)
        for term in ATTRIBUTION_TERMS:
            np.testing.assert_allclose(history[f"pnl_{term}"].cumsum(), history[f"cum_pnl_{term}"], atol=1e-9)
        # the book was opened at the model price without costs, so its whole P&L is the simulation's
        assert history["pnl_cum"].iloc[-1] == pytest.approx(sim.book.pnl(sim.current_market), abs=1e-9)
        assert history["value"].iloc[-1] == pytest.approx(sim.book.value(sim.current_market))
        np.testing.assert_allclose(history["value"], history["cash"] + history["positions_value"])
        totals = attribution_totals(history)
        assert sum(totals[t] for t in ATTRIBUTION_TERMS) == pytest.approx(totals["actual"], abs=1e-9)

    def test_daily_steps_are_well_explained(self, mkt0, call):
        sim = _long_call_sim(mkt0, call, DeltaHedgeEveryN(1), realized_vol=0.2)
        history = sim.run().iloc[1:50]  # away from expiry, where the Greeks blow up
        explained = history["pnl_gamma"].abs() + history["pnl_theta"].abs() + history["pnl_delta"].abs()
        assert history["pnl_unexplained"].abs().sum() < 0.05 * explained.sum()

    def test_delta_hedged_book_has_no_delta_pnl_but_unhedged_does(self, mkt0, call):
        hedged = _long_call_sim(mkt0, call, DeltaHedgeEveryN(1)).run()
        naked = _long_call_sim(mkt0, call, NoHedge()).run()
        assert hedged["cum_pnl_delta"].abs().max() < 1e-9
        assert naked["cum_pnl_delta"].abs().max() > 10.0
        assert np.all(hedged["delta"].abs() < 1e-9)
        assert hedged["hedge_trades"].iloc[0] == 1 and naked["hedge_trades"].sum() == 0
        # each hedge trade is minus the delta found before hedging
        np.testing.assert_allclose(hedged["hedge_shares"], -hedged["delta_before_hedge"], atol=1e-9)
        np.testing.assert_allclose(hedged["stock_position"].diff().iloc[1:], hedged["hedge_shares"].iloc[1:],
                                   atol=1e-9)

    def test_pure_stock_book(self):
        mkt = Market(100.0, 0.2, rate=0.0, div=0.03)
        book = Book(cash=50_000.0)
        book.trade(Underlying(), 300, mkt)
        scenario = gbm_scenario(mkt, 0.5, 26, realized_vol=0.25, drift=0.05, seed=4)
        history = TradingSimulator(book, scenario).run()
        d_spot = np.diff(scenario.spot)
        dividends = 300 * scenario.spot[:-1] * 0.03 * np.diff(scenario.times)
        np.testing.assert_allclose(history["pnl_step"].iloc[1:], 300 * d_spot + dividends, atol=1e-8)
        np.testing.assert_allclose(history["pnl_delta"].iloc[1:], 300 * d_spot, atol=1e-9)
        np.testing.assert_allclose(history["pnl_carry"].iloc[1:], dividends, atol=1e-9)
        assert history["pnl_unexplained"].abs().max() < 1e-8
        assert np.all(history["stock_position"] == 300)
        assert book.dividends_accrued == pytest.approx(dividends.sum())

    def test_cash_accrues_at_the_rate(self):
        mkt = Market(100.0, 0.2, rate=0.05)
        book = Book(cash=1_000.0)
        scenario = gbm_scenario(mkt, 2.0, 24, seed=1)
        history = TradingSimulator(book, scenario, DeltaHedgeEveryN(1)).run()
        assert history["value"].iloc[-1] == pytest.approx(1_000.0 * math.exp(0.05 * 2.0))
        np.testing.assert_allclose(history["pnl_carry"], history["pnl_step"], atol=1e-10)
        assert history["hedge_trades"].sum() == 0

    def test_expiry_inside_scenario_settles_with_continuous_value(self, mkt0):
        option = EuropeanOption("call", 95.0, 0.25)
        book = Book(cash=500.0)
        book.trade(option, -10, mkt0)
        scenario = gbm_scenario(mkt0, horizon=0.5, n_steps=126, seed=8)  # expiry lands on date 63
        assert scenario.times[63] == pytest.approx(0.25)
        sim = TradingSimulator(book, scenario)
        sim.run(62)
        shadow = book.copy()
        shadow.accrue(sim.current_market, scenario.dt(62))
        expected_value = shadow.value(scenario.market_at(63))  # marked at intrinsic, NOT settled
        row = sim.step()
        assert row["n_settled"] == 1 and row["n_positions"] == 0 and book.is_empty
        intrinsic = max(scenario.spot[63] - 95.0, 0.0)
        assert row["settlement_cash"] == pytest.approx(-10 * intrinsic)
        assert row["value"] == pytest.approx(expected_value, abs=1e-9)
        assert row["positions_value"] == 0.0 and row["value"] == pytest.approx(book.cash)
        assert sum(row[c] for c in TERM_COLUMNS) == pytest.approx(row["pnl_step"], abs=1e-9)
        history = sim.run()
        assert len(history) == 127
        # after expiry only cash remains: P&L is pure interest
        np.testing.assert_allclose(history["pnl_step"].iloc[64:], history["pnl_carry"].iloc[64:], atol=1e-10)
        assert len(book.settlements) == 1

    def test_path_dependent_instruments_are_observed(self, mkt0):
        tracker = MaxTracker(expiry=0.25)
        book = Book()
        book.trade(tracker, 2, mkt0)
        scenario = gbm_scenario(mkt0, 0.25, 40, realized_vol=0.4, seed=12)
        sim = TradingSimulator(book, scenario)
        history = sim.run()
        assert len(book.settlements) == 1
        assert book.settlements[0].unit_value == pytest.approx(scenario.spot.max())
        assert book.settlements[0].instrument.running_max == pytest.approx(scenario.spot.max())
        np.testing.assert_allclose(history[TERM_COLUMNS].sum(axis=1), history["pnl_step"], atol=1e-9)

    def test_reproducible_and_reset(self, mkt0, call):
        first = _long_call_sim(mkt0, call, DeltaHedgeEveryN(2), seed=21)
        second = _long_call_sim(mkt0, call, DeltaHedgeEveryN(2), seed=21)
        history = first.run()
        pd.testing.assert_frame_equal(history, second.run())
        original_book = first.book
        first.reset()
        assert first.step_index == 0 and not first.is_finished and len(first.history) == 1
        assert first.book is not original_book
        assert first.book.quantity(call) == 100 and len(first.book.trades) == 2  # option + initial hedge
        pd.testing.assert_frame_equal(first.run(), history)

    def test_manual_trades_between_steps(self, mkt0, call):
        sim = _long_call_sim(mkt0, call, NoHedge(), costs=TransactionCosts(option_pct=0.01))
        sim.run(5)
        put = EuropeanOption("put", 95.0, 0.25)
        mkt = sim.current_market
        model = put.price(mkt)
        cheap = sim.trade(put, 10, price=model - 0.25, fee=1.0, note="bought cheap")
        assert cheap.t == mkt.t and sim.book.quantity(put) == 10
        row = sim.step()
        assert row["pnl_trading"] == pytest.approx(10 * 0.25, abs=1e-9)
        assert row["pnl_fees"] == pytest.approx(-cheap.fees) and cheap.fees > 1.0
        assert row["fees_paid"] == pytest.approx(cheap.fees)
        assert sum(row[c] for c in TERM_COLUMNS) == pytest.approx(row["pnl_step"], abs=1e-9)
        next_row = sim.step()
        assert next_row["pnl_trading"] == pytest.approx(0.0, abs=1e-9) and next_row["pnl_fees"] == 0.0

    def test_deposit_is_not_pnl(self, mkt0, call):
        sim = _long_call_sim(mkt0, call, NoHedge())
        sim.run(3)
        reference = _long_call_sim(mkt0, call, NoHedge())
        reference.run(3)
        sim.book.deposit(10_000.0)
        assert sim.step()["pnl_trading"] == pytest.approx(0.0, abs=1e-9)
        # only the extra interest differs
        assert sim.last_row["pnl_step"] - reference.step()["pnl_step"] == pytest.approx(
            10_000.0 * math.expm1(0.02 * sim.scenario.dt(3)), rel=1e-6)

    def test_costs_and_hedge_at_start(self, mkt0, call):
        costs = TransactionCosts(stock_bps=10.0)
        sim = _long_call_sim(mkt0, call, DeltaHedgeEveryN(1), costs=costs)
        assert sim.book.costs is costs
        first = sim.history.iloc[0]
        assert first["hedge_trades"] == 1 and first["fees_paid"] > 0
        assert first["pnl_step"] == pytest.approx(-first["fees_paid"]) == pytest.approx(first["pnl_fees"])
        assert first["fees_paid"] == pytest.approx(abs(first["hedge_shares"]) * 100.0 * 10.0 / 1e4)
        history = sim.run()
        assert history["cum_pnl_fees"].iloc[-1] == pytest.approx(-history["fees_paid"].sum())
        assert history["cum_pnl_fees"].iloc[-1] < -first["fees_paid"]

        book = Book()
        book.trade(call, 100, mkt0)
        lazy = TradingSimulator(book, sim.scenario, DeltaHedgeEveryN(1), hedge_at_start=False)
        assert lazy.history["hedge_trades"].iloc[0] == 0 and len(book.trades) == 1
        assert lazy.step()["hedge_trades"] == 1

    def test_stepping_api(self, mkt0, call):
        sim = _long_call_sim(mkt0, call)
        assert sim.steps_remaining == 63 and sim.current_market == mkt0
        assert len(sim.run(10)) == 11 and sim.step_index == 10
        assert sim.current_market.t == pytest.approx(sim.scenario.times[10])
        assert len(sim.run(1000)) == 64
        with pytest.raises(RuntimeError):
            sim.step()
        assert len(sim.run()) == 64  # running a finished simulation is harmless
        assert sim.last_row["t"] == pytest.approx(0.25)
        with pytest.raises(ValueError):
            sim.run(-1)

    def test_policy_can_be_swapped_between_steps(self, mkt0, call):
        sim = _long_call_sim(mkt0, call, NoHedge())
        sim.run(5)
        sim.policy = DeltaHedgeEveryN(1)
        assert sim.step()["hedge_trades"] == 1 and abs(sim.last_row["delta"]) < 1e-9

    def test_constructor_validation(self, mkt0, call):
        scenario = gbm_scenario(mkt0, 0.25, 5, seed=1)
        with pytest.raises(ValueError):
            TradingSimulator("book", scenario)
        with pytest.raises(ValueError):
            TradingSimulator(Book(), "scenario")
        with pytest.raises(ValueError):
            TradingSimulator(Book(), scenario, policy="daily")
        with pytest.raises(ValueError):
            TradingSimulator(Book(), scenario, costs=0.01)

    def test_gamma_scalping_on_one_long_path(self, call):
        mkt = Market(100.0, 0.2)
        option = EuropeanOption("call", 100.0, 1.0)
        results = {}
        for realized in (0.35, 0.10):
            book = Book()
            book.trade(option, 100, mkt)
            scenario = gbm_scenario(mkt, 1.0, 252, realized_vol=realized, seed=5)
            history = TradingSimulator(book, scenario, DeltaHedgeEveryN(1)).run()
            results[realized] = gamma_scalping_summary(history)
        rich, poor = results[0.35], results[0.10]
        assert rich["gamma_plus_theta"] > 0 and rich["gamma_theta_ratio"] > 1 and rich["vol_edge"] > 0
        assert poor["gamma_plus_theta"] < 0 and poor["gamma_theta_ratio"] < 1 and poor["vol_edge"] < 0
        assert rich["theta_pnl"] < 0 < rich["gamma_pnl"]
        assert rich["realized_vol"] == pytest.approx(0.35, rel=0.15)
        assert rich["mean_implied_vol"] == pytest.approx(0.2)
        parts = ("gamma_pnl", "theta_pnl", "delta_pnl", "vega_pnl", "other_pnl")
        assert sum(rich[k] for k in parts) == pytest.approx(rich["total_pnl"])

    def test_history_helpers_validate(self):
        with pytest.raises(ValueError):
            gamma_scalping_summary(pd.DataFrame({"t": [0.0]}))
        with pytest.raises(ValueError):
            attribution_totals("nope")


# ---------------------------------------------------------------------- #
# Hedging experiments
# ---------------------------------------------------------------------- #
class TestHedgingExperiment:
    def test_mean_zero_and_sqrt_n_law(self, mkt0):
        option = EuropeanOption("call", 100.0, 1.0)
        experiment = delta_hedging_experiment(option, mkt0, n_paths=6000, seed=1,
                                              rebalance_steps_list=(13, 52, 252))
        assert len(experiment) == 3 * 6000
        assert set(experiment.columns) >= {"path", "rebalance_steps", "pnl", "pnl_pct_premium",
                                           "terminal_spot", "theory_pnl"}
        assert experiment.attrs["vectorised"] is True
        assert experiment.attrs["premium"] == pytest.approx(option.price(mkt0))
        summary = hedging_experiment_summary(experiment)
        assert list(summary.index) == [13, 52, 252]
        assert np.all(summary["mean"].abs() < 4 * summary["stderr"])
        assert summary.loc[13, "std"] / summary.loc[52, "std"] == pytest.approx(2.0, abs=0.3)
        assert summary.loc[52, "std"] / summary.loc[252, "std"] == pytest.approx(math.sqrt(252 / 52), rel=0.15)
        assert summary["std_x_sqrt_n"].max() / summary["std_x_sqrt_n"].min() < 1.2
        np.testing.assert_allclose(summary["theory_mean"], 0.0, atol=1e-12)
        # the same paths are used for every frequency
        spots = experiment.pivot(index="path", columns="rebalance_steps", values="terminal_spot")
        np.testing.assert_allclose(spots[13], spots[252])

    def test_daily_hedging_error_matches_the_textbook_size(self, mkt0):
        option = EuropeanOption("call", 100.0, 1.0)
        experiment = delta_hedging_experiment(option, mkt0, n_paths=4000, seed=2, rebalance_steps_list=(252,))
        vega = option.greeks(mkt0)["vega"]
        textbook = math.sqrt(math.pi / 4) * vega * 0.2 / math.sqrt(252)  # Derman-Kamal rule of thumb
        assert experiment["pnl"].std() == pytest.approx(textbook, rel=0.25)

    def test_realised_above_implied_pays_long_gamma(self, mkt0):
        option = EuropeanOption("put", 100.0, 0.5)
        kwargs = dict(n_paths=4000, seed=3, rebalance_steps_list=(126,))
        rich = hedging_experiment_summary(delta_hedging_experiment(option, mkt0, realized_vol=0.3, **kwargs))
        poor = hedging_experiment_summary(delta_hedging_experiment(option, mkt0, realized_vol=0.1, **kwargs))
        short = hedging_experiment_summary(
            delta_hedging_experiment(option, mkt0, realized_vol=0.3, quantity=-1.0, **kwargs))
        assert rich.loc[126, "mean"] > 10 * rich.loc[126, "stderr"]
        assert poor.loc[126, "mean"] < -10 * poor.loc[126, "stderr"]
        assert short.loc[126, "mean"] == pytest.approx(-rich.loc[126, "mean"], rel=1e-9)
        for summary in (rich, poor):
            assert summary.loc[126, "mean"] == pytest.approx(summary.loc[126, "theory_mean"], rel=0.05)
        # the mean is the vol edge: the option revalued at the realised vol
        edge = (option.price(mkt0.bumped(vol=0.3)) - option.price(mkt0)) * math.exp(0.02 * 0.5)
        assert rich.loc[126, "mean"] == pytest.approx(edge, rel=0.05)

    def test_hedging_at_the_realised_vol_locks_the_profit(self, mkt0):
        option = EuropeanOption("call", 100.0, 0.5)
        kwargs = dict(n_paths=2000, seed=4, rebalance_steps_list=(252,), realized_vol=0.3)
        at_implied = delta_hedging_experiment(option, mkt0, **kwargs)
        at_realised = delta_hedging_experiment(option, mkt0, hedge_vol=0.3, **kwargs)
        assert at_realised.attrs["hedge_vol"] == 0.3
        assert at_realised["pnl"].std() < 0.35 * at_implied["pnl"].std()
        edge = (option.price(mkt0.bumped(vol=0.3)) - option.price(mkt0)) * math.exp(0.02 * 0.5)
        assert at_realised["pnl"].mean() == pytest.approx(edge, rel=0.02)
        np.testing.assert_allclose(at_realised["theory_pnl"], edge, rtol=1e-9)

    @pytest.mark.parametrize("hedge_vol", [0.2, 0.3])
    def test_vectorised_route_matches_the_book(self, mkt0, hedge_vol):
        option = EuropeanOption("put", 105.0, 0.25)
        costs = TransactionCosts(stock_bps=5.0, option_pct=0.01)
        times, paths = simulate_gbm_paths(100.0, 0.3, 0.06, 0.01, 0.25, 12, n_paths=6, seed=5)
        fast, _ = _vectorised_hedging_pnl(option, -3.0, mkt0, times, paths, hedge_vol, 0.3, costs)
        policy = DeltaHedgeEveryN(1) if hedge_vol == 0.2 else DeltaHedgeAtVol(hedge_vol)
        slow = _generic_hedging_pnl(option, -3.0, mkt0, times, paths, policy, costs)
        np.testing.assert_allclose(fast, slow, atol=1e-9)

    def test_vectorised_route_handles_packages_and_positions(self, mkt0):
        straddle = CompositeInstrument(
            "straddle", (EuropeanOption("call", 100.0, 0.5), EuropeanOption("put", 100.0, 0.5)))
        experiment = delta_hedging_experiment(Position(straddle, 2.0), mkt0, n_paths=500, seed=6,
                                              rebalance_steps_list=(26,), realized_vol=0.3, quantity=5.0)
        assert experiment.attrs["vectorised"] is True and experiment.attrs["quantity"] == 10.0
        assert experiment["pnl"].mean() > 0

    def test_generic_route_for_path_dependent_products(self, mkt0):
        tracker = MaxTracker(expiry=0.25)
        experiment = delta_hedging_experiment(tracker, mkt0, n_paths=8, seed=7, rebalance_steps_list=(4, 8))
        assert experiment.attrs["vectorised"] is False
        assert experiment["theory_pnl"].isna().all() and np.isfinite(experiment["pnl"]).all()
        assert math.isnan(hedging_experiment_summary(experiment).loc[4, "theory_mean"])

    def test_generic_route_for_calendars(self, mkt0):
        calendar = CompositeInstrument(
            "calendar", (Position(EuropeanOption("call", 100.0, 0.5), 1.0),
                         Position(EuropeanOption("call", 100.0, 0.25), -1.0)))
        experiment = delta_hedging_experiment(calendar, mkt0, n_paths=6, seed=8, rebalance_steps_list=(8,))
        assert experiment.attrs["vectorised"] is False and np.isfinite(experiment["pnl"]).all()
        assert experiment["pnl"].abs().max() < calendar.price(mkt0) * 3

    def test_seed_reproducibility(self, mkt0, call):
        a = delta_hedging_experiment(call, mkt0, n_paths=50, seed=9, rebalance_steps_list=(4, 13))
        b = delta_hedging_experiment(call, mkt0, n_paths=50, seed=9, rebalance_steps_list=(4, 13))
        c = delta_hedging_experiment(call, mkt0, n_paths=50, seed=10, rebalance_steps_list=(4, 13))
        pd.testing.assert_frame_equal(a, b)
        assert not np.allclose(a["pnl"], c["pnl"])

    def test_costs_reduce_the_mean(self, mkt0, call):
        kwargs = dict(n_paths=500, seed=11, rebalance_steps_list=(63,))
        free = delta_hedging_experiment(call, mkt0, **kwargs)
        costly = delta_hedging_experiment(call, mkt0, costs=TransactionCosts(stock_bps=20.0), **kwargs)
        assert np.all(costly["pnl"].to_numpy() < free["pnl"].to_numpy())

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_paths": 1}, {"rebalance_steps_list": ()}, {"rebalance_steps_list": (0,)},
            {"realized_vol": -0.2}, {"quantity": 0.0}, {"hedge_vol": 0.0}, {"costs": 0.01},
        ],
    )
    def test_validation(self, mkt0, call, kwargs):
        with pytest.raises(ValueError):
            delta_hedging_experiment(call, mkt0, **{"n_paths": 10, **kwargs})

    def test_needs_a_live_expiry_and_a_scalar_market(self, mkt0, call):
        with pytest.raises(ValueError):
            delta_hedging_experiment(Underlying(), mkt0, n_paths=10)
        with pytest.raises(ValueError):
            delta_hedging_experiment(call, mkt0.bumped(t=0.25), n_paths=10)
        with pytest.raises(ValueError):
            delta_hedging_experiment(call, Market(np.array([99.0, 100.0]), 0.2), n_paths=10)
        with pytest.raises(ValueError):
            delta_hedging_experiment("call", mkt0, n_paths=10)
        with pytest.raises(ValueError):
            hedging_experiment_summary(pd.DataFrame({"pnl": [1.0]}))


# ---------------------------------------------------------------------- #
# Figures
# ---------------------------------------------------------------------- #
class TestFigures:
    @pytest.fixture(scope="class")
    def history(self) -> pd.DataFrame:
        mkt = Market(spot=100.0, vol=0.2, rate=0.02, div=0.01)
        book = Book("figures")
        book.trade(EuropeanOption("call", 100.0, 0.25), 100, mkt)
        scenario = gbm_scenario(mkt, 0.25, 30, realized_vol=0.3, implied_vol_model="spot_correlated", seed=3)
        return TradingSimulator(book, scenario, DeltaHedgeEveryN(1), costs=TransactionCosts(stock_bps=5)).run()

    @pytest.fixture(scope="class")
    def experiment(self) -> pd.DataFrame:
        mkt = Market(spot=100.0, vol=0.2)
        return delta_hedging_experiment(EuropeanOption("call", 100.0, 0.25), mkt, n_paths=300, seed=1,
                                        rebalance_steps_list=(4, 16, 64))

    @pytest.mark.parametrize("mode", ["auto", "light", "dark"])
    def test_simulation_figures_build(self, history, mode):
        figures = [
            sim_plots.simulation_dashboard(history, mode=mode),
            sim_plots.simulation_dashboard(history, x="step", mode=mode),
            sim_plots.pnl_attribution_chart(history, mode=mode),
            sim_plots.pnl_attribution_chart(history, cumulative=False, x="step", mode=mode),
            sim_plots.pnl_attribution_waterfall(history, mode=mode),
            sim_plots.gamma_theta_chart(history, mode=mode),
        ]
        for fig in figures:
            assert isinstance(fig, go.Figure) and len(fig.data) >= 1
            assert fig.layout.title.text
            json.loads(fig.to_json())

    def test_dashboard_has_one_axis_per_panel(self, history):
        fig = sim_plots.simulation_dashboard(history)
        assert len({trace.yaxis for trace in fig.data}) == 8  # eight panels, no secondary axis
        assert all(getattr(axis, "overlaying", None) is None for axis in fig.select_yaxes())

    def test_attribution_chart_content(self, history):
        fig = sim_plots.pnl_attribution_chart(history)
        names = [trace.name for trace in fig.data]
        assert names[-1] == "Actual P&L" and {"Delta", "Gamma", "Theta", "Vega", "Unexplained"} <= set(names)
        np.testing.assert_allclose(fig.data[-1].y, history["pnl_cum"])
        colors = {trace.name: trace.line.color for trace in fig.data}
        bars = sim_plots.pnl_attribution_chart(history, cumulative=False)
        assert bars.layout.barmode == "relative"
        for trace in bars.data[:-1]:
            assert trace.marker.color == colors[trace.name]  # colour follows the term

    def test_waterfall_sums_to_actual(self, history):
        fig = sim_plots.pnl_attribution_waterfall(history)
        trace = fig.data[0]
        assert trace.measure[-1] == "total" and trace.x[-1] == "Actual P&L"
        assert sum(trace.y[:-1]) == pytest.approx(trace.y[-1], abs=1e-6)
        assert trace.y[-1] == pytest.approx(history["pnl_cum"].iloc[-1])
        assert "Rho" not in trace.x and "Fees" in trace.x

    def test_experiment_figures_build(self, experiment):
        hist = sim_plots.hedging_error_histogram(experiment)
        assert len(hist.data) == 3
        for trace in hist.data:
            assert sum(trace.y) == pytest.approx(100.0)
        titles = [a.text for a in hist.layout.annotations if "rebalances" in (a.text or "")]
        assert len(titles) == 3 and all("std" in t for t in titles)
        pct = sim_plots.hedging_error_histogram(experiment, pct_of_premium=True, bins=21)
        assert "% of premium" in pct.layout.title.text
        line = sim_plots.hedging_error_vs_frequency(experiment)
        assert line.layout.xaxis.type == "log" and line.layout.yaxis.type == "log"
        assert list(line.data[1].x) == [4, 16, 64]
        json.loads(hist.to_json()), json.loads(line.to_json())

    def test_figure_validation(self, history, experiment):
        with pytest.raises(ValueError):
            sim_plots.simulation_dashboard(history.drop(columns=["spot"]))
        with pytest.raises(ValueError):
            sim_plots.simulation_dashboard(history, x="days")
        with pytest.raises(ValueError):
            sim_plots.pnl_attribution_chart(history.iloc[0:0])
        with pytest.raises(ValueError):
            sim_plots.pnl_attribution_waterfall(experiment)
        with pytest.raises(ValueError):
            sim_plots.gamma_theta_chart("history")
        with pytest.raises(ValueError):
            sim_plots.hedging_error_histogram(experiment, bins=1)
        with pytest.raises(ValueError):
            sim_plots.hedging_error_histogram(history)
        with pytest.raises(ValueError):
            sim_plots.hedging_error_vs_frequency(history)
        with pytest.raises(ValueError):
            sim_plots.simulation_dashboard(history, mode="sepia")


# ---------------------------------------------------------------------- #
# The terminal trading game (examples/05_trading_game.py)
# ---------------------------------------------------------------------- #
GAME_PATH = Path(__file__).resolve().parents[1] / "examples" / "05_trading_game.py"


@pytest.fixture(scope="module")
def game_module():
    spec = importlib.util.spec_from_file_location("trading_game_example", GAME_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTradingGameExample:
    @staticmethod
    def _game(module, *argv):
        args = module.parse_args(["--seed", "3", "--days", "10", *argv])
        game, hidden_vol, seed = module.build_game(args)
        lines: list[str] = []
        game.say = lines.append
        assert seed == 3 and 0.6 * 0.2 <= hidden_vol <= 1.5 * 0.2
        return game, lines

    def test_orders_trade_the_book(self, game_module):
        game, lines = self._game(game_module)
        game.execute("buy 10 call 105 0.25")
        game.execute("SELL 5 Put 95 5d")
        game.execute("buy 2 straddle 100 5d")
        game.execute("stock -30")
        game.execute("sell 20 stock")
        book = game.book
        assert book.quantity(EuropeanOption("call", 105.0, 0.25)) == 10
        five_days = float(game.sim.scenario.times[5])
        assert book.quantity(EuropeanOption("put", 95.0, five_days)) == -5
        assert book.quantity(EuropeanOption("call", 100.0, five_days)) == 2
        assert book.underlying_quantity() == -50
        assert not any(line.lstrip().startswith("!") for line in lines)

        game.execute("hedge")
        assert abs(book.greeks(game.mkt)["delta"]) < 1e-9
        game.execute("hedge 25")
        assert book.greeks(game.mkt)["delta"] == pytest.approx(25.0)
        game.execute("close 1")
        assert book.quantity(EuropeanOption("call", 105.0, 0.25)) == 0
        game.execute("close all")
        assert book.is_empty and book.pnl(game.mkt) < 0  # the round trip only cost the fees

    def test_bad_input_never_raises_or_trades(self, game_module):
        game, lines = self._game(game_module)
        bad = [
            "frobnicate", "buy", "buy lots of calls", "buy ten call 100 5d", "buy 10 call 100", "buy -5 call 100 5d",
            "buy 10 call -100 5d", "buy 10 call 100 0d", "buy 10 call 100 1.5d", "buy 10 call 100 -1",
            "buy 10 banana 100 5d", "buy nan call 100 5d", "buy 1e999 stock", "stock", "stock 0", "stock many",
            "hedge a", "hedge 1 2", "autohedge", "autohedge perhaps", "close", "close 1", "close all", "quote",
            "quote call 100", "chain 1 2", "next 0", "next 1.5", "next soon", "next 1 2",
        ]
        for line in bad:
            before = len(lines)
            game.execute(line)
            assert any(text.lstrip().startswith("!") for text in lines[before:]), line
        assert game.book.is_empty and not game.book.trades and game.sim.step_index == 0 and game.running
        for line in ["", "   ", "help", "?", "book", "risk", "explain", "blotter", "chain", "chain 5d",
                     "quote straddle 100 5d"]:
            game.execute(line)  # information commands work on an empty book too
        assert game.running and game.sim.step_index == 0

    def test_clock_autohedge_settlement_and_end_of_game(self, game_module):
        game, lines = self._game(game_module, "--no-costs")
        game.execute("buy 50 straddle 100 5d")
        game.execute("autohedge on")
        assert isinstance(game.sim.policy, DeltaHedgeEveryN)
        assert abs(game.book.greeks(game.mkt)["delta"]) < 1e-9  # hedged immediately
        game.execute("next 4")
        assert game.sim.step_index == 4 and len(game.book.settlements) == 0
        game.execute("n")
        assert game.sim.step_index == 5 and len(game.book.settlements) == 2
        assert any("EXPIRED" in line for line in lines)
        game.execute("autohedge off")
        assert isinstance(game.sim.policy, NoHedge)
        game.execute("risk"), game.execute("explain"), game.execute("blotter")
        game.execute("next 500")
        assert game.sim.is_finished and not game.running

        history = game.sim.history
        assert history["pnl_cum"].iloc[-1] == pytest.approx(game.book.pnl(game.mkt))
        game.final_report(0.25)
        assert any("FINAL REPORT" in line for line in lines)

    def test_quit_and_margin_call(self, game_module):
        game, _ = self._game(game_module)
        game.execute("quit")
        assert not game.running
        game.final_report(0.2)  # nothing happened: the report must still work

        game, lines = self._game(game_module, "--capital", "100")
        game.execute("sell 5000 straddle 100 0.5")  # absurd leverage: the first move wipes the capital out
        game.execute("next 10")
        assert not game.running and any("MARGIN CALL" in line for line in lines)

    def test_coach_reads_the_attribution(self, game_module):
        zero = dict.fromkeys([*ATTRIBUTION_TERMS, "actual"], 0.0)
        summary = {"gamma_plus_theta": 50.0, "vol_edge": 0.05, "realized_vol": 0.25, "mean_implied_vol": 0.2}
        assert "no market risk" in game_module.coach(zero, summary)[0]
        long_gamma = {**zero, "gamma": 150.0, "theta": -100.0, "actual": 50.0}
        text = " ".join(game_module.coach(long_gamma, summary))
        assert "LONG gamma" in text and "should do" in text
        against = " ".join(game_module.coach(long_gamma, {**summary, "vol_edge": -0.05}))
        assert "path dependent" in against
        short_gamma = {**zero, "delta": 900.0, "gamma": -150.0, "theta": 100.0, "fees": -40.0, "actual": 810.0}
        text = " ".join(game_module.coach(short_gamma, {**summary, "gamma_plus_theta": -50.0}))
        assert "DIRECTIONAL" in text and "SHORT gamma" in text

    def test_scripted_main_writes_outputs(self, game_module, tmp_path, capsys):
        script = "buy 20 straddle 100 5d; hedge; next 3; oops; autohedge on; next 100"
        game_module.main(["--seed", "5", "--days", "8", "--script", script, "--outdir", str(tmp_path)])
        printed = capsys.readouterr().out
        assert "FINAL REPORT" in printed and "unknown command 'oops'" in printed and "hidden number" in printed
        written = {path.name for path in tmp_path.iterdir()}
        assert {"05_01_dashboard.html", "05_02_attribution.html", "05_03_waterfall.html",
                "05_04_gamma_vs_theta.html", "05_history.csv", "05_blotter.csv"} <= written
        replay = pd.read_csv(tmp_path / "05_history.csv")
        assert len(replay) == 9 and replay["pnl_cum"].iloc[-1] != 0.0
