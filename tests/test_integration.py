"""Cross-module tests: every product through serialisation, the book, the simulator and the plots."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

import optionlab as ol
from optionlab import black_scholes as bs
from optionlab.exotics import EXOTIC_REGISTRY
from optionlab.plotting import book_plots, profiles, sim_plots, strategy_plots
from optionlab.simulator import ATTRIBUTION_TERMS
from optionlab.strategies import STRATEGY_REGISTRY, summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MKT = ol.Market(spot=100.0, vol=0.2, rate=0.03, div=0.01)

VANILLA = ol.EuropeanOption("call", 100.0, 0.5)
DIGITAL = ol.DigitalOption("call", 102.0, 0.04, payout=10.0)  # expires inside the simulated scenario
KNOCK_OUT = ol.BarrierOption("call", strike=100.0, barrier=106.0, expiry=0.5,
                             barrier_type="up-and-out", rebate=2.0)
ASIAN = ol.AsianOption("put", 100.0, 0.5)
LOOKBACK = ol.LookbackOption("call", 0.5)
IRON_CONDOR = STRATEGY_REGISTRY["iron_condor"].build_default(spot=100.0, expiry=0.5)
CALENDAR = STRATEGY_REGISTRY["calendar_spread"].build_default(spot=100.0, expiry=0.25)

PRODUCTS = {
    "underlying": ol.Underlying(),
    "vanilla": VANILLA,
    "composite": ol.CompositeInstrument("risk reversal", ((VANILLA, 1.0), (ol.EuropeanOption("put", 90.0, 0.5), -1.0))),
    "strategy": IRON_CONDOR,
    "calendar": CALENDAR,
    "digital": DIGITAL,
    "barrier": KNOCK_OUT,
    "asian": ASIAN,
    "lookback": LOOKBACK,
}


def mixed_book() -> ol.Book:
    """A vanilla, a registry strategy, a digital, a barrier, an Asian and a lookback (plus stock)."""
    book = ol.Book("everything", cash=10_000.0, costs=ol.TransactionCosts(stock_bps=1.0, option_pct=0.002))
    book.trade(VANILLA, 10, MKT, note="vanilla")
    book.trade(IRON_CONDOR, -5, MKT, note="strategy")
    book.trade(DIGITAL, 3, MKT, note="digital")
    book.trade(KNOCK_OUT, 4, MKT, note="barrier")
    book.trade(ASIAN, -6, MKT, note="asian")
    book.trade(LOOKBACK, 2, MKT, note="lookback")
    book.hedge_delta(MKT)
    return book


# ---------------------------------------------------------------------- #
# One Greek convention everywhere
# ---------------------------------------------------------------------- #
class TestGreekConventions:
    @pytest.mark.parametrize("name", list(PRODUCTS))
    def test_same_keys_floats_for_scalars_arrays_for_grids(self, name):
        product = PRODUCTS[name]
        scalar = product.greeks(MKT)
        assert tuple(scalar) == ol.GREEK_KEYS == ("price",) + bs.GREEK_NAMES
        assert all(isinstance(value, float) for value in scalar.values())
        assert scalar["price"] == pytest.approx(float(product.price(MKT)))

        spots = np.array([85.0, 100.0, 104.0, 115.0])
        grid = product.greeks(MKT.bumped(spot=spots))
        assert tuple(grid) == ol.GREEK_KEYS
        assert all(np.shape(value) == (4,) for value in grid.values())
        np.testing.assert_allclose(grid["price"], product.price(MKT.bumped(spot=spots)))
        assert grid["delta"][1] == pytest.approx(scalar["delta"])

    @pytest.mark.parametrize("name", ["digital", "barrier", "asian", "lookback", "strategy"])
    def test_greeks_are_raw_derivatives(self, name):
        # With the spot sitting ON its running extreme a lookback has a kink (a down bump drags the
        # minimum along), so the plain-bump comparison uses a lookback away from its extremes.
        seasoned = ol.LookbackOption("call", 0.5, running_min=92.0, running_max=104.0)
        product = seasoned if name == "lookback" else PRODUCTS[name]
        greeks = product.greeks(MKT)
        h, dv = 1e-3, 1e-4
        up, down = product.price(MKT.bumped(spot=100.0 + h)), product.price(MKT.bumped(spot=100.0 - h))
        assert greeks["delta"] == pytest.approx((up - down) / (2 * h), rel=1e-4, abs=1e-6)
        vol_up, vol_down = product.price(MKT.bumped(vol=0.2 + dv)), product.price(MKT.bumped(vol=0.2 - dv))
        assert greeks["vega"] == pytest.approx((vol_up - vol_down) / (2 * dv), rel=1e-4, abs=1e-5)  # per 1.00 of vol
        later = product.price(MKT.bumped(t=1e-5))
        assert greeks["theta"] == pytest.approx((later - greeks["price"]) / 1e-5, rel=2e-2, abs=1e-3)  # per YEAR

    def test_position_and_book_use_the_same_keys(self):
        book = mixed_book()
        assert tuple(book.greeks(MKT)) == ol.GREEK_KEYS
        assert tuple(ol.Position(ASIAN, -2.0).greeks(MKT)) == ol.GREEK_KEYS
        total = sum(position.greeks(MKT)["vega"] for position in book.positions)
        assert book.greeks(MKT)["vega"] == pytest.approx(total)

    def test_trader_units_leave_non_greek_entries_alone(self):
        row = {"leg": "+1 x C 100", "strike": None, "quantity": 2, "vega": 50.0, "theta": -7.3}
        assert bs.to_trader_units(row) == {
            "leg": "+1 x C 100", "strike": None, "quantity": 2, "vega": 0.5, "theta": -7.3 / 365.0,
        }


# ---------------------------------------------------------------------- #
# Registries and serialisation
# ---------------------------------------------------------------------- #
class TestRegistriesAndSerialisation:
    def test_every_instrument_class_is_registered_by_importing_the_package(self):
        assert set(ol.INSTRUMENT_REGISTRY) >= {
            "Underlying", "EuropeanOption", "CompositeInstrument", "Strategy",
            "DigitalOption", "BarrierOption", "AsianOption", "LookbackOption",
        }

    @pytest.mark.parametrize("name", list(EXOTIC_REGISTRY))
    def test_exotic_defaults_round_trip_through_json(self, name):
        spec = EXOTIC_REGISTRY[name]
        product = spec.build_default(spot=250.0, expiry=0.75, t=0.25)
        assert isinstance(product, spec.instrument_class) and product.expiry == 0.75
        clone = ol.instrument_from_dict(json.loads(json.dumps(product.to_dict())))
        assert clone == product and hash(clone) == hash(product)
        mkt = ol.Market(250.0, 0.3, 0.02, 0.0, t=0.25)
        assert clone.price(mkt) == product.price(mkt) > 0.0

    @pytest.mark.parametrize("name", list(STRATEGY_REGISTRY))
    def test_strategy_defaults_round_trip_through_json(self, name):
        strategy = STRATEGY_REGISTRY[name].build_default(spot=80.0, expiry=0.5)
        clone = ol.instrument_from_dict(json.loads(json.dumps(strategy.to_dict())))
        assert clone == strategy and isinstance(clone, ol.Strategy) and clone.key == name

    def test_exotic_registry_schema(self):
        spec = ol.exotics.get_exotic_spec("Up-and-out barrier")
        assert spec is EXOTIC_REGISTRY["up_and_out_barrier"]
        assert list(spec.schema) == ["option_type", "strike", "barrier", "expiry", "rebate"]
        assert spec.default_params(200.0, 1.0) == {
            "option_type": "call", "strike": 200.0, "barrier": 240.0, "expiry": 1.0, "rebate": 0.0,
        }
        built = spec.build_default(200.0, 1.0, barrier=230.0, rebate=1.5)
        assert (built.barrier, built.rebate, built.barrier_type) == (230.0, 1.5, "up-and-out")
        assert ol.list_exotics("barrier") == [
            "up_and_out_barrier", "up_and_in_barrier", "down_and_out_barrier", "down_and_in_barrier",
        ]
        assert ol.list_exotics() == list(EXOTIC_REGISTRY)
        assert all(len(s.description) > 80 and s.title for s in EXOTIC_REGISTRY.values())
        asian = ol.build_exotic("asian", option_type="put", strike=95.0, expiry=1.0, averaging="geometric")
        assert asian == ol.AsianOption("put", 95.0, 1.0, averaging="geometric")
        assert EXOTIC_REGISTRY["asian"].schema["averaging"].choices == ("arithmetic", "geometric")
        forward_start = EXOTIC_REGISTRY["asian"].default_params(100.0, expiry=1.0, t=0.5)
        assert forward_start["avg_start"] == 0.5 and forward_start["expiry"] == 1.0

    def test_exotic_registry_errors(self):
        spec = EXOTIC_REGISTRY["cash_digital"]
        with pytest.raises(ValueError, match="Unknown exotic"):
            ol.build_exotic("quanto")
        with pytest.raises(ValueError, match="unknown parameter"):
            spec.build_default(100.0, 1.0, barrier=120.0)
        with pytest.raises(ValueError, match="unknown parameter"):
            spec.build(strike=100.0, expiry=1.0, kind="asset")  # fixed by the spec, not a user parameter
        with pytest.raises(ValueError, match="expected"):
            spec.build(option_type="call")  # strike and expiry missing
        with pytest.raises(ValueError, match="strike"):
            spec.build(option_type="call", strike=-1.0, expiry=1.0)
        with pytest.raises(ValueError, match="spot"):
            spec.default_params(0.0, 1.0)
        with pytest.raises(ValueError, match="later"):
            spec.default_params(100.0, expiry=0.5, t=0.5)
        with pytest.raises(ValueError, match="kind"):
            ol.exotics.ExoticParam("x", "level", 1.0)
        with pytest.raises(ValueError, match="not one of"):
            ol.exotics.ExoticParam("x", "choice", "c", choices=("a", "b"))

    def test_book_with_every_product_round_trips_through_json(self, tmp_path):
        book = mixed_book()
        path = tmp_path / "book.json"
        clone = ol.Book.from_json(book.to_json(path))
        assert clone.positions == book.positions and clone.trades == book.trades
        assert clone.cash == book.cash and clone.costs == book.costs
        assert clone.value(MKT) == book.value(MKT)
        assert clone.greeks(MKT) == book.greeks(MKT)
        ladder = MKT.bumped(spot=np.linspace(80.0, 120.0, 9))
        np.testing.assert_array_equal(clone.value(ladder), book.value(ladder))
        assert ol.Book.from_json(path).to_dict() == book.to_dict()

    def test_a_fresh_interpreter_importing_only_the_book_module_can_load_everything(self, tmp_path):
        path = tmp_path / "book.json"
        book = mixed_book()
        book.to_json(path)
        code = (
            "import sys; from optionlab.book import Book; from optionlab.market import Market; "
            "book = Book.from_json(sys.argv[1]); print(repr(book.value(Market(100.0, 0.2, 0.03, 0.01)))); "
            "print('plotly' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(path)], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        )
        value, plotly_loaded = result.stdout.split()
        assert float(value) == book.value(MKT)
        assert plotly_loaded == "False"  # `import optionlab` stays light: plotly only with optionlab.plotting

    def test_composite_subclasses_survive_observe_and_scaled(self):
        package = ol.Strategy("KO + stock", ((KNOCK_OUT, 2.0), (ol.Underlying(), -1.0)),
                              description="teaching text", view=("bullish",), key="custom")
        observed = package.observe(107.0, 0.1)
        assert type(observed) is ol.Strategy and observed.description == "teaching text"
        assert observed.legs[0].instrument.knocked and observed.view == ("bullish",)
        assert package.observation_cash_flow(observed) == pytest.approx(2.0 * KNOCK_OUT.rebate)
        assert package.observe(101.0, 0.1) is package
        scaled = package.scaled(-3.0)
        assert type(scaled) is ol.Strategy and scaled.key == "custom" and scaled.legs[0].quantity == -6.0


# ---------------------------------------------------------------------- #
# Life cycle: simulator across a barrier hit and an expiry
# ---------------------------------------------------------------------- #
DT = 0.01
SPOTS = [100.0, 101.0, 103.0, 104.5, 103.5, 107.0, 105.0, 102.0, 99.0]  # digital expires at step 4, barrier hit at 5
VOLS = [0.20, 0.20, 0.21, 0.21, 0.20, 0.19, 0.20, 0.22, 0.23]


def run_simulation(policy=None):
    scenario = ol.scenario_from_arrays(SPOTS, VOLS, dt=DT, rate=0.03, div=0.01, name="hit and expiry")
    sim = ol.TradingSimulator(mixed_book(), scenario, policy)
    return sim, sim.run()


class TestSimulatorLifeCycle:
    def test_attribution_identity_holds_on_every_row(self):
        _, history = run_simulation(ol.DeltaHedgeEveryN(1))
        assert len(history) == len(SPOTS) and np.all(np.isfinite(history.to_numpy()))
        terms = history[[f"pnl_{term}" for term in ATTRIBUTION_TERMS]].sum(axis=1)
        np.testing.assert_allclose(terms, history["pnl_step"], atol=1e-9)
        np.testing.assert_allclose(history["pnl_cum"], history["pnl_step"].cumsum(), atol=1e-9)
        np.testing.assert_allclose(history["value"], history["cash"] + history["positions_value"])

    def test_digital_expiry_settles_in_cash(self):
        sim, history = run_simulation()
        expiry_row = history.iloc[4]
        assert expiry_row["t"] == pytest.approx(DIGITAL.expiry) and expiry_row["n_settled"] == 1
        assert expiry_row["settlement_cash"] == pytest.approx(3 * 10.0)  # 103.5 > 102: three digitals pay 10
        assert DIGITAL not in sim.book
        record = sim.book.settlements[0]
        assert (record.instrument, record.quantity, record.unit_value, record.spot) == (DIGITAL, 3.0, 10.0, 103.5)

    def test_knock_out_pays_its_rebate_when_the_barrier_is_touched(self):
        sim, history = run_simulation()
        hit_row = history.iloc[5]
        assert hit_row["n_settled"] == 1 and hit_row["settlement_cash"] == pytest.approx(4 * KNOCK_OUT.rebate)
        knocked = [p.instrument for p in sim.book.positions if isinstance(p.instrument, ol.BarrierOption)]
        assert len(knocked) == 1 and knocked[0].knocked and sim.book.quantity(knocked[0]) == 4.0
        assert knocked[0].price(sim.current_market) == 0.0
        record = sim.book.settlements[1]
        assert record.instrument == KNOCK_OUT and record.cash_flow == pytest.approx(8.0)
        assert history["n_settled"].sum() == 2  # nothing else was paid afterwards

    def test_rebate_keeps_the_value_continuous_through_the_knock_out(self):
        book = ol.Book("one barrier")
        book.trade(KNOCK_OUT, 1.0, MKT)
        just_below = MKT.bumped(spot=105.999, t=0.05)
        at_the_barrier = MKT.bumped(spot=106.0, t=0.05)
        before = book.value(just_below)
        changed = book.observe(at_the_barrier)
        assert list(changed) == [KNOCK_OUT] and changed[KNOCK_OUT].knocked
        assert book.positions_value(at_the_barrier) == 0.0
        assert book.value(at_the_barrier) == pytest.approx(before, abs=5e-3)  # option value -> rebate cash
        assert book.observe(at_the_barrier) == {}  # observing again pays nothing more
        assert len(book.settlements) == 1

    def test_short_knock_out_pays_the_rebate_and_knock_in_pays_nothing_at_the_hit(self):
        knock_in = ol.BarrierOption("call", 100.0, 106.0, 0.5, "up-and-in", rebate=2.0)
        book = ol.Book("short")
        book.trade(KNOCK_OUT, -3.0, MKT)
        book.trade(knock_in, 5.0, MKT)
        cash = book.cash
        book.observe(MKT.bumped(spot=108.0, t=0.02))
        assert book.cash == pytest.approx(cash - 3.0 * 2.0)
        assert [s.instrument for s in book.settlements] == [KNOCK_OUT]

    def test_path_state_is_carried_by_the_book_and_survives_json(self):
        sim, _ = run_simulation()
        instruments = {type(p.instrument).__name__: p.instrument for p in sim.book.positions}
        lookback, asian = instruments["LookbackOption"], instruments["AsianOption"]
        assert (lookback.running_min, lookback.running_max) == (min(SPOTS), max(SPOTS))
        assert asian.observed_until == pytest.approx(DT * (len(SPOTS) - 1))
        assert asian.running_average == pytest.approx(np.trapezoid(SPOTS, dx=DT) / (DT * (len(SPOTS) - 1)))
        clone = ol.Book.from_json(sim.book.to_json())
        assert clone.positions == sim.book.positions
        assert clone.value(sim.current_market) == sim.book.value(sim.current_market)

    def test_delta_hedging_removes_most_of_the_delta_pnl(self):
        _, hedged = run_simulation(ol.DeltaHedgeEveryN(1))
        _, unhedged = run_simulation(ol.NoHedge())
        assert abs(hedged["pnl_delta"]).sum() < 0.05 * abs(unhedged["pnl_delta"]).sum()
        np.testing.assert_allclose(hedged["delta"], 0.0, atol=1e-9)

    def test_simulation_figures_accept_an_exotic_history(self):
        _, history = run_simulation(ol.DeltaHedgeEveryN(1))
        for build in (sim_plots.simulation_dashboard, sim_plots.pnl_attribution_chart,
                      sim_plots.pnl_attribution_waterfall, sim_plots.gamma_theta_chart):
            assert isinstance(build(history), go.Figure)

    def test_delta_hedged_knock_out_with_rebate_breaks_even_on_average(self):
        option = ol.BarrierOption("call", strike=100.0, barrier=90.0, expiry=0.25,
                                  barrier_type="down-and-out", rebate=5.0)
        mkt = ol.Market(100.0, 0.25, 0.02, 0.0)
        experiment = ol.delta_hedging_experiment(option, mkt, n_paths=120, rebalance_steps_list=(25,), seed=3)
        assert not experiment.attrs["vectorised"]  # path-dependent: the generic book-based route
        mean = experiment["pnl"].mean()
        stderr = experiment["pnl"].std(ddof=1) / math.sqrt(len(experiment))
        # About a quarter of the paths knock out: forgetting the rebate would cost ~1.2 on average.
        assert abs(mean) < 4.0 * stderr + 0.15  # 0.15: daily-vs-continuous monitoring bias of the premium
        assert stderr < 0.2


class TestBookStartsTheLifeOfPathDependentProducts:
    def test_trade_stores_the_product_as_observed_at_the_trade(self):
        book = ol.Book("fresh")
        record = book.trade(LOOKBACK, 2.0, MKT)
        book.trade(ASIAN, 1.0, MKT)
        book.trade(KNOCK_OUT, 1.0, MKT)
        lookback, asian, barrier = (p.instrument for p in book.positions)
        assert record.instrument == LOOKBACK  # the blotter keeps the contract that was traded
        assert (lookback.running_min, lookback.running_max) == (100.0, 100.0)
        assert (asian.observed_until, asian.last_spot, asian.running_average) == (0.0, 100.0, None)
        assert barrier == KNOCK_OUT and KNOCK_OUT in book  # nothing to record: stored unchanged
        assert book.pnl(MKT) == pytest.approx(0.0, abs=1e-12)  # observing does not change the value
        assert book.quantity(lookback) == 2.0
        book.close_position(lookback, MKT)
        assert len(book) == 2

    def test_a_lookback_in_the_book_keeps_its_extreme_on_a_spot_ladder(self):
        book = ol.Book("lookback")
        book.trade(LOOKBACK, 1.0, MKT)
        ladder = book.spot_ladder(MKT, shocks=(-0.1, 0.0, 0.1), trader_units=False)
        seasoned = ol.LookbackOption("call", 0.5, running_min=100.0, running_max=100.0)
        np.testing.assert_allclose(ladder["pnl"], seasoned.price(MKT.bumped(spot=ladder["spot"].to_numpy())) - seasoned.price(MKT))
        # Up 10% the minimum stays at 100; a FRESH lookback repriced at 110 would restart its
        # minimum there and gain only 10% of its value (the price would be linear in the spot).
        fresh_gain = 0.1 * LOOKBACK.price(MKT)
        assert LOOKBACK.price(MKT.bumped(spot=110.0)) - LOOKBACK.price(MKT) == pytest.approx(fresh_gain)
        assert ladder["pnl"].iloc[2] > 3.0 * fresh_gain

    def test_trading_a_knock_out_beyond_its_barrier_is_a_wash(self):
        book = ol.Book("late")
        beyond = MKT.bumped(spot=107.0)
        trade = book.trade(KNOCK_OUT, 1.0, beyond)
        assert trade.price == KNOCK_OUT.rebate  # worth its rebate: the hit happens now
        assert book.positions[0].instrument.knocked and book.cash == pytest.approx(0.0)
        assert book.value(beyond) == pytest.approx(0.0)

    def test_default_observation_cash_flow_is_zero(self):
        assert VANILLA.observation_cash_flow(VANILLA) == 0.0
        assert LOOKBACK.observation_cash_flow(LOOKBACK.observe(90.0, 0.1)) == 0.0
        assert IRON_CONDOR.observation_cash_flow(IRON_CONDOR) == 0.0


class TestMonteCarloRebate:
    def test_mc_price_carries_a_knock_out_rebate_from_the_hit_to_expiry(self):
        mkt = ol.Market(100.0, 0.25, 0.08, 0.04)
        option = ol.BarrierOption("call", 100.0, 112.0, 0.5, "up-and-out", rebate=6.0)
        n_steps = 126
        value, stderr = ol.mc_price(option, mkt, n_paths=40_000, n_steps=n_steps, seed=21)
        reference = option.discrete_barrier_adjusted(0.25, dt=0.5 / n_steps).price(mkt)
        assert abs(value - reference) < 3.5 * stderr
        times, paths = ol.simulate_gbm_paths(100.0, 0.25, 0.08, 0.04, 0.5, n_steps, 40_000, seed=21)
        at_expiry = math.exp(-0.08 * 0.5) * option.path_payoff(paths, times).mean()  # rebate paid at expiry
        assert value - at_expiry > 5.0 * stderr

    def test_rebate_inside_a_position_and_a_composite(self):
        mkt = ol.Market(100.0, 0.25, 0.08, 0.04)
        option = ol.BarrierOption("put", 100.0, 90.0, 0.5, "down-and-out", rebate=4.0)
        single = ol.mc_price(option, mkt, n_paths=4_000, n_steps=50, seed=5)[0]
        assert ol.mc_price(ol.Position(option, -2.0), mkt, n_paths=4_000, n_steps=50, seed=5)[0] == pytest.approx(-2 * single)
        package = ol.CompositeInstrument("pair", ((option, 1.0), (option.vanilla_equivalent(), 1.0)))
        vanilla = ol.mc_price(option.vanilla_equivalent(), mkt, n_paths=4_000, n_steps=50, seed=5)[0]
        assert ol.mc_price(package, mkt, n_paths=4_000, n_steps=50, seed=5)[0] == pytest.approx(single + vanilla)


# ---------------------------------------------------------------------- #
# Generic plots on every product
# ---------------------------------------------------------------------- #
def finite_traces(fig: go.Figure) -> bool:
    for trace in fig.data:
        for axis in ("x", "y", "z"):
            values = getattr(trace, axis, None)
            if values is not None and not np.all(np.isfinite(np.asarray(values, dtype=float))):
                return False
    return True


class TestGenericPlotsOnEveryProduct:
    @pytest.mark.parametrize("name", list(PRODUCTS))
    def test_profiles_and_dashboards(self, name):
        product = PRODUCTS[name]
        figures = [
            profiles.greek_profile(product, MKT, "gamma", n=25),
            profiles.greek_dashboard(product, MKT, n=25),
            profiles.greek_profile(product, MKT, "vega", x="vol", n=15),
            profiles.pnl_profile(product, MKT, n=25),
            profiles.taylor_pnl_explain(product, MKT, dvol=0.01, dt=1 / 365, n=25),
            profiles.gamma_theta_tradeoff(ol.Position(product, -2.0), MKT, n=25),
        ]
        assert all(isinstance(fig, go.Figure) and fig.data and finite_traces(fig) for fig in figures)

    @pytest.mark.parametrize("name", [n for n in PRODUCTS if n != "underlying"])
    def test_time_axes_and_surfaces(self, name):
        product = PRODUCTS[name]
        figures = [
            profiles.greek_evolution(product, MKT, "delta", n=25),
            profiles.greek_vs_time(product, MKT, "theta", n=15),
            profiles.greek_surface(product, MKT, "gamma", kind="heatmap", nx=9, ny=7),
            profiles.greek_surface(product, MKT, "price", y="vol", nx=9, ny=7),
        ]
        assert all(isinstance(fig, go.Figure) and fig.data and finite_traces(fig) for fig in figures)

    def test_all_exotics_on_one_comparison(self):
        exotics = [DIGITAL, KNOCK_OUT, ASIAN, LOOKBACK, VANILLA]
        fig = profiles.compare_instruments(exotics, MKT, ("price", "delta"), n=25)
        assert len(fig.data) == len(exotics) * 2 and finite_traces(fig)

    def test_barrier_and_strike_levels_are_found_on_any_product(self):
        assert profiles.key_levels(KNOCK_OUT) == [("K", 100.0), ("B", 106.0)]
        assert profiles.key_levels(LOOKBACK) == []  # a floating lookback has no strike at all
        assert [level for _, level in profiles.key_levels(IRON_CONDOR)] == pytest.approx([85.0, 95.0, 105.0, 115.0])

    def test_strategy_tools_accept_exotic_legs(self):
        package = ol.CompositeInstrument(
            "call spread financed by a digital",
            ((VANILLA, 1.0), (ol.EuropeanOption("call", 110.0, 0.5), -1.0),
             (ol.DigitalOption("call", 110.0, 0.5, payout=3.0), -1.0)),
        )
        report = summary(package, MKT)
        assert report["max_profit"] == pytest.approx(10.0 - report["net_premium"])  # reached just below 110
        assert math.isfinite(report["max_loss"]) and json.dumps(report)
        for build in (strategy_plots.payoff_diagram, strategy_plots.strategy_greeks_dashboard,
                      strategy_plots.strategy_time_decay, strategy_plots.strategy_vol_sensitivity):
            assert isinstance(build(package, MKT), go.Figure)
        barrier_package = ol.CompositeInstrument("KO + put", ((KNOCK_OUT, 1.0), (ol.EuropeanOption("put", 95.0, 0.5), 1.0)))
        assert isinstance(strategy_plots.payoff_diagram(barrier_package, MKT), go.Figure)

    def test_book_plots_with_exotics_in_the_book(self):
        book = mixed_book()
        figures = [
            book_plots.book_risk_profile(book, MKT, n_points=21),
            book_plots.book_scenario_heatmap(book, MKT),
            book_plots.book_scenario_heatmap(book, MKT, horizon_days=20, quantity="vega"),
            book_plots.book_greeks_breakdown(book, MKT, greek="gamma"),
            book_plots.book_pnl_by_horizon(book, MKT, days=(0, 10, 30), n_points=21),
            book_plots.stress_test_chart(book, MKT),
        ]
        assert all(isinstance(fig, go.Figure) and fig.data for fig in figures)
        frame = book.positions_frame(MKT)
        assert list(frame.columns[5:]) == list(bs.GREEK_NAMES)
        assert np.all(np.isfinite(book.spot_ladder(MKT).to_numpy()))
        assert np.all(np.isfinite(book.scenario_grid(MKT, horizon_days=20).to_numpy()))


def test_top_level_api_is_complete():
    assert all(hasattr(ol, name) for name in ol.__all__)
    assert ol.numerical.NUMERICAL_GREEK_KEYS == ol.GREEK_KEYS
    assert len(set(ol.__all__)) == len(ol.__all__)
    assert ol.__version__ == "0.1.0"
