"""Tests of optionlab.plotting.exotic_plots (figures are built, never shown)."""

from __future__ import annotations

import importlib.util
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from optionlab.exotics import AsianOption, BarrierOption, DigitalOption, LookbackOption
from optionlab.instruments import EuropeanOption, Position, Underlying
from optionlab.market import Market
from optionlab.plotting import exotic_plots, theme
from optionlab.plotting.exotic_plots import (
    asian_averaging_figure,
    barrier_paths_figure,
    digital_replication_figure,
    exotic_vs_vanilla,
    vanilla_benchmark,
)

MKT = Market(spot=100.0, vol=0.2, rate=0.04, div=0.01)
UP_AND_OUT = BarrierOption("call", strike=100.0, barrier=120.0, expiry=0.5, barrier_type="up-and-out")
DOWN_AND_IN = BarrierOption("put", strike=100.0, barrier=85.0, expiry=0.5, barrier_type="down-and-in")
DIGITAL = DigitalOption("call", 100.0, 0.5, payout=10.0)
ASIAN = AsianOption("call", 100.0, 0.5)
EXOTICS = [
    UP_AND_OUT,
    DOWN_AND_IN,
    DIGITAL,
    DigitalOption("put", 95.0, 0.5, kind="asset"),
    ASIAN,
    AsianOption("put", 100.0, 0.5, averaging="geometric"),
    LookbackOption("call", 0.5),
    LookbackOption("put", 0.5, strike=100.0, kind="fixed"),
]


def all_numbers(fig: go.Figure) -> np.ndarray:
    chunks = []
    for trace in fig.data:
        for axis in ("x", "y"):
            values = getattr(trace, axis, None)
            if values is not None:
                chunks.append(np.asarray(values, dtype=float).ravel())
    return np.concatenate(chunks)


# ---------------------------------------------------------------------- #
# vanilla benchmark and the overlay
# ---------------------------------------------------------------------- #
class TestVanillaBenchmark:
    @pytest.mark.parametrize("exotic", [UP_AND_OUT, DIGITAL, ASIAN], ids=lambda e: e.label)
    def test_same_type_strike_and_expiry(self, exotic):
        assert vanilla_benchmark(exotic, MKT) == EuropeanOption(exotic.option_type, exotic.strike, exotic.expiry)

    def test_fresh_floating_lookback_is_compared_with_the_atm_vanilla(self):
        assert vanilla_benchmark(LookbackOption("call", 0.5), MKT) == EuropeanOption("call", 100.0, 0.5)

    def test_seasoned_floating_lookback_uses_its_running_extreme(self):
        call = LookbackOption("call", 0.5, running_min=90.0, running_max=105.0)
        put = LookbackOption("put", 0.5, running_min=90.0, running_max=105.0)
        assert vanilla_benchmark(call, MKT).strike == 90.0
        assert vanilla_benchmark(put, MKT).strike == 105.0
        assert call.vanilla_equivalent().strike == 90.0  # no spot needed once an extreme is stored

    def test_fresh_floating_lookback_needs_a_spot(self):
        with pytest.raises(ValueError, match="spot"):
            LookbackOption("call", 0.5).vanilla_equivalent()

    def test_instrument_without_vanilla_equivalent(self):
        with pytest.raises(ValueError, match="vanilla_equivalent"):
            vanilla_benchmark(Underlying(), MKT)


class TestExoticVsVanilla:
    @pytest.mark.parametrize("exotic", EXOTICS, ids=lambda e: e.label)
    def test_every_exotic_draws(self, exotic):
        fig = exotic_vs_vanilla(exotic, MKT, n=41)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2 * 4  # exotic + vanilla in each of the four default panels
        assert np.all(np.isfinite(all_numbers(fig)))
        assert exotic.label in fig.layout.title.text

    def test_curves_are_the_model_values(self):
        fig = exotic_vs_vanilla(UP_AND_OUT, MKT, greeks=("price",), n=21)
        exotic_trace, vanilla_trace = fig.data
        spots = np.asarray(exotic_trace.x, dtype=float)
        np.testing.assert_allclose(exotic_trace.y, UP_AND_OUT.price(MKT.bumped(spot=spots)))
        np.testing.assert_allclose(vanilla_trace.y, UP_AND_OUT.vanilla_equivalent().price(MKT.bumped(spot=spots)))
        assert vanilla_trace.line.dash == "dash" and exotic_trace.line.dash == "solid"
        assert vanilla_trace.line.color == theme.COLORS["neutral"]

    def test_default_range_stops_just_beyond_an_outer_barrier(self):
        up = np.asarray(exotic_vs_vanilla(UP_AND_OUT, MKT, greeks=("price",), n=21).data[0].x)
        assert up.min() == pytest.approx(70.0) and up.max() == pytest.approx(132.0)
        down = np.asarray(exotic_vs_vanilla(DOWN_AND_IN, MKT, greeks=("price",), n=21).data[0].x)
        assert down.min() == pytest.approx(0.9 * 85.0) and down.max() == pytest.approx(130.0)

    def test_put_is_orange_and_call_is_blue(self):
        assert exotic_vs_vanilla(DOWN_AND_IN, MKT, n=21).data[0].line.color == theme.COLORS["put"]
        assert exotic_vs_vanilla(UP_AND_OUT, MKT, n=21).data[0].line.color == theme.COLORS["call"]

    def test_short_position_flips_both_curves(self):
        long_fig = exotic_vs_vanilla(DIGITAL, MKT, greeks=("delta",), n=21)
        short_fig = exotic_vs_vanilla(Position(DIGITAL, -2.0), MKT, greeks=("delta",), n=21)
        for long_trace, short_trace in zip(long_fig.data, short_fig.data):
            np.testing.assert_allclose(short_trace.y, -2.0 * np.asarray(long_trace.y))

    def test_other_axis_and_explicit_benchmark(self):
        benchmark = EuropeanOption("call", 110.0, 0.5)
        fig = exotic_vs_vanilla(ASIAN, MKT, greeks=("vega",), x="vol", vanilla=benchmark, n=15,
                                trader_units=False, mode="dark")
        assert len(fig.data) == 2
        np.testing.assert_allclose(fig.data[1].y, benchmark.greeks(MKT.bumped(vol=np.asarray(fig.data[1].x)))["vega"])

    def test_validation(self):
        with pytest.raises(ValueError, match="scalar"):
            exotic_vs_vanilla(DIGITAL, MKT.bumped(spot=np.array([90.0, 100.0])))
        with pytest.raises(ValueError, match="Instrument"):
            exotic_vs_vanilla("digital", MKT)
        with pytest.raises(ValueError, match="vanilla"):
            exotic_vs_vanilla(DIGITAL, MKT, vanilla="call")
        with pytest.raises(ValueError, match="mode"):
            exotic_vs_vanilla(DIGITAL, MKT, mode="sepia")


# ---------------------------------------------------------------------- #
# barrier paths
# ---------------------------------------------------------------------- #
class TestBarrierPaths:
    def test_traces_split_the_paths_by_outcome(self):
        fig = barrier_paths_figure(UP_AND_OUT, MKT, n_paths=200, seed=1, n_steps=50)
        names = [trace.name for trace in fig.data]
        assert names[0].startswith("Knocked out") and names[1].startswith("Survives")
        assert names[2] == "First touch of the barrier"
        counts = [int(name.rsplit("(", 1)[1].rstrip(")")) for name in names[:2]]
        assert sum(counts) == 200
        assert len(fig.data[2].x) == counts[0]  # one first-touch marker per knocked path
        assert np.all(np.asarray(fig.data[2].y) >= 120.0)

    def test_survivors_never_reach_the_barrier(self):
        fig = barrier_paths_figure(UP_AND_OUT, MKT, n_paths=100, seed=2, n_steps=50)
        survivors = np.asarray(fig.data[1].y, dtype=float)
        assert np.nanmax(survivors) < 120.0
        knocked = np.asarray(fig.data[0].y, dtype=float)
        assert np.nanmax(knocked) >= 120.0

    def test_knock_in_colours_the_touching_paths_as_alive(self):
        fig = barrier_paths_figure(DOWN_AND_IN, MKT, n_paths=200, seed=3, n_steps=50)
        dead, alive = fig.data[0], fig.data[1]
        assert dead.name.startswith("Never activated") and alive.name.startswith("Knocked in")
        assert np.nanmin(np.asarray(alive.y, dtype=float)) <= 85.0 < np.nanmin(np.asarray(dead.y, dtype=float))
        assert alive.line.color == theme.COLORS["call"] and dead.line.color == theme.COLORS["neutral"]

    def test_seed_makes_it_reproducible_and_paths_start_now(self):
        mkt = MKT.bumped(t=0.1, spot=104.0)
        first = barrier_paths_figure(UP_AND_OUT, mkt, n_paths=20, seed=5, n_steps=30)
        second = barrier_paths_figure(UP_AND_OUT, mkt, n_paths=20, seed=5, n_steps=30)
        np.testing.assert_array_equal(np.asarray(first.data[0].y), np.asarray(second.data[0].y))
        x = np.asarray(first.data[0].x, dtype=float)
        assert np.nanmin(x) == pytest.approx(0.1) and np.nanmax(x) == pytest.approx(0.5)

    def test_reference_lines_and_subtitle(self):
        fig = barrier_paths_figure(UP_AND_OUT, MKT, n_paths=10, seed=1, n_steps=20, mode="light")
        levels = sorted(shape.y0 for shape in fig.layout.shapes)
        assert levels == [100.0, 120.0]
        assert "vanilla" in fig.layout.title.text and "H = 120" in fig.layout.title.text

    def test_already_knocked_option_counts_every_path(self):
        knocked = BarrierOption("call", 100.0, 120.0, 0.5, "up-and-out", knocked=True)
        fig = barrier_paths_figure(knocked, MKT, n_paths=15, seed=1, n_steps=20)
        assert fig.data[0].name == "Knocked out (15)"

    def test_validation(self):
        with pytest.raises(ValueError, match="BarrierOption"):
            barrier_paths_figure(DIGITAL, MKT)
        with pytest.raises(ValueError, match="expired"):
            barrier_paths_figure(UP_AND_OUT, MKT.bumped(t=0.5))
        with pytest.raises(ValueError, match="n_paths"):
            barrier_paths_figure(UP_AND_OUT, MKT, n_paths=0)
        with pytest.raises(ValueError, match="n_paths"):
            barrier_paths_figure(UP_AND_OUT, MKT, n_paths=10_000)
        with pytest.raises(ValueError, match="n_steps"):
            barrier_paths_figure(UP_AND_OUT, MKT, n_steps=0)


# ---------------------------------------------------------------------- #
# digital replication
# ---------------------------------------------------------------------- #
class TestDigitalReplication:
    def test_default_figure(self):
        fig = digital_replication_figure(DIGITAL, MKT, n=81)
        assert len(fig.data) == (4 + 1) * 3
        assert np.all(np.isfinite(all_numbers(fig)))
        names = [trace.name for trace in fig.data[:5]]
        assert names[:4] == ["Spread, width 20", "Spread, width 10", "Spread, width 5", "Spread, width 2"]
        assert names[4].startswith("Digital")
        assert [trace.showlegend for trace in fig.data[:5]] == [True] * 5
        assert not any(trace.showlegend for trace in fig.data[5:])

    def test_tighter_spreads_converge_to_the_digital(self):
        fig = digital_replication_figure(DIGITAL, MKT, widths=(20.0, 5.0, 1.0), quantities=("price",), n=61)
        *spreads, digital = (np.asarray(trace.y, dtype=float) for trace in fig.data)
        errors = [np.max(np.abs(spread - digital)) for spread in spreads]
        assert errors[0] > errors[1] > errors[2]
        assert errors[2] < 1e-3 * DIGITAL.payout * 10

    def test_payoff_panel_is_the_step_and_the_ramps(self):
        fig = digital_replication_figure(DIGITAL, MKT, widths=(10.0,), quantities=("payoff",), n=41)
        spots = np.asarray(fig.data[0].x, dtype=float)
        np.testing.assert_allclose(fig.data[1].y, DIGITAL.payoff(spots))
        np.testing.assert_allclose(fig.data[0].y, np.clip(spots - 95.0, 0.0, 10.0))

    def test_conservative_spread_dominates_the_digital(self):
        fig = digital_replication_figure(DIGITAL, MKT, widths=(5.0,), quantities=("price",),
                                         placement="conservative", n=41)
        assert np.all(np.asarray(fig.data[0].y) >= np.asarray(fig.data[1].y) - 1e-12)

    def test_put_digital_and_trader_units(self):
        put = DigitalOption("put", 100.0, 0.5, payout=5.0)
        raw = digital_replication_figure(put, MKT, widths=(4.0,), quantities=("vega",), trader_units=False, n=21)
        desk = digital_replication_figure(put, MKT, widths=(4.0,), quantities=("vega",), trader_units=True, n=21)
        np.testing.assert_allclose(np.asarray(desk.data[1].y) * 100.0, raw.data[1].y)
        assert "put spreads" in desk.layout.title.text

    def test_validation(self):
        with pytest.raises(ValueError, match="DigitalOption"):
            digital_replication_figure(UP_AND_OUT, MKT)
        with pytest.raises(ValueError, match="quantities"):
            digital_replication_figure(DIGITAL, MKT, quantities=("pnl",))
        with pytest.raises(ValueError, match="widths"):
            digital_replication_figure(DIGITAL, MKT, widths=(1, 2, 3, 4, 5, 6, 7))
        with pytest.raises(ValueError, match="width"):
            digital_replication_figure(DIGITAL, MKT, widths=(-1.0,))
        with pytest.raises(ValueError, match="x_range"):
            digital_replication_figure(DIGITAL, MKT, x_range=(120.0, 80.0))
        with pytest.raises(ValueError, match="placement"):
            digital_replication_figure(DIGITAL, MKT, placement="wide")


# ---------------------------------------------------------------------- #
# Asian averaging
# ---------------------------------------------------------------------- #
class TestAsianAveraging:
    def test_running_average_matches_the_instrument_state(self):
        fig = asian_averaging_figure(ASIAN, MKT, seed=4, n_steps=60)
        spot_trace, average_trace = fig.data
        times, path = np.asarray(spot_trace.x, dtype=float), np.asarray(spot_trace.y, dtype=float)
        state = ASIAN
        for t, spot in zip(times, path):
            state = state.observe(float(spot), float(t))
        average = np.asarray(average_trace.y, dtype=float)
        assert average[-1] == pytest.approx(state.running_average)
        assert average[-1] == pytest.approx(np.trapezoid(path, times) / 0.5)  # continuous average, trapezoid rule
        assert np.isnan(average[0]) and np.all(np.isfinite(average[1:]))

    def test_average_moves_less_than_the_spot(self):
        fig = asian_averaging_figure(ASIAN, MKT, seed=9, n_steps=250)
        spot, average = (np.asarray(trace.y, dtype=float) for trace in fig.data)
        late = slice(125, None)
        assert np.std(np.diff(average[late])) < 0.2 * np.std(np.diff(spot[late]))

    def test_seasoned_and_forward_starting(self):
        seasoned = AsianOption("call", 100.0, 0.5, running_average=110.0, observed_until=0.25, last_spot=108.0)
        fig = asian_averaging_figure(seasoned, MKT.bumped(t=0.25, spot=108.0), seed=1, n_steps=40)
        average = np.asarray(fig.data[1].y, dtype=float)
        assert average[0] == pytest.approx(110.0)
        forward = AsianOption("put", 100.0, 1.0, avg_start=0.5)
        fig = asian_averaging_figure(forward, MKT, seed=1, n_steps=40, mode="dark")
        average = np.asarray(fig.data[1].y, dtype=float)
        times = np.asarray(fig.data[1].x, dtype=float)
        assert np.all(np.isnan(average[times <= 0.5])) and np.all(np.isfinite(average[times > 0.5]))
        assert any("averaging starts" in (a.text or "") for a in fig.layout.annotations)

    def test_subtitle_quotes_both_payoffs(self):
        fig = asian_averaging_figure(ASIAN, MKT, seed=4, n_steps=30)
        assert "the Asian pays" in fig.layout.title.text and "vanilla would pay" in fig.layout.title.text

    def test_validation(self):
        with pytest.raises(ValueError, match="AsianOption"):
            asian_averaging_figure(DIGITAL, MKT)
        with pytest.raises(ValueError, match="expired"):
            asian_averaging_figure(ASIAN, MKT.bumped(t=0.75))
        with pytest.raises(ValueError, match="n_steps"):
            asian_averaging_figure(ASIAN, MKT, n_steps=0)


@pytest.mark.parametrize("mode", ["auto", "light", "dark"])
def test_every_figure_in_every_mode(mode):
    figures = [
        exotic_vs_vanilla(UP_AND_OUT, MKT, n=21, mode=mode),
        barrier_paths_figure(UP_AND_OUT, MKT, n_paths=10, seed=1, n_steps=20, mode=mode),
        digital_replication_figure(DIGITAL, MKT, n=21, mode=mode),
        asian_averaging_figure(ASIAN, MKT, seed=1, n_steps=20, mode=mode),
    ]
    for fig in figures:
        assert isinstance(fig, go.Figure)
        assert fig.layout.template.layout.paper_bgcolor == "rgba(0,0,0,0)"


def test_public_names_are_exported():
    import optionlab.plotting as plotting

    for name in exotic_plots.__all__:
        assert getattr(plotting, name) is getattr(exotic_plots, name)
    assert all(hasattr(plotting, name) for name in plotting.__all__)


# ---------------------------------------------------------------------- #
# examples/03_exotics.py (not a package, and its name starts with a digit: loaded by path)
# ---------------------------------------------------------------------- #
def load_exotics_example():
    path = Path(__file__).resolve().parents[1] / "examples" / "03_exotics.py"
    spec = importlib.util.spec_from_file_location("example_03_exotics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exotics_example_runs_without_opening_a_browser(tmp_path, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the example must not open a browser unless --show is passed")

    monkeypatch.setattr(webbrowser, "open", forbidden)
    example = load_exotics_example()
    example.main(["--outdir", str(tmp_path), "--paths", "4000", "--seed", "2"])
    written = {path.name for path in tmp_path.iterdir()}
    assert len([name for name in written if name.startswith("03_") and name.endswith(".html")]) == 10
    assert "plotly.min.js" in written  # shared bundle: the pages work offline
    report = capsys.readouterr().out
    assert "Closed form vs Monte Carlo" in report and "BGK-shifted" in report and "Brownian-bridge" in report
    assert "nan" not in report.lower()


def test_exotics_example_monte_carlo_agrees_with_the_right_reference():
    example = load_exotics_example()
    exotics = example.build_exotics(100.0, 0.5)
    assert list(exotics) == list(example.EXOTIC_REGISTRY)
    for name in ("cash_digital", "up_and_out_barrier", "down_and_in_barrier", "fixed_lookback"):
        method, reference, price, stderr = example.monte_carlo_rows(exotics[name], MKT, 20_000, 7)[-1]
        assert abs(price - reference) < 4.0 * stderr, (name, method)
