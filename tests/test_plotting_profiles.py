"""Tests for optionlab.plotting.profiles (generic Greek visualisation).

Every figure is exercised with three kinds of instrument:

* a closed-form ``EuropeanOption`` (vectorised path),
* a hand-built ``CompositeInstrument`` straddle (vectorised, two strikes-worth of legs),
* ``ScalarOnlyCall``, a dummy instrument whose pricer REFUSES array markets, to
  exercise the point-by-point fallback of ``evaluate_on_grid``.

No figure is ever shown: tests only inspect the returned ``go.Figure``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import plotly.graph_objects as go
import pytest

from optionlab import black_scholes as bs
from optionlab.instruments import (
    CompositeInstrument,
    EuropeanOption,
    Instrument,
    Position,
    Underlying,
)
from optionlab.market import Market
from optionlab.plotting import profiles as P
from optionlab.plotting import theme

_N = NormalDist()


@dataclass(frozen=True)
class ScalarOnlyCall(Instrument):
    """Black-Scholes call written with ``math``: any array market raises."""

    strike: float
    expiry: float

    def price(self, mkt: Market) -> float:
        tau = self.expiry - mkt.t
        if tau <= 0:  # ambiguous truth value for arrays -> ValueError
            return max(mkt.spot - self.strike, 0.0)
        sig = mkt.vol * math.sqrt(tau)
        d1 = (math.log(mkt.spot / self.strike) + (mkt.rate - mkt.div) * tau + 0.5 * sig**2) / sig
        return (
            mkt.spot * math.exp(-mkt.div * tau) * _N.cdf(d1)
            - self.strike * math.exp(-mkt.rate * tau) * _N.cdf(d1 - sig)
        )

    def payoff(self, spot_T):
        return np.maximum(np.asarray(spot_T, dtype=float) - self.strike, 0.0)


@dataclass(frozen=True)
class KnockOutStub(Instrument):
    """Only there to check that barrier-like fields are picked up as chart levels."""

    strike: float
    upper_barrier: float
    barrier_hit: bool
    expiry: float

    def price(self, mkt: Market):
        return EuropeanOption("call", self.strike, self.expiry).price(mkt)

    def payoff(self, spot_T):
        return np.maximum(np.asarray(spot_T, dtype=float) - self.strike, 0.0)


MKT = Market(spot=100.0, vol=0.2, rate=0.03, div=0.01)
CALL = EuropeanOption("call", 100.0, 1.0)
PUT = EuropeanOption("put", 100.0, 1.0)
STRADDLE = CompositeInstrument("Straddle 100", (Position(CALL, 1.0), Position(PUT, 1.0)))
SCALAR_ONLY = ScalarOnlyCall(100.0, 1.0)

VECTORISED = [pytest.param(CALL, id="european"), pytest.param(STRADDLE, id="composite")]
ALL_KINDS = VECTORISED + [pytest.param(SCALAR_ONLY, id="scalar-only")]


def _grid_size(instrument) -> int:
    """Coarse grids for the scalar-only instrument (19 repricings per point)."""
    return 9 if isinstance(instrument, ScalarOnlyCall) else 41


def assert_clean(fig: go.Figure, n_traces: int) -> None:
    """A themed Figure with the expected number of traces and no NaN/inf in its data."""
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == n_traces, [t.name for t in fig.data]
    for trace in fig.data:
        for attr in ("x", "y", "z"):
            values = getattr(trace, attr, None)
            if values is not None:
                arr = np.asarray(values, dtype=float)
                assert arr.size > 0
                assert np.all(np.isfinite(arr)), f"non-finite {attr} in trace {trace.name!r}"
    assert fig.layout.template.layout.colorway is not None  # went through theme.apply_theme
    assert fig.layout.title.text


# ---------------------------------------------------------------------- #
# evaluate_on_grid
# ---------------------------------------------------------------------- #
def test_scalar_only_instrument_really_rejects_arrays():
    with pytest.raises((TypeError, ValueError)):
        SCALAR_ONLY.price(MKT.bumped(spot=np.array([90.0, 110.0])))


def test_evaluate_on_grid_vectorised_matches_black_scholes():
    spots = np.linspace(60.0, 140.0, 17)
    out = P.evaluate_on_grid(CALL, MKT, ("price", "delta", "vega"), spot=spots)
    assert set(out) == {"price", "delta", "vega"}
    np.testing.assert_allclose(out["price"], bs.price(spots, 100.0, 1.0, 0.2, 0.03, 0.01, "call"))
    np.testing.assert_allclose(out["vega"], bs.vega(spots, 100.0, 1.0, 0.2, 0.03, 0.01, "call"))


def test_evaluate_on_grid_fallback_loop_matches_vectorised_result():
    spots = np.linspace(70.0, 130.0, 7)
    taus = np.array([1.0, 0.5, 0.1])
    axes = dict(spot=spots[None, :], tau=taus[:, None])
    slow = P.evaluate_on_grid(SCALAR_ONLY, MKT, ("price", "delta", "gamma", "theta"), **axes)
    fast = P.evaluate_on_grid(CALL, MKT, ("price", "delta", "gamma", "theta"), **axes)
    assert slow["price"].shape == (3, 7)
    np.testing.assert_allclose(slow["price"], fast["price"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(slow["delta"], fast["delta"], atol=1e-5)
    np.testing.assert_allclose(slow["gamma"], fast["gamma"], atol=1e-5)
    np.testing.assert_allclose(slow["theta"], fast["theta"], atol=1e-3)


def test_evaluate_on_grid_tau_axis_moves_the_clock_not_the_instrument():
    mkt = MKT.bumped(t=0.25)
    out = P.evaluate_on_grid(CALL, mkt, "price", tau=np.array([0.75, 0.5]))["price"]
    expected = bs.price(100.0, 100.0, np.array([0.75, 0.5]), 0.2, 0.03, 0.01, "call")
    np.testing.assert_allclose(out, expected)
    assert CALL.expiry == 1.0


def test_evaluate_on_grid_trader_units_and_scalar_point():
    raw = P.evaluate_on_grid(CALL, MKT, ("vega", "theta"))
    desk = P.evaluate_on_grid(CALL, MKT, ("vega", "theta"), trader_units=True)
    assert raw["vega"].shape == ()
    assert float(desk["vega"]) == pytest.approx(float(raw["vega"]) / 100.0)
    assert float(desk["theta"]) == pytest.approx(float(raw["theta"]) / 365.0)


def test_evaluate_on_grid_accepts_positions():
    spots = np.array([90.0, 100.0, 110.0])
    short = P.evaluate_on_grid(Position(CALL, -2.0), MKT, "delta", spot=spots)["delta"]
    np.testing.assert_allclose(short, -2.0 * P.evaluate_on_grid(CALL, MKT, "delta", spot=spots)["delta"])


def test_evaluate_on_grid_errors():
    with pytest.raises(ValueError, match="unknown grid axis"):
        P.evaluate_on_grid(CALL, MKT, "price", strike=np.array([1.0]))
    with pytest.raises(ValueError, match="no Greek"):
        P.evaluate_on_grid(CALL, MKT, "ultima", spot=np.array([100.0]))
    with pytest.raises(ValueError, match="never expires"):
        P.evaluate_on_grid(Underlying(), MKT, "price", tau=np.array([0.5]))
    with pytest.raises(ValueError, match="either 'tau' or 't'"):
        P.evaluate_on_grid(CALL, MKT, "price", tau=np.array([0.5]), t=np.array([0.1]))
    with pytest.raises(ValueError, match="Instrument or a Position"):
        P.evaluate_on_grid("call", MKT, "price")
    with pytest.raises(ValueError, match="at least one"):
        P.evaluate_on_grid(CALL, MKT, ())


# ---------------------------------------------------------------------- #
# Small helpers
# ---------------------------------------------------------------------- #
def test_key_levels_finds_strikes_and_barriers_recursively():
    assert P.key_levels(CALL) == [("K", 100.0)]
    spread = CompositeInstrument(
        "Spread",
        (Position(EuropeanOption("call", 95.0, 1.0), 1), Position(EuropeanOption("call", 105.0, 1.0), -1)),
    )
    assert P.key_levels(Position(spread, -3.0)) == [("K", 95.0), ("K", 105.0)]
    assert P.key_levels(KnockOutStub(100.0, 130.0, False, 1.0)) == [("K", 100.0), ("B", 130.0)]
    assert P.key_levels(Underlying()) == []


def test_greek_axis_label_states_the_units():
    assert P.greek_axis_label("vega", trader_units=True) == "Vega (per vol point)"
    assert P.greek_axis_label("vega", trader_units=False) == "Vega (per 1.00 vol)"
    assert P.greek_axis_label("theta") == "Theta (per day)"
    assert P.greek_axis_label("price") == "Price"
    assert P.greek_axis_label("my_greek") == "My greek"


@pytest.mark.parametrize(
    "tau, expected",
    [(2.0, "2y"), (1.0, "1y"), (0.5, "6m"), (1 / 12, "1m"), (1 / 52, "1w"), (1 / 365, "1d"),
     (0.125, "46d")],
)
def test_format_tenor(tau, expected):
    assert P._format_tenor(tau) == expected


def test_zero_crossings_ignore_flat_zero_stretches():
    x = np.arange(7.0)
    y = np.array([0.0, 0.0, -1.0, 1.0, 2.0, 0.0, -2.0])
    np.testing.assert_allclose(P._zero_crossings(x, y), [2.5, 5.0])


# ---------------------------------------------------------------------- #
# greek_profile / greek_dashboard
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("instrument", ALL_KINDS)
@pytest.mark.parametrize("x", ["spot", "vol", "tau", "t", "rate", "div"])
def test_greek_profile_every_axis(instrument, x):
    fig = P.greek_profile(instrument, MKT, "delta", x=x, n=_grid_size(instrument))
    assert_clean(fig, 2)  # curve + current-market marker
    assert fig.data[0].mode == "lines" and fig.data[1].mode == "markers+text"
    assert fig.layout.hovermode == "x unified"


def test_greek_profile_curve_and_marker_values():
    fig = P.greek_profile(CALL, MKT, "vega", trader_units=True, n=51)
    curve, marker = fig.data
    expected = bs.vega(np.asarray(curve.x), 100.0, 1.0, 0.2, 0.03, 0.01, "call") / 100.0
    np.testing.assert_allclose(np.asarray(curve.y), expected)
    assert marker.x[0] == pytest.approx(100.0)
    assert marker.y[0] == pytest.approx(bs.vega(100.0, 100.0, 1.0, 0.2, 0.03, 0.01, "call") / 100.0)
    assert "per vol point" in fig.layout.yaxis.title.text
    raw = P.greek_profile(CALL, MKT, "vega", trader_units=False, n=51)
    np.testing.assert_allclose(np.asarray(raw.data[0].y), expected * 100.0)
    assert "per 1.00 vol" in raw.layout.yaxis.title.text


def test_greek_profile_default_spot_range_and_refinement_around_strike():
    fig = P.greek_profile(CALL, MKT, "gamma", n=201)
    x = np.asarray(fig.data[0].x)
    assert x[0] == pytest.approx(50.0) and x[-1] == pytest.approx(150.0)
    assert np.all(np.diff(x) > 0)
    assert x.size > 201  # extra points packed around the strike
    assert np.sum(np.abs(x - 100.0) < 0.5) > 10
    assert len(fig.layout.shapes) == 1  # the strike line


def test_greek_profile_put_is_orange_and_short_position_flips_sign():
    assert P.greek_profile(PUT, MKT, n=21).data[0].line.color == theme.COLORS["put"]
    assert P.greek_profile(CALL, MKT, n=21).data[0].line.color == theme.COLORS["call"]
    long_gamma = np.asarray(P.greek_profile(CALL, MKT, "gamma", n=21).data[0].y)
    short_gamma = np.asarray(P.greek_profile(Position(CALL, -1.0), MKT, "gamma", n=21).data[0].y)
    np.testing.assert_allclose(short_gamma, -long_gamma)


def test_greek_profile_tau_axis_is_reversed_and_stays_away_from_expiry():
    fig = P.greek_profile(CALL, MKT, "theta", x="tau", n=31)
    x = np.asarray(fig.data[0].x)
    assert fig.layout.xaxis.autorange == "reversed"
    assert x.min() >= 1e-4 and x.max() == pytest.approx(1.0)


def test_greek_profile_marker_dropped_when_market_is_outside_the_range():
    assert_clean(P.greek_profile(CALL, MKT, "delta", x_range=(110.0, 150.0), n=21), 1)
    assert_clean(P.greek_profile(Underlying(), MKT, "delta", n=21), 2)


def test_greek_profile_errors():
    with pytest.raises(ValueError, match="axis must be one of"):
        P.greek_profile(CALL, MKT, x="strike")
    with pytest.raises(ValueError, match="no Greek"):
        P.greek_profile(CALL, MKT, "ultima")
    with pytest.raises(ValueError, match="scalar base Market"):
        P.greek_profile(CALL, MKT.bumped(spot=np.array([90.0, 100.0])))
    with pytest.raises(ValueError, match="low < high"):
        P.greek_profile(CALL, MKT, x_range=(120.0, 80.0))
    with pytest.raises(ValueError, match="strictly positive"):
        P.greek_profile(CALL, MKT, x_range=(0.0, 80.0))
    with pytest.raises(ValueError, match="at least 2"):
        P.greek_profile(CALL, MKT, n=1)
    with pytest.raises(ValueError, match="never expires"):
        P.greek_profile(Underlying(), MKT, x="tau")
    with pytest.raises(ValueError, match="Instrument or a Position"):
        P.greek_profile(object(), MKT)


@pytest.mark.parametrize("instrument", ALL_KINDS)
def test_greek_dashboard(instrument):
    fig = P.greek_dashboard(instrument, MKT, n=_grid_size(instrument))
    assert_clean(fig, 2 * len(P.DEFAULT_DASHBOARD_GREEKS))
    panel_titles = [a.text for a in fig.layout.annotations]
    assert panel_titles[:6] == ["Price", "Delta", "Gamma (per 1 spot)", "Vega (per vol point)",
                                "Theta (per day)", "Rho (per 1% rate)"]
    assert not any(trace.showlegend for trace in fig.data)  # one series: the title names it


def test_greek_dashboard_custom_greeks_and_columns():
    greeks = ("vanna", "volga", "charm", "speed", "color")
    fig = P.greek_dashboard(STRADDLE, MKT, greeks, x="vol", ncols=2, n=25)
    assert_clean(fig, 10)
    assert fig.layout.xaxis5.tickformat == ".0%"


# ---------------------------------------------------------------------- #
# Families of curves
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("instrument", ALL_KINDS)
def test_greek_evolution_default_tenor_ladder(instrument):
    fig = P.greek_evolution(instrument, MKT, "gamma", n=_grid_size(instrument))
    assert_clean(fig, 6)
    assert [t.name for t in fig.data] == ["1y", "6m", "3m", "1m", "1w", "1d"]
    assert len({t.line.color for t in fig.data}) == 6  # an ordered ramp, one step per curve
    assert fig.layout.legend.title.text == "Time to expiry"


def test_greek_evolution_gamma_concentrates_at_the_strike():
    fig = P.greek_evolution(CALL, MKT, "gamma", taus=(1 / 52, 1.0, 0.25), x_range=(80.0, 120.0), n=81)
    peaks = [np.max(np.asarray(t.y)) for t in fig.data]
    assert [t.name for t in fig.data] == ["1y", "3m", "1w"]  # sorted so that time passes
    assert peaks[0] < peaks[1] < peaks[2]


def test_greek_evolution_never_moves_time_backwards_by_default():
    mkt = MKT.bumped(t=0.8)  # 0.2y left
    fig = P.greek_evolution(CALL, mkt, "delta", n=21)
    assert [t.name for t in fig.data] == ["73d", "1m", "1w", "1d"]


@pytest.mark.parametrize("instrument", ALL_KINDS)
@pytest.mark.parametrize("vary, n_curves", [("vol", 5), ("rate", 6), ("div", 6), ("spot", 3)])
def test_greek_evolution_other_dimensions(instrument, vary, n_curves):
    x = "tau" if vary == "spot" else "spot"
    fig = P.greek_evolution(instrument, MKT, "vega", x=x, vary=vary, n=_grid_size(instrument))
    assert_clean(fig, n_curves)


def test_greek_evolution_explicit_values_are_sorted():
    fig = P.greek_evolution(CALL, MKT, "vega", vary="vol", values=(0.4, 0.1, 0.2), n=21)
    assert [t.name for t in fig.data] == ["vol 10%", "vol 20%", "vol 40%"]


def test_greek_evolution_errors():
    with pytest.raises(ValueError, match="same market dimension"):
        P.greek_evolution(CALL, MKT, "gamma", x="tau", vary="tau")
    with pytest.raises(ValueError, match="same market dimension"):
        P.greek_evolution(CALL, MKT, "gamma", x="t", vary="tau")
    with pytest.raises(ValueError, match="only applies with vary='tau'"):
        P.greek_evolution(CALL, MKT, "gamma", taus=(1.0,), vary="vol")
    with pytest.raises(ValueError, match="at most 8"):
        P.greek_evolution(CALL, MKT, "gamma", taus=np.linspace(0.1, 1.0, 9))
    with pytest.raises(ValueError, match="strictly positive"):
        P.greek_evolution(CALL, MKT, "gamma", taus=(0.5, 0.0))
    with pytest.raises(ValueError, match="never expires"):
        P.greek_evolution(Underlying(), MKT, "delta")


@pytest.mark.parametrize("instrument", ALL_KINDS)
def test_greek_vs_time(instrument):
    fig = P.greek_vs_time(instrument, MKT, "gamma", n=_grid_size(instrument))
    assert_clean(fig, 3)
    assert [t.name for t in fig.data] == ["S = 90", "S = 100", "S = 110"]
    assert [t.line.color for t in fig.data] == theme.PALETTE[:3]
    t = np.asarray(fig.data[0].x)
    assert t[0] == pytest.approx(0.0) and t[-1] < 1.0  # stops just before expiry
    assert np.all(np.diff(t) > 0)
    assert len(fig.layout.shapes) == 1  # expiry line


def test_greek_vs_time_atm_gamma_blows_up_and_wings_die():
    fig = P.greek_vs_time(CALL, MKT, "gamma", n=101)
    otm, atm, itm = (np.asarray(t.y) for t in fig.data)
    assert atm[-1] > 5.0 * atm[0]
    assert otm[-1] < otm[0] and itm[-1] < itm[0]


def test_greek_vs_time_custom_spots_tau_axis_and_errors():
    fig = P.greek_vs_time(STRADDLE, MKT, "theta", spots=(80.0, 100.0), x="tau", n=21)
    assert_clean(fig, 2)
    assert fig.layout.xaxis.autorange == "reversed"
    with pytest.raises(ValueError, match="'t' or 'tau'"):
        P.greek_vs_time(CALL, MKT, x="spot")
    with pytest.raises(ValueError, match="never expires"):
        P.greek_vs_time(Underlying(), MKT)
    with pytest.raises(ValueError, match="has expired"):
        P.greek_vs_time(CALL, MKT.bumped(t=1.0))


# ---------------------------------------------------------------------- #
# Surfaces
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("instrument", ALL_KINDS)
@pytest.mark.parametrize("kind, trace_type", [("surface", "surface"), ("heatmap", "heatmap")])
def test_greek_surface(instrument, kind, trace_type):
    small = isinstance(instrument, ScalarOnlyCall)
    fig = P.greek_surface(instrument, MKT, "gamma", kind=kind, nx=6 if small else 31, ny=5 if small else 21)
    assert_clean(fig, 2)  # surface + current-market marker
    assert fig.data[0].type == trace_type
    z = np.asarray(fig.data[0].z)
    assert z.shape == ((5, 6) if small else (21, 31))
    assert np.asarray(fig.data[0].y).min() >= 0.02 - 1e-12  # 2% of the remaining life, not tau = 0


def test_greek_surface_spot_vol_and_colour_choice():
    signed = P.greek_surface(CALL, MKT, "vanna", y="vol", kind="heatmap", nx=21, ny=11)
    assert_clean(signed, 2)
    assert signed.data[0].zmin == pytest.approx(-signed.data[0].zmax)  # diverging, centred on 0
    assert signed.layout.yaxis.tickformat == ".0%"
    positive = P.greek_surface(CALL, MKT, "gamma", kind="surface", nx=21, ny=11)
    assert positive.data[0].cmin is None and not positive.data[0].reversescale
    negative = P.greek_surface(CALL, MKT, "theta", kind="surface", nx=21, ny=11)
    assert negative.data[0].reversescale  # dark must still mean "large"


def test_greek_surface_errors():
    with pytest.raises(ValueError, match="kind must be"):
        P.greek_surface(CALL, MKT, kind="contour")
    with pytest.raises(ValueError, match="same market dimension"):
        P.greek_surface(CALL, MKT, x="tau", y="t")
    with pytest.raises(ValueError, match="never expires"):
        P.greek_surface(Underlying(), MKT, "delta")


# ---------------------------------------------------------------------- #
# Comparisons
# ---------------------------------------------------------------------- #
def test_compare_instruments_list_and_dict():
    fig = P.compare_instruments([CALL, PUT, STRADDLE, SCALAR_ONLY], MKT, "delta", n=9)
    assert_clean(fig, 4)
    assert [t.name for t in fig.data] == ["C 100 T=1.00", "P 100 T=1.00", "Straddle 100", "ScalarOnlyCall"]
    assert [t.line.color for t in fig.data] == theme.PALETTE[:4]
    assert len({t.line.dash for t in fig.data}) == 4  # identity never rests on colour alone
    np.testing.assert_allclose(np.asarray(fig.data[3].y), np.asarray(fig.data[0].y), atol=1e-5)

    both_sides = {"long": CALL, "short": Position(CALL, -1.0)}
    named = P.compare_instruments(both_sides, MKT, ("delta", "gamma"), n=21)
    assert_clean(named, 4)
    assert [t.showlegend for t in named.data] == [True, True, False, False]  # one legend entry each


def test_compare_instruments_range_covers_every_strike():
    low, high = EuropeanOption("put", 60.0, 1.0), EuropeanOption("call", 140.0, 1.0)
    x = np.asarray(P.compare_instruments([low, high], MKT, "gamma", n=21).data[0].x)
    assert x[0] == pytest.approx(30.0) and x[-1] == pytest.approx(210.0)


def test_compare_instruments_errors():
    with pytest.raises(ValueError, match="at least one"):
        P.compare_instruments([], MKT)
    with pytest.raises(ValueError, match="unique"):
        P.compare_instruments([CALL, CALL], MKT)
    with pytest.raises(ValueError, match="at most 8"):
        P.compare_instruments([EuropeanOption("call", 90.0 + k, 1.0) for k in range(9)], MKT)
    with pytest.raises(ValueError, match="colours"):
        P.compare_instruments([CALL, PUT], MKT, colors=["#3987e5"])


def test_call_put_comparison():
    fig = P.call_put_comparison(100.0, 1.0, MKT, n=41)
    assert_clean(fig, 2 * len(P.DEFAULT_DASHBOARD_GREEKS))
    assert fig.data[0].line.color == theme.COLORS["call"]
    assert fig.data[1].line.color == theme.COLORS["put"]
    gamma_call, gamma_put = (np.asarray(t.y) for t in fig.data[4:6])
    np.testing.assert_allclose(gamma_call, gamma_put, atol=1e-12)  # parity: same gamma
    assert_clean(P.call_put_comparison(100.0, 1.0, MKT, greeks=("delta", "charm"), x="tau", n=21), 4)


# ---------------------------------------------------------------------- #
# P&L views
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("instrument", ALL_KINDS)
def test_pnl_profile_default(instrument):
    fig = P.pnl_profile(instrument, MKT, n=_grid_size(instrument))
    # 4 horizons + payoff + current-market marker + breakeven markers
    assert_clean(fig, 7)
    assert [t.name for t in fig.data[:5]] == ["today", "in 3m", "in 6m", "in 9m", "at expiry (payoff)"]
    x, payoff_line = np.asarray(fig.data[4].x), np.asarray(fig.data[4].y)
    entry = float(instrument.price(MKT))
    np.testing.assert_allclose(payoff_line, np.asarray(instrument.payoff(x)) - entry)
    today = np.asarray(fig.data[0].y)
    assert np.interp(100.0, x, today) == pytest.approx(0.0, abs=1e-9)  # P&L is zero where we stand


def test_pnl_profile_breakevens_of_a_straddle():
    fig = P.pnl_profile(STRADDLE, MKT, entry_price=12.0, horizons=(0.0,), n=101)
    assert_clean(fig, 4)
    breakevens = fig.data[-1]
    assert breakevens.name == "Breakeven"
    np.testing.assert_allclose(np.asarray(breakevens.x), [88.0, 112.0], atol=1e-9)
    assert "88.00, 112.00" in fig.layout.title.text


def test_pnl_profile_value_mode_and_dropped_horizons():
    fig = P.pnl_profile(CALL, MKT, entry_price=0.0, horizons=(0.5, 2.0), n=21)
    assert_clean(fig, 2)  # 0.5y + payoff; 2y is beyond expiry; no "today" marker; no breakeven on values
    assert fig.layout.yaxis.title.text == "Value"
    assert np.all(np.asarray(fig.data[0].y) >= 0.0)


def test_pnl_profile_calendar_ends_at_first_expiry_with_a_value_line():
    calendar = CompositeInstrument(
        "Calendar", (Position(EuropeanOption("call", 100.0, 0.25), -1.0), Position(CALL, 1.0))
    )
    fig = P.pnl_profile(calendar, MKT, n=41)
    terminal = next(t for t in fig.data if t.name == "at first expiry (value)")
    x = np.asarray(terminal.x)
    expected = calendar.price(MKT.bumped(spot=x, t=0.25)) - calendar.price(MKT)
    np.testing.assert_allclose(np.asarray(terminal.y), expected)
    assert x[np.argmax(np.asarray(terminal.y))] == pytest.approx(100.0, abs=2.0)  # the "tent"


def test_pnl_profile_without_expiry_and_errors():
    # No terminal date: today's line and the current-market marker, no breakeven.
    assert_clean(P.pnl_profile(Underlying(), MKT, n=21), 2)
    with pytest.raises(ValueError, match="finite and >= 0"):
        P.pnl_profile(CALL, MKT, horizons=(-0.1,))
    with pytest.raises(ValueError, match="at most 7"):
        P.pnl_profile(CALL, MKT, horizons=np.linspace(0.0, 0.8, 8))


@pytest.mark.parametrize("instrument", ALL_KINDS)
def test_taylor_pnl_explain(instrument):
    fig = P.taylor_pnl_explain(instrument, MKT, n=_grid_size(instrument))
    assert_clean(fig, 4)  # repricing, delta, delta+gamma, marker
    assert [t.name for t in fig.data[:3]] == ["Full repricing", "Delta", "Delta + gamma"]
    x = np.asarray(fig.data[0].x)
    actual, first, second = (np.asarray(t.y) for t in fig.data[:3])
    near = np.abs(x - 100.0) <= 5.0
    assert np.max(np.abs(second[near] - actual[near])) < np.max(np.abs(first[near] - actual[near]))
    far = np.argmax(x)
    assert abs(second[far] - actual[far]) < abs(first[far] - actual[far])


def test_taylor_pnl_explain_with_vol_and_time_scenario():
    fig = P.taylor_pnl_explain(CALL, MKT, dvol=0.02, dt=7 / 365, n=41)
    assert_clean(fig, 5)
    assert fig.data[3].name == "Delta + gamma + vega + theta"
    x = np.asarray(fig.data[0].x)
    at_spot = lambda trace: float(np.interp(100.0, x, np.asarray(trace.y)))  # noqa: E731
    assert abs(at_spot(fig.data[3]) - at_spot(fig.data[0])) < 0.05  # vega + theta explain the no-move P&L
    assert abs(at_spot(fig.data[2]) - at_spot(fig.data[0])) > 0.3
    assert "per vol point" in fig.layout.title.text
    raw = P.taylor_pnl_explain(CALL, MKT, dvol=0.02, trader_units=False, n=21)
    assert "per 1.00 vol" in raw.layout.title.text
    with pytest.raises(ValueError, match="negative"):
        P.taylor_pnl_explain(CALL, MKT, dvol=-0.5)
    with pytest.raises(ValueError, match=">= 0"):
        P.taylor_pnl_explain(CALL, MKT, dt=-0.1)


@pytest.mark.parametrize("instrument", ALL_KINDS)
def test_gamma_theta_tradeoff(instrument):
    fig = P.gamma_theta_tradeoff(instrument, MKT, n=_grid_size(instrument))
    assert_clean(fig, 4)  # profit wash, loss wash, repricing, approximation
    x = np.asarray(fig.data[2].x)
    hedged, approx = np.asarray(fig.data[2].y), np.asarray(fig.data[3].y)
    assert np.interp(100.0, x, hedged) < 0.0  # nothing moves: the long option pays theta
    assert hedged[0] > 0.0 and hedged[-1] > 0.0  # a 4-sigma move pays for it
    np.testing.assert_allclose(hedged, approx, atol=0.02)
    assert "breakeven move" in fig.layout.title.text
    assert len(fig.layout.shapes) == 4  # zero line, no-move line, two breakevens


def test_gamma_theta_tradeoff_short_position_and_no_breakeven():
    short = P.gamma_theta_tradeoff(Position(STRADDLE, -1.0), MKT, n=41)
    x, hedged = np.asarray(short.data[2].x), np.asarray(short.data[2].y)
    assert np.interp(100.0, x, hedged) > 0.0 and hedged[0] < 0.0
    stock = P.gamma_theta_tradeoff(Underlying(), MKT, n=21)  # no gamma, no theta: flat, no breakeven
    assert_clean(stock, 4)
    assert "breakeven" not in stock.layout.title.text
    with pytest.raises(ValueError, match="strictly positive"):
        P.gamma_theta_tradeoff(CALL, MKT, dt=0.0)


# ---------------------------------------------------------------------- #
# Other kinds of instrument
# ---------------------------------------------------------------------- #
def test_numerical_greeks_instrument_with_barrier_levels():
    stub = KnockOutStub(100.0, 130.0, False, 1.0)  # vectorised price, bump-and-reprice Greeks
    fig = P.greek_profile(stub, MKT, "gamma", n=41)
    assert_clean(fig, 2)
    x = np.asarray(fig.data[0].x)
    assert x[0] == pytest.approx(50.0) and x[-1] == pytest.approx(195.0)  # covers the barrier
    assert [a.text for a in fig.layout.annotations] == ["K 100", "B 130"]
    np.testing.assert_allclose(
        np.asarray(fig.data[0].y), bs.gamma(x, 100.0, 1.0, 0.2, 0.03, 0.01, "call"), atol=1e-6
    )
    assert_clean(P.greek_surface(stub, MKT, "theta", nx=9, ny=7), 2)  # numerical theta on a (t, spot) grid


def test_monte_carlo_priced_instrument_goes_through_the_scalar_loop():
    from optionlab.monte_carlo import mc_price

    @dataclass(frozen=True)
    class McCall(Instrument):
        strike: float
        expiry: float

        def price(self, mkt: Market) -> float:  # mc_price refuses array markets
            return mc_price(EuropeanOption("call", self.strike, self.expiry), mkt, 2_000, 1, seed=7)[0]

        def payoff(self, spot_T):
            return np.maximum(np.asarray(spot_T, dtype=float) - self.strike, 0.0)

    fig = P.greek_profile(McCall(100.0, 1.0), MKT, "price", n=5)
    assert_clean(fig, 2)
    x, y = np.asarray(fig.data[0].x), np.asarray(fig.data[0].y)
    np.testing.assert_allclose(y, bs.price(x, 100.0, 1.0, 0.2, 0.03, 0.01, "call"), atol=1.0)
    assert_clean(P.pnl_profile(McCall(100.0, 1.0), MKT, horizons=(0.0, 0.5), n=5), 5)


# ---------------------------------------------------------------------- #
# Cross-cutting behaviour
# ---------------------------------------------------------------------- #
def test_modes_and_purity():
    before = (CALL, MKT, STRADDLE)
    for mode in ("auto", "light", "dark"):
        assert_clean(P.greek_evolution(CALL, MKT, "gamma", mode=mode, n=21), 6)
        assert_clean(P.greek_surface(CALL, MKT, "gamma", mode=mode, nx=11, ny=7), 2)
    with pytest.raises(ValueError, match="mode must be one of"):
        P.greek_profile(CALL, MKT, mode="sepia")
    assert before == (CALL, MKT, STRADDLE)  # nothing was mutated


def test_figures_are_json_serialisable():
    for fig in (
        P.greek_dashboard(STRADDLE, MKT, n=21),
        P.greek_surface(CALL, MKT, "vega", y="vol", nx=9, ny=7),
        P.pnl_profile(STRADDLE, MKT, n=21),
    ):
        assert fig.to_json()


def test_expired_instrument_still_plots_against_spot():
    expired = MKT.bumped(t=1.0)
    assert_clean(P.greek_profile(CALL, expired, "delta", n=21), 2)
    assert_clean(P.pnl_profile(CALL, expired, entry_price=5.0, n=41), 2)  # payoff line + breakeven
