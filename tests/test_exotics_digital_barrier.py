"""Tests for optionlab.exotics.digital and optionlab.exotics.barrier.

The barrier closed forms are checked four independent ways: in-out parity,
reference values from Haug's book, a numerical integration of the
reflection-principle density (knock-outs) and of the first-passage density
(rebates), and Monte Carlo with the Broadie-Glasserman-Kou correction.
"""

import itertools
import json
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from optionlab import black_scholes as bs
from optionlab.exotics.barrier import (
    BARRIER_TYPES,
    BGK_BETA,
    BarrierOption,
    bgk_adjusted_barrier,
)
from optionlab.exotics.digital import DigitalOption
from optionlab.instruments import (
    GREEK_KEYS,
    INSTRUMENT_REGISTRY,
    CompositeInstrument,
    EuropeanOption,
    Position,
    instrument_from_dict,
)
from optionlab.market import Market
from optionlab.monte_carlo import mc_price, simulate_gbm_paths
from optionlab.numerical import numerical_greeks

pytestmark = pytest.mark.filterwarnings("error")

OPTION_TYPES = ("call", "put")


def assert_greeks_close(actual, expected, rel=2e-3, floor=1e-4, keys=GREEK_KEYS):
    """Compare two Greek dicts key by key with a relative tolerance and an absolute floor."""
    for key in keys:
        a, e = np.asarray(actual[key], dtype=float), np.asarray(expected[key], dtype=float)
        scale = floor + np.abs(e)
        worst = float(np.max(np.abs(a - e) / scale))
        assert worst < rel, f"{key}: worst relative error {worst:.2e}"


# ====================================================================== #
# DigitalOption
# ====================================================================== #
class TestDigitalPricing:
    def test_haug_cash_or_nothing_put(self):
        # Haug, Complete Guide, cash-or-nothing example: 2.6710
        put = DigitalOption("put", 80.0, 0.75, payout=10.0)
        assert put.price(Market(100.0, 0.35, rate=0.06, div=0.06)) == pytest.approx(2.6710, abs=5e-5)

    def test_closed_forms(self):
        S, K, tau, vol, r, q = 105.0, 100.0, 0.6, 0.3, 0.04, 0.015
        d1, d2 = bs.d1_d2(S, K, tau, vol, r, q)
        mkt = Market(S, vol, r, q, t=0.4)
        assert DigitalOption("call", K, 1.0, 5.0).price(mkt) == pytest.approx(
            5.0 * math.exp(-r * tau) * norm.cdf(d2), rel=1e-13)
        assert DigitalOption("put", K, 1.0, 5.0).price(mkt) == pytest.approx(
            5.0 * math.exp(-r * tau) * norm.cdf(-d2), rel=1e-13)
        assert DigitalOption("call", K, 1.0, 2.0, "asset").price(mkt) == pytest.approx(
            2.0 * S * math.exp(-q * tau) * norm.cdf(d1), rel=1e-12)
        assert DigitalOption("put", K, 1.0, 2.0, "asset").price(mkt) == pytest.approx(
            2.0 * S * math.exp(-q * tau) * norm.cdf(-d1), rel=1e-12)

    @pytest.mark.parametrize("rate,div", [(0.05, 0.0), (0.01, 0.06), (-0.01, 0.02)])
    def test_cash_call_plus_put_is_a_zero_coupon_bond(self, rate, div):
        spot = np.linspace(40.0, 180.0, 29)[:, None]
        vol = np.array([0.05, 0.2, 0.6])[None, :]
        mkt = Market(spot, vol, rate, div, t=0.25)
        total = (DigitalOption("call", 100.0, 1.0, 7.5).price(mkt)
                 + DigitalOption("put", 100.0, 1.0, 7.5).price(mkt))
        assert total.shape == (29, 3)
        np.testing.assert_allclose(total, 7.5 * math.exp(-rate * 0.75), rtol=1e-13)

    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    def test_asset_minus_strike_times_cash_is_the_vanilla(self, option_type):
        mkt = Market(np.linspace(50.0, 160.0, 23), 0.3, 0.04, 0.01, t=0.1)
        omega = 1.0 if option_type == "call" else -1.0
        asset = DigitalOption(option_type, 95.0, 0.9, kind="asset").price(mkt)
        cash = DigitalOption(option_type, 95.0, 0.9, kind="cash").price(mkt)
        vanilla = EuropeanOption(option_type, 95.0, 0.9).price(mkt)
        np.testing.assert_allclose(omega * (asset - 95.0 * cash), vanilla, atol=1e-11)

    def test_asset_call_plus_put_is_the_prepaid_forward(self):
        mkt = Market(np.linspace(50.0, 160.0, 12), 0.3, 0.04, 0.03)
        total = (DigitalOption("call", 100.0, 2.0, kind="asset").price(mkt)
                 + DigitalOption("put", 100.0, 2.0, kind="asset").price(mkt))
        np.testing.assert_allclose(total, mkt.spot * math.exp(-0.03 * 2.0), rtol=1e-12)

    def test_probability_itm(self):
        mkt = Market(np.array([80.0, 100.0, 120.0]), 0.25, 0.03, 0.01)
        call, put = DigitalOption("call", 100.0, 1.0, 9.0), DigitalOption("put", 100.0, 1.0, 9.0)
        _, d2 = bs.d1_d2(mkt.spot, 100.0, 1.0, 0.25, 0.03, 0.01)
        np.testing.assert_allclose(call.probability_itm(mkt), norm.cdf(d2), rtol=1e-12)
        np.testing.assert_allclose(call.probability_itm(mkt) + put.probability_itm(mkt), 1.0, rtol=1e-12)
        assert isinstance(call.probability_itm(Market(100.0, 0.25)), float)

    @pytest.mark.parametrize("kind,option_type", list(itertools.product(("cash", "asset"), OPTION_TYPES)))
    def test_monte_carlo_agrees(self, kind, option_type):
        digital = DigitalOption(option_type, 105.0, 0.75, 1.0 if kind == "asset" else 20.0, kind)
        mkt = Market(100.0, 0.3, 0.05, 0.02)
        value, stderr = mc_price(digital, mkt, n_paths=100_000, n_steps=1, seed=5)
        assert abs(value - digital.price(mkt)) < 4.0 * stderr


class TestDigitalGreeks:
    @pytest.mark.parametrize("kind,option_type", list(itertools.product(("cash", "asset"), OPTION_TYPES)))
    def test_analytic_greeks_match_bump_and_reprice(self, kind, option_type):
        digital = DigitalOption(option_type, 100.0, 0.85, 7.0 if kind == "cash" else 0.5, kind)
        mkt = Market(np.array([80.0, 95.0, 100.0, 104.0, 125.0]), 0.25, 0.05, 0.02, t=0.1)
        analytic = digital.greeks(mkt)
        assert tuple(analytic) == GREEK_KEYS
        assert_greeks_close(analytic, numerical_greeks(digital, mkt), rel=3e-3)

    def test_greeks_match_bump_and_reprice_on_a_vol_and_time_grid(self):
        digital = DigitalOption("call", 100.0, 1.0, 3.0)
        mkt = Market(np.array([90.0, 110.0])[:, None, None], np.array([0.15, 0.4])[None, :, None],
                     0.03, 0.01, t=np.array([0.0, 0.7])[None, None, :])
        analytic = digital.greeks(mkt)
        assert analytic["gamma"].shape == (2, 2, 2)
        assert_greeks_close(analytic, numerical_greeks(digital, mkt), rel=3e-3)

    def test_scalar_market_gives_floats(self):
        for kind in ("cash", "asset"):
            digital = DigitalOption("put", 100.0, 1.0, kind=kind)
            assert isinstance(digital.price(Market(100.0, 0.2)), float)
            assert all(isinstance(v, float) for v in digital.greeks(Market(100.0, 0.2)).values())

    def test_greeks_price_entry_is_the_price(self):
        mkt = Market(np.linspace(60.0, 140.0, 9), 0.2, 0.03)
        for kind in ("cash", "asset"):
            digital = DigitalOption("call", 100.0, 1.0, 4.0, kind)
            np.testing.assert_allclose(digital.greeks(mkt)["price"], digital.price(mkt), rtol=1e-14)

    def test_delta_spike_grows_towards_expiry(self):
        digital = DigitalOption("call", 100.0, 1.0, payout=1.0)
        peaks = [digital.greeks(Market(100.0, 0.2, t=t))["delta"] for t in (0.0, 0.9, 0.99, 0.999)]
        assert all(later > 2.0 * earlier for earlier, later in zip(peaks, peaks[1:]))
        # ... while far from the strike the delta dies out.
        assert digital.greeks(Market(90.0, 0.2, t=0.999))["delta"] < 1e-10

    def test_gamma_and_vega_change_sign_at_the_strike(self):
        call = DigitalOption("call", 100.0, 1.0)
        below, above = call.greeks(Market(92.0, 0.2, t=0.9)), call.greeks(Market(108.0, 0.2, t=0.9))
        assert below["gamma"] > 0 > above["gamma"]
        assert below["vega"] > 0 > above["vega"]  # OTM wants movement, ITM wants calm
        put = DigitalOption("put", 100.0, 1.0)
        assert put.greeks(Market(92.0, 0.2, t=0.9))["vega"] < 0 < put.greeks(Market(108.0, 0.2, t=0.9))["vega"]

    def test_call_delta_positive_put_delta_negative(self):
        mkt = Market(np.linspace(70.0, 130.0, 13), 0.25, 0.02)
        assert np.all(DigitalOption("call", 100.0, 1.0).greeks(mkt)["delta"] > 0)
        assert np.all(DigitalOption("put", 100.0, 1.0).greeks(mkt)["delta"] < 0)


class TestDigitalEdgeCases:
    def test_payoff(self):
        spots = np.array([90.0, 100.0, 100.01, 130.0])
        np.testing.assert_array_equal(DigitalOption("call", 100.0, 1.0, 5.0).payoff(spots), [0, 0, 5, 5])
        np.testing.assert_array_equal(DigitalOption("put", 100.0, 1.0, 5.0).payoff(spots), [5, 0, 0, 0])
        np.testing.assert_allclose(
            DigitalOption("call", 100.0, 1.0, 2.0, "asset").payoff(spots), [0, 0, 200.02, 260.0])
        np.testing.assert_allclose(
            DigitalOption("put", 100.0, 1.0, 2.0, "asset").payoff(spots), [180.0, 0, 0, 0])

    @pytest.mark.parametrize("t", [1.0, 1.5])
    @pytest.mark.parametrize("kind,option_type", list(itertools.product(("cash", "asset"), OPTION_TYPES)))
    def test_settlement_at_and_after_expiry(self, kind, option_type, t):
        digital = DigitalOption(option_type, 100.0, 1.0, 3.0, kind)
        spots = np.array([80.0, 100.0, 125.0])
        mkt = Market(spots, 0.2, 0.05, 0.01, t=t)
        np.testing.assert_allclose(digital.price(mkt), digital.payoff(spots))
        greeks = digital.greeks(mkt)
        expected_delta = np.where(digital.payoff(spots) > 0, 3.0, 0.0) if kind == "asset" else np.zeros(3)
        np.testing.assert_allclose(greeks["delta"], expected_delta)
        for key in GREEK_KEYS[2:]:
            np.testing.assert_array_equal(greeks[key], np.zeros(3))

    def test_zero_vol_is_the_discounted_forward_indicator(self):
        mkt = Market(np.array([90.0, 99.0, 110.0]), 0.0, 0.05, 0.0)  # forwards 94.6, 104.1, 115.6
        df = math.exp(-0.05)
        np.testing.assert_allclose(DigitalOption("call", 100.0, 1.0, 2.0).price(mkt), [0, 2 * df, 2 * df])
        np.testing.assert_allclose(DigitalOption("put", 100.0, 1.0, 2.0).price(mkt), [2 * df, 0, 0])
        np.testing.assert_allclose(
            DigitalOption("call", 100.0, 1.0, kind="asset").price(mkt), [0.0, 99.0, 110.0])
        greeks = DigitalOption("call", 100.0, 1.0, 2.0).greeks(mkt)
        np.testing.assert_array_equal(greeks["gamma"], np.zeros(3))
        np.testing.assert_allclose(greeks["theta"], 0.05 * greeks["price"])
        np.testing.assert_allclose(greeks["rho"], -1.0 * greeks["price"])

    def test_tiny_vol_is_continuous_with_zero_vol(self):
        spots = np.array([90.0, 99.0, 110.0])
        for kind in ("cash", "asset"):
            digital = DigitalOption("call", 100.0, 1.0, 2.0, kind)
            np.testing.assert_allclose(digital.price(Market(spots, 1e-6, 0.05)),
                                       digital.price(Market(spots, 0.0, 0.05)), atol=1e-12)

    def test_zero_spot(self):
        mkt = Market(0.0, 0.2, 0.05)
        assert DigitalOption("call", 100.0, 1.0, 4.0).price(mkt) == 0.0
        assert DigitalOption("put", 100.0, 1.0, 4.0).price(mkt) == pytest.approx(4.0 * math.exp(-0.05))
        assert DigitalOption("call", 100.0, 1.0, kind="asset").price(mkt) == 0.0
        assert DigitalOption("put", 100.0, 1.0, kind="asset").price(mkt) == pytest.approx(0.0, abs=1e-12)

    def test_nan_inputs_propagate(self):
        mkt = Market(np.array([100.0, np.nan]), 0.2)
        digital = DigitalOption("call", 100.0, 1.0)
        assert np.isnan(digital.price(mkt)[1]) and np.isfinite(digital.price(mkt)[0])
        assert np.isnan(digital.greeks(mkt)["delta"][1])

    def test_validation(self):
        with pytest.raises(ValueError, match="option_type"):
            DigitalOption("straddle", 100.0, 1.0)
        with pytest.raises(ValueError, match="strike"):
            DigitalOption("call", 0.0, 1.0)
        with pytest.raises(ValueError, match="payout"):
            DigitalOption("call", 100.0, 1.0, payout=-1.0)
        with pytest.raises(ValueError, match="kind"):
            DigitalOption("call", 100.0, 1.0, kind="gap")
        with pytest.raises(ValueError, match="expiry"):
            DigitalOption("call", 100.0, math.inf)
        with pytest.raises(ValueError, match="numbers"):
            DigitalOption("call", "abc", 1.0)

    def test_normalisation_label_hash_and_serialisation(self):
        digital = DigitalOption(" CALL ", 100, 1, payout=2, kind="Asset")
        assert (digital.option_type, digital.kind) == ("call", "asset")
        assert isinstance(digital.strike, float) and isinstance(digital.payout, float)
        assert digital.label == "AoN C 100 T=1.00"
        assert DigitalOption("put", 97.5, 0.25).label == "Dig P 97.5 T=0.25"
        assert digital == DigitalOption("call", 100.0, 1.0, 2.0, "asset")
        assert len({digital, DigitalOption("call", 100.0, 1.0, 2.0, "asset")}) == 1
        assert INSTRUMENT_REGISTRY["DigitalOption"] is DigitalOption
        data = json.loads(json.dumps(digital.to_dict()))
        assert data["type"] == "DigitalOption"
        assert instrument_from_dict(data) == digital
        assert digital.vanilla_equivalent() == EuropeanOption("call", 100.0, 1.0)

    def test_works_inside_positions_and_composites(self):
        digital = DigitalOption("call", 100.0, 1.0, 10.0)
        package = CompositeInstrument("digital risk reversal", (
            Position(digital, 2.0), Position(DigitalOption("put", 90.0, 1.0, 10.0), -1.0)))
        mkt = Market(np.array([85.0, 100.0]), 0.2, 0.02)
        expected = 2.0 * digital.price(mkt) - DigitalOption("put", 90.0, 1.0, 10.0).price(mkt)
        np.testing.assert_allclose(package.price(mkt), expected)
        assert instrument_from_dict(json.loads(json.dumps(package.to_dict()))) == package
        assert digital.observe(120.0, 0.5) is digital


class TestCallSpreadReplication:
    MKT = Market(100.0, 0.2, 0.05, 0.01)

    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    def test_structure(self, option_type):
        digital = DigitalOption(option_type, 100.0, 1.0, payout=10.0)
        spread = digital.replicating_call_spread(2.0)
        assert isinstance(spread, CompositeInstrument)
        assert spread.expiry == digital.expiry
        legs = {leg.instrument.strike: leg.quantity for leg in spread.legs}
        long_strike, short_strike = (99.0, 101.0) if option_type == "call" else (101.0, 99.0)
        assert legs == {long_strike: pytest.approx(5.0), short_strike: pytest.approx(-5.0)}
        assert all(leg.instrument.option_type == option_type for leg in spread.legs)

    @pytest.mark.parametrize("kind,option_type", list(itertools.product(("cash", "asset"), OPTION_TYPES)))
    def test_centered_spread_converges_quadratically(self, kind, option_type):
        digital = DigitalOption(option_type, 105.0, 1.0, 10.0 if kind == "cash" else 1.0, kind)
        exact = digital.price(self.MKT)
        errors = [abs(digital.replicating_call_spread(w).price(self.MKT) - exact) for w in (8.0, 4.0, 2.0)]
        assert errors[2] < 0.01 * exact
        assert 3.5 < errors[0] / errors[1] < 4.5 and 3.5 < errors[1] / errors[2] < 4.5

    @pytest.mark.parametrize("kind,option_type", list(itertools.product(("cash", "asset"), OPTION_TYPES)))
    def test_payoff_matches_outside_the_ramp(self, kind, option_type):
        digital = DigitalOption(option_type, 100.0, 1.0, 3.0, kind)
        spots = np.array([50.0, 95.0, 98.99, 101.01, 105.0, 170.0])
        np.testing.assert_allclose(
            digital.replicating_call_spread(2.0).payoff(spots), digital.payoff(spots), atol=1e-9)

    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    def test_conservative_spread_dominates_the_digital(self, option_type):
        digital = DigitalOption(option_type, 100.0, 1.0, payout=10.0)
        spread = digital.replicating_call_spread(5.0, placement="conservative")
        spots = np.linspace(60.0, 140.0, 321)
        assert np.all(spread.payoff(spots) >= digital.payoff(spots) - 1e-12)
        mkt = Market(spots, 0.2, 0.03)
        assert np.all(spread.price(mkt) > digital.price(mkt))
        # the ramp sits entirely on the out-of-the-money side of the strike
        strikes = sorted(leg.instrument.strike for leg in spread.legs)
        assert strikes == ([95.0, 100.0] if option_type == "call" else [100.0, 105.0])

    def test_spread_caps_the_delta_that_the_digital_lets_explode(self):
        digital = DigitalOption("call", 100.0, 1.0, payout=10.0)
        spread = digital.replicating_call_spread(5.0)
        near_expiry = Market(100.0, 0.2, t=0.9999)
        assert digital.greeks(near_expiry)["delta"] > 10.0
        assert spread.greeks(near_expiry)["delta"] <= 10.0 / 5.0 + 1e-9

    def test_tight_spread_reproduces_the_greeks(self):
        digital = DigitalOption("call", 100.0, 1.0, payout=10.0)
        mkt = Market(np.array([90.0, 100.0, 112.0]), 0.25, 0.03, 0.01)
        assert_greeks_close(digital.replicating_call_spread(0.05).greeks(mkt), digital.greeks(mkt),
                            rel=1e-3, keys=("price", "delta", "gamma", "vega", "theta", "rho"))

    def test_validation(self):
        digital = DigitalOption("call", 100.0, 1.0)
        for bad in (0.0, -1.0, math.nan, "wide"):
            with pytest.raises(ValueError, match="width"):
                digital.replicating_call_spread(bad)
        with pytest.raises(ValueError, match="too large"):
            digital.replicating_call_spread(250.0)
        with pytest.raises(ValueError, match="placement"):
            digital.replicating_call_spread(1.0, placement="aggressive")


# ====================================================================== #
# BarrierOption: closed form
# ====================================================================== #
def make_pair(option_type, direction, strike, barrier, expiry=1.0, rebate=0.0):
    """Matching (knock-out, knock-in) pair."""
    return (BarrierOption(option_type, strike, barrier, expiry, f"{direction}-and-out", rebate),
            BarrierOption(option_type, strike, barrier, expiry, f"{direction}-and-in", rebate))


def reflection_knock_out_price(option_type, strike, barrier, tau, spot, vol, rate, div):
    """Knock-out price (no rebate) by integrating the reflection-principle density.

    For ``x = ln(S_T / S)`` and ``a = ln(H / S)``, the density of paths that
    end at ``x`` without touching the barrier is the Gaussian density minus
    its image reflected in the barrier, weighted by ``exp(2 nu a / vol^2)``.
    """
    nu = rate - div - 0.5 * vol**2
    a = math.log(barrier / spot)
    sd = vol * math.sqrt(tau)
    omega = 1.0 if option_type == "call" else -1.0

    def integrand(x):
        density = (norm.pdf((x - nu * tau) / sd)
                   - math.exp(2.0 * nu * a / vol**2) * norm.pdf((x - 2.0 * a - nu * tau) / sd)) / sd
        return max(omega * (spot * math.exp(x) - strike), 0.0) * density

    wide = 12.0 * sd + abs(nu) * tau
    lower, upper = (-wide, a) if barrier > spot else (a, wide)
    kink = math.log(strike / spot)
    value, _ = quad(integrand, lower, upper, points=[kink] if lower < kink < upper else None,
                    epsabs=1e-12, epsrel=1e-12, limit=400)
    return math.exp(-rate * tau) * value


def first_passage_integrals(barrier, tau, spot, vol, rate, div):
    """``(E[exp(-r t_hit); t_hit <= tau], P[t_hit <= tau])`` by numerical integration."""
    nu = rate - div - 0.5 * vol**2
    a = math.log(barrier / spot)

    def density(t):
        return abs(a) / (vol * math.sqrt(2.0 * math.pi * t**3)) * math.exp(-((a - nu * t) ** 2) / (2.0 * vol**2 * t))

    discounted = quad(lambda t: math.exp(-rate * t) * density(t), 0.0, tau, epsabs=1e-13, epsrel=1e-12)[0]
    probability = quad(density, 0.0, tau, epsabs=1e-13, epsrel=1e-12)[0]
    return discounted, probability


MARKETS = [  # (vol, rate, div)
    (0.20, 0.05, 0.01),
    (0.35, 0.01, 0.06),
    (0.10, 0.08, 0.00),
    (0.50, 0.00, 0.00),
    (0.15, -0.01, 0.00),
]


class TestBarrierParity:
    @pytest.mark.parametrize("vol,rate,div", MARKETS)
    @pytest.mark.parametrize("strike", [80.0, 100.0, 130.0])  # below, between and above both barriers
    @pytest.mark.parametrize("direction,barrier", [("up", 120.0), ("down", 85.0)])
    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    def test_knock_in_plus_knock_out_is_vanilla(self, option_type, direction, barrier, strike, vol, rate, div):
        knock_out, knock_in = make_pair(option_type, direction, strike, barrier)
        spots = np.array([50.0, 70.0, 84.0, 85.0, 86.0, 95.0, 100.0, 110.0, 119.0, 120.0, 121.0, 150.0])
        mkt = Market(spots, vol, rate, div, t=0.3)
        vanilla = EuropeanOption(option_type, strike, 1.0).price(mkt)
        ko, ki = knock_out.price(mkt), knock_in.price(mkt)
        np.testing.assert_allclose(ko + ki, vanilla, atol=1e-10)
        assert np.all(ko >= 0) and np.all(ki >= -1e-12)
        assert np.all(ko <= vanilla + 1e-10) and np.all(ki <= vanilla + 1e-10)

    @pytest.mark.parametrize("strike", [80.0, 100.0, 130.0])
    @pytest.mark.parametrize("direction,barrier", [("up", 120.0), ("down", 85.0)])
    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    def test_parity_holds_for_the_greeks_too(self, option_type, direction, barrier, strike):
        knock_out, knock_in = make_pair(option_type, direction, strike, barrier)
        spots = (np.array([70.0, 100.0, 119.0, 119.95, 120.0, 120.5, 140.0]) if direction == "up"
                 else np.array([60.0, 84.5, 85.0, 85.04, 86.0, 100.0, 130.0]))
        mkt = Market(spots, 0.25, 0.05, 0.02, t=0.2)
        ko, ki = knock_out.greeks(mkt), knock_in.greeks(mkt)
        total = {k: ko[k] + ki[k] for k in GREEK_KEYS}
        assert_greeks_close(total, EuropeanOption(option_type, strike, 1.0).greeks(mkt), rel=3e-3)

    @pytest.mark.parametrize("strike", [80.0, 130.0])
    @pytest.mark.parametrize("direction,barrier", [("up", 120.0), ("down", 85.0)])
    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    def test_with_zero_rates_the_two_rebates_add_up_to_the_rebate(self, option_type, direction, barrier, strike):
        # One of "hit" / "never hit" happens for sure, and with r = 0 timing does not matter.
        knock_out, knock_in = make_pair(option_type, direction, strike, barrier, rebate=4.0)
        mkt = Market(np.array([90.0, 100.0, 110.0]), 0.3, 0.0, 0.03)
        vanilla = EuropeanOption(option_type, strike, 1.0).price(mkt)
        np.testing.assert_allclose(knock_out.price(mkt) + knock_in.price(mkt), vanilla + 4.0, atol=1e-10)


# Haug (2007), "The Complete Guide to Option Pricing Formulas", standard barrier table:
# S = 100, T = 0.5, r = 0.08, b = 0.04 (so div = 0.04), rebate = 3; columns vol = 25% and 30%.
HAUG_TASK_VALUES = [  # the values quoted in the project brief
    ("call", "down-and-out", 90.0, 95.0, 0.25, 9.0246),
    ("call", "down-and-out", 90.0, 95.0, 0.30, 8.8334),
    ("call", "up-and-out", 90.0, 105.0, 0.25, 2.6789),
    ("call", "down-and-in", 90.0, 95.0, 0.25, 7.7627),
]
HAUG_TABLE = {  # (type, barrier type, strike, barrier): (vol 25%, vol 30%)
    ("call", "down-and-out", 90, 95): (9.0246, 8.8334),
    ("call", "down-and-out", 100, 95): (6.7924, 7.0285),
    ("call", "down-and-out", 110, 95): (4.8759, 5.4137),
    ("call", "down-and-out", 90, 100): (3.0000, 3.0000),
    ("call", "up-and-out", 90, 105): (2.6789, 2.6341),
    ("call", "up-and-out", 100, 105): (2.3580, 2.4389),
    ("call", "up-and-out", 110, 105): (2.3453, 2.4315),
    ("put", "down-and-out", 90, 95): (2.2798, 2.4170),
    ("put", "down-and-out", 100, 95): (2.2947, 2.4258),
    ("put", "down-and-out", 110, 95): (2.6252, 2.6246),
    ("put", "up-and-out", 90, 105): (3.7760, 4.2293),
    ("put", "up-and-out", 100, 105): (5.4932, 5.8032),
    ("put", "up-and-out", 110, 105): (7.5187, 7.5649),
    ("call", "down-and-in", 90, 95): (7.7627, 9.0093),
    ("call", "down-and-in", 100, 95): (4.0109, 5.1370),
    ("call", "down-and-in", 110, 95): (2.0576, 2.8517),
    ("call", "down-and-in", 90, 100): (13.8333, 14.8816),
    ("call", "down-and-in", 100, 100): (7.8494, 9.2045),
    ("call", "down-and-in", 110, 100): (3.9795, 5.3043),
    ("call", "up-and-in", 90, 105): (14.1112, 15.2098),
    ("call", "up-and-in", 100, 105): (8.4482, 9.7278),
    ("call", "up-and-in", 110, 105): (4.5910, 5.8350),
    ("put", "down-and-in", 90, 95): (2.9586, 3.8769),
    ("put", "down-and-in", 100, 95): (6.5677, 7.7989),
    ("put", "down-and-in", 110, 95): (11.9752, 13.3078),
    ("put", "down-and-in", 90, 100): (2.2845, 3.3328),
    ("put", "down-and-in", 100, 100): (5.9085, 7.2636),
    ("put", "down-and-in", 110, 100): (11.6465, 12.9713),
    ("put", "up-and-in", 90, 105): (1.4653, 2.0658),
    ("put", "up-and-in", 100, 105): (3.3721, 4.4226),
    ("put", "up-and-in", 110, 105): (7.0846, 8.3686),
}


class TestBarrierReferenceValues:
    @pytest.mark.parametrize("option_type,barrier_type,strike,barrier,vol,expected", HAUG_TASK_VALUES)
    def test_haug_values_from_the_brief(self, option_type, barrier_type, strike, barrier, vol, expected):
        option = BarrierOption(option_type, strike, barrier, 0.5, barrier_type, rebate=3.0)
        assert option.price(Market(100.0, vol, 0.08, 0.04)) == pytest.approx(expected, abs=5e-5)

    @pytest.mark.parametrize("case", list(HAUG_TABLE), ids=lambda c: f"{c[0]}-{c[1]}-K{c[2]}-H{c[3]}")
    def test_haug_table(self, case):
        # Table values as recalled from the book (4 decimals). 56 of the 62 numbers agree to
        # 5e-5; six vol = 30% entries differ by one unit in the last digit (e.g. 4.22924 here
        # versus 4.2293 printed), which is within rounding / normal-CDF precision of the
        # original, hence atol = 1.5e-4. The sharp arbiters are the integration tests below
        # (1e-8) and the Monte Carlo tests.
        option_type, barrier_type, strike, barrier = case
        option = BarrierOption(option_type, strike, barrier, 0.5, barrier_type, rebate=3.0)
        prices = option.price(Market(100.0, np.array([0.25, 0.30]), 0.08, 0.04))
        np.testing.assert_allclose(prices, HAUG_TABLE[case], atol=1.5e-4)

    @pytest.mark.parametrize("vol,rate,div", MARKETS)
    @pytest.mark.parametrize("strike", [80.0, 100.0, 130.0])
    @pytest.mark.parametrize("direction,barrier", [("up", 120.0), ("down", 85.0)])
    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    def test_knock_out_matches_reflection_principle_integral(
        self, option_type, direction, barrier, strike, vol, rate, div
    ):
        option = BarrierOption(option_type, strike, barrier, 0.75, f"{direction}-and-out")
        reference = reflection_knock_out_price(option_type, strike, barrier, 0.75, 100.0, vol, rate, div)
        assert option.price(Market(100.0, vol, rate, div)) == pytest.approx(reference, abs=1e-8)

    @pytest.mark.parametrize("vol,rate,div", MARKETS)
    @pytest.mark.parametrize("direction,barrier", [("up", 120.0), ("down", 85.0)])
    def test_rebates_match_first_passage_integrals(self, direction, barrier, vol, rate, div):
        mkt = Market(100.0, vol, rate, div)
        discounted_hit, hit_probability = first_passage_integrals(barrier, 0.75, 100.0, vol, rate, div)
        for option_type, strike in itertools.product(OPTION_TYPES, (80.0, 130.0)):
            out_0, in_0 = make_pair(option_type, direction, strike, barrier, 0.75)
            out_r, in_r = make_pair(option_type, direction, strike, barrier, 0.75, rebate=5.0)
            # knock-out: rebate paid AT THE HIT
            assert out_r.price(mkt) - out_0.price(mkt) == pytest.approx(5.0 * discounted_hit, abs=1e-9)
            # knock-in: rebate paid AT EXPIRY if never hit
            no_hit_bond = 5.0 * math.exp(-rate * 0.75) * (1.0 - hit_probability)
            assert in_r.price(mkt) - in_0.price(mkt) == pytest.approx(no_hit_bond, abs=1e-9)

    def test_up_and_out_call_struck_above_the_barrier_is_worth_only_its_rebate(self):
        mkt = Market(np.array([80.0, 100.0, 115.0]), 0.3, 0.04)
        np.testing.assert_array_equal(BarrierOption("call", 125.0, 120.0, 1.0, "uo").price(mkt), np.zeros(3))
        np.testing.assert_array_equal(BarrierOption("put", 80.0, 85.0, 1.0, "do").price(
            mkt.bumped(spot=np.array([90.0, 100.0, 115.0]))), np.zeros(3))
        with_rebate = BarrierOption("call", 125.0, 120.0, 1.0, "uo", rebate=2.0).price(mkt)
        assert np.all((with_rebate > 0) & (with_rebate < 2.0))

    def test_deeply_negative_rates_with_a_rebate_at_hit_are_rejected(self):
        mkt = Market(100.0, 0.2, rate=-0.02, div=-0.02)
        with pytest.raises(ValueError, match="rebate-at-hit"):
            BarrierOption("call", 100.0, 120.0, 1.0, "uo", rebate=1.0).price(mkt)
        # no rebate at hit -> no problem
        assert BarrierOption("call", 100.0, 120.0, 1.0, "uo").price(mkt) > 0
        assert BarrierOption("call", 100.0, 120.0, 1.0, "ui", rebate=1.0).price(mkt) > 0


class TestBarrierLimits:
    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    @pytest.mark.parametrize("direction,barrier", [("up", 1e4), ("down", 1e-2)])
    def test_far_away_barrier(self, option_type, direction, barrier):
        knock_out, knock_in = make_pair(option_type, direction, 100.0, barrier, rebate=5.0)
        mkt = Market(np.array([80.0, 100.0, 125.0]), 0.3, 0.04, 0.01)
        vanilla = EuropeanOption(option_type, 100.0, 1.0).price(mkt)
        np.testing.assert_allclose(knock_out.price(mkt), vanilla, atol=1e-9)
        # never knocked in: only the rebate, paid at expiry for sure
        np.testing.assert_allclose(knock_in.price(mkt), 5.0 * math.exp(-0.04), atol=1e-9)

    @pytest.mark.parametrize("strike", [80.0, 100.0, 130.0])
    @pytest.mark.parametrize("direction,barrier", [("up", 120.0), ("down", 85.0)])
    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    def test_value_is_continuous_at_the_barrier(self, option_type, direction, barrier, strike):
        knock_out, knock_in = make_pair(option_type, direction, strike, barrier, rebate=2.5)
        inside = barrier * (1.0 - 1e-9) if direction == "up" else barrier * (1.0 + 1e-9)
        mkt = Market(inside, 0.3, 0.05, 0.02, t=0.4)
        assert knock_out.price(mkt) == pytest.approx(2.5, abs=1e-5)
        assert knock_in.price(mkt) == pytest.approx(EuropeanOption(option_type, strike, 1.0).price(mkt), abs=1e-5)

    def test_barrier_option_is_cheaper_than_the_vanilla_and_monotone_in_the_barrier(self):
        mkt = Market(100.0, 0.25, 0.03)
        vanilla = EuropeanOption("call", 100.0, 1.0).price(mkt)
        prices = [BarrierOption("call", 100.0, h, 1.0, "up-and-out").price(mkt) for h in (105, 120, 150, 400)]
        assert all(a < b for a, b in zip(prices, prices[1:])) and prices[-1] < vanilla
        assert prices[0] < 0.01 * vanilla
        assert prices[-1] == pytest.approx(vanilla, rel=1e-4)

    def test_reverse_knock_out_is_short_vol_near_the_barrier(self):
        option = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-out")
        assert option.is_reverse and not BarrierOption("call", 100.0, 85.0, 1.0, "do").is_reverse
        greeks = option.greeks(Market(115.0, 0.25, 0.03, t=0.5))
        assert greeks["vega"] < 0 and greeks["delta"] < 0 and greeks["gamma"] < 0


class TestBarrierDegenerateMarkets:
    def test_zero_vol_follows_the_forward_curve(self):
        # rate 25%: a spot of 90 grows to 115.6 (no hit), 100 reaches 120 at t* = ln(1.2) / 0.25
        mkt = Market(np.array([90.0, 100.0, 125.0]), 0.0, 0.25)
        t_hit = math.log(1.2) / 0.25
        rebate_pv = 2.0 * math.exp(-0.25 * t_hit)
        vanilla = EuropeanOption("call", 100.0, 1.0).price(mkt)
        uoc = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-out", rebate=2.0).price(mkt)
        np.testing.assert_allclose(uoc, [vanilla[0], rebate_pv, 2.0])
        uic = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-in", rebate=2.0).price(mkt)
        np.testing.assert_allclose(uic, [2.0 * math.exp(-0.25), vanilla[1], vanilla[2]])
        # down barrier with a rising forward is never hit
        doc = BarrierOption("call", 100.0, 85.0, 1.0, "down-and-out", rebate=2.0).price(mkt)
        np.testing.assert_allclose(doc, vanilla)

    @pytest.mark.parametrize("barrier_type,barrier", [("uo", 120.0), ("ui", 120.0), ("do", 85.0), ("di", 85.0)])
    @pytest.mark.parametrize("rate,div", [(0.25, 0.0), (0.0, 0.25), (0.03, 0.03)])
    def test_tiny_vol_is_continuous_with_zero_vol(self, barrier_type, barrier, rate, div):
        spots = np.array([70.0, 90.0, 100.0, 110.0, 125.0])
        for option_type in OPTION_TYPES:
            option = BarrierOption(option_type, 100.0, barrier, 1.0, barrier_type, rebate=2.0)
            tiny, zero = option.price(Market(spots, 1e-5, rate, div)), option.price(Market(spots, 0.0, rate, div))
            assert np.all(np.isfinite(tiny))
            np.testing.assert_allclose(tiny, zero, atol=1e-3)

    def test_zero_spot(self):
        mkt = Market(0.0, 0.3, 0.05)
        assert BarrierOption("put", 100.0, 120.0, 1.0, "uo").price(mkt) == pytest.approx(100 * math.exp(-0.05))
        assert BarrierOption("put", 100.0, 120.0, 1.0, "ui", rebate=1.0).price(mkt) == pytest.approx(math.exp(-0.05))
        assert BarrierOption("put", 100.0, 50.0, 1.0, "do", rebate=1.0).price(mkt) == 1.0  # breached now

    def test_nan_inputs_propagate(self):
        prices = BarrierOption("call", 100.0, 120.0, 1.0, "uo").price(Market(np.array([100.0, np.nan]), 0.2))
        assert np.isfinite(prices[0]) and np.isnan(prices[1])

    def test_an_instant_before_expiry_is_finite_and_close_to_settlement(self):
        spots = np.array([90.0, 110.0, 119.0, 125.0])
        for option in make_pair("call", "up", 100.0, 120.0, rebate=1.0):
            before = option.price(Market(spots, 0.3, 0.05, t=1.0 - 1e-12))
            np.testing.assert_allclose(before, option.price(Market(spots, 0.3, 0.05, t=1.0)), atol=1e-4)


# ====================================================================== #
# BarrierOption: state, settlement, vectorisation
# ====================================================================== #
class TestBarrierState:
    def test_observe_up_barrier_uses_greater_or_equal(self):
        option = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-out")
        assert option.observe(119.999, 0.5) is option
        hit = option.observe(120.0, 0.5)
        assert hit is not option and hit.knocked and not option.knocked
        assert hit == BarrierOption("call", 100.0, 120.0, 1.0, "up-and-out", knocked=True)
        assert option.observe(80.0, 0.5) is option

    def test_observe_down_barrier_uses_less_or_equal(self):
        option = BarrierOption("put", 100.0, 85.0, 1.0, "down-and-in")
        assert option.observe(85.001, 0.2) is option
        assert option.observe(85.0, 0.2).knocked and option.observe(60.0, 0.2).knocked
        assert option.observe(150.0, 0.2) is option

    def test_observe_is_sticky_and_ignores_the_time_after_expiry(self):
        option = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-in")
        knocked = option.observe(125.0, 0.3)
        assert knocked.observe(90.0, 0.4) is knocked  # cannot be un-knocked
        assert option.observe(125.0, 1.0).knocked     # the expiry date is still monitored
        assert option.observe(125.0, 1.0 + 1e-6) is option

    def test_observe_propagates_through_positions_and_composites(self):
        option = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-out")
        package = CompositeInstrument("capped call", (Position(option, 2.0), Position(EuropeanOption("put", 90.0, 1.0), -1.0)))
        assert package.observe(110.0, 0.1) is package
        observed = package.observe(121.0, 0.1)
        assert observed.legs[0].instrument.knocked and observed.legs[0].quantity == 2.0
        assert observed.legs[1] is package.legs[1]

    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    @pytest.mark.parametrize("direction,barrier", [("up", 120.0), ("down", 85.0)])
    def test_knocked_options(self, option_type, direction, barrier):
        knock_out, knock_in = (o.observe(barrier, 0.2) for o in make_pair(option_type, direction, 100.0, barrier, rebate=3.0))
        mkt = Market(np.array([70.0, 100.0, 140.0]), 0.25, 0.04, 0.01, t=0.3)
        vanilla = EuropeanOption(option_type, 100.0, 1.0)
        np.testing.assert_array_equal(knock_out.price(mkt), np.zeros(3))  # rebate already paid
        np.testing.assert_allclose(knock_in.price(mkt), vanilla.price(mkt))
        for key, value in knock_out.greeks(mkt).items():
            np.testing.assert_array_equal(value, np.zeros(3), err_msg=key)
        assert_greeks_close(knock_in.greeks(mkt), vanilla.greeks(mkt), rel=1e-12)
        spots = np.array([60.0, 100.0, 150.0])
        np.testing.assert_array_equal(knock_out.payoff(spots), np.zeros(3))
        np.testing.assert_allclose(knock_in.payoff(spots), vanilla.payoff(spots))
        # settlement of knocked options
        expired = mkt.bumped(t=1.0)
        np.testing.assert_array_equal(knock_out.price(expired), np.zeros(3))
        np.testing.assert_allclose(knock_in.price(expired), vanilla.payoff(mkt.spot))

    def test_spot_beyond_the_barrier_prices_as_a_hit_happening_now(self):
        spots = np.array([100.0, 119.0, 120.0, 121.0, 160.0])
        mkt = Market(spots, 0.25, 0.04, t=0.5)
        knock_out, knock_in = make_pair("call", "up", 100.0, 120.0, rebate=3.0)
        vanilla = EuropeanOption("call", 100.0, 1.0).price(mkt)
        np.testing.assert_array_equal(knock_out.price(mkt)[2:], [3.0, 3.0, 3.0])
        np.testing.assert_allclose(knock_in.price(mkt)[2:], vanilla[2:])
        # on the live side the closed form applies: strictly between 0 and the vanilla (no rebate)
        plain_out, plain_in = make_pair("call", "up", 100.0, 120.0)
        for live in (plain_out.price(mkt)[:2], plain_in.price(mkt)[:2]):
            assert np.all((live > 0) & (live < vanilla[:2]))

        down_out, down_in = make_pair("put", "down", 100.0, 85.0, rebate=3.0)
        mkt = Market(np.array([60.0, 85.0, 86.0, 100.0]), 0.25, 0.04, t=0.5)
        np.testing.assert_array_equal(down_out.price(mkt)[:2], [3.0, 3.0])
        np.testing.assert_allclose(down_in.price(mkt)[:2], EuropeanOption("put", 100.0, 1.0).price(mkt)[:2])

    @pytest.mark.parametrize("t", [1.0, 1.3])
    def test_settlement_of_unknocked_options(self, t):
        spots = np.array([90.0, 110.0, 119.99, 120.0, 130.0])
        mkt = Market(spots, 0.25, 0.04, t=t)
        knock_out, knock_in = make_pair("call", "up", 100.0, 120.0, rebate=3.0)
        np.testing.assert_allclose(knock_out.price(mkt), [0.0, 10.0, 19.99, 3.0, 3.0])
        np.testing.assert_allclose(knock_in.price(mkt), [3.0, 3.0, 3.0, 20.0, 30.0])
        np.testing.assert_allclose(knock_out.payoff(spots), knock_out.price(mkt))
        np.testing.assert_allclose(knock_in.payoff(spots), knock_in.price(mkt))
        assert knock_out.is_expired(mkt) is True

    def test_vectorised_over_spot_vol_and_time(self):
        option = BarrierOption("put", 100.0, 85.0, 1.0, "down-and-out", rebate=1.0)
        spot = np.linspace(70.0, 130.0, 13)[:, None, None]
        vol = np.array([0.1, 0.2, 0.4])[None, :, None]
        t = np.array([0.0, 0.5, 1.0, 1.2])[None, None, :]
        grid = option.price(Market(spot, vol, 0.03, 0.01, t=t))
        assert grid.shape == (13, 3, 4)
        for i, j, k in [(0, 0, 0), (5, 1, 1), (12, 2, 2), (3, 0, 3), (7, 2, 0)]:
            single = option.price(Market(float(spot[i, 0, 0]), float(vol[0, j, 0]), 0.03, 0.01, t=float(t[0, 0, k])))
            assert isinstance(single, float)
            assert grid[i, j, k] == pytest.approx(single, abs=1e-13)


class TestBarrierGreeks:
    def test_keys_types_and_price_entry(self):
        option = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-out", rebate=1.0)
        scalar = option.greeks(Market(100.0, 0.25, 0.03))
        assert tuple(scalar) == GREEK_KEYS and all(isinstance(v, float) for v in scalar.values())
        assert scalar["price"] == option.price(Market(100.0, 0.25, 0.03))
        mkt = Market(np.array([90.0, 119.99, 120.0, 140.0]), 0.25, 0.03)
        vector = option.greeks(mkt)
        assert all(v.shape == (4,) for v in vector.values())
        np.testing.assert_array_equal(vector["price"], option.price(mkt))

    def test_away_from_the_barrier_they_are_plain_numerical_greeks(self):
        option = BarrierOption("put", 100.0, 85.0, 1.0, "down-and-in", rebate=1.0)
        mkt = Market(np.array([95.0, 100.0, 120.0]), 0.3, 0.04, 0.01, t=0.25)
        assert_greeks_close(option.greeks(mkt), numerical_greeks(option, mkt), rel=1e-9)

    def test_delta_is_one_sided_right_below_an_up_and_out_barrier(self):
        option = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-out")
        mkt = Market(119.95, 0.25, 0.05, 0.02, t=0.2)
        h = 1e-4
        one_sided = (option.price(mkt) - option.price(mkt.bumped(spot=119.95 - h))) / h
        greeks = option.greeks(mkt)
        assert one_sided < -0.05
        assert greeks["delta"] == pytest.approx(one_sided, rel=1e-3)
        # bumping across the barrier instead mixes in the dead side and misses the true slope
        naive = numerical_greeks(option, mkt)["delta"]
        assert abs(naive - one_sided) > 0.2 * abs(one_sided)

    def test_greeks_beyond_the_barrier(self):
        mkt = Market(np.array([120.0, 135.0]), 0.25, 0.05, t=0.2)
        knock_out, knock_in = make_pair("call", "up", 100.0, 120.0, rebate=2.0)
        out_greeks = knock_out.greeks(mkt)
        np.testing.assert_array_equal(out_greeks["price"], [2.0, 2.0])
        for key in GREEK_KEYS[1:]:
            np.testing.assert_array_equal(out_greeks[key], np.zeros(2), err_msg=key)
        assert_greeks_close(knock_in.greeks(mkt), EuropeanOption("call", 100.0, 1.0).greeks(mkt), rel=1e-12)

    def test_gamma_near_the_barrier_explodes_towards_expiry(self):
        option = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-out")
        early, late = (option.greeks(Market(118.0, 0.25, t=t)) for t in (0.5, 0.99))
        assert late["delta"] < 10.0 * early["delta"] < 0
        assert late["gamma"] < 10.0 * early["gamma"] < 0

    def test_expired_greeks(self):
        option = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-out")
        greeks = option.greeks(Market(np.array([90.0, 110.0]), 0.25, t=1.0))
        np.testing.assert_allclose(greeks["price"], [0.0, 10.0])
        np.testing.assert_allclose(greeks["delta"], [0.0, 1.0])
        for key in GREEK_KEYS[2:]:
            np.testing.assert_array_equal(greeks[key], np.zeros(2))


# ====================================================================== #
# BarrierOption: path payoff and Monte Carlo
# ====================================================================== #
class TestBarrierPathPayoff:
    TIMES = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    PATHS = np.array([
        [100.0, 110.0, 119.0, 105.0, 130.0],   # ends above 120 -> hit at expiry
        [100.0, 121.0, 110.0, 125.0, 108.0],   # first hit at t = 0.25
        [100.0, 95.0, 90.0, 99.0, 112.0],      # never hits 120
        [100.0, 110.0, 120.0, 100.0, 95.0],    # touches exactly at t = 0.5
    ])

    def test_knock_out_with_rebate_compounded_from_the_hit(self):
        option = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-out", rebate=2.0)
        np.testing.assert_allclose(option.path_payoff(self.PATHS, self.TIMES), [2.0, 2.0, 12.0, 2.0])
        rate = 0.1
        expected = [2.0, 2.0 * math.exp(rate * 0.75), 12.0, 2.0 * math.exp(rate * 0.5)]
        np.testing.assert_allclose(option.path_payoff(self.PATHS, self.TIMES, rate=rate), expected)

    def test_knock_in_with_rebate_at_expiry(self):
        option = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-in", rebate=2.0)
        expected = [30.0, 8.0, 2.0, 0.0]
        np.testing.assert_allclose(option.path_payoff(self.PATHS, self.TIMES), expected)
        np.testing.assert_allclose(option.path_payoff(self.PATHS, self.TIMES, rate=0.1), expected)

    def test_down_barriers(self):
        down_out, down_in = make_pair("put", "down", 100.0, 92.0)
        np.testing.assert_allclose(down_out.path_payoff(self.PATHS, self.TIMES), [0.0, 0.0, 0.0, 5.0])
        np.testing.assert_allclose(down_in.path_payoff(self.PATHS, self.TIMES), [0.0, 0.0, 0.0, 0.0])
        down_in_call = BarrierOption("call", 100.0, 92.0, 1.0, "down-and-in")
        np.testing.assert_allclose(down_in_call.path_payoff(self.PATHS, self.TIMES), [0.0, 0.0, 12.0, 0.0])

    def test_initial_knocked_state_is_honoured(self):
        vanilla = EuropeanOption("call", 100.0, 1.0).payoff(self.PATHS[:, -1])
        knocked_in = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-in", rebate=2.0, knocked=True)
        np.testing.assert_allclose(knocked_in.path_payoff(self.PATHS, self.TIMES), vanilla)
        knocked_out = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-out", rebate=2.0, knocked=True)
        np.testing.assert_array_equal(knocked_out.path_payoff(self.PATHS, self.TIMES), np.zeros(4))

    def test_the_current_spot_is_monitored(self):
        paths = np.array([[125.0, 110.0, 105.0]])
        times = np.array([0.5, 0.75, 1.0])
        option = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-out", rebate=1.0)
        np.testing.assert_allclose(option.path_payoff(paths, times, rate=0.2), [math.exp(0.2 * 0.5)])

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="times"):
            BarrierOption("call", 100.0, 120.0, 1.0, "uo").path_payoff(self.PATHS, self.TIMES[:-1])


class TestBarrierMonteCarlo:
    """MC monitors the barrier at the time steps only: compare with the BGK-shifted closed form."""

    MKT = Market(100.0, 0.25, 0.08, 0.04)
    N_STEPS = 126  # T = 0.5 -> roughly daily monitoring

    @pytest.mark.parametrize("barrier_type,barrier", [("up-and-out", 115.0), ("up-and-in", 115.0),
                                                      ("down-and-out", 88.0), ("down-and-in", 88.0)])
    @pytest.mark.parametrize("option_type", OPTION_TYPES)
    def test_mc_matches_bgk_adjusted_closed_form(self, option_type, barrier_type, barrier):
        option = BarrierOption(option_type, 100.0, barrier, 0.5, barrier_type)
        value, stderr = mc_price(option, self.MKT, n_paths=40_000, n_steps=self.N_STEPS, seed=11)
        adjusted = option.discrete_barrier_adjusted(0.25, dt=0.5 / self.N_STEPS)
        assert abs(value - adjusted.price(self.MKT)) < 3.5 * stderr

    def test_the_continuity_correction_matters(self):
        option = BarrierOption("call", 100.0, 115.0, 0.5, "up-and-out")
        value, stderr = mc_price(option, self.MKT, n_paths=40_000, n_steps=self.N_STEPS, seed=11)
        assert value - option.price(self.MKT) > 6.0 * stderr  # discrete monitoring knocks out less often

    def test_inverse_shift_lets_a_coarse_simulation_estimate_the_continuous_price(self):
        option = BarrierOption("put", 100.0, 88.0, 0.5, "down-and-out")
        shifted = BarrierOption("put", 100.0, bgk_adjusted_barrier(88.0, 0.25, 0.5 / self.N_STEPS, "down", inverse=True),
                                0.5, "down-and-out")
        value, stderr = mc_price(shifted, self.MKT, n_paths=40_000, n_steps=self.N_STEPS, seed=3)
        assert abs(value - option.price(self.MKT)) < 3.5 * stderr

    @pytest.mark.parametrize("barrier_type,barrier", [("up-and-out", 112.0), ("down-and-in", 90.0)])
    def test_rebates_with_the_rate_passed_to_path_payoff(self, barrier_type, barrier):
        option = BarrierOption("call", 100.0, barrier, 0.5, barrier_type, rebate=6.0)
        times, paths = simulate_gbm_paths(100.0, 0.25, 0.08, 0.04, horizon=0.5, n_steps=self.N_STEPS,
                                          n_paths=40_000, seed=21)
        pv = math.exp(-0.08 * 0.5) * option.path_payoff(paths, times, rate=0.08)
        pairs = 0.5 * (pv[:20_000] + pv[20_000:])
        stderr = pairs.std(ddof=1) / math.sqrt(pairs.size)
        adjusted = option.discrete_barrier_adjusted(0.25, n_monitoring_per_year=self.N_STEPS / 0.5)
        assert abs(pairs.mean() - adjusted.price(self.MKT)) < 3.5 * stderr

    def test_mc_from_a_knocked_state_and_mid_life(self):
        mkt = Market(104.0, 0.25, 0.08, 0.04, t=0.2)
        knocked_in = BarrierOption("call", 100.0, 115.0, 0.5, "up-and-in", knocked=True)
        value, stderr = mc_price(knocked_in, mkt, n_paths=40_000, n_steps=10, seed=2)
        assert abs(value - EuropeanOption("call", 100.0, 0.5).price(mkt)) < 3.5 * stderr
        knocked_out = BarrierOption("call", 100.0, 115.0, 0.5, "up-and-out", rebate=5.0, knocked=True)
        assert mc_price(knocked_out, mkt, n_paths=1_000, n_steps=10, seed=2) == (0.0, 0.0)

    def test_mc_inside_a_composite(self):
        knock_out, knock_in = make_pair("put", "down", 100.0, 88.0, expiry=0.5)
        package = CompositeInstrument("in + out", (Position(knock_out, 1.0), Position(knock_in, 1.0)))
        value, stderr = mc_price(package, self.MKT, n_paths=20_000, n_steps=50, seed=8)
        assert abs(value - EuropeanOption("put", 100.0, 0.5).price(self.MKT)) < 3.5 * stderr


# ====================================================================== #
# BGK helper, validation, serialisation
# ====================================================================== #
class TestBgkAdjustment:
    def test_constant(self):
        riemann_zeta_half = -1.4603545088095868
        assert BGK_BETA == pytest.approx(-riemann_zeta_half / math.sqrt(2.0 * math.pi), rel=1e-12)
        assert BGK_BETA == pytest.approx(0.5826, abs=5e-5)

    def test_direction_and_inverse(self):
        shift = math.exp(BGK_BETA * 0.2 * math.sqrt(1 / 252))
        assert bgk_adjusted_barrier(120.0, 0.2, 1 / 252, "up") == pytest.approx(120.0 * shift)
        assert bgk_adjusted_barrier(85.0, 0.2, 1 / 252, "down") == pytest.approx(85.0 / shift)
        assert bgk_adjusted_barrier(120.0, 0.2, 1 / 252, "up", inverse=True) == pytest.approx(120.0 / shift)
        assert bgk_adjusted_barrier(85.0, 0.2, 1 / 252, "down", inverse=True) == pytest.approx(85.0 * shift)
        assert bgk_adjusted_barrier(120.0, 0.2, 0.0, "up") == 120.0
        assert isinstance(bgk_adjusted_barrier(120.0, 0.2, 0.01, "up"), float)

    def test_vectorised(self):
        out = bgk_adjusted_barrier(120.0, np.array([0.1, 0.2, 0.4]), 1 / 52, "up")
        assert out.shape == (3,) and np.all(np.diff(out) > 0) and np.all(out > 120.0)

    def test_validation(self):
        with pytest.raises(ValueError, match="direction"):
            bgk_adjusted_barrier(120.0, 0.2, 0.01, "sideways")
        with pytest.raises(ValueError, match="barrier"):
            bgk_adjusted_barrier(0.0, 0.2, 0.01, "up")
        with pytest.raises(ValueError, match="non-negative"):
            bgk_adjusted_barrier(120.0, 0.2, -0.01, "up")

    def test_method_moves_the_barrier_away_and_keeps_everything_else(self):
        up = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-out", rebate=1.5)
        adjusted = up.discrete_barrier_adjusted(0.3, dt=1 / 252)
        assert adjusted.barrier == pytest.approx(bgk_adjusted_barrier(120.0, 0.3, 1 / 252, "up")) and adjusted.barrier > 120.0
        assert adjusted == BarrierOption("call", 100.0, adjusted.barrier, 1.0, "up-and-out", rebate=1.5)
        down = BarrierOption("put", 100.0, 85.0, 1.0, "down-and-in")
        assert down.discrete_barrier_adjusted(0.3, n_monitoring_per_year=252).barrier == pytest.approx(
            bgk_adjusted_barrier(85.0, 0.3, 1 / 252, "down"))
        assert down.discrete_barrier_adjusted(0.3, n_monitoring_per_year=252).barrier < 85.0
        # fewer knock-outs -> a discretely monitored knock-out is worth more
        mkt = Market(100.0, 0.3, 0.03)
        assert adjusted.price(mkt) > up.price(mkt)

    def test_method_validation(self):
        option = BarrierOption("call", 100.0, 120.0, 1.0, "up-and-out")
        with pytest.raises(ValueError, match="exactly one"):
            option.discrete_barrier_adjusted(0.3)
        with pytest.raises(ValueError, match="exactly one"):
            option.discrete_barrier_adjusted(0.3, dt=0.01, n_monitoring_per_year=100)
        with pytest.raises(ValueError, match="strictly positive"):
            option.discrete_barrier_adjusted(0.3, n_monitoring_per_year=0)
        with pytest.raises(ValueError, match="scalars"):
            option.discrete_barrier_adjusted(np.array([0.2, 0.3]), dt=0.01)


class TestBarrierConstruction:
    @pytest.mark.parametrize("raw,expected", [
        ("Up-And-Out", "up-and-out"), ("down_and_in", "down-and-in"), (" up and in ", "up-and-in"),
        ("DO", "down-and-out"), ("ui", "up-and-in"),
    ])
    def test_barrier_type_normalisation(self, raw, expected):
        option = BarrierOption("CALL", 100, 120, 1, raw)
        assert option.barrier_type == expected and option.option_type == "call"
        assert expected in BARRIER_TYPES
        assert all(isinstance(getattr(option, name), float) for name in ("strike", "barrier", "expiry", "rebate"))

    def test_descriptors_and_label(self):
        option = BarrierOption("put", 100.0, 85.0, 0.5, "down-and-in")
        assert (option.is_call, option.is_up, option.is_knock_in, option.direction) == (False, False, True, "down")
        assert option.label == "DI P 100 H=85 T=0.50"
        assert option.observe(80.0, 0.1).label == "DI P 100 H=85 T=0.50 [knocked]"
        assert BarrierOption("call", 100.0, 120.0, 1.0, "uo").label == "UO C 100 H=120 T=1.00"
        assert option.vanilla_equivalent() == EuropeanOption("put", 100.0, 0.5)

    def test_validation(self):
        with pytest.raises(ValueError, match="barrier_type"):
            BarrierOption("call", 100.0, 120.0, 1.0, "up-and-away")
        with pytest.raises(ValueError, match="barrier_type"):
            BarrierOption("call", 100.0, 120.0, 1.0, None)
        with pytest.raises(ValueError, match="option_type"):
            BarrierOption("both", 100.0, 120.0, 1.0, "uo")
        with pytest.raises(ValueError, match="strike"):
            BarrierOption("call", -1.0, 120.0, 1.0, "uo")
        with pytest.raises(ValueError, match="barrier"):
            BarrierOption("call", 100.0, 0.0, 1.0, "uo")
        with pytest.raises(ValueError, match="expiry"):
            BarrierOption("call", 100.0, 120.0, math.nan, "uo")
        with pytest.raises(ValueError, match="rebate"):
            BarrierOption("call", 100.0, 120.0, 1.0, "uo", rebate=-1.0)
        with pytest.raises(ValueError, match="knocked"):
            BarrierOption("call", 100.0, 120.0, 1.0, "uo", knocked="yes")
        with pytest.raises(ValueError, match="numbers"):
            BarrierOption("call", "high", 120.0, 1.0, "uo")

    @pytest.mark.parametrize("knocked", [False, True])
    def test_serialisation_round_trip(self, knocked):
        option = BarrierOption("put", 95.0, 80.0, 0.75, "down-and-out", rebate=1.25, knocked=knocked)
        assert INSTRUMENT_REGISTRY["BarrierOption"] is BarrierOption
        data = json.loads(json.dumps(option.to_dict()))
        assert data == {"type": "BarrierOption", "option_type": "put", "strike": 95.0, "barrier": 80.0,
                        "expiry": 0.75, "barrier_type": "down-and-out", "rebate": 1.25, "knocked": knocked}
        rebuilt = instrument_from_dict(data)
        assert rebuilt == option and rebuilt.knocked is knocked and hash(rebuilt) == hash(option)
        # through a Position / composite as well
        package = CompositeInstrument("pkg", (Position(option, -3.0),))
        assert instrument_from_dict(json.loads(json.dumps(package.to_dict()))) == package

    def test_hashable_and_state_distinguishes_instruments(self):
        option = BarrierOption("call", 100.0, 120.0, 1.0, "uo")
        book = {option: 1.0, option.observe(125.0, 0.1): 2.0}
        assert len(book) == 2
        assert book[BarrierOption("call", 100.0, 120.0, 1.0, "up-and-out", knocked=True)] == 2.0

    def test_payoff_of_a_live_option(self):
        spots = np.array([80.0, 110.0, 120.0, 140.0])
        knock_out, knock_in = make_pair("call", "up", 100.0, 120.0, rebate=1.0)
        np.testing.assert_allclose(knock_out.payoff(spots), [0.0, 10.0, 1.0, 1.0])
        np.testing.assert_allclose(knock_in.payoff(spots), [1.0, 1.0, 20.0, 40.0])
