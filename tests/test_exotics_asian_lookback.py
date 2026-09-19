"""Tests for optionlab.exotics.asian and optionlab.exotics.lookback."""

import json
import math

import numpy as np
import pytest

from optionlab import black_scholes as bs
from optionlab.exotics.asian import AsianOption, mc_price_with_control_variate
from optionlab.exotics.lookback import LookbackOption, mc_price_brownian_bridge
from optionlab.instruments import (
    GREEK_KEYS,
    CompositeInstrument,
    EuropeanOption,
    Position,
    instrument_from_dict,
)
from optionlab.market import Market
from optionlab.monte_carlo import mc_price, simulate_gbm_paths

pytestmark = pytest.mark.filterwarnings("error")


# ====================================================================== #
# ASIAN OPTIONS
# ====================================================================== #
def _black(forward, strike, var, omega):
    sd = math.sqrt(var)
    d1 = math.log(forward / strike) / sd + 0.5 * sd
    ncdf = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))  # noqa: E731
    return omega * (forward * ncdf(omega * d1) - strike * ncdf(omega * (d1 - sd)))


def _textbook_turnbull_wakeman(spot, strike, tau1, tau, vol, rate, carry, omega):
    """Turnbull-Wakeman price from the closed-form moments (regular parameters only)."""
    ell = tau - tau1
    b, v2 = carry, vol**2
    m1 = spot * (math.exp(b * tau) - math.exp(b * tau1)) / (b * ell)
    m2 = (2.0 * spot**2 / ell**2) * (
        (math.exp((2 * b + v2) * tau) - math.exp((2 * b + v2) * tau1)) / ((2 * b + v2) * (b + v2))
        - math.exp((b + v2) * tau1) * (math.exp(b * tau) - math.exp(b * tau1)) / (b * (b + v2))
    )
    return math.exp(-rate * tau) * _black(m1, strike, math.log(m2 / m1**2), omega)


class TestAsianReferences:
    def test_haug_geometric_put(self):
        # Haug, "The Complete Guide to Option Pricing Formulas": b = 0.08 -> div = r - b = -0.03.
        put = AsianOption("put", 85.0, 0.25, averaging="geometric")
        assert put.price(Market(80.0, 0.20, rate=0.05, div=-0.03)) == pytest.approx(4.6922, abs=1e-4)

    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_kemna_vorst_is_a_vanilla_with_adjusted_inputs(self, option_type):
        # Geometric Asian = BS with vol / sqrt(3) and carry (b - vol^2 / 6) / 2.
        spot, strike, T, vol, rate, div = 100.0, 95.0, 0.75, 0.3, 0.06, 0.02
        carry_adj = 0.5 * (rate - div - vol**2 / 6.0)
        expected = bs.price(spot, strike, T, vol / math.sqrt(3.0), rate, rate - carry_adj, option_type)
        asian = AsianOption(option_type, strike, T, averaging="geometric")
        assert asian.price(Market(spot, vol, rate, div)) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize(
        "spot,strike,avg_start,T,vol,rate,div",
        [
            (100.0, 100.0, 0.0, 1.0, 0.2, 0.05, 0.0),
            (90.0, 100.0, 0.0, 0.5, 0.4, 0.08, 0.03),
            (110.0, 100.0, 0.25, 1.0, 0.3, 0.02, 0.06),
            (100.0, 105.0, 0.5, 2.0, 0.25, 0.10, 0.0),
        ],
    )
    @pytest.mark.parametrize("option_type", ["call", "put"])
    def test_arithmetic_matches_textbook_turnbull_wakeman(
        self, spot, strike, avg_start, T, vol, rate, div, option_type
    ):
        omega = 1.0 if option_type == "call" else -1.0
        expected = _textbook_turnbull_wakeman(spot, strike, avg_start, T, vol, rate, rate - div, omega)
        asian = AsianOption(option_type, strike, T, avg_start=avg_start)
        assert asian.price(Market(spot, vol, rate, div)) == pytest.approx(expected, rel=1e-10)

    # Exact continuous arithmetic Asian calls (Linetsky 2004 / Fu-Madan-Wang test cases), K = 2.
    LINETSKY = [
        (0.02, 0.10, 1.0, 2.0, 0.0559860415),
        (0.18, 0.30, 1.0, 2.0, 0.2183875466),
        (0.0125, 0.25, 2.0, 2.0, 0.1722687410),
        (0.05, 0.50, 1.0, 1.9, 0.1931737903),
        (0.05, 0.50, 1.0, 2.0, 0.2464156905),
        (0.05, 0.50, 1.0, 2.1, 0.3062203648),
        (0.05, 0.50, 2.0, 2.0, 0.3500952190),
    ]

    @pytest.mark.parametrize("rate,vol,T,spot,exact", LINETSKY)
    def test_turnbull_wakeman_accuracy_against_exact_values(self, rate, vol, T, spot, exact):
        price = AsianOption("call", 2.0, T).price(Market(spot, vol, rate))
        tolerance = 0.01 if vol**2 * T <= 0.16 else 0.03  # the lognormal fit degrades with vol^2 T
        assert price == pytest.approx(exact, rel=tolerance)
        assert price >= exact  # the known bias of the approximation: it overprices

    @pytest.mark.parametrize("rate,vol,T,spot,exact", LINETSKY[:3] + LINETSKY[4:5])
    def test_control_variate_mc_recovers_exact_values(self, rate, vol, T, spot, exact):
        mc, stderr = mc_price_with_control_variate(
            AsianOption("call", 2.0, T), Market(spot, vol, rate), n_paths=20_000, n_steps=200, seed=11
        )
        assert abs(mc - exact) < 4.0 * stderr + 1e-5
        assert stderr < 0.002 * exact  # the control variate makes 20k paths very sharp


class TestAsianMonteCarlo:
    @pytest.mark.parametrize("vol", [0.1, 0.2, 0.4])
    @pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
    def test_turnbull_wakeman_against_mc(self, vol, strike):
        # About 1% of the price at the money. Put-call parity is exact, so a call and its put
        # share the same ABSOLUTE error; on the cheap out-of-the-money side that is a larger
        # percentage (the true average is less skewed than a lognormal), hence the bound
        # expressed in spot terms for the wings: below 0.2% of the spot everywhere.
        mkt = Market(100.0, vol, rate=0.05, div=0.01)
        for option_type in ("call", "put"):
            asian = AsianOption(option_type, strike, 1.0)
            mc, stderr = mc_price_with_control_variate(asian, mkt, n_paths=10_000, n_steps=100, seed=5)
            error = abs(asian.price(mkt) - mc)
            assert error < 0.002 * 100.0 + 4.0 * stderr
            if strike == 100.0:
                assert error < 0.0125 * mc + 4.0 * stderr

    def test_forward_starting_window(self):
        mkt = Market(100.0, 0.3, rate=0.04, div=0.01)
        asian = AsianOption("call", 100.0, 1.0, avg_start=0.5)
        mc, stderr = mc_price_with_control_variate(asian, mkt, n_paths=20_000, n_steps=100, seed=2)
        assert abs(asian.price(mkt) - mc) < 0.01 * mc + 4.0 * stderr
        # avg_start off the uniform grid: it is inserted, the result barely moves.
        odd = AsianOption("call", 100.0, 1.0, avg_start=0.5037)
        mc_odd, stderr_odd = mc_price_with_control_variate(odd, mkt, n_paths=20_000, n_steps=100, seed=2)
        assert abs(odd.price(mkt) - mc_odd) < 0.01 * mc_odd + 4.0 * stderr_odd

    @pytest.mark.parametrize("averaging", ["arithmetic", "geometric"])
    def test_seasoned_price_matches_mc_of_the_remaining_path(self, averaging):
        mkt = Market(102.0, 0.3, rate=0.05, div=0.02, t=0.4)
        asian = AsianOption(
            "call", 100.0, 1.0, averaging=averaging,
            running_average=95.0, observed_until=0.4, last_spot=102.0,
        )
        mc, stderr = mc_price_with_control_variate(asian, mkt, n_paths=20_000, n_steps=150, seed=8)
        assert abs(asian.price(mkt) - mc) < 0.005 * mc + 4.0 * stderr
        plain, plain_err = mc_price(asian, mkt, n_paths=40_000, n_steps=150, seed=9)
        assert abs(asian.price(mkt) - plain) < 0.005 * plain + 4.0 * plain_err

    def test_geometric_control_variate_is_exact(self):
        mkt = Market(100.0, 0.25, rate=0.03)
        asian = AsianOption("put", 100.0, 1.0, averaging="geometric")
        mc, stderr = mc_price_with_control_variate(asian, mkt, n_paths=2_000, n_steps=252, seed=1)
        assert stderr < 1e-10
        assert mc == pytest.approx(asian.price(mkt), rel=1e-4)  # trapezoid grid vs continuous: O(dt^2)

    def test_generic_mc_pricer_and_composites(self):
        mkt = Market(100.0, 0.2, rate=0.03)
        call = AsianOption("call", 100.0, 0.5, averaging="geometric")
        package = CompositeInstrument("asian vs vanilla", (call * 2.0, -EuropeanOption("call", 100.0, 0.5)))
        mc, stderr = mc_price(package, mkt, n_paths=40_000, n_steps=100, seed=4)
        assert abs(mc - package.price(mkt)) < 4.0 * stderr + 1e-3

    def test_mc_validation_and_expired(self):
        asian = AsianOption("call", 100.0, 1.0)
        with pytest.raises(ValueError):
            mc_price_with_control_variate(EuropeanOption("call", 100.0, 1.0), Market(100.0, 0.2))
        with pytest.raises(ValueError):
            mc_price_with_control_variate(asian, Market(np.array([100.0, 101.0]), 0.2))
        with pytest.raises(ValueError):
            mc_price_with_control_variate(asian, Market(100.0, 0.2), n_paths=2)
        with pytest.raises(ValueError):
            mc_price_with_control_variate(asian, Market(100.0, 0.2), n_steps=0)
        done = asian.observe(100.0, 0.0).observe(120.0, 1.0)
        assert mc_price_with_control_variate(done, Market(120.0, 0.2, t=1.0)) == (10.0, 0.0)


class TestAsianBoundsAndLimits:
    @pytest.mark.parametrize("vol", [0.1, 0.3, 0.6])
    @pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
    def test_orderings(self, vol, strike):
        mkt = Market(100.0, vol, rate=0.05, div=0.02)
        arithmetic = AsianOption("call", strike, 1.0).price(mkt)
        geometric = AsianOption("call", strike, 1.0, averaging="geometric").price(mkt)
        vanilla = EuropeanOption("call", strike, 1.0).price(mkt)
        assert geometric <= arithmetic <= vanilla
        # Puts: a geometric mean is lower, so the geometric put is the more valuable one. The
        # arithmetic price is an approximation, so the ordering holds up to its far-wing error.
        assert (
            AsianOption("put", strike, 1.0, averaging="geometric").price(mkt)
            >= AsianOption("put", strike, 1.0).price(mkt) - 1e-4
        )

    def test_volatility_of_the_average_is_vol_over_sqrt_three(self):
        mkt = Market(100.0, 0.3, rate=0.0)
        assert AsianOption("call", 100.0, 1.0, averaging="geometric").effective_vol(mkt) == pytest.approx(
            0.3 / math.sqrt(3.0), rel=1e-12
        )
        assert AsianOption("call", 100.0, 1.0).effective_vol(mkt) == pytest.approx(0.3 / math.sqrt(3.0), rel=0.02)
        assert AsianOption("call", 100.0, 1.0).effective_vol(mkt.bumped(t=1.0)) == 0.0

    @pytest.mark.parametrize("averaging", ["arithmetic", "geometric"])
    def test_put_call_parity_on_the_expected_average(self, averaging):
        mkt = Market(100.0, 0.3, rate=0.06, div=0.01)
        call = AsianOption("call", 97.0, 1.0, averaging=averaging).price(mkt)
        put = AsianOption("put", 97.0, 1.0, averaging=averaging).price(mkt)
        if averaging == "arithmetic":
            expected_average = 100.0 * math.expm1(0.05) / 0.05
        else:
            expected_average = 100.0 * math.exp(0.5 * (0.05 - 0.3**2 / 6.0))
        assert call - put == pytest.approx(math.exp(-0.06) * (expected_average - 97.0), rel=1e-12)

    @pytest.mark.parametrize("averaging", ["arithmetic", "geometric"])
    def test_tiny_window_is_a_vanilla(self, averaging):
        mkt = Market(100.0, 0.25, rate=0.04, div=0.01)
        asian = AsianOption("put", 105.0, 1.0, averaging=averaging, avg_start=1.0 - 1e-9)
        assert asian.price(mkt) == pytest.approx(EuropeanOption("put", 105.0, 1.0).price(mkt), abs=1e-6)

    @pytest.mark.parametrize("averaging", ["arithmetic", "geometric"])
    @pytest.mark.parametrize("option_type", ["call", "put"])
    @pytest.mark.parametrize("carry_center", [0.0, -0.09, -0.045])  # b = 0, b = -vol^2, 2b = -vol^2
    def test_continuity_at_the_singular_carries(self, averaging, option_type, carry_center):
        asian = AsianOption(option_type, 100.0, 1.0, averaging=averaging, avg_start=0.2)
        price = lambda b: asian.price(Market(100.0, 0.3, rate=0.03, div=0.03 - b))  # noqa: E731
        center = price(carry_center)
        assert math.isfinite(center) and center > 0
        assert price(carry_center + 1e-10) == pytest.approx(center, abs=1e-7)
        assert price(carry_center - 1e-10) == pytest.approx(center, abs=1e-7)
        h = 1e-3  # a smooth function: the symmetric average is second-order accurate
        assert 0.5 * (price(carry_center + h) + price(carry_center - h)) == pytest.approx(center, abs=1e-4)

    def test_zero_vol_and_zero_spot_are_deterministic(self):
        asian = AsianOption("call", 100.0, 1.0)
        forward_average = 105.0 * math.expm1(0.05) / 0.05
        assert asian.price(Market(105.0, 0.0, rate=0.05)) == pytest.approx(
            math.exp(-0.05) * (forward_average - 100.0), rel=1e-12
        )
        for averaging in ("arithmetic", "geometric"):
            put = AsianOption("put", 100.0, 1.0, averaging=averaging)
            assert put.price(Market(0.0, 0.2, rate=0.05)) == pytest.approx(100.0 * math.exp(-0.05))
            assert AsianOption("call", 100.0, 1.0, averaging=averaging).price(Market(0.0, 0.2)) == 0.0


    def test_absurd_volatility_reaches_the_no_arbitrage_limits(self):
        # vol^2 T overflows a double: call -> discounted expected average, put -> discounted strike
        mkt = Market(100.0, np.array([20.0, 40.0]), rate=0.05)
        call = AsianOption("call", 100.0, 1.0).price(mkt)
        put = AsianOption("put", 100.0, 1.0).price(mkt)
        assert np.allclose(call, math.exp(-0.05) * 100.0 * math.expm1(0.05) / 0.05)
        assert np.allclose(put, 100.0 * math.exp(-0.05))
        lookback = LookbackOption("put", 1.0, strike=100.0, kind="fixed").price(mkt)
        assert np.allclose(lookback, 100.0 * math.exp(-0.05))


class TestAsianSeasoning:
    def test_certain_exercise_branch(self):
        # 90% of the window fixed at 150: even a spot of 0 for the rest gives A = 135 > K.
        state = dict(running_average=150.0, observed_until=0.9, last_spot=100.0)
        mkt = Market(100.0, 0.4, rate=0.05, div=0.01, t=0.9)
        call = AsianOption("call", 100.0, 1.0, **state)
        put = AsianOption("put", 100.0, 1.0, **state)
        expected_average = 0.9 * 150.0 + 0.1 * 100.0 * math.expm1(0.04 * 0.1) / (0.04 * 0.1)
        assert call.price(mkt) == pytest.approx(math.exp(-0.05 * 0.1) * (expected_average - 100.0), rel=1e-12)
        assert put.price(mkt) == 0.0
        assert call.greeks(mkt)["vega"] == pytest.approx(0.0, abs=1e-9)
        assert call.greeks(mkt)["delta"] == pytest.approx(0.1 * math.exp(-0.05 * 0.1) * 1.002, rel=1e-3)

    def test_price_is_continuous_where_the_adjusted_strike_changes_sign(self):
        mkt = Market(100.0, 0.3, rate=0.03, t=0.9)
        pivot = 100.0 / 0.9  # running average at which K* = 0

        def price(average, option_type="call"):
            option = AsianOption(option_type, 100.0, 1.0, running_average=average, observed_until=0.9)
            return option.price(mkt)

        assert price(pivot - 1e-7) == pytest.approx(price(pivot + 1e-7), abs=1e-6)
        assert price(pivot - 1e-7, "put") == pytest.approx(0.0, abs=1e-9)
        assert price(pivot - 0.5) < price(pivot) < price(pivot + 0.5)

    def test_greeks_die_as_the_average_fills_in(self):
        fresh = AsianOption("call", 100.0, 1.0)
        mkt = Market(100.0, 0.3, rate=0.02)
        previous = fresh.greeks(mkt)
        assert fresh.remaining_weight(mkt) == 1.0
        for t in (0.5, 0.9, 0.99):
            seasoned = AsianOption("call", 100.0, 1.0, running_average=100.0, observed_until=t, last_spot=100.0)
            now = seasoned.greeks(mkt.bumped(t=t))
            assert seasoned.remaining_weight(mkt.bumped(t=t)) == pytest.approx(1.0 - t)
            assert 0.0 < now["delta"] < previous["delta"]
            assert 0.0 < now["vega"] < previous["vega"]
            previous = now
        assert previous["delta"] < 0.01 and previous["vega"] < 0.05

    def test_unobserved_stretch_is_held_at_the_current_spot(self):
        mkt = Market(107.0, 0.25, rate=0.03, div=0.01, t=0.45)
        stale = AsianOption("call", 100.0, 1.0, running_average=104.0, observed_until=0.3, last_spot=98.0)
        caught_up = AsianOption(
            "call", 100.0, 1.0, running_average=(104.0 * 0.3 + 107.0 * 0.15) / 0.45, observed_until=0.45
        )
        assert stale.price(mkt) == pytest.approx(caught_up.price(mkt), rel=1e-13)
        geometric = AsianOption("put", 100.0, 1.0, averaging="geometric", running_average=104.0, observed_until=0.3)
        folded = math.exp((math.log(104.0) * 0.3 + math.log(107.0) * 0.15) / 0.45)
        caught_up = AsianOption("put", 100.0, 1.0, averaging="geometric", running_average=folded, observed_until=0.45)
        assert geometric.price(mkt) == pytest.approx(caught_up.price(mkt), rel=1e-13)
        # last_spot only serves observe(): it never enters a price, a payoff or a Greek
        for t in (0.3, 0.45, 1.0):
            m = mkt.bumped(t=t)
            assert stale.price(m) == AsianOption(
                "call", 100.0, 1.0, running_average=104.0, observed_until=0.3, last_spot=140.0
            ).price(m)

    @pytest.mark.parametrize("spot", [80.0, 103.0, 125.0])
    def test_geometric_price_satisfies_the_pricing_pde_on_a_spot_ladder(self, spot):
        # theta (time passing, the average filling in at the ladder spot) + carry + gamma = r V
        asian = AsianOption(
            "call", 100.0, 1.0, averaging="geometric", running_average=97.0, observed_until=0.3, last_spot=103.0
        )
        g = asian.greeks(Market(spot, 0.3, rate=0.05, div=0.02, t=0.3))
        residual = g["theta"] + 0.5 * 0.3**2 * spot**2 * g["gamma"] + 0.03 * spot * g["delta"] - 0.05 * g["price"]
        assert abs(residual) < 1e-4 * g["price"]

    def test_arithmetic_approximation_nearly_satisfies_the_pricing_pde(self):
        asian = AsianOption("call", 100.0, 1.0, running_average=97.0, observed_until=0.3, last_spot=103.0)
        g = asian.greeks(Market(110.0, 0.3, rate=0.05, div=0.02, t=0.3))
        residual = g["theta"] + 0.5 * 0.3**2 * 110.0**2 * g["gamma"] + 0.03 * 110.0 * g["delta"] - 0.05 * g["price"]
        assert abs(residual) < 0.05 * abs(g["theta"])

    def test_settlement_at_and_after_expiry(self):
        mkt = Market(130.0, 0.2, rate=0.05)
        done = AsianOption("call", 100.0, 1.0, running_average=108.0, observed_until=1.0, last_spot=130.0)
        for t in (1.0, 1.5):
            assert done.price(mkt.bumped(t=t)) == pytest.approx(8.0)
        assert np.allclose(done.payoff([50.0, 100.0, 200.0]), 8.0)
        # half observed, then the clock jumps to expiry: the rest counts at the settlement spot
        half = AsianOption("call", 100.0, 1.0, running_average=100.0, observed_until=0.5, last_spot=100.0)
        assert half.price(mkt.bumped(t=1.0)) == pytest.approx(0.5 * 100.0 + 0.5 * 130.0 - 100.0)
        assert half.payoff(130.0) == pytest.approx(15.0)
        fresh = AsianOption("put", 100.0, 1.0)
        spots = np.array([60.0, 100.0, 140.0])
        assert np.allclose(fresh.payoff(spots), [40.0, 0.0, 0.0])
        assert np.allclose(fresh.price(Market(spots, 0.2, t=1.0)), fresh.payoff(spots))


class TestAsianObserve:
    def test_trapezoid_running_average(self):
        asian = AsianOption("call", 100.0, 1.0)
        first = asian.observe(100.0, 0.0)
        assert (first.running_average, first.observed_until, first.last_spot) == (None, 0.0, 100.0)
        mid = first.observe(110.0, 0.5)
        assert mid.running_average == pytest.approx(105.0) and mid.observed_until == 0.5
        end = mid.observe(90.0, 1.0)
        assert end.running_average == pytest.approx(102.5) and end.observed_until == 1.0
        assert end.price(Market(90.0, 0.2, t=1.0)) == pytest.approx(2.5)
        assert asian.running_average is None  # the original is untouched

    def test_geometric_running_average_uses_logs(self):
        asian = AsianOption("put", 100.0, 1.0, averaging="geometric").observe(100.0, 0.0).observe(121.0, 1.0)
        assert asian.running_average == pytest.approx(110.0)

    def test_observations_outside_the_window(self):
        asian = AsianOption("call", 100.0, 1.0, avg_start=0.5)
        before = asian.observe(80.0, 0.1).observe(90.0, 0.5)
        assert before.running_average is None and before.last_spot == 90.0
        inside = before.observe(110.0, 0.75)
        assert inside.running_average == pytest.approx(100.0)
        assert inside.observe(500.0, 0.6) is inside  # time never moves backwards
        assert inside.observe(500.0, 0.75) is inside
        end = inside.observe(110.0, 3.0)  # late observation: counted at expiry
        assert end.observed_until == 1.0 and end.running_average == pytest.approx(105.0)
        assert end.observe(1.0, 4.0) is end
        with pytest.raises(ValueError):
            asian.observe(0.0, 0.2)

    def test_step_straddling_the_window_start_is_interpolated(self):
        early = AsianOption("call", 100.0, 1.0, avg_start=0.5).observe(100.0, 0.4)
        inside = early.observe(120.0, 0.6)  # linear path: 110 at the window start, 115 on average
        assert inside.running_average == pytest.approx(115.0) and inside.observed_until == 0.6
        geometric = AsianOption("put", 100.0, 1.0, averaging="geometric", avg_start=0.5)
        geometric = geometric.observe(100.0, 0.4).observe(144.0, 0.6)  # log-linear: 120 at the start
        assert geometric.running_average == pytest.approx(math.sqrt(120.0 * 144.0))

    def test_first_observation_inside_the_window_is_held_flat(self):
        asian = AsianOption("call", 100.0, 1.0).observe(104.0, 0.25)
        assert asian.running_average == pytest.approx(104.0) and asian.observed_until == 0.25

    @pytest.mark.parametrize("averaging", ["arithmetic", "geometric"])
    def test_observe_and_path_payoff_agree(self, averaging):
        times, paths = simulate_gbm_paths(100.0, 0.4, 0.03, 0.0, horizon=1.0, n_steps=40, n_paths=6, seed=3)
        fresh = AsianOption("call", 95.0, 1.0, averaging=averaging, avg_start=0.13)
        full = fresh.path_payoff(paths, times)
        for i, path in enumerate(paths):
            walked, halfway = fresh, None
            for k, (spot, t) in enumerate(zip(path, times)):
                walked = walked.observe(spot, t)
                if k == 17:
                    halfway = walked
            assert walked.price(Market(path[-1], 0.4, t=1.0)) == pytest.approx(full[i], abs=1e-9)
            # state so far + the rest of the path = the whole path
            rest = halfway.path_payoff(paths[i : i + 1, 17:], times[17:])
            assert rest[0] == pytest.approx(full[i], abs=1e-9)

    def test_path_payoff_honours_state_and_short_paths(self):
        asian = AsianOption("call", 100.0, 1.0, running_average=120.0, observed_until=0.5, last_spot=100.0)
        times = np.array([0.5, 0.75, 1.0, 1.25])  # the last column is beyond expiry and ignored
        paths = np.array([[100.0, 100.0, 100.0, 900.0], [100.0, 80.0, 60.0, 900.0]])
        assert np.allclose(asian.path_payoff(paths, times), [10.0, 0.0])
        # a path that stops early is held flat until expiry
        assert np.allclose(asian.path_payoff(paths[:, :2], times[:2]), [10.0, max(60.0 + 42.5 - 100.0, 0.0)])


class TestAsianInterface:
    def test_vectorisation_matches_scalar_calls(self):
        spots = np.array([70.0, 100.0, 130.0])[:, None]
        vols = np.array([0.1, 0.25, 0.5, 0.8])[None, :]
        for asian in (
            AsianOption("call", 100.0, 1.0),
            AsianOption("put", 100.0, 1.0, averaging="geometric", avg_start=0.3),
            AsianOption("call", 100.0, 1.0, running_average=90.0, observed_until=0.4, last_spot=95.0),
        ):
            grid = asian.price(Market(spots, vols, rate=0.03, div=0.01, t=0.4))
            assert grid.shape == (3, 4)
            for i in range(3):
                for j in range(4):
                    scalar = asian.price(Market(spots[i, 0], vols[0, j], rate=0.03, div=0.01, t=0.4))
                    assert isinstance(scalar, float)
                    assert grid[i, j] == pytest.approx(scalar, rel=1e-12, abs=1e-14)

    def test_vectorisation_over_time_covers_every_regime(self):
        asian = AsianOption("call", 100.0, 1.0, avg_start=0.25)
        ts = np.array([-0.5, 0.0, 0.25, 0.6, 1.0, 1.4])
        prices = asian.price(Market(110.0, 0.3, rate=0.02, t=ts))
        assert prices.shape == (6,) and np.all(np.isfinite(prices))
        for t, value in zip(ts, prices):
            assert value == pytest.approx(asian.price(Market(110.0, 0.3, rate=0.02, t=float(t))), rel=1e-12)
        assert prices[-1] == pytest.approx(10.0) and prices[-2] == pytest.approx(10.0)
        assert asian.remaining_weight(Market(110.0, 0.3, t=ts)).tolist() == pytest.approx([1, 1, 1, 8 / 15, 0, 0])

    def test_greeks_are_finite_and_sensible(self):
        mkt = Market(100.0, 0.25, rate=0.04, div=0.01)
        for asian in (
            AsianOption("call", 100.0, 1.0),
            AsianOption("put", 100.0, 1.0, averaging="geometric"),
            AsianOption("call", 100.0, 1.0, avg_start=0.5),
        ):
            g = asian.greeks(mkt)
            assert tuple(g) == GREEK_KEYS
            assert all(isinstance(v, float) and math.isfinite(v) for v in g.values())
            assert g["gamma"] > 0 and g["vega"] > 0 and g["theta"] < 0
            assert (g["delta"] > 0) == asian.is_call
        vanilla = EuropeanOption("call", 100.0, 1.0).greeks(mkt)
        assert AsianOption("call", 100.0, 1.0).greeks(mkt)["vega"] < vanilla["vega"]
        ladder = AsianOption("call", 100.0, 1.0).greeks(Market(np.linspace(60.0, 140.0, 9), 0.25))
        assert all(v.shape == (9,) and np.all(np.isfinite(v)) for v in ladder.values())
        assert np.all(np.diff(ladder["delta"]) > 0)

    def test_serialisation_round_trip_with_state(self):
        asian = AsianOption("PUT", 100, 1, averaging="Geometric", avg_start=0.25)
        asian = asian.observe(101.0, 0.25).observe(97.0, 0.5)
        data = json.loads(json.dumps(asian.to_dict()))
        assert data["type"] == "AsianOption" and data["observed_until"] == 0.5
        clone = instrument_from_dict(data)
        assert clone == asian and hash(clone) == hash(asian)
        assert clone.price(Market(97.0, 0.2, t=0.5)) == asian.price(Market(97.0, 0.2, t=0.5))
        position = Position.from_dict(json.loads(json.dumps(Position(asian, -3.0).to_dict())))
        assert position == Position(asian, -3.0)
        fresh = AsianOption("call", 100.0, 1.0)
        assert instrument_from_dict(fresh.to_dict()) == fresh

    def test_normalisation_label_and_attributes(self):
        asian = AsianOption(" Call ", 100, 2, averaging="ARITHMETIC")
        assert (asian.option_type, asian.averaging, asian.strike, asian.expiry) == ("call", "arithmetic", 100.0, 2.0)
        assert asian.label == "Asian arith C 100 T=2.00"
        assert AsianOption("put", 97.5, 0.5, averaging="geometric").label == "Asian geo P 97.5 T=0.50"
        assert asian.is_expired(Market(100.0, 0.2, t=2.0)) and not asian.is_expired(Market(100.0, 0.2, t=1.0))

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(option_type="straddle"),
            dict(strike=0.0),
            dict(strike=float("nan")),
            dict(averaging="harmonic"),
            dict(avg_start=1.0),
            dict(expiry=float("inf")),
            dict(running_average=100.0),  # no observed_until
            dict(running_average=-5.0, observed_until=0.5),
            dict(running_average=100.0, observed_until=1.5),
            dict(running_average=100.0, observed_until=-0.5),
            dict(observed_until=0.5),  # inside the window without an average
            dict(last_spot=100.0),  # no observed_until
            dict(strike="abc"),
        ],
    )
    def test_validation(self, kwargs):
        base = dict(option_type="call", strike=100.0, expiry=1.0)
        with pytest.raises(ValueError):
            AsianOption(**{**base, **kwargs})


# ====================================================================== #
# LOOKBACK OPTIONS
# ====================================================================== #
def _lookback(option_type, kind, strike=None, **state):
    return LookbackOption(option_type, 1.0, strike=strike if kind == "fixed" else None, kind=kind, **state)


ALL_LOOKBACKS = [("call", "floating"), ("put", "floating"), ("call", "fixed"), ("put", "fixed")]


class TestLookbackReferences:
    def test_haug_floating_strike_call(self):
        call = LookbackOption("call", 0.5, running_min=100.0)
        assert call.price(Market(120.0, 0.30, rate=0.10, div=0.06)) == pytest.approx(25.3533, abs=1e-4)

    # Haug's fixed-strike table: S = S_max = S_min = 100, r = b = 0.10, vol = 10% / 20% / 30%.
    HAUG_FIXED = {
        ("call", 95.0, 0.5): (13.2687, 18.9263, 24.9857),
        ("call", 100.0, 0.5): (8.5126, 14.1702, 20.2296),
        ("call", 105.0, 0.5): (4.3908, 9.8905, 15.8512),
        ("call", 95.0, 1.0): (18.3241, 26.0731, 34.7116),
        ("call", 100.0, 1.0): (13.8000, 21.5489, 30.1874),
        ("call", 105.0, 1.0): (9.5445, 17.2965, 25.9002),
        ("put", 95.0, 0.5): (0.6899, 4.4448, 8.9213),
        ("put", 100.0, 0.5): (3.3917, 8.3177, 13.1579),
        ("put", 105.0, 0.5): (8.1478, 13.0739, 17.9140),
        ("put", 95.0, 1.0): (1.0534, 6.2813, 12.2376),
        ("put", 100.0, 1.0): (3.8079, 10.1294, 16.3889),
        ("put", 105.0, 1.0): (8.3321, 14.6536, 20.9130),
    }

    @pytest.mark.parametrize("key", list(HAUG_FIXED))
    def test_haug_fixed_strike_table(self, key):
        option_type, strike, expiry = key
        option = LookbackOption(option_type, expiry, strike=strike, kind="fixed", running_min=100.0, running_max=100.0)
        prices = option.price(Market(100.0, np.array([0.1, 0.2, 0.3]), rate=0.10))
        assert np.allclose(prices, self.HAUG_FIXED[key], atol=1.5e-4)

    @pytest.mark.parametrize("option_type,kind", ALL_LOOKBACKS)
    @pytest.mark.parametrize("rate,div", [(0.06, 0.01), (0.03, 0.03), (0.0, 0.05)])
    def test_brownian_bridge_mc_confirms_the_closed_forms(self, option_type, kind, rate, div):
        mkt = Market(100.0, 0.25, rate=rate, div=div, t=0.2)
        for state in ({}, {"running_min": 88.0, "running_max": 113.0}):
            option = _lookback(option_type, kind, strike=104.0, **state)
            mc, stderr = mc_price_brownian_bridge(option, mkt, n_paths=100_000, n_steps=3, seed=21)
            assert abs(option.price(mkt) - mc) < 4.0 * stderr

    def test_bridge_mc_is_unbiased_for_any_step_count(self):
        mkt = Market(100.0, 0.3, rate=0.04)
        option = LookbackOption("put", 1.0)
        for n_steps in (1, 12):
            mc, stderr = mc_price_brownian_bridge(option, mkt, n_paths=100_000, n_steps=n_steps, seed=n_steps)
            assert abs(option.price(mkt) - mc) < 4.0 * stderr

    def test_discrete_monitoring_is_cheaper_by_about_the_bgk_shift(self):
        mkt = Market(100.0, 0.3, rate=0.05)
        option = LookbackOption("call", 1.0)
        n_steps = 50
        mc, stderr = mc_price(option, mkt, n_paths=60_000, n_steps=n_steps, seed=7)
        gap = option.price(mkt) - mc
        shift = 0.5826 * 0.3 * math.sqrt(1.0 / n_steps) * 100.0  # the part of the minimum the grid misses
        assert gap > 5.0 * stderr
        assert 0.6 * shift < gap < 1.4 * shift

    def test_bridge_mc_validation_and_expired(self):
        option = LookbackOption("call", 1.0, running_min=90.0)
        with pytest.raises(ValueError):
            mc_price_brownian_bridge(EuropeanOption("call", 100.0, 1.0), Market(100.0, 0.2))
        with pytest.raises(ValueError):
            mc_price_brownian_bridge(option, Market(np.array([100.0, 101.0]), 0.2))
        with pytest.raises(ValueError):
            mc_price_brownian_bridge(option, Market(100.0, 0.2), n_paths=1)
        with pytest.raises(ValueError):
            mc_price_brownian_bridge(option, Market(100.0, 0.2), n_steps=0)
        assert mc_price_brownian_bridge(option, Market(100.0, 0.2, t=1.0)) == (10.0, 0.0)


class TestLookbackBoundsAndLimits:
    @pytest.mark.parametrize("vol", [0.1, 0.3])
    @pytest.mark.parametrize("rate,div", [(0.05, 0.0), (0.02, 0.02), (0.0, 0.04)])
    def test_lookbacks_dominate_vanillas(self, vol, rate, div):
        mkt = Market(100.0, vol, rate=rate, div=div)
        for option_type in ("call", "put"):
            atm = EuropeanOption(option_type, 100.0, 1.0).price(mkt)
            assert LookbackOption(option_type, 1.0).price(mkt) >= atm
            for strike in (90.0, 100.0, 110.0):
                fixed = LookbackOption(option_type, 1.0, strike=strike, kind="fixed")
                assert fixed.price(mkt) >= EuropeanOption(option_type, strike, 1.0).price(mkt)
        # a stored extreme is the strike of the vanilla that is dominated
        seasoned = LookbackOption("call", 1.0, running_min=85.0)
        assert seasoned.price(mkt) >= EuropeanOption("call", 85.0, 1.0).price(mkt)

    def test_fresh_lookback_costs_about_twice_the_atm_vanilla(self):
        mkt = Market(100.0, 0.1, rate=0.0)
        vanilla = EuropeanOption("call", 100.0, 0.25).price(mkt)
        assert LookbackOption("call", 0.25).price(mkt) == pytest.approx(2.0 * vanilla, rel=0.03)
        assert LookbackOption("put", 0.25).price(mkt) == pytest.approx(2.0 * vanilla, rel=0.03)

    @pytest.mark.parametrize("rate,div", [(0.07, 0.02), (0.03, 0.03), (0.01, 0.06)])
    def test_fixed_and_floating_strikes_are_linked(self, rate, div):
        # fixed call (K <= S_max) - floating put = forward on the spot struck at K; mirror for puts
        mkt = Market(100.0, 0.3, rate=rate, div=div, t=0.25)
        state = dict(running_min=91.0, running_max=108.0)
        forward = 100.0 * math.exp(-div * 0.75) - 105.0 * math.exp(-rate * 0.75)
        fixed_call = LookbackOption("call", 1.0, strike=105.0, kind="fixed", **state).price(mkt)
        assert fixed_call - LookbackOption("put", 1.0, **state).price(mkt) == pytest.approx(forward, abs=1e-10)
        forward = 95.0 * math.exp(-rate * 0.75) - 100.0 * math.exp(-div * 0.75)
        fixed_put = LookbackOption("put", 1.0, strike=95.0, kind="fixed", **state).price(mkt)
        assert fixed_put - LookbackOption("call", 1.0, **state).price(mkt) == pytest.approx(forward, abs=1e-10)

    @pytest.mark.parametrize("option_type,kind", ALL_LOOKBACKS)
    @pytest.mark.parametrize("state", [{}, {"running_min": 85.0, "running_max": 120.0}])
    def test_continuity_at_zero_carry(self, option_type, kind, state):
        option = _lookback(option_type, kind, strike=102.0, **state)
        price = lambda b: option.price(Market(100.0, 0.25, rate=0.04, div=0.04 - b))  # noqa: E731
        center = price(0.0)
        assert math.isfinite(center) and center > 0
        for b in (1e-12, 1e-10, 1e-9, 1e-8):  # on both sides of the switch to the limit formula
            assert price(b) == pytest.approx(center, abs=1e-5)
            assert price(-b) == pytest.approx(center, abs=1e-5)
        h = 1e-3  # smooth in b: the symmetric average is second-order accurate
        assert 0.5 * (price(h) + price(-h)) == pytest.approx(center, abs=1e-4)
        slope = (price(h) - price(-h)) / (2 * h)
        assert price(1e-6) - center == pytest.approx(1e-6 * slope, abs=1e-7)

    def test_low_volatility_is_stable(self):
        # 2 b / vol^2 is huge: the power (S/H)^(-lambda) would overflow if evaluated naively
        vols = np.array([1e-4, 1e-3, 1e-2])
        put = LookbackOption("put", 1.0, running_max=130.0).price(Market(100.0, vols, rate=0.08))
        assert np.allclose(put, 130.0 * math.exp(-0.08) - 100.0, atol=1e-6)
        call = LookbackOption("call", 1.0, running_min=70.0).price(Market(100.0, vols, rate=0.0, div=0.08))
        assert np.allclose(call, 100.0 * math.exp(-0.08) - 70.0, atol=1e-6)

    def test_zero_vol_and_zero_spot_are_deterministic(self):
        up = Market(100.0, 0.0, rate=0.05)  # the path only rises: min = spot now, max = the forward
        assert LookbackOption("call", 1.0).price(up) == pytest.approx(100.0 - 100.0 * math.exp(-0.05))
        assert LookbackOption("put", 1.0).price(up) == pytest.approx(0.0, abs=1e-12)
        fixed = LookbackOption("call", 1.0, strike=102.0, kind="fixed")
        assert fixed.price(up) == pytest.approx(math.exp(-0.05) * (100.0 * math.exp(0.05) - 102.0))
        dead = Market(0.0, 0.2, rate=0.05)
        assert LookbackOption("put", 1.0, strike=90.0, kind="fixed").price(dead) == pytest.approx(90.0 * math.exp(-0.05))
        assert LookbackOption("call", 1.0).price(dead) == 0.0


class TestLookbackInterface:
    def test_missing_extremes_start_at_the_current_spot(self):
        mkt = Market(100.0, 0.2, rate=0.03, div=0.01)
        for option_type, kind in ALL_LOOKBACKS:
            fresh = _lookback(option_type, kind, strike=100.0)
            stored = _lookback(option_type, kind, strike=100.0, running_min=100.0, running_max=100.0)
            assert fresh.price(mkt) == stored.price(mkt)

    def test_spot_ladder_through_the_stored_extreme(self):
        call = LookbackOption("call", 1.0, running_min=95.0)
        spots = np.array([80.0, 90.0, 95.0, 100.0, 120.0])
        ladder = call.price(Market(spots, 0.2, rate=0.03))
        assert ladder.shape == (5,)
        for spot, value in zip(spots, ladder):
            new_min = min(95.0, spot)  # below the stored minimum, the ladder point is the new minimum
            expected = LookbackOption("call", 1.0, running_min=new_min).price(Market(spot, 0.2, rate=0.03))
            assert value == pytest.approx(expected, rel=1e-13)
        assert np.allclose(ladder[:2] / spots[:2], ladder[0] / spots[0])  # linear once the minimum is dragged

    def test_vectorisation_over_vol_and_time(self):
        spots = np.array([80.0, 100.0, 125.0])[:, None]
        vols = np.array([0.1, 0.3, 0.6])[None, :]
        for option_type, kind in ALL_LOOKBACKS:
            option = _lookback(option_type, kind, strike=105.0, running_min=90.0, running_max=110.0)
            grid = option.price(Market(spots, vols, rate=0.04, div=0.04, t=0.5))
            assert grid.shape == (3, 3)
            for i in range(3):
                for j in range(3):
                    scalar = option.price(Market(spots[i, 0], vols[0, j], rate=0.04, div=0.04, t=0.5))
                    assert isinstance(scalar, float)
                    assert grid[i, j] == pytest.approx(scalar, rel=1e-12)
            over_time = option.price(Market(100.0, 0.2, rate=0.02, t=np.array([0.0, 0.5, 1.0, 2.0])))
            assert over_time.shape == (4,) and np.all(np.isfinite(over_time))
            assert over_time[2] == over_time[3] == pytest.approx(float(option.payoff(100.0)))

    def test_payoff_and_settlement(self):
        spots = np.array([70.0, 100.0, 130.0])
        state = dict(running_min=90.0, running_max=110.0)
        assert np.allclose(LookbackOption("call", 1.0, **state).payoff(spots), [0.0, 10.0, 40.0])
        assert np.allclose(LookbackOption("put", 1.0, **state).payoff(spots), [40.0, 10.0, 0.0])
        fixed_call = LookbackOption("call", 1.0, strike=105.0, kind="fixed", **state)
        assert np.allclose(fixed_call.payoff(spots), [5.0, 5.0, 25.0])
        fixed_put = LookbackOption("put", 1.0, strike=105.0, kind="fixed", **state)
        assert np.allclose(fixed_put.payoff(spots), [35.0, 15.0, 15.0])
        assert np.allclose(LookbackOption("call", 1.0).payoff(spots), 0.0)
        assert np.allclose(fixed_call.price(Market(spots, 0.3, rate=0.05, t=1.0)), fixed_call.payoff(spots))

    def test_observe_updates_both_extremes(self):
        option = LookbackOption("call", 1.0)
        seen = option.observe(100.0, 0.0)
        assert (seen.running_min, seen.running_max) == (100.0, 100.0)
        assert seen.observe(100.0, 0.1) is seen and seen.observe(100.0, 0.1) == seen
        lower = seen.observe(92.0, 0.2).observe(97.0, 0.3).observe(108.0, 0.4)
        assert (lower.running_min, lower.running_max) == (92.0, 108.0)
        assert lower.observe(50.0, 1.5) is lower  # after expiry nothing counts any more
        assert option.running_min is None
        with pytest.raises(ValueError):
            option.observe(-1.0, 0.1)
        position = Position(option, 2.0).observe(100.0, 0.0)
        assert position.instrument == seen and position.quantity == 2.0

    def test_path_payoff_honours_stored_extremes(self):
        times = np.array([0.5, 0.75, 1.0, 1.25])  # last column is after expiry
        paths = np.array([[100.0, 95.0, 104.0, 1.0], [100.0, 120.0, 111.0, 999.0]])
        assert np.allclose(LookbackOption("call", 1.0).path_payoff(paths, times), [9.0, 11.0])
        assert np.allclose(LookbackOption("call", 1.0, running_min=90.0).path_payoff(paths, times), [14.0, 21.0])
        assert np.allclose(LookbackOption("put", 1.0, running_max=115.0).path_payoff(paths, times), [11.0, 9.0])
        fixed = LookbackOption("call", 1.0, strike=110.0, kind="fixed", running_max=112.0)
        assert np.allclose(fixed.path_payoff(paths, times), [2.0, 10.0])
        fixed_put = LookbackOption("put", 1.0, strike=98.0, kind="fixed")
        assert np.allclose(fixed_put.path_payoff(paths, times), [3.0, 0.0])

    def test_greeks_are_finite_and_use_frozen_extremes(self):
        mkt = Market(100.0, 0.25, rate=0.04, div=0.01)
        for option_type, kind in ALL_LOOKBACKS:
            for state in ({}, {"running_min": 90.0, "running_max": 112.0}):
                g = _lookback(option_type, kind, strike=103.0, **state).greeks(mkt)
                assert tuple(g) == GREEK_KEYS
                assert all(isinstance(v, float) and math.isfinite(v) for v in g.values())
                assert g["gamma"] > 0 and g["vega"] > 0
        fresh = LookbackOption("call", 1.0).greeks(mkt)
        stored = LookbackOption("call", 1.0, running_min=100.0, running_max=100.0).greeks(mkt)
        assert fresh == pytest.approx(stored)
        # Homogeneity: at S = S_min the delta is price / spot. Gamma jumps there (it is 0 on the
        # side where the minimum is dragged along), so the central difference is only O(bump) accurate.
        assert fresh["delta"] == pytest.approx(fresh["price"] / 100.0, rel=5e-3)
        away = LookbackOption("call", 1.0, running_min=80.0)
        bumped = (away.price(mkt.bumped(spot=100.01)) - away.price(mkt.bumped(spot=99.99))) / 0.02
        assert away.greeks(mkt)["delta"] == pytest.approx(bumped, rel=1e-5)
        ladder = LookbackOption("put", 1.0, running_max=110.0).greeks(Market(np.linspace(70.0, 130.0, 7), 0.25))
        assert all(v.shape == (7,) and np.all(np.isfinite(v)) for v in ladder.values())
        expired = LookbackOption("call", 1.0, running_min=90.0).greeks(mkt.bumped(t=1.0))
        assert expired["price"] == pytest.approx(10.0) and expired["vega"] == 0.0

    def test_serialisation_round_trip_with_state(self):
        floating = LookbackOption("PUT", 1).observe(100.0, 0.0).observe(117.5, 0.3)
        fixed = LookbackOption("call", 0.5, strike=105, kind="Fixed", running_max=111.0)
        for option in (floating, fixed, LookbackOption("call", 2.0)):
            data = json.loads(json.dumps(option.to_dict()))
            assert data["type"] == "LookbackOption"
            clone = instrument_from_dict(data)
            assert clone == option and hash(clone) == hash(option)
            assert clone.price(Market(100.0, 0.2)) == option.price(Market(100.0, 0.2))
        package = CompositeInstrument("lb", (Position(floating, 1.0), Position(fixed, -2.0)))
        assert instrument_from_dict(json.loads(json.dumps(package.to_dict()))) == package

    def test_normalisation_label_and_attributes(self):
        floating = LookbackOption(" Call ", 1)
        assert (floating.option_type, floating.kind, floating.strike, floating.expiry) == ("call", "floating", None, 1.0)
        assert floating.label == "LB float C T=1.00" and floating.is_floating and floating.is_call
        fixed = LookbackOption("put", 0.5, strike=97.5, kind="FIXED")
        assert fixed.label == "LB fixed P 97.5 T=0.50" and fixed.strike == 97.5 and not fixed.is_floating
        assert fixed.is_expired(Market(100.0, 0.2, t=0.5))

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(option_type="both"),
            dict(kind="partial"),
            dict(kind="fixed"),  # no strike
            dict(strike=100.0),  # floating with a strike
            dict(kind="fixed", strike=-1.0),
            dict(expiry=float("nan")),
            dict(expiry="soon"),
            dict(running_min=0.0),
            dict(running_max=float("inf")),
            dict(running_min=110.0, running_max=100.0),
        ],
    )
    def test_validation(self, kwargs):
        base = dict(option_type="call", expiry=1.0)
        with pytest.raises(ValueError):
            LookbackOption(**{**base, **kwargs})
