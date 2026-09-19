"""Tests for optionlab.monte_carlo (GBM simulation and generic MC pricer)."""

import math
from dataclasses import dataclass

import numpy as np
import pytest

from optionlab.instruments import (
    CompositeInstrument,
    EuropeanOption,
    Instrument,
    Position,
    Underlying,
)
from optionlab.market import Market
from optionlab.monte_carlo import (
    mc_price,
    simulate_gbm_path,
    simulate_gbm_paths,
    simulate_gbm_paths_on_grid,
)

pytestmark = pytest.mark.filterwarnings("error")


# ---------------------------------------------------------------------- #
# Toy path-dependent products with known values
# ---------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpotAtDate(Instrument):
    """Pays, at ``expiry``, the spot observed at the ABSOLUTE time ``fixing``."""

    fixing: float
    expiry: float

    def price(self, mkt):
        growth = np.exp((mkt.rate - mkt.div) * (self.fixing - mkt.t))
        return mkt.spot * growth * np.exp(-mkt.rate * (self.expiry - mkt.t))

    def payoff(self, spot_T):
        return np.asarray(spot_T, dtype=float)

    def path_payoff(self, paths, times):
        column = int(np.argmin(np.abs(times - self.fixing)))
        assert abs(times[column] - self.fixing) < 1e-12, "fixing date must be on the grid"
        return paths[:, column]


@dataclass(frozen=True)
class RunningMaxPayer(Instrument):
    """Pays the maximum of the spot over the whole life, including the past (``max_so_far``)."""

    expiry: float
    max_so_far: float

    def price(self, mkt):
        raise NotImplementedError("priced by Monte Carlo only")

    def payoff(self, spot_T):
        return np.maximum(np.asarray(spot_T, dtype=float), self.max_so_far)

    def path_payoff(self, paths, times):
        return np.maximum(paths.max(axis=1), self.max_so_far)


# ---------------------------------------------------------------------- #
# Path simulation
# ---------------------------------------------------------------------- #
class TestSimulation:
    def test_shapes_and_times(self):
        times, paths = simulate_gbm_paths(100.0, 0.2, 0.03, 0.01, horizon=2.0, n_steps=8, n_paths=50, seed=1)
        assert times.shape == (9,) and paths.shape == (50, 9)
        np.testing.assert_allclose(times, np.linspace(0.0, 2.0, 9))
        np.testing.assert_array_equal(paths[:, 0], 100.0)
        assert np.all(paths > 0)

    def test_absolute_start_time(self):
        times, _ = simulate_gbm_paths(100.0, 0.2, horizon=0.5, n_steps=5, n_paths=4, seed=1, t0=0.25)
        assert times[0] == 0.25 and times[-1] == pytest.approx(0.75)

    def test_seed_reproducibility(self):
        a = simulate_gbm_paths(100.0, 0.2, 0.0, 0.0, 1.0, 10, 20, seed=42)[1]
        b = simulate_gbm_paths(100.0, 0.2, 0.0, 0.0, 1.0, 10, 20, seed=42)[1]
        c = simulate_gbm_paths(100.0, 0.2, 0.0, 0.0, 1.0, 10, 20, seed=43)[1]
        d = simulate_gbm_paths(100.0, 0.2, 0.0, 0.0, 1.0, 10, 20, seed=np.random.default_rng(42))[1]
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, d)
        assert not np.array_equal(a, c)

    def test_antithetic_pairs_mirror_each_other(self):
        r, q, vol = 0.03, 0.01, 0.25
        times, paths = simulate_gbm_paths(100.0, vol, r, q, 1.0, 12, 200, seed=5, antithetic=True)
        log_ret = np.log(paths / 100.0)
        drift = (r - q - 0.5 * vol**2) * times
        np.testing.assert_allclose(log_ret[:100] + log_ret[100:], 2 * drift[None, :] * np.ones((100, 1)), atol=1e-12)

    def test_odd_number_of_paths(self):
        _, paths = simulate_gbm_paths(100.0, 0.2, n_steps=3, n_paths=7, seed=0)
        assert paths.shape == (7, 4)
        _, plain = simulate_gbm_paths(100.0, 0.2, n_steps=3, n_paths=7, seed=0, antithetic=False)
        assert plain.shape == (7, 4)

    def test_terminal_distribution(self):
        S0, vol, r, q, T = 100.0, 0.3, 0.05, 0.02, 1.5
        _, paths = simulate_gbm_paths(S0, vol, r, q, T, 6, 200_000, seed=11, antithetic=False)
        terminal = paths[:, -1]
        stderr = terminal.std(ddof=1) / math.sqrt(terminal.size)
        assert abs(terminal.mean() - S0 * math.exp((r - q) * T)) < 4 * stderr
        log_ret = np.log(terminal / S0)
        assert log_ret.mean() == pytest.approx((r - q - 0.5 * vol**2) * T, abs=4 * vol * math.sqrt(T / terminal.size))
        assert log_ret.var(ddof=1) == pytest.approx(vol**2 * T, rel=0.02)

    def test_scheme_is_exact_whatever_the_step(self):
        # One giant step and 100 small ones sample the same terminal law.
        kwargs = dict(spot=100.0, vol=0.4, rate=0.02, div=0.0, horizon=2.0, n_paths=100_000, antithetic=False)
        one = simulate_gbm_paths(n_steps=1, seed=1, **kwargs)[1][:, -1]
        many = simulate_gbm_paths(n_steps=100, seed=2, **kwargs)[1][:, -1]
        assert np.log(one).mean() == pytest.approx(np.log(many).mean(), abs=0.01)
        assert np.log(one).std() == pytest.approx(np.log(many).std(), rel=0.02)

    def test_zero_vol_is_deterministic(self):
        times, paths = simulate_gbm_paths(100.0, 0.0, 0.05, 0.01, 1.0, 4, 3, seed=0)
        np.testing.assert_allclose(paths, np.ones((3, 1)) * 100.0 * np.exp(0.04 * times)[None, :])

    def test_non_uniform_grid(self):
        times = np.array([0.5, 0.51, 0.75, 2.0])
        paths = simulate_gbm_paths_on_grid(50.0, 0.2, 0.01, 0.0, times, 100_000, seed=3)
        assert paths.shape == (100_000, 4)
        # Variance of log-returns accrues with elapsed time, not with step count.
        var = np.log(paths / 50.0).var(axis=0, ddof=1)
        np.testing.assert_allclose(var[1:], 0.2**2 * (times[1:] - 0.5), rtol=0.03)
        single = simulate_gbm_paths_on_grid(50.0, 0.2, 0.0, 0.0, np.array([0.0]), 5, seed=0)
        np.testing.assert_array_equal(single, np.full((5, 1), 50.0))

    def test_single_path(self):
        times, path = simulate_gbm_path(100.0, 0.2, 0.01, 0.0, horizon=1.0, n_steps=252, seed=9, t0=1.0)
        assert times.shape == path.shape == (253,)
        assert path[0] == 100.0 and times[0] == 1.0
        _, again = simulate_gbm_path(100.0, 0.2, 0.01, 0.0, horizon=1.0, n_steps=252, seed=9)
        np.testing.assert_array_equal(path, again)

    def test_validation(self):
        with pytest.raises(ValueError, match="horizon"):
            simulate_gbm_paths(100.0, 0.2, horizon=0.0)
        with pytest.raises(ValueError, match="n_steps"):
            simulate_gbm_paths(100.0, 0.2, n_steps=0)
        with pytest.raises(ValueError, match="n_paths"):
            simulate_gbm_paths(100.0, 0.2, n_paths=0)
        with pytest.raises(ValueError, match="spot"):
            simulate_gbm_paths(0.0, 0.2)
        with pytest.raises(ValueError, match="vol"):
            simulate_gbm_paths(100.0, -0.2)
        with pytest.raises(ValueError, match="scalar spot"):
            simulate_gbm_paths(np.array([100.0, 101.0]), 0.2)
        with pytest.raises(ValueError, match="increasing"):
            simulate_gbm_paths_on_grid(100.0, 0.2, 0.0, 0.0, np.array([0.0, 0.5, 0.5]), 10)


# ---------------------------------------------------------------------- #
# Generic pricer
# ---------------------------------------------------------------------- #
class TestMcPrice:
    @pytest.mark.parametrize("option_type,strike", [("call", 100.0), ("put", 100.0), ("call", 120.0), ("put", 85.0)])
    def test_european_within_three_stderr(self, option_type, strike):
        opt = EuropeanOption(option_type, strike, 1.0)
        mkt = Market(100.0, 0.2, 0.05, 0.02)
        price, stderr = mc_price(opt, mkt, n_paths=100_000, n_steps=1, seed=2024)
        assert isinstance(price, float) and isinstance(stderr, float)
        assert 0 < stderr < 0.05
        assert abs(price - opt.price(mkt)) < 3 * stderr

    def test_uses_time_to_maturity_from_market_clock(self):
        opt = EuropeanOption("call", 100.0, 1.0)
        mkt = Market(105.0, 0.25, 0.03, t=0.6)
        price, stderr = mc_price(opt, mkt, n_paths=100_000, n_steps=4, seed=7)
        assert abs(price - opt.price(mkt)) < 3 * stderr

    def test_reproducible_and_converging(self):
        opt = EuropeanOption("call", 100.0, 1.0)
        mkt = Market(100.0, 0.2, 0.05)
        assert mc_price(opt, mkt, 5_000, 2, seed=1) == mc_price(opt, mkt, 5_000, 2, seed=1)
        _, small = mc_price(opt, mkt, 4_000, 1, seed=1)
        _, large = mc_price(opt, mkt, 64_000, 1, seed=1)
        assert large == pytest.approx(small / 4, rel=0.15)  # 16x paths -> 4x less noise

    def test_antithetic_reduces_noise(self):
        opt = EuropeanOption("call", 80.0, 1.0)  # ITM: payoff close to linear, antithetic shines
        mkt = Market(100.0, 0.2, 0.05)
        _, with_av = mc_price(opt, mkt, 50_000, 1, seed=3, antithetic=True)
        _, without = mc_price(opt, mkt, 50_000, 1, seed=3, antithetic=False)
        assert with_av < 0.6 * without

    def test_odd_paths_with_antithetic(self):
        price, stderr = mc_price(EuropeanOption("put", 100.0, 1.0), Market(100.0, 0.2), 2_001, 1, seed=0)
        assert math.isfinite(price) and stderr > 0

    def test_path_payoff_receives_absolute_times(self):
        # Fixing at t=0.75 seen from t=0.5, paid at 1.0; 4 steps of 0.125 put 0.75 on the grid.
        product = SpotAtDate(fixing=0.75, expiry=1.0)
        mkt = Market(100.0, 0.3, 0.04, 0.01, t=0.5)
        price, stderr = mc_price(product, mkt, n_paths=100_000, n_steps=4, seed=5)
        assert abs(price - product.price(mkt)) < 3 * stderr

    def test_stored_path_state_is_used(self):
        mkt = Market(100.0, 0.2, 0.05, t=0.5)
        # Past maximum so high that the future cannot beat it: deterministic payoff.
        price, stderr = mc_price(RunningMaxPayer(1.0, max_so_far=1e6), mkt, 2_000, 16, seed=0)
        assert price == pytest.approx(1e6 * math.exp(-0.05 * 0.5)) and stderr == pytest.approx(0.0, abs=1e-6)
        # Otherwise the running max is worth more than the forward and more with finer monitoring.
        coarse, _ = mc_price(RunningMaxPayer(1.0, max_so_far=0.0), mkt, 20_000, 2, seed=0)
        fine, _ = mc_price(RunningMaxPayer(1.0, max_so_far=0.0), mkt, 20_000, 64, seed=0)
        assert fine > coarse > 100.0 * math.exp(-0.05 * 0.5)

    def test_composite_calendar_with_stock_and_expired_leg(self):
        mkt = Market(100.0, 0.25, 0.05, 0.01, t=0.3)
        combo = CompositeInstrument(
            "combo",
            (
                Position(EuropeanOption("call", 100.0, 0.5), -2),
                Position(EuropeanOption("call", 105.0, 1.37), 3),
                Position(EuropeanOption("put", 120.0, 0.2), 1),  # already expired: settles at intrinsic
                Position(Underlying(), -0.5),
                Position(
                    CompositeInstrument("nested", (Position(EuropeanOption("put", 90.0, 1.0), 1),)), 2
                ),
            ),
        )
        price, stderr = mc_price(combo, mkt, n_paths=200_000, n_steps=10, seed=8)
        assert stderr > 0 and abs(price - combo.price(mkt)) < 3 * stderr

    def test_per_leg_discounting(self):
        # Short-dated deep ITM leg: its value is dominated by the discounting of its own expiry.
        mkt = Market(100.0, 0.1, 0.10)
        near = EuropeanOption("call", 1.0, 0.5)
        combo = CompositeInstrument("c", (Position(near, 1), Position(EuropeanOption("call", 300.0, 3.0), 1)))
        price, stderr = mc_price(combo, mkt, n_paths=50_000, n_steps=6, seed=4)
        assert abs(price - combo.price(mkt)) < 3 * stderr
        wrong_discounting = 100.0 - 1.0 * math.exp(-0.10 * 3.0)
        assert abs(price - wrong_discounting) > 10 * stderr

    def test_no_simulation_needed(self):
        mkt = Market(100.0, 0.2, 0.05, t=2.0)
        combo = CompositeInstrument(
            "done", (Position(EuropeanOption("call", 90.0, 1.0), 2), Position(Underlying(), 1))
        )
        assert mc_price(combo, mkt, seed=0) == (120.0, 0.0)
        assert mc_price(Underlying(), mkt) == (100.0, 0.0)

    def test_position_scaling(self):
        opt = EuropeanOption("call", 100.0, 1.0)
        mkt = Market(100.0, 0.2, 0.05)
        price, stderr = mc_price(opt, mkt, 10_000, 1, seed=1)
        short_price, short_stderr = mc_price(Position(opt, -3), mkt, 10_000, 1, seed=1)
        assert short_price == pytest.approx(-3 * price) and short_stderr == pytest.approx(3 * stderr)

    def test_does_not_mutate_inputs(self):
        opt = EuropeanOption("call", 100.0, 1.0)
        mkt = Market(100.0, 0.2, 0.05)
        before = (opt.to_dict(), mkt.to_dict())
        mc_price(opt, mkt, 1_000, 3, seed=1)
        assert (opt.to_dict(), mkt.to_dict()) == before

    def test_validation(self):
        opt = EuropeanOption("call", 100.0, 1.0)
        with pytest.raises(ValueError, match="scalar Market"):
            mc_price(opt, Market(np.array([100.0, 101.0]), 0.2))
        with pytest.raises(ValueError, match="Instrument"):
            mc_price("call", Market(100.0, 0.2))
        with pytest.raises(ValueError, match="n_paths"):
            mc_price(opt, Market(100.0, 0.2), n_paths=1)
        with pytest.raises(ValueError, match="n_steps"):
            mc_price(opt, Market(100.0, 0.2), n_steps=0)

        @dataclass(frozen=True)
        class BadShape(Instrument):
            expiry: float = 1.0

            def price(self, mkt):
                return 0.0

            def payoff(self, spot_T):
                return np.zeros_like(spot_T)

            def path_payoff(self, paths, times):
                return paths  # wrong: one value per path expected

        with pytest.raises(ValueError, match="path_payoff must return shape"):
            mc_price(BadShape(), Market(100.0, 0.2), 100, 2, seed=0)
