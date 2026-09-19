"""Tests for optionlab.numerical (generic bump-and-reprice Greeks)."""

import copy
import math

import numpy as np
import pytest

from optionlab.instruments import GREEK_KEYS, CompositeInstrument, EuropeanOption, Position
from optionlab.market import Market
from optionlab.numerical import NUMERICAL_GREEK_KEYS, numerical_greeks

pytestmark = pytest.mark.filterwarnings("error")

# Relative tolerance per Greek for default bumps against closed forms.
RTOL = {
    "price": 1e-14,
    "delta": 1e-5,
    "gamma": 1e-5,
    "vega": 1e-5,
    "theta": 1e-5,
    "rho": 1e-6,
    "vanna": 2e-4,
    "volga": 1e-3,
    "charm": 2e-4,
    "speed": 5e-4,
    "color": 5e-4,
    "zomma": 1e-3,
}


def assert_greeks_close(numeric, analytic, atol_scale=1e-6):
    for key in GREEK_KEYS:
        a, n = np.asarray(analytic[key]), np.asarray(numeric[key])
        atol = atol_scale * max(float(np.max(np.abs(a))), 1e-8)
        np.testing.assert_allclose(n, a, rtol=RTOL[key], atol=atol, err_msg=key)


def test_keys_match_library_convention():
    assert NUMERICAL_GREEK_KEYS == GREEK_KEYS


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("strike", [85.0, 100.0, 120.0])
@pytest.mark.parametrize("div", [0.0, 0.03])
def test_matches_analytic_european(option_type, strike, div):
    opt = EuropeanOption(option_type, strike, 1.25)
    mkt = Market(spot=100.0, vol=0.25, rate=0.04, div=div, t=0.25)
    numeric = numerical_greeks(opt, mkt)
    assert tuple(numeric) == GREEK_KEYS
    assert all(isinstance(v, float) for v in numeric.values())
    assert_greeks_close(numeric, opt.greeks(mkt))


def test_vectorised_over_market_fields():
    opt = EuropeanOption("call", 100.0, 1.0)
    spots = np.linspace(70.0, 130.0, 13)
    vols = np.array([0.15, 0.3])[:, None]
    mkt = Market(spot=spots[None, :], vol=vols, rate=0.03, div=0.01)
    numeric = numerical_greeks(opt, mkt)
    assert all(v.shape == (2, 13) for v in numeric.values())
    assert_greeks_close(numeric, opt.greeks(mkt), atol_scale=1e-5)
    # A cell of the grid equals the scalar computation.
    scalar = numerical_greeks(opt, Market(spots[4], 0.3, 0.03, 0.01))
    for key in GREEK_KEYS:
        assert numeric[key][1, 4] == pytest.approx(scalar[key], rel=1e-9, abs=1e-12)


def test_composite_through_generic_engine():
    straddle = CompositeInstrument(
        "Straddle",
        (Position(EuropeanOption("call", 100, 1.0), 1), Position(EuropeanOption("put", 100, 1.0), 1)),
    )
    mkt = Market(102.0, 0.2, 0.02, 0.01)
    assert_greeks_close(numerical_greeks(straddle, mkt), straddle.greeks(mkt))


def test_polynomial_price_function_is_differentiated_exactly():
    # V = S^3 vol^2 + 5 r - 2 t : every derivative is known in closed form.
    def price_fn(m):
        return m.spot**3 * m.vol**2 + 5.0 * m.rate - 2.0 * m.t

    S, v = 3.0, 0.5
    out = numerical_greeks(price_fn, Market(S, v, 0.01, 0.0, t=1.0))
    expected = {
        "price": S**3 * v**2 + 0.05 - 2.0,
        "delta": 3 * S**2 * v**2,
        "gamma": 6 * S * v**2,
        "speed": 6 * v**2,
        "vega": 2 * S**3 * v,
        "volga": 2 * S**3,
        "vanna": 6 * S**2 * v,
        "zomma": 12 * S * v,
        "theta": -2.0,
        "rho": 5.0,
        "charm": 0.0,
        "color": 0.0,
    }
    for key, value in expected.items():
        assert out[key] == pytest.approx(value, rel=1e-5, abs=1e-4), key


def test_callable_without_expiry_and_explicit_expiry():
    # Zero-coupon bond paying 1 at T=2: theta = r P, rho = -(T - t) P.
    def bond(m):
        return np.exp(-m.rate * (2.0 - m.t)) + 0.0 * m.spot

    mkt = Market(100.0, 0.2, rate=0.05, t=0.5)
    price = math.exp(-0.05 * 1.5)
    for kwargs in ({}, {"expiry": 2.0}):
        out = numerical_greeks(bond, mkt, **kwargs)
        assert out["theta"] == pytest.approx(0.05 * price, rel=1e-7)
        assert out["rho"] == pytest.approx(-1.5 * price, rel=1e-7)
        assert out["delta"] == 0.0 and out["vega"] == 0.0


def test_rejects_bad_input():
    mkt = Market(100.0, 0.2)
    with pytest.raises(ValueError, match="callable"):
        numerical_greeks(42, mkt)
    with pytest.raises(ValueError, match="spot_bump_rel"):
        numerical_greeks(EuropeanOption("call", 100, 1.0), mkt, spot_bump_rel=0.0)
    with pytest.raises(ValueError, match="t_bump"):
        numerical_greeks(EuropeanOption("call", 100, 1.0), mkt, t_bump=-1.0)
    with pytest.raises(ValueError, match="positive spot"):
        numerical_greeks(EuropeanOption("call", 100, 1.0), Market(0.0, 0.2))


def test_bump_sizes_are_parameters():
    opt = EuropeanOption("call", 100.0, 1.0)
    mkt = Market(100.0, 0.2, 0.05)
    coarse = numerical_greeks(opt, mkt, spot_bump_rel=5e-2, vol_bump=2e-2, t_bump=1e-2, rate_bump=1e-2)
    fine = numerical_greeks(opt, mkt)
    exact = opt.greeks(mkt)
    assert abs(coarse["gamma"] - exact["gamma"]) > abs(fine["gamma"] - exact["gamma"])
    assert coarse["delta"] == pytest.approx(exact["delta"], rel=1e-2)


class RecordingOption:
    """Wraps an option and records every market it is asked to price."""

    def __init__(self, option):
        self.option = option
        self.expiry = option.expiry
        self.markets = []

    def price(self, mkt):
        self.markets.append(mkt)
        return self.option.price(mkt)


@pytest.mark.parametrize("tau", [1e-7, 1e-5, 3e-4, 0.01])
def test_time_bump_never_crosses_expiry(tau):
    spy = RecordingOption(EuropeanOption("call", 100.0, 1.0))
    mkt = Market(101.0, 0.2, 0.03, t=1.0 - tau)
    out = numerical_greeks(spy, mkt)
    assert max(m.t for m in spy.markets) < 1.0
    assert min(m.t for m in spy.markets) == mkt.t  # time is never bumped backwards
    assert all(math.isfinite(v) for v in out.values())
    assert out["theta"] < 0 and 0.0 < out["delta"] <= 1.0
    assert len(spy.markets) == 19


def test_near_expiry_theta_is_still_accurate():
    opt = EuropeanOption("put", 100.0, 1.0)
    mkt = Market(98.0, 0.3, 0.02, 0.01, t=1.0 - 2e-4)  # tau smaller than 2 * t_bump
    out = numerical_greeks(opt, mkt)
    exact = opt.greeks(mkt)
    assert out["theta"] == pytest.approx(exact["theta"], rel=2e-2)
    assert out["delta"] == pytest.approx(exact["delta"], rel=1e-3)


@pytest.mark.parametrize("t", [1.0, 1.7])
def test_expired_instrument(t):
    opt = EuropeanOption("call", 100.0, 1.0)
    itm = numerical_greeks(opt, Market(120.0, 0.2, 0.05, t=t))
    assert itm["price"] == 20.0 and itm["delta"] == pytest.approx(1.0)
    otm = numerical_greeks(opt, Market(80.0, 0.2, 0.05, t=t))
    assert otm["price"] == 0.0 and otm["delta"] == 0.0
    for out in (itm, otm):
        assert all(out[k] == 0.0 for k in GREEK_KEYS if k not in ("price", "delta"))


def test_array_clock_mixes_live_and_expired():
    opt = EuropeanOption("call", 100.0, 1.0)
    mkt = Market(110.0, 0.2, 0.05, t=np.array([0.0, 0.999, 1.0, 1.5]))
    out = numerical_greeks(opt, mkt)
    exact = opt.greeks(mkt)
    assert all(np.all(np.isfinite(v)) for v in out.values())
    np.testing.assert_allclose(out["price"], exact["price"])
    np.testing.assert_allclose(out["theta"][0], exact["theta"][0], rtol=1e-6)
    np.testing.assert_array_equal(out["theta"][2:], [0.0, 0.0])
    np.testing.assert_array_equal(out["gamma"][2:], [0.0, 0.0])
    np.testing.assert_allclose(out["delta"][2:], [1.0, 1.0])


def test_small_and_zero_vol():
    opt = EuropeanOption("call", 100.0, 1.0)
    tiny = numerical_greeks(opt, Market(100.0, 5e-4, 0.05))  # vol < vol_bump: bump shrinks
    assert all(math.isfinite(v) for v in tiny.values())
    assert tiny["delta"] == pytest.approx(1.0, abs=1e-6)  # forward is ITM, no vol: pure forward
    zero = numerical_greeks(opt, Market(100.0, 0.0, 0.05))
    assert zero["vega"] == 0.0 and zero["volga"] == 0.0 and zero["vanna"] == 0.0 and zero["zomma"] == 0.0
    assert zero["price"] == pytest.approx(100 - 100 * math.exp(-0.05))


def test_nothing_is_mutated():
    opt = EuropeanOption("call", 100.0, 1.0)
    spots = np.linspace(80.0, 120.0, 5)
    mkt = Market(spots, 0.2, 0.05, 0.01, 0.1)
    before_mkt, before_opt, before_spots = copy.deepcopy(mkt.to_dict()), opt.to_dict(), spots.copy()
    numerical_greeks(opt, mkt)
    assert mkt.to_dict() == before_mkt and opt.to_dict() == before_opt
    np.testing.assert_array_equal(spots, before_spots)
