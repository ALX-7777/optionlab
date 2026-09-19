"""Tests for optionlab.black_scholes.

Every analytic Greek is checked against 4th-order central finite differences
of the price (or of a lower-order analytic Greek) on a grid of moneyness,
maturities, vols, rates and dividend yields, for calls and puts.
"""

import math

import numpy as np
import pytest

from optionlab import black_scholes as bs

# Any numpy warning (divide by zero, invalid value...) is a test failure.
pytestmark = pytest.mark.filterwarnings("error")

OPTION_TYPES = ("call", "put")


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def make_grid():
    """Broadcastable grid: ITM/ATM/OTM x short/long tau x low/high vol x r x q."""
    spot = np.array([70.0, 90.0, 100.0, 110.0, 140.0])
    tau = np.array([0.05, 0.5, 2.0, 5.0])
    vol = np.array([0.15, 0.45])
    rate = np.array([0.0, 0.05])
    div = np.array([0.0, 0.03])
    S, T, V, R, Q = np.meshgrid(spot, tau, vol, rate, div, indexing="ij")
    return dict(spot=S, strike=100.0, tau=T, vol=V, rate=R, div=Q)


def fd(fn, kwargs, name, rel_step=2e-4, abs_step=None):
    """4th-order central difference of ``fn(**kwargs)`` w.r.t. ``kwargs[name]``."""
    x = np.asarray(kwargs[name], dtype=float)
    h = np.full(x.shape, abs_step) if abs_step is not None else rel_step * np.abs(x)

    def f(shift):
        return fn(**{**kwargs, name: x + shift})

    return (-f(2 * h) + 8 * f(h) - 8 * f(-h) + f(-2 * h)) / (12 * h)


def assert_close(analytic, numeric, rtol, scale_atol=1e-9):
    """Relative check with an absolute floor proportional to the largest value."""
    atol = scale_atol * max(float(np.max(np.abs(analytic))), 1e-12)
    np.testing.assert_allclose(analytic, numeric, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------- #
# Reference values and model identities
# ---------------------------------------------------------------------- #
def test_reference_values():
    call = bs.price(100, 100, 1.0, 0.2, 0.05, 0.0, "call")
    put = bs.price(100, 100, 1.0, 0.2, 0.05, 0.0, "put")
    assert round(call, 4) == 10.4506
    assert round(put, 4) == 5.5735
    assert isinstance(call, float)


def test_reference_value_with_dividends():
    # Hull-style example: S=K=100, T=1, r=5%, q=2%, vol=20%.
    assert bs.price(100, 100, 1.0, 0.2, 0.05, 0.02, "call") == pytest.approx(9.22700551, abs=1e-7)


def test_put_call_parity_with_dividends():
    g = make_grid()
    call = bs.price(option_type="call", **g)
    put = bs.price(option_type="put", **g)
    forward_value = g["spot"] * np.exp(-g["div"] * g["tau"]) - 100.0 * np.exp(-g["rate"] * g["tau"])
    np.testing.assert_allclose(call - put, forward_value, rtol=0, atol=1e-10)


def test_delta_parity():
    g = make_grid()
    diff = bs.delta(option_type="call", **g) - bs.delta(option_type="put", **g)
    np.testing.assert_allclose(diff, np.exp(-g["div"] * g["tau"]), rtol=0, atol=1e-13)


@pytest.mark.parametrize("name", ["gamma", "vega", "vanna", "volga", "speed", "color", "zomma"])
def test_greeks_shared_by_call_and_put(name):
    g = make_grid()
    fn = getattr(bs, name)
    np.testing.assert_allclose(fn(option_type="call", **g), fn(option_type="put", **g), rtol=1e-12, atol=1e-14)


def test_theta_and_rho_parity():
    # Differentiate C - P = S e^{-q tau} - K e^{-r tau} in t and in r.
    g = make_grid()
    S, T, r, q = g["spot"], g["tau"], g["rate"], g["div"]
    theta_diff = bs.theta(option_type="call", **g) - bs.theta(option_type="put", **g)
    np.testing.assert_allclose(
        theta_diff, q * S * np.exp(-q * T) - r * 100.0 * np.exp(-r * T), rtol=0, atol=1e-10
    )
    rho_diff = bs.rho(option_type="call", **g) - bs.rho(option_type="put", **g)
    np.testing.assert_allclose(rho_diff, 100.0 * T * np.exp(-r * T), rtol=0, atol=1e-10)


# ---------------------------------------------------------------------- #
# Analytic Greeks vs finite differences
# ---------------------------------------------------------------------- #
# greek -> (function differentiated, variable, sign)
FD_RELATIONS = {
    "delta": ("price", "spot", 1.0),
    "gamma": ("delta", "spot", 1.0),
    "speed": ("gamma", "spot", 1.0),
    "vega": ("price", "vol", 1.0),
    "volga": ("vega", "vol", 1.0),
    "vanna": ("delta", "vol", 1.0),
    "zomma": ("gamma", "vol", 1.0),
    "theta": ("price", "tau", -1.0),
    "charm": ("delta", "tau", -1.0),
    "color": ("gamma", "tau", -1.0),
    "dual_delta": ("price", "strike", 1.0),
}


@pytest.mark.parametrize("option_type", OPTION_TYPES)
@pytest.mark.parametrize("greek", sorted(FD_RELATIONS))
def test_greek_matches_finite_difference(greek, option_type):
    base_name, variable, sign = FD_RELATIONS[greek]
    g = make_grid()
    g["strike"] = np.full(g["spot"].shape, 100.0)
    base_fn = getattr(bs, base_name)

    def f(**kw):
        return base_fn(option_type=option_type, **kw)

    numeric = sign * fd(f, g, variable)
    analytic = getattr(bs, greek)(option_type=option_type, **g)
    assert_close(analytic, numeric, rtol=1e-6)


@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_rho_matches_finite_difference(option_type):
    g = make_grid()
    numeric = fd(lambda **kw: bs.price(option_type=option_type, **kw), g, "rate", abs_step=1e-4)
    assert_close(bs.rho(option_type=option_type, **g), numeric, rtol=1e-7)


@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_vanna_is_also_dvega_dspot(option_type):
    g = make_grid()
    numeric = fd(lambda **kw: bs.vega(option_type=option_type, **kw), g, "spot")
    assert_close(bs.vanna(option_type=option_type, **g), numeric, rtol=1e-6)


@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_gamma_matches_second_difference_of_price(option_type):
    g = make_grid()
    h = 1e-3 * g["spot"]

    def p(shift):
        return bs.price(option_type=option_type, **{**g, "spot": g["spot"] + shift})

    # 4th-order second-derivative stencil.
    numeric = (-p(2 * h) + 16 * p(h) - 30 * p(0 * h) + 16 * p(-h) - p(-2 * h)) / (12 * h**2)
    assert_close(bs.gamma(option_type=option_type, **g), numeric, rtol=1e-5, scale_atol=1e-7)


def test_black_scholes_pde_holds():
    # theta + (r - q) S delta + 0.5 vol^2 S^2 gamma - r V = 0
    g = make_grid()
    for option_type in OPTION_TYPES:
        out = bs.greeks(option_type=option_type, **g)
        residual = (
            out["theta"]
            + (g["rate"] - g["div"]) * g["spot"] * out["delta"]
            + 0.5 * g["vol"] ** 2 * g["spot"] ** 2 * out["gamma"]
            - g["rate"] * out["price"]
        )
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)


# ---------------------------------------------------------------------- #
# greeks() dict, vectorisation
# ---------------------------------------------------------------------- #
def test_greeks_dict_matches_individual_functions():
    g = make_grid()
    for option_type in OPTION_TYPES:
        out = bs.greeks(option_type=option_type, **g)
        assert tuple(out) == ("price",) + bs.GREEK_NAMES
        for name, value in out.items():
            np.testing.assert_array_equal(value, getattr(bs, name)(option_type=option_type, **g))


def test_scalar_input_returns_floats():
    out = bs.greeks(100.0, 95.0, 0.5, 0.25, 0.01, 0.02, "put")
    assert all(isinstance(v, float) for v in out.values())
    d1, d2 = bs.d1_d2(100.0, 95.0, 0.5, 0.25, 0.01, 0.02)
    assert isinstance(d1, float) and d2 == pytest.approx(d1 - 0.25 * math.sqrt(0.5))


def test_broadcast_shapes():
    spot = np.linspace(50, 150, 201)
    assert bs.price(spot, 100, 1.0, 0.2).shape == (201,)
    vols = np.array([0.1, 0.2, 0.3])[:, None]
    grid = bs.price(spot[None, :], 100, 1.0, vols)
    assert grid.shape == (3, 201)
    np.testing.assert_allclose(grid[1], bs.price(spot, 100, 1.0, 0.2))
    strikes = np.array([90.0, 100.0, 110.0])[:, None, None]
    taus = np.array([0.25, 1.0])[None, :, None]
    assert bs.vega(spot[None, None, :], strikes, taus, 0.2).shape == (3, 2, 201)


def test_vomma_is_volga():
    assert bs.vomma is bs.volga


def test_option_type_validation():
    assert bs.price(100, 100, 1, 0.2, option_type="CALL") == bs.price(100, 100, 1, 0.2, option_type="call")
    assert bs.normalize_option_type(" Put ") == "put"
    for bad in ("c", "", None, 1, "calls"):
        with pytest.raises(ValueError, match="option_type"):
            bs.price(100, 100, 1, 0.2, option_type=bad)


def test_strike_must_be_positive():
    with pytest.raises(ValueError, match="strike"):
        bs.price(100, 0.0, 1, 0.2)
    with pytest.raises(ValueError, match="strike"):
        bs.implied_vol(5.0, 100, np.array([100.0, -1.0]), 1)


# ---------------------------------------------------------------------- #
# Edge cases: no warnings (module-level filter), no NaN
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("tau", [0.0, -0.25])
def test_at_and_after_expiry(tau):
    spot = np.array([80.0, 100.0, 120.0])
    call = bs.greeks(spot, 100.0, tau, 0.2, 0.05, 0.02, "call")
    put = bs.greeks(spot, 100.0, tau, 0.2, 0.05, 0.02, "put")
    np.testing.assert_array_equal(call["price"], [0.0, 0.0, 20.0])
    np.testing.assert_array_equal(put["price"], [20.0, 0.0, 0.0])
    np.testing.assert_array_equal(call["delta"], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(put["delta"], [-1.0, 0.0, 0.0])
    for out in (call, put):
        for name in bs.GREEK_NAMES:
            if name != "delta":
                np.testing.assert_array_equal(out[name], np.zeros(3), err_msg=name)
    np.testing.assert_array_equal(bs.dual_delta(spot, 100.0, tau, 0.2, 0.05, 0.02, "call"), [0.0, 0.0, -1.0])


@pytest.mark.parametrize("vol", [0.0, -0.1])
def test_zero_vol_is_discounted_forward_intrinsic(vol):
    spot = np.array([80.0, 100.0, 120.0])
    tau, r, q = 2.0, 0.05, 0.01
    fwd_value = spot * math.exp(-q * tau) - 100.0 * math.exp(-r * tau)
    np.testing.assert_allclose(bs.price(spot, 100.0, tau, vol, r, q, "call"), np.maximum(fwd_value, 0.0))
    np.testing.assert_allclose(bs.price(spot, 100.0, tau, vol, r, q, "put"), np.maximum(-fwd_value, 0.0))
    out = bs.greeks(spot, 100.0, tau, vol, r, q, "call")
    np.testing.assert_allclose(out["delta"], math.exp(-q * tau) * (fwd_value > 0))
    for name in ("gamma", "vega", "vanna", "volga", "speed", "color", "zomma"):
        np.testing.assert_array_equal(out[name], np.zeros(3))
    assert np.all(np.isfinite(out["theta"])) and np.all(np.isfinite(out["rho"]))


def test_limits_are_continuous():
    # vol -> 0+ and tau -> 0+ converge to the degenerate branches.
    for option_type in OPTION_TYPES:
        args = (np.array([80.0, 120.0]), 100.0)
        small_vol = bs.price(*args, 1.0, 1e-8, 0.03, 0.01, option_type)
        zero_vol = bs.price(*args, 1.0, 0.0, 0.03, 0.01, option_type)
        np.testing.assert_allclose(small_vol, zero_vol, atol=1e-10)
        small_tau = bs.price(*args, 1e-12, 0.2, 0.03, 0.01, option_type)
        np.testing.assert_allclose(small_tau, bs.intrinsic_value(args[0], 100.0, option_type), atol=1e-8)


def test_extreme_moneyness_is_finite():
    spot = np.array([0.0, 1e-12, 1e-3, 1.0, 1e4, 1e8, 1e12])
    for option_type in OPTION_TYPES:
        for tau in (1e-6, 1.0, 30.0):
            for vol in (1e-4, 0.2, 3.0):
                out = bs.greeks(spot, 100.0, tau, vol, 0.03, 0.01, option_type)
                for name, value in out.items():
                    assert np.all(np.isfinite(value)), (option_type, tau, vol, name)
    # Spot at zero: the call is worthless, the put is worth the discounted strike.
    assert bs.price(0.0, 100.0, 1.0, 0.2, 0.05, 0.0, "call") == 0.0
    assert bs.price(0.0, 100.0, 1.0, 0.2, 0.05, 0.0, "put") == pytest.approx(100 * math.exp(-0.05))


def test_mixed_regular_and_degenerate_entries():
    tau = np.array([-1.0, 0.0, 1.0])
    vol = np.array([0.2, 0.2, 0.0])
    out = bs.greeks(110.0, 100.0, tau, vol, 0.0, 0.0, "call")
    np.testing.assert_allclose(out["price"], [10.0, 10.0, 10.0])
    np.testing.assert_allclose(out["gamma"], 0.0)


def test_nan_inputs_propagate():
    out = bs.price(np.array([100.0, np.nan]), 100.0, 1.0, np.array([np.nan, 0.2]))
    assert np.all(np.isnan(out))
    assert math.isnan(bs.delta(100.0, 100.0, 1.0, float("nan")))


def test_no_negative_zeros():
    out = bs.greeks(100.0, 100.0, 1.0, 0.0, 0.05)
    assert all(math.copysign(1.0, v) > 0 for v in out.values() if v == 0.0)


def test_d1_d2_degenerate():
    d1, d2 = bs.d1_d2(np.array([90.0, 100.0, 110.0]), 100.0, 0.0, 0.2)
    np.testing.assert_array_equal(d1, [-np.inf, 0.0, np.inf])
    np.testing.assert_array_equal(d2, d1)


# ---------------------------------------------------------------------- #
# Trading intuition encoded as tests
# ---------------------------------------------------------------------- #
def test_theta_signs():
    # Long ATM options bleed...
    assert bs.theta(100, 100, 1.0, 0.2, 0.05, 0.0, "call") < 0
    assert bs.theta(100, 100, 1.0, 0.2, 0.0, 0.0, "put") < 0
    # ...but a deep ITM put earns carry on the strike it is waiting to receive,
    assert bs.theta(40, 100, 1.0, 0.2, 0.08, 0.0, "put") > 0
    # and so does a deep ITM call on a high-dividend stock.
    assert bs.theta(250, 100, 1.0, 0.2, 0.0, 0.08, "call") > 0


def test_rho_and_vanna_signs():
    assert bs.rho(100, 100, 1.0, 0.2, 0.02, 0.0, "call") > 0
    assert bs.rho(100, 100, 1.0, 0.2, 0.02, 0.0, "put") < 0
    assert bs.vanna(80, 100, 1.0, 0.2) > 0  # OTM call / ITM put side
    assert bs.vanna(125, 100, 1.0, 0.2) < 0


# ---------------------------------------------------------------------- #
# Trader units and metadata
# ---------------------------------------------------------------------- #
def test_to_trader_units():
    raw = bs.greeks(100, 100, 1.0, 0.2, 0.05)
    snapshot = dict(raw)
    trader = bs.to_trader_units(raw)
    assert raw == snapshot  # input untouched
    assert trader["vega"] == pytest.approx(raw["vega"] / 100)
    assert trader["theta"] == pytest.approx(raw["theta"] / 365)
    assert trader["rho"] == pytest.approx(raw["rho"] / 100)
    assert trader["vanna"] == pytest.approx(raw["vanna"] / 100)
    assert trader["volga"] == pytest.approx(raw["volga"] / 1e4)
    assert trader["charm"] == pytest.approx(raw["charm"] / 365)
    assert trader["color"] == pytest.approx(raw["color"] / 365)
    assert trader["zomma"] == pytest.approx(raw["zomma"] / 100)
    for name in ("price", "delta", "gamma", "speed"):
        assert trader[name] == raw[name]
    # Works on arrays and ignores unknown keys.
    arr = bs.to_trader_units({"vega": np.array([100.0, 200.0]), "custom": 3.0})
    np.testing.assert_allclose(arr["vega"], [1.0, 2.0])
    assert arr["custom"] == 3.0


def test_one_vol_point_vega_predicts_repricing():
    vega_pt = bs.to_trader_units(bs.greeks(100, 100, 1.0, 0.2, 0.05))["vega"]
    bumped = bs.price(100, 100, 1.0, 0.21, 0.05) - bs.price(100, 100, 1.0, 0.20, 0.05)
    assert bumped == pytest.approx(vega_pt, rel=1e-2)


def test_greek_info_is_complete():
    assert set(bs.GREEK_INFO) == {"price", *bs.GREEK_NAMES}
    for name, info in bs.GREEK_INFO.items():
        assert {"label", "symbol", "raw_unit", "trader_unit", "description"} <= set(info), name
        assert all(isinstance(v, str) and v for v in info.values())
    assert set(bs.TRADER_UNIT_SCALES) <= set(bs.GREEK_NAMES)


# ---------------------------------------------------------------------- #
# Implied volatility
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("option_type", OPTION_TYPES)
def test_implied_vol_round_trip(option_type):
    spot = np.array([80.0, 95.0, 100.0, 105.0, 125.0])[:, None, None]
    tau = np.array([0.05, 0.5, 3.0])[None, :, None]
    vol = np.array([0.05, 0.2, 0.6, 1.5])[None, None, :]
    prices = bs.price(spot, 100.0, tau, vol, 0.03, 0.01, option_type)
    recovered = bs.implied_vol(prices, spot, 100.0, tau, 0.03, 0.01, option_type)
    assert recovered.shape == prices.shape
    # Only meaningful where the price carries information about vol (non-negligible vega).
    informative = bs.vega(spot, 100.0, tau, vol, 0.03, 0.01) > 1e-6
    assert informative.sum() > 50
    np.testing.assert_allclose(recovered[informative], np.broadcast_to(vol, prices.shape)[informative], rtol=1e-8)
    # Everywhere it returns a number, that number reprices the option.
    ok = ~np.isnan(recovered)
    repriced = bs.price(spot, 100.0, tau, np.where(ok, recovered, 0.2), 0.03, 0.01, option_type)
    np.testing.assert_allclose(repriced[ok], prices[ok], rtol=0, atol=1e-9)


def test_implied_vol_scalar_and_extremes():
    iv = bs.implied_vol(10.450583572185565, 100, 100, 1.0, 0.05)
    assert isinstance(iv, float) and iv == pytest.approx(0.2, abs=1e-10)
    high = bs.price(100, 100, 1.0, 8.0)
    assert bs.implied_vol(high, 100, 100, 1.0) == pytest.approx(8.0, rel=1e-8)
    low = bs.price(100, 100, 1.0, 0.003)
    assert bs.implied_vol(low, 100, 100, 1.0) == pytest.approx(0.003, rel=1e-8)


def test_implied_vol_arbitrage_bounds():
    # Below the forward intrinsic value, above the upper bound, expired: nan.
    assert math.isnan(bs.implied_vol(4.0, 100, 100, 1.0, 0.05))  # < 100 - 100 e^{-0.05} = 4.877
    assert math.isnan(bs.implied_vol(100.0, 100, 100, 1.0))  # call >= spot
    assert math.isnan(bs.implied_vol(101.0, 100, 100, 1.0))
    assert math.isnan(bs.implied_vol(96.0, 100, 100, 1.0, 0.05, 0.0, "put"))  # put >= K e^{-rT}
    assert math.isnan(bs.implied_vol(-1.0, 100, 100, 1.0))
    assert math.isnan(bs.implied_vol(5.0, 100, 100, 0.0))
    assert math.isnan(bs.implied_vol(float("nan"), 100, 100, 1.0))
    # Exactly on the lower bound: zero vol.
    assert bs.implied_vol(0.0, 90, 100, 1.0) == 0.0
    lower = 100 - 100 * math.exp(-0.05)
    assert bs.implied_vol(lower, 100, 100, 1.0, 0.05) == 0.0


def test_implied_vol_vectorised_mixed_validity():
    prices = np.array([1.0, 5.0, 99.0, 101.0, -2.0])
    iv = bs.implied_vol(prices, 100.0, 100.0, 1.0)
    assert iv.shape == (5,)
    assert np.isnan(iv[3]) and np.isnan(iv[4])
    np.testing.assert_allclose(bs.price(100.0, 100.0, 1.0, iv[:3]), prices[:3], atol=1e-9)
