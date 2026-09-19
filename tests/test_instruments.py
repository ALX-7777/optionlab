"""Tests for optionlab.market and optionlab.instruments."""

import json
import math
from dataclasses import FrozenInstanceError, dataclass, replace

import numpy as np
import pytest

from optionlab import black_scholes as bs
from optionlab.instruments import (
    EXPIRY_TOL,
    GREEK_KEYS,
    INSTRUMENT_REGISTRY,
    CompositeInstrument,
    EuropeanOption,
    Instrument,
    Position,
    Underlying,
    instrument_from_dict,
    register_instrument,
    slice_paths_to_expiry,
)
from optionlab.market import Market

pytestmark = pytest.mark.filterwarnings("error")


@register_instrument
@dataclass(frozen=True)
class FlaggedCall(Instrument):
    """Toy path-dependent product used to test the generic machinery.

    A call that dies the first time the spot is observed at or above
    ``barrier``. (The price ignores the future knock-out risk on purpose: this
    class tests plumbing, not barrier maths.)
    """

    strike: float
    expiry: float
    barrier: float
    hit: bool = False
    observations: tuple = ()

    def price(self, mkt):
        alive = 0.0 if self.hit else 1.0
        return alive * bs.price(mkt.spot, self.strike, self.expiry - mkt.t, mkt.vol, mkt.rate, mkt.div)

    def payoff(self, spot_T):
        alive = 0.0 if self.hit else 1.0
        return alive * np.maximum(np.asarray(spot_T, dtype=float) - self.strike, 0.0)

    def observe(self, spot, t):
        return replace(
            self,
            hit=self.hit or spot >= self.barrier,
            observations=self.observations + (float(t),),
        )


@pytest.fixture
def mkt():
    return Market(spot=100.0, vol=0.2, rate=0.05, div=0.02)


# ---------------------------------------------------------------------- #
# Market
# ---------------------------------------------------------------------- #
class TestMarket:
    def test_defaults_and_frozen(self):
        m = Market(100, 0.2)
        assert (m.spot, m.vol, m.rate, m.div, m.t) == (100.0, 0.2, 0.0, 0.0, 0.0)
        assert isinstance(m.spot, float)
        with pytest.raises(FrozenInstanceError):
            m.spot = 101.0

    def test_bumped_returns_modified_copy(self, mkt):
        bumped = mkt.bumped(spot=105.0, vol=0.25, t=0.5)
        assert (bumped.spot, bumped.vol, bumped.t) == (105.0, 0.25, 0.5)
        assert (bumped.rate, bumped.div) == (mkt.rate, mkt.div)
        assert mkt.spot == 100.0 and mkt.t == 0.0
        assert mkt.bumped() == mkt

    def test_bumped_rejects_unknown_field(self, mkt):
        with pytest.raises(ValueError, match="Unknown Market field"):
            mkt.bumped(sigma=0.3)

    def test_validation(self):
        with pytest.raises(ValueError, match="spot"):
            Market(-1.0, 0.2)
        with pytest.raises(ValueError, match="vol"):
            Market(100.0, np.array([0.2, -0.1]))
        with pytest.raises(ValueError, match="numeric"):
            Market("abc", 0.2)
        with pytest.raises(ValueError, match="broadcast"):
            Market(np.ones(3), np.ones(4))

    def test_array_fields(self):
        spots = np.linspace(50, 150, 5)
        m = Market(spot=spots, vol=np.array([[0.1], [0.2]]))
        assert m.shape == (2, 5) and not m.is_scalar
        assert Market(100, 0.2).shape == () and Market(100, 0.2).is_scalar
        # Stored arrays are read-only views; the caller's array stays writable.
        with pytest.raises(ValueError):
            m.spot[0] = 1.0
        spots[0] = 55.0
        assert Market([1.0, 2.0], 0.2).spot.tolist() == [1.0, 2.0]

    def test_equality_and_hash(self, mkt):
        assert mkt == Market(100.0, 0.2, 0.05, 0.02)
        assert mkt != mkt.bumped(spot=101.0)
        assert hash(mkt) == hash(Market(100.0, 0.2, 0.05, 0.02))
        arr = Market(np.array([1.0, 2.0]), 0.2)
        assert arr == Market(np.array([1.0, 2.0]), 0.2)
        assert arr != Market(np.array([1.0, 3.0]), 0.2)
        with pytest.raises(TypeError):
            hash(arr)

    def test_helpers(self):
        m = Market(100.0, 0.2, rate=0.05, div=0.02, t=0.25)
        assert m.tau(1.0) == 0.75
        assert m.discount_factor(1.0) == pytest.approx(math.exp(-0.05 * 0.75))
        assert m.forward(1.0) == pytest.approx(100 * math.exp(0.03 * 0.75))
        assert m.discount_factor(0.1) == 1.0 and m.forward(0.1) == 100.0

    def test_dict_round_trip(self):
        m = Market(np.array([90.0, 110.0]), 0.2, 0.01, 0.0, 0.5)
        assert Market.from_dict(json.loads(json.dumps(m.to_dict()))) == m
        with pytest.raises(ValueError):
            Market.from_dict({"spot": 1.0})
        with pytest.raises(ValueError):
            Market.from_dict({"spot": 1.0, "vol": 0.2, "foo": 1})


# ---------------------------------------------------------------------- #
# Underlying
# ---------------------------------------------------------------------- #
class TestUnderlying:
    def test_price_and_greeks(self, mkt):
        stock = Underlying()
        assert stock.price(mkt) == 100.0
        g = stock.greeks(mkt)
        assert tuple(g) == GREEK_KEYS
        assert g["price"] == 100.0 and g["delta"] == 1.0
        assert all(g[k] == 0.0 for k in GREEK_KEYS if k not in ("price", "delta"))

    def test_vectorised(self):
        m = Market(spot=100.0, vol=np.array([0.1, 0.2, 0.3]))
        np.testing.assert_array_equal(Underlying().price(m), [100.0] * 3)
        g = Underlying().greeks(m)
        assert all(np.shape(v) == (3,) for v in g.values())

    def test_life_cycle(self, mkt):
        stock = Underlying()
        assert stock.expiry is None
        assert stock.is_expired(mkt.bumped(t=1e9)) is False
        assert stock.tau(mkt) == math.inf
        assert stock.observe(120.0, 0.5) is stock
        np.testing.assert_array_equal(stock.payoff([90.0, 110.0]), [90.0, 110.0])
        assert stock == Underlying() and hash(stock) == hash(Underlying())
        assert instrument_from_dict(json.loads(json.dumps(stock.to_dict()))) == stock


# ---------------------------------------------------------------------- #
# EuropeanOption
# ---------------------------------------------------------------------- #
class TestEuropeanOption:
    def test_reference_prices(self):
        m = Market(100.0, 0.2, 0.05)
        assert round(EuropeanOption("call", 100, 1.0).price(m), 4) == 10.4506
        assert round(EuropeanOption("put", 100, 1.0).price(m), 4) == 5.5735

    def test_validation_and_normalisation(self):
        opt = EuropeanOption("CALL", 100, 1)
        assert opt.option_type == "call" and opt.is_call
        assert isinstance(opt.strike, float) and isinstance(opt.expiry, float)
        assert opt == EuropeanOption("call", 100.0, 1.0)
        for bad in ("straddle", None):
            with pytest.raises(ValueError, match="option_type"):
                EuropeanOption(bad, 100, 1.0)
        for bad_strike in (0.0, -5.0, float("nan"), "x"):
            with pytest.raises(ValueError):
                EuropeanOption("call", bad_strike, 1.0)
        with pytest.raises(ValueError):
            EuropeanOption("call", 100.0, float("inf"))

    def test_frozen_and_hashable(self):
        opt = EuropeanOption("put", 95, 0.5)
        with pytest.raises(FrozenInstanceError):
            opt.strike = 90.0
        book = {opt: 3, EuropeanOption("put", 95, 0.5): 4}
        assert book == {opt: 4}

    def test_absolute_expiry_uses_market_clock(self, mkt):
        opt = EuropeanOption("call", 100, 1.0)
        later = mkt.bumped(t=0.4)
        assert opt.price(later) == pytest.approx(bs.price(100, 100, 0.6, 0.2, 0.05, 0.02))
        assert opt.tau(later) == pytest.approx(0.6)

    def test_vectorised_price_and_greeks(self):
        opt = EuropeanOption("call", 100, 1.0)
        spots = np.linspace(50, 150, 201)
        prices = opt.price(Market(spot=spots, vol=0.2))
        assert prices.shape == (201,) and np.all(np.diff(prices) > 0)
        grid = Market(spot=spots[None, :], vol=np.array([0.1, 0.3])[:, None], t=0.5)
        g = opt.greeks(grid)
        assert tuple(g) == GREEK_KEYS
        assert all(v.shape == (2, 201) for v in g.values())
        assert g["price"][1, 100] == pytest.approx(opt.price(Market(100.0, 0.3, t=0.5)))

    def test_greeks_are_analytic(self, mkt):
        opt = EuropeanOption("put", 105, 0.75)
        assert opt.greeks(mkt) == bs.greeks(100.0, 105.0, 0.75, 0.2, 0.05, 0.02, "put")

    def test_settlement_at_and_after_expiry(self, mkt):
        call, put = EuropeanOption("call", 100, 1.0), EuropeanOption("put", 100, 1.0)
        for t in (1.0, 1.5):
            m = mkt.bumped(spot=np.array([80.0, 100.0, 130.0]), t=t)
            np.testing.assert_array_equal(call.price(m), [0.0, 0.0, 30.0])
            np.testing.assert_array_equal(put.price(m), [20.0, 0.0, 0.0])
            np.testing.assert_array_equal(call.greeks(m)["delta"], [0.0, 0.0, 1.0])
            np.testing.assert_array_equal(call.greeks(m)["gamma"], [0.0, 0.0, 0.0])

    def test_is_expired(self, mkt):
        opt = EuropeanOption("call", 100, 1.0)
        assert opt.is_expired(mkt) is False
        assert opt.is_expired(mkt.bumped(t=1.0)) is True
        # Accumulated floating point steps still land "on" the expiry.
        assert opt.is_expired(mkt.bumped(t=sum([0.1] * 10) - 1e-15)) is True
        assert opt.is_expired(mkt.bumped(t=1.0 - 10 * EXPIRY_TOL)) is False
        np.testing.assert_array_equal(
            opt.is_expired(mkt.bumped(t=np.array([0.5, 1.0, 2.0]))), [False, True, True]
        )

    def test_payoff_and_label(self):
        call, put = EuropeanOption("call", 100, 1.0), EuropeanOption("put", 97.5, 0.25)
        np.testing.assert_array_equal(call.payoff([90, 100, 115]), [0.0, 0.0, 15.0])
        np.testing.assert_array_equal(put.payoff(np.array([90.0, 100.0])), [7.5, 0.0])
        assert call.label == "C 100 T=1.00"
        assert put.label == "P 97.5 T=0.25"

    def test_path_payoff_default_uses_last_column(self):
        call = EuropeanOption("call", 100, 1.0)
        paths = np.array([[100.0, 150.0, 90.0], [100.0, 80.0, 130.0]])
        np.testing.assert_array_equal(call.path_payoff(paths, np.array([0.0, 0.5, 1.0])), [0.0, 30.0])

    def test_implied_vol(self, mkt):
        opt = EuropeanOption("put", 110, 1.0)
        assert opt.implied_vol(opt.price(mkt), mkt.bumped(vol=0.9)) == pytest.approx(0.2, abs=1e-10)

    def test_serialisation_round_trip(self):
        opt = EuropeanOption("put", 95.5, 0.75)
        d = opt.to_dict()
        assert d == {"type": "EuropeanOption", "option_type": "put", "strike": 95.5, "expiry": 0.75}
        restored = instrument_from_dict(json.loads(json.dumps(d)))
        assert restored == opt and isinstance(restored, EuropeanOption)


# ---------------------------------------------------------------------- #
# Position
# ---------------------------------------------------------------------- #
class TestPosition:
    def test_scaling(self, mkt):
        opt = EuropeanOption("call", 100, 1.0)
        short = Position(opt, -3)
        assert short.quantity == -3.0 and isinstance(short.quantity, float)
        assert short.price(mkt) == pytest.approx(-3 * opt.price(mkt))
        g, base = short.greeks(mkt), opt.greeks(mkt)
        assert tuple(g) == GREEK_KEYS
        assert all(g[k] == pytest.approx(-3 * base[k]) for k in GREEK_KEYS)
        assert g["theta"] > 0 and g["gamma"] < 0  # short option: earn theta, short gamma
        np.testing.assert_array_equal(short.payoff([90.0, 110.0]), [-0.0, -30.0])
        assert short.expiry == 1.0 and short.is_expired(mkt.bumped(t=1.0))
        assert short.scaled(-1) == Position(opt, 3)
        assert Position(opt).quantity == 1.0

    def test_label_and_sugar(self):
        opt = EuropeanOption("call", 100, 1.0)
        assert Position(opt, 2).label == "+2 x C 100 T=1.00"
        assert 2 * opt == Position(opt, 2.0) == opt * 2
        assert -opt == Position(opt, -1.0)
        with pytest.raises(TypeError):
            opt * "two"

    def test_validation(self):
        with pytest.raises(ValueError, match="Instrument"):
            Position("call", 1)
        with pytest.raises(ValueError, match="finite"):
            Position(Underlying(), float("nan"))
        with pytest.raises(ValueError, match="number"):
            Position(Underlying(), "many")

    def test_observe(self):
        vanilla = Position(EuropeanOption("call", 100, 1.0), 2)
        assert vanilla.observe(150.0, 0.5) is vanilla
        flagged = Position(FlaggedCall(100, 1.0, 120), -2)
        after = flagged.observe(125.0, 0.5)
        assert after.quantity == -2 and after.instrument.hit and not flagged.instrument.hit

    def test_round_trip_and_hash(self):
        pos = Position(EuropeanOption("put", 90, 2.0), -1.5)
        restored = Position.from_dict(json.loads(json.dumps(pos.to_dict())))
        assert restored == pos and hash(restored) == hash(pos)
        with pytest.raises(ValueError):
            Position.from_dict({"quantity": 1.0})


# ---------------------------------------------------------------------- #
# CompositeInstrument
# ---------------------------------------------------------------------- #
def make_straddle(strike=100.0, expiry=1.0):
    return CompositeInstrument(
        "Straddle",
        (Position(EuropeanOption("call", strike, expiry), 1), Position(EuropeanOption("put", strike, expiry), 1)),
    )


class TestComposite:
    def test_price_greeks_payoff_are_weighted_sums(self, mkt):
        call, put = EuropeanOption("call", 105, 1.0), EuropeanOption("put", 95, 1.0)
        rr = CompositeInstrument("Risk reversal", (Position(call, 2), Position(put, -3)))
        assert rr.price(mkt) == pytest.approx(2 * call.price(mkt) - 3 * put.price(mkt))
        g, gc, gp = rr.greeks(mkt), call.greeks(mkt), put.greeks(mkt)
        assert tuple(g) == GREEK_KEYS
        for k in GREEK_KEYS:
            assert g[k] == pytest.approx(2 * gc[k] - 3 * gp[k])
        np.testing.assert_allclose(rr.payoff([80.0, 100.0, 120.0]), [-45.0, 0.0, 30.0])
        assert rr.label == "Risk reversal" and rr.expiry == 1.0

    def test_vectorised(self):
        straddle = make_straddle()
        spots = np.linspace(60, 140, 81)
        m = Market(spot=spots, vol=0.25, rate=0.01)
        assert straddle.price(m).shape == (81,)
        assert straddle.greeks(m)["vega"].shape == (81,)
        assert straddle.payoff(spots).min() == 0.0

    def test_leg_coercion(self):
        call = EuropeanOption("call", 100, 1.0)
        combo = CompositeInstrument("Covered call", [Underlying(), (call, -1), Position(call, 0.5)])
        assert combo.legs == (Position(Underlying(), 1.0), Position(call, -1.0), Position(call, 0.5))
        assert CompositeInstrument("single", call).legs == (Position(call, 1.0),)
        with pytest.raises(ValueError, match="at least one leg"):
            CompositeInstrument("empty", ())
        with pytest.raises(ValueError, match="leg"):
            CompositeInstrument("bad", (3.0,))
        with pytest.raises(ValueError, match="name"):
            CompositeInstrument(None, (call,))

    def test_calendar_with_expired_leg(self, mkt):
        near, far = EuropeanOption("call", 100, 0.25), EuropeanOption("call", 100, 1.0)
        calendar = CompositeInstrument("Calendar", (Position(near, -1), Position(far, 1)))
        assert calendar.expiry == 1.0 and calendar.first_expiry == 0.25
        assert calendar.is_expired(mkt.bumped(t=0.5)) is False
        at_near = mkt.bumped(spot=np.array([90.0, 100.0, 115.0]), t=0.25)
        expected = far.price(at_near) - np.array([0.0, 0.0, 15.0])
        np.testing.assert_allclose(calendar.price(at_near), expected)
        # The classic tent shape: most valuable when the spot sits at the strike.
        assert np.argmax(calendar.price(at_near)) == 1
        # Stock-only composites never expire.
        stock_only = CompositeInstrument("Stock", (Underlying(),))
        assert stock_only.expiry is None and stock_only.first_expiry is None

    def test_flatten_nested_and_merge(self):
        call, put = EuropeanOption("call", 100, 1.0), EuropeanOption("put", 100, 1.0)
        inner = CompositeInstrument("inner", (Position(call, 1), Position(put, -1)))
        middle = CompositeInstrument("middle", (Position(inner, 2), Position(Underlying(), -1)))
        outer = CompositeInstrument("outer", (Position(middle, -3), Position(call, 1)))
        assert outer.flatten() == (
            Position(call, -6.0),
            Position(put, 6.0),
            Position(Underlying(), 3.0),
            Position(call, 1.0),
        )
        assert outer.flatten(merge=True) == (
            Position(call, -5.0),
            Position(put, 6.0),
            Position(Underlying(), 3.0),
        )
        flat_zero = CompositeInstrument("flat", (Position(call, 1), Position(call, -1), Position(put, 1)))
        assert flat_zero.flatten(merge=True) == (Position(put, 1.0),)

    def test_nested_pricing_matches_flat(self, mkt):
        call, put = EuropeanOption("call", 100, 1.0), EuropeanOption("put", 100, 1.0)
        inner = CompositeInstrument("inner", (Position(call, 1), Position(put, -1)))
        outer = CompositeInstrument("outer", (Position(inner, 2), Position(Underlying(), -1)))
        flat = CompositeInstrument("flat", outer.flatten())
        assert outer.price(mkt) == pytest.approx(flat.price(mkt))
        assert outer.greeks(mkt)["delta"] == pytest.approx(flat.greeks(mkt)["delta"])

    def test_scaled(self, mkt):
        straddle = make_straddle()
        short = straddle.scaled(-2)
        assert short.name == "Straddle" and [leg.quantity for leg in short.legs] == [-2.0, -2.0]
        assert short.price(mkt) == pytest.approx(-2 * straddle.price(mkt))
        assert [leg.quantity for leg in straddle.legs] == [1.0, 1.0]

    def test_observe_maps_over_legs(self):
        vanilla = make_straddle()
        assert vanilla.observe(130.0, 0.5) is vanilla
        nested = CompositeInstrument("n", (Position(FlaggedCall(100, 1.0, 120), 1),))
        combo = CompositeInstrument(
            "combo", (Position(nested, 2), Position(EuropeanOption("put", 90, 1.0), 1))
        )
        after = combo.observe(125.0, 0.3)
        assert after is not combo and after.name == "combo"
        knocked = after.legs[0].instrument.legs[0].instrument
        assert knocked.hit and knocked.observations == (0.3,)
        assert after.legs[1] is combo.legs[1]
        assert after.payoff(np.array([150.0]))[0] == 0.0
        assert combo.payoff(np.array([150.0]))[0] == 100.0

    def test_path_payoff_reads_each_leg_at_its_expiry(self):
        near, far = EuropeanOption("call", 100, 0.5), EuropeanOption("call", 100, 1.0)
        calendar = CompositeInstrument("Calendar", (Position(near, -1), Position(far, 1)))
        times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        paths = np.array([[100.0, 105.0, 110.0, 120.0, 130.0], [100.0, 95.0, 90.0, 100.0, 104.0]])
        np.testing.assert_array_equal(calendar.path_payoff(paths, times), [-10.0 + 30.0, 4.0])

    def test_hashable_and_json_round_trip_with_path_state(self):
        knocked = FlaggedCall(100, 1.0, 120).observe(125.0, 0.3)
        inner = CompositeInstrument("inner", (Position(knocked, 1), Position(Underlying(), -0.5)))
        outer = CompositeInstrument("outer", (Position(inner, 2), Position(EuropeanOption("put", 90, 2.0), -1)))
        text = json.dumps(outer.to_dict())
        restored = instrument_from_dict(json.loads(text))
        assert restored == outer and hash(restored) == hash(outer)
        restored_knocked = restored.legs[0].instrument.legs[0].instrument
        assert restored_knocked.hit is True and restored_knocked.observations == (0.3,)
        assert {outer: "x"}[restored] == "x"


# ---------------------------------------------------------------------- #
# Registry, default Greeks, helpers
# ---------------------------------------------------------------------- #
class TestRegistry:
    def test_builtin_types_are_registered(self):
        assert INSTRUMENT_REGISTRY["EuropeanOption"] is EuropeanOption
        assert INSTRUMENT_REGISTRY["Underlying"] is Underlying
        assert INSTRUMENT_REGISTRY["CompositeInstrument"] is CompositeInstrument
        assert INSTRUMENT_REGISTRY["FlaggedCall"] is FlaggedCall

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown instrument type 'Swaption'"):
            instrument_from_dict({"type": "Swaption", "strike": 1.0})

    def test_malformed_dicts(self):
        with pytest.raises(ValueError, match="'type'"):
            instrument_from_dict({"strike": 100})
        with pytest.raises(ValueError, match="'type'"):
            instrument_from_dict("EuropeanOption")
        with pytest.raises(ValueError, match="unknown field"):
            instrument_from_dict({"type": "EuropeanOption", "option_type": "call", "strike": 1, "expiry": 1, "x": 0})
        with pytest.raises(ValueError, match="cannot build"):
            instrument_from_dict({"type": "EuropeanOption", "option_type": "call"})

    def test_register_rejects_non_instruments(self):
        with pytest.raises(ValueError):
            register_instrument(dict)


def test_default_greeks_are_numerical(mkt):
    toy = FlaggedCall(100, 1.0, 1e9)
    numeric = toy.greeks(mkt)
    analytic = EuropeanOption("call", 100, 1.0).greeks(mkt)
    assert tuple(numeric) == GREEK_KEYS
    for k in GREEK_KEYS:
        assert numeric[k] == pytest.approx(analytic[k], rel=1e-3, abs=1e-7), k
    assert toy.label == "FlaggedCall"
    dead = toy.observe(2e9, 0.1)
    assert all(v == 0.0 for v in dead.greeks(mkt).values())


def test_slice_paths_to_expiry():
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    paths = np.arange(10.0).reshape(2, 5)
    p, t = slice_paths_to_expiry(paths, times, 0.5)
    assert t.tolist() == [0.0, 0.25, 0.5] and p.shape == (2, 3)
    p, t = slice_paths_to_expiry(paths, times, None)
    assert p.shape == (2, 5)
    p, t = slice_paths_to_expiry(paths, times, 0.5 - 1e-13)  # float noise still includes 0.5
    assert t[-1] == 0.5
    p, t = slice_paths_to_expiry(paths, times, -1.0)  # never empty
    assert p.shape == (2, 1)
