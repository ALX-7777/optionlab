"""Tests for the Greeks explorer page (``app_pages/greeks.py``).

The page is driven through Streamlit's headless ``AppTest``: the app is started
from its entry point, the Greeks page is opened, and the widgets it adds are
set the way a user would set them.

One quirk is worth knowing before reading the tests. The page puts its seven
views in ``st.tabs(..., key="greeks_view", on_change="rerun")`` and only
renders the open one, which keeps a rerun to a couple of hundred milliseconds.
``AppTest`` models a tab as a layout block rather than as a widget, so the
selected tab is *not* carried from one ``run()`` to the next: it has to be
seeded in the session state before every run. :func:`run_on` does exactly that,
and every test that needs a view other than the first one goes through it.
"""

from __future__ import annotations

import math

import pytest
from streamlit.testing.v1 import AppTest

from optionlab.plotting import profiles

APP = "../app.py"
PAGE = "app_pages/greeks.py"
VIEW_KEY = "greeks_view"

VIEWS = [
    "Against spot",
    "Through time",
    "Against volatility",
    "Surface",
    "Compare",
    "P&L intuition",
    "Implied vol",
]

# Column headers of the page's tables come from the library, so the tests ask
# the library for them instead of hard-coding the units.
GAMMA_COLUMN = profiles.greek_axis_label("gamma", True)
THETA_COLUMN = profiles.greek_axis_label("theta", True)
VEGA_COLUMN = profiles.greek_axis_label("vega", True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def open_page(view: str | None = None) -> AppTest:
    """Start the app, open the Greeks page and optionally select a view."""
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.switch_page(PAGE).run()
    assert not at.exception, at.exception
    if view is not None:
        at.session_state[VIEW_KEY] = view
        at.run()
    return at


def run_on(at: AppTest, view: str) -> AppTest:
    """Rerun the app with ``view`` open (see the module docstring)."""
    at.session_state[VIEW_KEY] = view
    return at.run()


def chart_keys(at: AppTest) -> list[str]:
    return [element.key for element in at.get("plotly_chart")]


def metric(at: AppTest, label: str) -> str:
    """Value of the metric with this label, as the app formats it."""
    for element in at.metric:
        if element.label == label:
            return element.value
    raise AssertionError(f"no metric labelled {label!r}; got {[m.label for m in at.metric]}")


# --------------------------------------------------------------------------- #
# The page itself
# --------------------------------------------------------------------------- #
def test_page_renders_the_headline_risk() -> None:
    at = open_page()
    assert not at.exception

    # The five headline Greeks, in the library's own labels and trader units.
    for label in ("Price", "Delta", "Gamma", "Vega", "Theta"):
        assert metric(at, label)

    # An at-the-money 30-day call: delta near a half, and a 50/50 chance.
    assert 0.45 < float(metric(at, "Delta")) < 0.60
    assert metric(at, "Chance of finishing in the money") == "50.00%"
    assert metric(at, "Time to expiry") == "30 days"

    # The default view is the spot dashboard plus its single-Greek zoom.
    assert chart_keys(at) == ["greeks_spot_dashboard", "greeks_spot_profile"]


@pytest.mark.parametrize("view", VIEWS)
def test_every_view_renders_a_chart(view: str) -> None:
    at = open_page(view)
    assert not at.exception, f"{view} raised {at.exception}"
    assert chart_keys(at), f"{view} rendered no chart"
    assert all(key.startswith("greeks_") for key in chart_keys(at))


# --------------------------------------------------------------------------- #
# The option builder
# --------------------------------------------------------------------------- #
def test_option_controls_change_the_risk() -> None:
    at = open_page()
    call_delta = float(metric(at, "Delta"))

    at.segmented_control(key="greeks_type").set_value("put").run()
    assert not at.exception
    put_delta = float(metric(at, "Delta"))
    assert put_delta < 0 < call_delta
    # Put-call parity: the two deltas differ by the discounted dividend factor.
    assert call_delta - put_delta == pytest.approx(1.0, abs=0.01)

    # A far out-of-the-money put a week from expiry is almost worthless.
    at.number_input(key="greeks_strike").set_value(70.0).run()
    at.slider(key="greeks_days").set_value(7).run()
    assert not at.exception
    assert float(metric(at, "Price")) < 0.05
    assert metric(at, "Time to expiry") == "7 days"


def test_at_the_money_button_recentres_the_strike() -> None:
    at = open_page()
    at.number_input(key="greeks_strike").set_value(137.0).run()
    assert at.session_state.greeks_strike == 137.0

    at.button(key="greeks_atm").click().run()
    assert not at.exception
    assert at.session_state.greeks_strike == at.session_state.mkt_spot


def test_short_position_flips_every_greek() -> None:
    at = open_page()
    long_gamma = float(metric(at, "Gamma"))

    at.number_input(key="greeks_qty").set_value(-2.0).run()
    assert not at.exception
    assert float(metric(at, "Gamma")) == pytest.approx(-2.0 * long_gamma, rel=1e-6)
    # Short an option and theta is collected rather than paid.
    assert float(metric(at, "Theta")) > 0


def test_position_size_of_zero_is_explained_not_crashed() -> None:
    at = open_page()
    at.number_input(key="greeks_qty").set_value(0.0).run()
    assert not at.exception
    assert any("zero" in warning.value for warning in at.warning)
    # The page still shows a full set of Greeks, for one option.
    assert float(metric(at, "Gamma")) > 0


# --------------------------------------------------------------------------- #
# The views
# --------------------------------------------------------------------------- #
def test_spot_view_zoom_follows_the_greek_picker() -> None:
    at = open_page()
    at.segmented_control(key="greeks_spot_greek").set_value("gamma").run()
    assert not at.exception

    captions = [caption.value for caption in at.caption]
    assert any("delta changes when the spot moves" in text for text in captions), captions
    assert "greeks_spot_profile" in chart_keys(at)


def test_decay_table_shows_the_at_the_money_blow_up() -> None:
    at = open_page("Through time")
    table = at.dataframe[0].value

    assert list(table["Contract"]) == [
        "1 year",
        "6 months",
        "3 months",
        "1 month",
        "1 week",
        "1 day",
    ]
    # Gamma grows and theta falls monotonically as the expiry approaches; vega
    # does the opposite. That is the whole lesson of the view.
    assert table[GAMMA_COLUMN].is_monotonic_increasing
    assert table[THETA_COLUMN].is_monotonic_decreasing
    assert table[VEGA_COLUMN].is_monotonic_decreasing
    assert table[GAMMA_COLUMN].iloc[-1] > 10 * table[GAMMA_COLUMN].iloc[0]


def test_time_view_greek_picker_redraws_both_charts() -> None:
    at = open_page("Through time")
    assert chart_keys(at) == ["greeks_time_evolution", "greeks_time_vs_time"]

    at.segmented_control(key="greeks_time_greek").set_value("theta")
    run_on(at, "Through time")
    assert not at.exception
    assert chart_keys(at) == ["greeks_time_evolution", "greeks_time_vs_time"]
    captions = [caption.value for caption in at.caption]
    assert any("passage of time" in text for text in captions), captions


def test_surface_view_switches_between_axes_and_styles() -> None:
    at = open_page("Surface")
    assert chart_keys(at) == ["greeks_surface"]

    at.segmented_control(key="greeks_surface_axis").set_value("Volatility")
    at.segmented_control(key="greeks_surface_kind").set_value("Heatmap")
    at.segmented_control(key="greeks_surface_greek").set_value("vega")
    run_on(at, "Surface")
    assert not at.exception
    assert chart_keys(at) == ["greeks_surface"]
    assert any("Vega over spot and volatility" in text for text in (s.value for s in at.subheader))


def test_compare_view_builds_a_strike_ladder() -> None:
    at = open_page("Compare")
    # The default comparison is the call against the put at the same strike.
    assert list(at.dataframe[0].value["Contract"]) == ["Call 100", "Put 100"]

    at.segmented_control(key="greeks_compare_mode").set_value("Strike ladder")
    run_on(at, "Compare")
    assert not at.exception

    contracts = list(at.dataframe[0].value["Contract"])
    assert [name.split("(")[-1].rstrip(")") for name in contracts] == ["90%", "100%", "110%"]
    assert "greeks_compare_chart" in chart_keys(at)


def test_compare_view_asks_for_a_greek_instead_of_crashing() -> None:
    at = open_page("Compare")
    at.pills(key="greeks_compare_greeks").set_value([])
    run_on(at, "Compare")
    assert not at.exception
    assert any("at least one Greek" in info.value for info in at.info)
    assert "greeks_compare_chart" not in chart_keys(at)


def test_pnl_view_applies_the_scenario_to_the_break_even() -> None:
    at = open_page("P&L intuition")
    assert chart_keys(at) == [
        "greeks_pnl_profile",
        "greeks_pnl_taylor",
        "greeks_pnl_gamma_theta",
    ]
    one_day_breakeven = float(metric(at, "Break-even move over the period"))

    at.slider(key="greeks_period").set_value(9)
    run_on(at, "P&L intuition")
    assert not at.exception

    nine_day_breakeven = float(metric(at, "Break-even move over the period"))
    # The break-even move grows with the square root of the period.
    assert nine_day_breakeven == pytest.approx(3.0 * one_day_breakeven, rel=0.05)
    # And it lands within a few percent of one implied standard deviation —
    # which is what "the option is priced at that volatility" means.
    implied_move = float(metric(at, "Implied move over the period"))
    assert nine_day_breakeven == pytest.approx(implied_move, rel=0.1)


def test_implied_vol_tool_recovers_the_sidebar_volatility() -> None:
    at = open_page("Implied vol")
    implied = float(metric(at, "Implied volatility").rstrip("%"))
    # The input starts at the model price rounded to two decimals, so the
    # solver must return the sidebar volatility give or take the rounding.
    assert implied == pytest.approx(20.0, abs=0.5)
    assert metric(at, "Sidebar volatility") == "20.00%"
    assert "greeks_iv_curve" in chart_keys(at)


def test_implied_vol_tool_rejects_an_impossible_price() -> None:
    at = open_page("Implied vol")
    at.number_input(key="greeks_market_price").set_value(500.0)
    run_on(at, "Implied vol")
    assert not at.exception
    assert any("No volatility reproduces" in error.value for error in at.error)

    # And it recovers: the button puts the model price back.
    at.button(key="greeks_reset_price").click()
    run_on(at, "Implied vol")
    assert not at.exception
    assert not at.error
    assert float(metric(at, "Implied volatility").rstrip("%")) == pytest.approx(20.0, abs=0.5)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("view", VIEWS)
def test_an_option_one_day_from_expiry_still_renders(view: str) -> None:
    """The hardest case for every figure: gamma and theta are exploding."""
    at = open_page()
    at.slider(key="greeks_days").set_value(1)
    run_on(at, view)
    assert not at.exception, f"{view} raised {at.exception}"
    assert chart_keys(at) or at.warning, f"{view} showed neither a chart nor a warning"


def test_an_extreme_market_does_not_break_the_page() -> None:
    """A penny stock at 150% volatility, deep in the money, two years out."""
    at = open_page()
    at.sidebar.number_input(key="mkt_spot").set_value(1.0).run()
    at.sidebar.slider(key="mkt_vol_pct").set_value(150.0).run()
    at.number_input(key="greeks_strike").set_value(0.25)
    at.slider(key="greeks_days").set_value(730)
    at.run()
    assert not at.exception

    for view in VIEWS:
        run_on(at, view)
        assert not at.exception, f"{view} raised {at.exception}"


def test_raw_units_change_the_table_headers() -> None:
    at = open_page()
    at.sidebar.toggle(key="trader_units").set_value(False)
    run_on(at, "Through time")
    assert not at.exception

    columns = list(at.dataframe[0].value.columns)
    assert profiles.greek_axis_label("theta", False) in columns
    assert profiles.greek_axis_label("theta", True) not in columns
    # Raw theta is per year, so it is 365 times the per-day number.
    raw_theta = float(at.dataframe[0].value[profiles.greek_axis_label("theta", False)].iloc[0])
    assert raw_theta < -1.0
    assert math.isfinite(raw_theta)
