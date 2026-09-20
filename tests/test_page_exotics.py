"""Tests for the Exotics page (``app_pages/exotics.py``).

They drive the page the way a user does — pick a family, pick a product, change a
contract, switch view — and assert on what is rendered: the price against the
vanilla benchmark, the family-specific metric, the Monte Carlo verdict and the
error messages that guard bad inputs.

Everything runs headless through Streamlit's ``AppTest``: no browser, no server.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

APP = "../app.py"
PAGE = "app_pages/exotics.py"

#: The metric each family adds next to the price comparison.
FAMILY_METRIC = {
    "Digital": "Probability in the money",
    "Barrier": "The other half",
    "Asian": "Volatility of the average",
    "Lookback": "Cost of hindsight",
}

VIEWS = ("Against the vanilla", "Inside the product", "Model check")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def open_page(family: str | None = None, product: str | None = None) -> AppTest:
    """Run the app, switch to the Exotics page and optionally pick a product."""
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.switch_page(PAGE).run()
    if family is not None:
        at.segmented_control(key="exotics_family").set_value(family).run()
        if product is not None:
            at.selectbox(key=f"exotics_product_{family.lower()}").set_value(product).run()
    return at


def metrics(at: AppTest) -> dict[str, str]:
    """Rendered metrics as ``{label: value}`` (first occurrence wins)."""
    found: dict[str, str] = {}
    for element in at.metric:
        found.setdefault(element.label, element.value)
    return found


def number(at: AppTest, label: str) -> float:
    """The value of one metric, parsed back into a number."""
    return float(metrics(at)[label].replace(",", "").rstrip("%"))


@pytest.fixture(scope="module")
def page() -> AppTest:
    """One run of the page with its defaults (an up-and-out barrier call)."""
    return open_page()


# --------------------------------------------------------------------------- #
# The page renders
# --------------------------------------------------------------------------- #
def test_page_runs_and_prices_the_default_contract(page: AppTest) -> None:
    assert not page.exception
    shown = metrics(page)
    assert "Vanilla benchmark" in shown
    assert "Exotic / vanilla" in shown
    # The default contract is a knock-out, so it must be cheaper than its vanilla.
    assert float(shown["Up-and-out barrier"]) < float(shown["Vanilla benchmark"])


def test_greeks_are_compared_with_the_vanilla(page: AppTest) -> None:
    table = page.dataframe[0].value
    assert list(table.columns) == [
        "Greek",
        "Unit",
        "Up-and-out barrier",
        "Vanilla",
        "Difference",
    ]
    assert list(table["Greek"]) == ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"]


def test_the_page_teaches(page: AppTest) -> None:
    """Every page owes the reader explanations and at least two interview notes."""
    body = [element.value for element in page.markdown]
    assert sum("Interview angle" in text for text in body) >= 2
    assert any("What to notice" in text for text in body)


@pytest.mark.parametrize("family", list(FAMILY_METRIC))
def test_every_family_adds_its_own_metric(family: str) -> None:
    at = open_page(family)
    assert not at.exception
    shown = metrics(at)
    assert FAMILY_METRIC[family] in shown, shown
    assert "Vanilla benchmark" in shown


def test_switching_product_reprices_the_page() -> None:
    """The knock-in is worth more than the knock-out: it keeps the paths that rally."""
    at = open_page("Barrier", "up_and_out_barrier")
    knock_out = number(at, "Up-and-out barrier")

    at.selectbox(key="exotics_product_barrier").set_value("up_and_in_barrier").run()
    assert not at.exception
    knock_in = number(at, "Up-and-in barrier")

    vanilla = number(at, "Vanilla benchmark")
    assert knock_in > knock_out
    assert knock_out + knock_in == pytest.approx(vanilla, abs=1e-6)  # in-out parity


# --------------------------------------------------------------------------- #
# The three views
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("view", VIEWS)
def test_each_view_renders_a_figure(view: str) -> None:
    at = open_page()
    at.segmented_control(key="exotics_view").set_value(view).run()
    assert not at.exception, at.exception
    # Every view draws at least one optionlab figure (the model check draws none,
    # but it does show its verdict).
    assert at.get("plotly_chart") or at.success or at.warning or at.info


def test_model_check_agrees_with_the_closed_form() -> None:
    at = open_page("Barrier", "up_and_out_barrier")
    at.segmented_control(key="exotics_view").set_value("Model check").run()
    assert not at.exception

    shown = metrics(at)
    assert "Monte Carlo" in shown
    assert "Standard error" in shown
    # The simulation watches the barrier at a finite number of dates, so the page
    # compares it with the continuity-corrected closed form, and they must agree.
    assert "Closed form, monitoring-adjusted" in shown
    assert at.success, "the simulation should land within three standard errors"

    coarse = number(at, "Standard error")
    at.select_slider(key="exotics_mc_paths").set_value(5_000)
    at.button(key="exotics_mc_run").click().run()
    assert not at.exception
    assert number(at, "Standard error") > coarse  # fewer paths, more noise


def test_lookback_monte_carlo_without_the_bridge_is_biased_low() -> None:
    """Reading the extreme on grid dates only misses the excursions in between."""
    at = open_page("Lookback", "floating_lookback")
    at.segmented_control(key="exotics_view").set_value("Model check").run()
    assert at.success  # the Brownian bridge is on by default and agrees

    at.checkbox(key="exotics_mc_bridge").uncheck().run()
    assert not at.exception
    assert number(at, "Monte Carlo") < number(at, "Closed form")
    assert at.warning, "a discretisation bias is not sampling noise"


def test_digital_hedge_gets_harder_into_expiry() -> None:
    at = open_page("Digital", "cash_digital")
    at.segmented_control(key="exotics_view").set_value("Inside the product").run()
    assert not at.exception

    at.slider(key="exotics_digital_pin").set_value(30).run()
    far = number(at, "Delta of the digital")
    at.slider(key="exotics_digital_pin").set_value(1).run()
    near = number(at, "Delta of the digital")

    assert near > far, "the delta of a digital explodes as expiry approaches"
    # The replicating spread is bounded by payout / width, the digital is not.
    assert number(at, "Delta of the spread") <= number(at, "Options in the hedge") + 1e-6


def test_lookback_view_prices_a_new_running_extreme() -> None:
    at = open_page("Lookback", "floating_lookback")
    at.segmented_control(key="exotics_view").set_value("Inside the product").run()
    assert not at.exception
    assert number(at, "Value of that print") > 0


# --------------------------------------------------------------------------- #
# Path state
# --------------------------------------------------------------------------- #
def test_knocked_out_barrier_is_worth_nothing() -> None:
    at = open_page("Barrier", "up_and_out_barrier")
    assert number(at, "Up-and-out barrier") > 0

    at.toggle(key="exotics_knocked").set_value(True).run()
    assert not at.exception
    assert number(at, "Up-and-out barrier") == pytest.approx(0.0)
    # The knock-in twin has become the vanilla: parity still holds after the touch.
    assert number(at, "The other half") == pytest.approx(number(at, "Vanilla benchmark"))


def test_seasoning_an_asian_damps_it() -> None:
    """Half the window already printed below the strike: less time value, lower price."""
    at = open_page("Asian", "asian")
    fresh = number(at, "Asian (average price)")

    at.slider(key="exotics_asian_seasoned").set_value(180).run()
    at.slider(key="exotics_asian_average").set_value(90).run()
    assert not at.exception
    assert number(at, "Asian (average price)") < fresh

    at.segmented_control(key="exotics_view").set_value("Inside the product").run()
    assert not at.exception
    assert number(at, "Still random") < 100  # part of the average is fixed for good


def test_lookback_remembers_a_lower_extreme() -> None:
    at = open_page("Lookback", "floating_lookback")
    fresh = number(at, "Floating-strike lookback")

    at.slider(key="exotics_lookback_min").set_value(80).run()
    assert not at.exception
    assert number(at, "Floating-strike lookback") > fresh


# --------------------------------------------------------------------------- #
# Bad inputs are handled, never raised
# --------------------------------------------------------------------------- #
def test_averaging_window_starting_after_expiry_shows_an_error() -> None:
    at = open_page("Asian", "asian")
    at.slider(key="exotics_asian_avg_start").set_value(500).run()  # expiry defaults to 365 days
    assert not at.exception, "a bad contract must not raise"
    assert at.error
    assert "avg_start" in at.error[0].value


def test_barrier_already_beyond_the_spot_warns() -> None:
    at = open_page("Barrier", "up_and_out_barrier")
    at.number_input(key="exotics_up_and_out_barrier_barrier").set_value(90.0).run()
    assert not at.exception
    assert at.warning
    assert "already" in at.warning[0].value
    assert number(at, "Up-and-out barrier") == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# The ticket
# --------------------------------------------------------------------------- #
def test_adding_to_the_book_moves_cash() -> None:
    at = open_page("Barrier", "up_and_out_barrier")
    cash_before = at.session_state.book.cash

    at.button(key="exotics_add").click().run()
    assert not at.exception
    assert at.success
    book = at.session_state.book
    assert len(book.positions) == 1
    assert book.cash < cash_before  # ten contracts bought, cash paid away
