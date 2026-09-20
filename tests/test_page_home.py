"""Tests for the "Start here" landing page (``app_pages/home.py``).

The page is driven through the real app shell with Streamlit's ``AppTest``, so
``st.navigation`` and the sidebar market behave exactly as they do in a browser.

What is covered:

* the page renders with no exception and reacts to the shared market;
* the live snapshot quotes the library's own price and Greeks for the
  at-the-money call and put;
* both controls the page adds (maturity, Greek scope) change what is shown;
* the guards of rule 11: a deselected segmented control falls back to its
  default instead of raising.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from app_lib import ui
from optionlab import GREEK_INFO, EuropeanOption, Market, to_trader_units

APP = "../app.py"
PAGE = "app_pages/home.py"

#: Pages the landing page must link to, so nobody silently loses a destination.
#: ``st.page_link`` resolves the file path to the page's url path, which for
#: these pages is the file stem.
LINKED_PAGES = ("greeks", "strategies", "exotics", "book", "simulator", "interview")


def open_home() -> AppTest:
    """Start the app shell on the default market, then switch to the landing page."""
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.switch_page(PAGE).run()
    return at


def metric_values(at: AppTest, label: str) -> list[str]:
    return [m.value for m in at.metric if m.label == label]


def all_text(at: AppTest) -> str:
    """Every markdown and caption string on the page, concatenated."""
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption])


@pytest.fixture(scope="module")
def home() -> AppTest:
    return open_home()


# --------------------------------------------------------------------------- #
# The page renders
# --------------------------------------------------------------------------- #
def test_page_runs(home: AppTest) -> None:
    assert not home.exception
    assert home.title[0].value == "OptionLab"


def test_the_two_controls_are_present(home: AppTest) -> None:
    assert home.segmented_control(key="home_tenor").value == "3 months"
    assert home.segmented_control(key="home_greek_scope").value == "Headline five"


def test_market_snapshot_follows_the_sidebar(home: AppTest) -> None:
    assert metric_values(home, "Spot") == ["100.00"]
    assert metric_values(home, "Implied volatility") == ["20.0%"]
    assert metric_values(home, "Interest rate") == ["3.00%"]
    assert metric_values(home, "Dividend yield") == ["1.00%"]


def test_every_destination_page_is_linked(home: AppTest) -> None:
    links = home.get("page_link")
    targets = {link.proto.page for link in links}
    assert targets == set(LINKED_PAGES)
    # Six cards, four drills and the closing call to action.
    assert len(links) == 11
    assert all(link.proto.label for link in links)


def test_the_teaching_boxes_are_there(home: AppTest) -> None:
    text = all_text(home)
    # ui.explain renders an expander carrying an icon, which AppTest calls a status.
    labels = [s.label for s in home.status]
    # Every page opens with the same orientation box, in the same place.
    assert labels[0] == "How to use this page"
    assert "How to read this" in labels
    assert text.count("Interview angle") >= 2
    assert "What to notice" in text


# --------------------------------------------------------------------------- #
# The numbers are the library's numbers
# --------------------------------------------------------------------------- #
def test_snapshot_quotes_the_library_price_and_greeks(home: AppTest) -> None:
    mkt = Market(spot=100.0, vol=0.20, rate=0.03, div=0.01)
    call = to_trader_units(dict(EuropeanOption("call", 100.0, 0.25).greeks(mkt)))
    put = to_trader_units(dict(EuropeanOption("put", 100.0, 0.25).greeks(mkt)))

    # The call row is rendered first, the put row second.
    assert metric_values(home, "Price") == [f"{call['price']:,.2f}", f"{put['price']:,.2f}"]
    assert metric_values(home, "Delta") == [f"{call['delta']:,.3f}", f"{put['delta']:,.3f}"]
    # A call and a put on the same strike share their gamma and their vega.
    assert len(set(metric_values(home, "Gamma"))) == 1
    assert len(set(metric_values(home, "Vega"))) == 1


def test_put_call_parity_is_stated_with_the_computed_numbers(home: AppTest) -> None:
    mkt = Market(spot=100.0, vol=0.20, rate=0.03, div=0.01)
    call = EuropeanOption("call", 100.0, 0.25).price(mkt)
    put = EuropeanOption("put", 100.0, 0.25).price(mkt)
    parity = mkt.discount_factor(0.25) * (mkt.forward(0.25) - 100.0)
    assert call - put == pytest.approx(parity, abs=1e-9)
    assert f"+{parity:,.2f}" in all_text(home)


# --------------------------------------------------------------------------- #
# Interactions
# --------------------------------------------------------------------------- #
def test_changing_the_maturity_reprices_the_snapshot() -> None:
    at = open_home()
    three_month = metric_values(at, "Price")

    at.segmented_control(key="home_tenor").set_value("1 year").run()
    assert not at.exception

    one_year = metric_values(at, "Price")
    assert one_year != three_month

    mkt = Market(spot=100.0, vol=0.20, rate=0.03, div=0.01)
    expected = EuropeanOption("call", 100.0, 1.0).price(mkt)
    assert one_year[0] == f"{expected:,.2f}"
    # A longer option is worth more: that is the whole point of the control.
    assert float(one_year[0]) > float(three_month[0])


def test_greek_scope_switches_between_five_rows_and_twelve() -> None:
    at = open_home()
    assert len(at.dataframe[0].value) == len(ui.HEADLINE_GREEKS)

    at.segmented_control(key="home_greek_scope").set_value("All twelve").run()
    assert not at.exception

    table = at.dataframe[0].value
    assert len(table) == len(GREEK_INFO)
    assert set(table["Greek"]) == {info["label"] for info in GREEK_INFO.values()}
    # Every sentence comes from the library, never from the page.
    for name, info in GREEK_INFO.items():
        row = table.loc[table["Greek"] == info["label"], "In one sentence"].iloc[0]
        assert info["description"].startswith(row.rstrip(".")), name


def test_trader_units_toggle_changes_the_quoted_vega() -> None:
    at = open_home()
    trader_vega = metric_values(at, "Vega")[0]

    at.sidebar.toggle(key="trader_units").set_value(False).run()
    assert not at.exception

    raw_vega = metric_values(at, "Vega")[0]
    assert raw_vega != trader_vega
    # Raw vega is per 1.00 of volatility, so a hundred times the desk number.
    assert float(raw_vega.replace(",", "")) == pytest.approx(
        float(trader_vega.replace(",", "")) * 100.0, rel=1e-3
    )
    assert "Raw derivatives" in all_text(at)


def test_a_different_market_reprices_the_whole_page() -> None:
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.sidebar.number_input(key="mkt_spot").set_value(250.0).run()
    at.sidebar.slider(key="mkt_vol_pct").set_value(40.0).run()
    at.switch_page(PAGE).run()
    assert not at.exception

    assert metric_values(at, "Spot") == ["250.00"]
    assert metric_values(at, "Implied volatility") == ["40.0%"]

    mkt = Market(spot=250.0, vol=0.40, rate=0.03, div=0.01)
    expected = EuropeanOption("call", 250.0, 0.25).price(mkt)
    assert metric_values(at, "Price")[0] == f"{expected:,.2f}"


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_deselecting_the_maturity_falls_back_to_the_default() -> None:
    # Clicking the selected segment a second time clears it, which Streamlit
    # reports as None. The page must show the default instead of crashing.
    at = open_home()
    at.session_state["home_tenor"] = None
    at.run()

    assert not at.exception
    assert "No maturity selected" in all_text(at)
    mkt = Market(spot=100.0, vol=0.20, rate=0.03, div=0.01)
    expected = EuropeanOption("call", 100.0, 0.25).price(mkt)
    assert metric_values(at, "Price")[0] == f"{expected:,.2f}"


def test_deselecting_the_greek_scope_keeps_the_headline_table() -> None:
    at = open_home()
    at.session_state["home_greek_scope"] = None
    at.run()

    assert not at.exception
    assert len(at.dataframe[0].value) == 5


def test_an_extreme_market_still_renders() -> None:
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.sidebar.number_input(key="mkt_spot").set_value(1.0).run()
    at.sidebar.slider(key="mkt_vol_pct").set_value(150.0).run()
    at.sidebar.slider(key="mkt_div_pct").set_value(10.0).run()
    at.switch_page(PAGE).run()

    assert not at.exception
    assert not at.error
    # A ten percent dividend against a three percent rate puts the forward
    # below the strike, so the put is now the dearer of the two.
    prices = [float(value) for value in metric_values(at, "Price")]
    assert prices[1] > prices[0]
