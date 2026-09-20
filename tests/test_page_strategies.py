"""Tests for the "Strategies" page (``app_pages/strategies.py``).

They drive the page the way a user does -- pick a family, pick a structure,
change a strike, switch chart tabs, compare with a second structure and trade
it into the book -- and check both that nothing raises and that what renders
actually changes. Bad inputs (strikes in the wrong order, a ratio of one, no
Greek selected) must produce a readable message, never a traceback.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from optionlab import STRATEGY_REGISTRY

APP = "../app.py"
PAGE = "app_pages/strategies.py"


def open_page() -> AppTest:
    """Boot the app and switch to the strategies page (AppTest needs the entry point)."""
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.switch_page(PAGE).run()
    assert not at.exception, at.exception
    return at


def metric(at: AppTest, label: str) -> str:
    """Value of the first metric carrying that label."""
    for element in at.metric:
        if element.label == label:
            return element.value
    raise AssertionError(f"no metric labelled {label!r}; got {[m.label for m in at.metric]}")


def page_text(at: AppTest) -> str:
    """Everything the page wrote as markdown or caption, for substring checks."""
    return "\n".join([element.value for element in at.markdown] + [element.value for element in at.caption])


def comparison_table(at: AppTest):
    """The side-by-side table; it is the first dataframe of the page, the legs the second."""
    frame = at.dataframe[0].value
    assert list(frame.columns)[0] == "Measure"
    return frame


def legs_table(at: AppTest):
    frame = at.dataframe[1].value
    assert list(frame.columns)[0] == "Leg"
    return frame


def run_on_tab(at: AppTest, tab: str) -> AppTest:
    """Rerun with a given chart tab open.

    Tabs are lazy (``on_change="rerun"``), so only the open tab's chart is
    built; AppTest cannot click a tab, but it can seed the tab widget's state
    -- which has to be done before every rerun.
    """
    at.session_state["strategies_chart_tab"] = tab
    at.run()
    assert not at.exception, at.exception
    return at


def select_strategy(at: AppTest, category: str, name: str) -> AppTest:
    """Pick a family, then a structure inside it."""
    at.get_by_key("strategies_category").set_value(category).run()
    at.selectbox(key="strategies_name").set_value(name).run()
    assert not at.exception, at.exception
    return at


@pytest.fixture(scope="module")
def page() -> AppTest:
    return open_page()


# --------------------------------------------------------------------------- #
# The page itself
# --------------------------------------------------------------------------- #
def test_page_runs_and_describes_the_default_strategy(page: AppTest) -> None:
    assert not page.exception
    text = page_text(page)
    assert "Long straddle" in text
    # The teaching text comes from the library, not from the page.
    assert STRATEGY_REGISTRY["long_straddle"].description in text


def test_headline_numbers_are_present(page: AppTest) -> None:
    # A bought straddle is a debit with an unlimited upside and two breakevens.
    assert metric(page, "Net debit") == "15.69"
    assert metric(page, "Max profit") == "unlimited"
    assert metric(page, "Max loss") == "-15.69"
    assert "/" in metric(page, "Breakevens")
    assert metric(page, "Chance of profit").endswith("%")
    # ... and the Greeks row is there too.
    assert metric(page, "Vega")


def test_every_section_renders(page: AppTest) -> None:
    labels = [tab.label for tab in page.tabs]
    assert labels == ["Payoff", "Greeks by leg", "Time decay", "Vol sensitivity"]
    assert page.get("plotly_chart"), "the payoff chart should be drawn"
    assert comparison_table(page) is not None
    assert len(legs_table(page)) == 2  # a straddle has two legs
    # The parameter form and the trade ticket are both there.
    assert any(button.label == "Update strategy" for button in page.button)
    assert page.button(key="strategies_trade_add") is not None


# --------------------------------------------------------------------------- #
# Picking a strategy
# --------------------------------------------------------------------------- #
def test_choosing_another_family_changes_the_structure() -> None:
    at = select_strategy(open_page(), "Vertical spread", "bull_call_spread")
    text = page_text(at)
    assert "Bull call spread" in text
    assert STRATEGY_REGISTRY["bull_call_spread"].description in text
    # A debit spread: bounded on both sides, unlike the straddle above.
    assert metric(at, "Net debit")
    assert metric(at, "Max profit") != "unlimited"
    assert metric(at, "Max loss") != "unlimited"


def test_a_calendar_is_shown_at_its_front_expiry() -> None:
    at = select_strategy(open_page(), "Time spread", "calendar_spread")
    text = page_text(at)
    # The page must say that the back leg is valued with a model assumption.
    assert "front expiry" in text
    assert "model assumption" in text
    # The back leg outlives the horizon, so the loss is not capped there.
    assert metric(at, "Max loss") == "unlimited"
    assert len(legs_table(at)) == 2


def test_every_registered_structure_prices_on_the_default_market() -> None:
    """All 37, one after another, the way "Surprise me" can reach any of them.

    Changing family leaves the previous structure's name in session state, which
    is no longer in the new option list; Streamlit falls back to the index, and
    this test is what proves the page survives that on every family.
    """
    at = open_page()
    broken: list[str] = []
    for key, spec in STRATEGY_REGISTRY.items():
        at.get_by_key("strategies_category").set_value(spec.category).run()
        at.selectbox(key="strategies_name").set_value(key).run()
        if at.exception:
            broken.append(f"{key}: {at.exception[0].message.splitlines()[0]}")
            continue
        assert at.session_state["strategies_name"] == key
        assert spec.title in page_text(at), key
        # Every structure quotes a premium and a worst case, whatever its shape.
        assert metric(at, "Max loss")
    assert not broken, broken


def test_surprise_me_lands_on_a_registered_strategy() -> None:
    at = open_page()
    at.button(key="strategies_surprise").click().run()
    assert not at.exception
    assert at.session_state["strategies_name"] in STRATEGY_REGISTRY
    spec = STRATEGY_REGISTRY[at.session_state["strategies_name"]]
    assert at.session_state["strategies_category"] == spec.category
    assert spec.title in page_text(at)


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
def test_moving_a_strike_reprices_the_package() -> None:
    at = select_strategy(open_page(), "Vertical spread", "bull_call_spread")
    before = metric(at, "Net debit")
    at.number_input(key="strategies_bull_call_spread_high_strike").set_value(120.0).run()
    assert not at.exception
    after = metric(at, "Net debit")
    # A wider spread buys more upside, so it costs more.
    assert float(after.replace(",", "")) > float(before.replace(",", ""))
    assert "120.00" in page_text(at)
    # The legs table follows the new strike.
    assert 120.0 in set(legs_table(at)["Strike"])


def test_strikes_in_the_wrong_order_warn_instead_of_crashing() -> None:
    at = select_strategy(open_page(), "Vertical spread", "bull_call_spread")
    at.number_input(key="strategies_bull_call_spread_low_strike").set_value(130.0).run()
    assert not at.exception
    assert at.warning, "a bad strike order should show a warning"
    assert "strictly increasing" in at.warning[0].value
    # The page stops cleanly: no numbers are shown for an impossible structure.
    assert not at.metric


def test_a_ratio_of_one_warns_instead_of_crashing() -> None:
    at = select_strategy(open_page(), "Ratio & backspread", "call_ratio_spread")
    at.number_input(key="strategies_call_ratio_spread_ratio").set_value(1.0).run()
    assert not at.exception
    assert at.warning
    assert "ratio" in at.warning[0].value


# --------------------------------------------------------------------------- #
# The charts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("tab", "explanation"),
    [
        ("Payoff", "The **bold line** is the P&L at expiry"),
        ("Greeks by leg", "One panel per Greek"),
        ("Time decay", "time is what sharpens it"),
        ("Vol sensitivity", "is the vega there"),
    ],
)
def test_each_chart_tab_draws_an_explained_chart(tab: str, explanation: str) -> None:
    at = run_on_tab(open_page(), tab)
    assert at.get("plotly_chart"), f"{tab} drew no chart"
    assert not at.warning
    assert explanation in page_text(at), f"{tab} is not explained"


def test_the_payoff_tab_can_hide_the_legs() -> None:
    at = open_page()
    at.toggle(key="strategies_payoff_legs").set_value(False).run()
    assert not at.exception
    assert at.session_state["strategies_payoff_legs"] is False
    assert at.get("plotly_chart")


def test_the_greeks_tab_asks_for_at_least_one_greek() -> None:
    at = run_on_tab(open_page(), "Greeks by leg")
    charts_before = len(at.get("plotly_chart"))
    assert at.get_by_key("strategies_greek_pick").value == ["delta", "gamma", "vega", "theta"]

    at.get_by_key("strategies_greek_pick").set_value([])
    run_on_tab(at, "Greeks by leg")
    assert any("at least one Greek" in element.value for element in at.info)
    assert len(at.get("plotly_chart")) < charts_before


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def test_the_comparison_follows_the_second_strategy() -> None:
    at = open_page()
    assert list(comparison_table(at).columns) == ["Measure", "Long straddle", "Iron condor"]

    at.selectbox(key="strategies_compare_name").set_value("covered_call").run()
    assert not at.exception
    assert list(comparison_table(at).columns) == ["Measure", "Long straddle", "Covered call"]
    # The commentary is generated from the two sets of numbers.
    assert "**Cost.**" in page_text(at)
    assert "Covered call" in page_text(at)


def test_the_comparison_can_be_drawn_on_a_greek() -> None:
    at = open_page()
    charts = len(at.get("plotly_chart"))
    at.get_by_key("strategies_compare_what").set_value("Vega").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == charts
    assert not at.warning


def test_a_narrow_strike_width_still_builds() -> None:
    at = open_page()
    wide = comparison_table(at).set_index("Measure")["Iron condor"]["Max loss"]
    at.slider(key="strategies_compare_width").set_value(0.25).run()
    assert not at.exception
    narrow = comparison_table(at).set_index("Measure")["Iron condor"]["Max loss"]
    # Pulling the strikes in around the spot narrows the wings, so the worst case shrinks.
    assert narrow != wide


# --------------------------------------------------------------------------- #
# The shared market and the units toggle
# --------------------------------------------------------------------------- #
def test_the_page_follows_the_shared_market() -> None:
    at = open_page()
    before = metric(at, "Net debit")
    at.sidebar.slider(key="mkt_vol_pct").set_value(40.0).run()
    assert not at.exception
    # A long straddle is long vega, so doubling implied vol makes it dearer.
    assert float(metric(at, "Net debit").replace(",", "")) > float(before.replace(",", ""))
    assert "40.00%" in page_text(at)


def test_raw_units_change_the_quoted_greeks() -> None:
    at = open_page()
    trader_vega = metric(at, "Vega")
    at.sidebar.toggle(key="trader_units").set_value(False).run()
    assert not at.exception
    assert metric(at, "Vega") != trader_vega
    assert "per 1.00 of volatility" in page_text(at)
    assert "Raw derivatives." in page_text(at)


# --------------------------------------------------------------------------- #
# Trading it into the book
# --------------------------------------------------------------------------- #
def test_adding_the_strategy_to_the_book() -> None:
    at = open_page()
    assert at.session_state.book.is_empty

    at.button(key="strategies_trade_add").click().run()
    assert not at.exception
    assert at.success, "a confirmation should be shown"
    book = at.session_state.book
    assert not book.is_empty
    assert len(book.positions) == 2  # the two legs of the straddle
    assert book.quantity(book.positions[0].instrument) == 10.0


def test_selling_the_strategy_flips_the_sign() -> None:
    at = open_page()
    at.get_by_key("strategies_trade_side").set_value("Sell").run()
    at.number_input(key="strategies_trade_qty").set_value(5.0).run()
    at.button(key="strategies_trade_add").click().run()
    assert not at.exception
    book = at.session_state.book
    assert book.quantity(book.positions[0].instrument) == -5.0
    # Selling a debit package brings cash in.
    assert book.cash > 100_000.0
