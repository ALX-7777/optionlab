"""Tests for the "My book" page (``app_pages/book.py``).

The page is driven through Streamlit's headless AppTest: the app is started
from its entry point, the book page is opened, and the widgets it adds are
exercised one by one. What is asserted is what a user would see — the headline
metrics, the positions table, the risk charts — plus the state of the book in
session state, which is what the rest of the app shares.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from optionlab import Book, EuropeanOption, Market, Underlying

APP = "../app.py"
PAGE = "app_pages/book.py"


def open_page() -> AppTest:
    """A fresh app, already on the book page, with an empty book."""
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.switch_page(PAGE).run()
    assert not at.exception
    return at


def with_sample_book() -> AppTest:
    """The book page with the sample position loaded (six netted lines)."""
    at = open_page()
    at.button(key="book_sample").click().run()
    assert not at.exception
    return at


def metric_value(at: AppTest, label: str) -> str:
    """The value of the first metric carrying ``label``."""
    for metric in at.metric:
        if metric.label == label:
            return metric.value
    raise AssertionError(f"no metric labelled {label!r}")


def market_of(at: AppTest) -> Market:
    """The sidebar market as the page sees it, rebuilt from session state."""
    return Market(
        spot=at.session_state.mkt_spot,
        vol=at.session_state.mkt_vol_pct / 100.0,
        rate=at.session_state.mkt_rate_pct / 100.0,
        div=at.session_state.mkt_div_pct / 100.0,
    )


# --------------------------------------------------------------------------- #
# The page itself
# --------------------------------------------------------------------------- #
def test_empty_book_renders_headline_and_offers_a_sample() -> None:
    at = open_page()

    assert metric_value(at, "Net liquidation value") == "100,000.00"
    assert metric_value(at, "Open lines") == "0"
    for label in ("Delta cash", "Gamma cash (per 1%)", "Vega (per vol point)", "Theta (per day)"):
        assert metric_value(at, label) == "0"
    # An empty book explains itself instead of drawing empty charts.
    assert at.info
    assert at.button(key="book_sample") is not None
    assert at.session_state.book.is_empty


def test_sample_book_gives_a_delta_flat_short_gamma_position() -> None:
    at = with_sample_book()
    book = at.session_state.book

    assert len(book) == 6
    assert at.success  # the flash message of the trade
    assert metric_value(at, "Open lines") == "6"

    greeks = book.greeks(market_of(at))
    assert abs(greeks["delta"]) < 1e-9  # the sample opens with its delta hedge on
    assert greeks["gamma"] < 0  # short the condors: short gamma
    assert greeks["vega"] < 0
    # Trading at the model price is P&L neutral, so nothing has been made or lost yet.
    assert metric_value(at, "P&L since inception") == "0.00"

    # The page says the same thing in desk language, from the signs it just computed.
    spoken = [md.value for md in at.markdown if "Say it out loud" in md.value]
    assert spoken, "the desk read line is missing"
    assert "delta flat" in spoken[0].lower()
    assert "short gamma" in spoken[0] and "short vega" in spoken[0]
    assert "collecting theta" in spoken[0]


def test_showing_every_greek_widens_the_positions_table() -> None:
    at = with_sample_book()
    narrow = at.dataframe[0].value

    at.toggle(key="book_all_greeks").set_value(True).run()

    assert not at.exception
    wide = at.dataframe[0].value
    assert wide.shape[1] > narrow.shape[1]
    assert "zomma" in wide.columns
    assert wide.shape[0] == narrow.shape[0]  # same lines, plus cash and the total row


# --------------------------------------------------------------------------- #
# Adding and closing positions
# --------------------------------------------------------------------------- #
def test_adding_a_vanilla_updates_the_book_and_the_table() -> None:
    at = open_page()
    at.segmented_control(key="book_family").set_value("Vanilla option").run()
    at.number_input(key="book_vanilla_strike").set_value(110.0).run()
    at.slider(key="book_vanilla_days").set_value(180).run()
    at.number_input(key="book_vanilla_qty").set_value(-25.0).run()
    at.button(key="book_add").click().run()

    assert not at.exception
    book = at.session_state.book
    assert len(book) == 1
    position = book.positions[0]
    assert isinstance(position.instrument, EuropeanOption)
    assert position.instrument.strike == 110.0
    assert position.quantity == -25.0
    # Selling raises cash.
    assert book.cash > 100_000.0
    assert metric_value(at, "Open lines") == "1"


def test_the_moneyness_caption_reads_from_the_side_you_picked() -> None:
    """A 110 strike is out of the money for a call and in the money for a put."""
    at = open_page()
    at.segmented_control(key="book_family").set_value("Vanilla option").run()
    at.number_input(key="book_vanilla_strike").set_value(110.0).run()

    captions = "\n".join(caption.value for caption in at.caption)
    assert "Strike 110.00 is 10% out of the money." in captions

    at.segmented_control(key="book_vanilla_type").set_value("put").run()
    assert not at.exception
    captions = "\n".join(caption.value for caption in at.caption)
    assert "Strike 110.00 is 10% in the money." in captions
    # Never the other side's answer, and never an unqualified guess.
    assert "for a call" not in captions


def test_adding_a_strategy_books_its_legs_separately() -> None:
    at = open_page()
    at.segmented_control(key="book_family").set_value("Strategy").run()
    at.selectbox(key="book_strategy_key").select("long_straddle").run()
    at.number_input(key="book_strategy_qty").set_value(5.0).run()
    at.button(key="book_add").click().run()

    assert not at.exception
    book = at.session_state.book
    assert len(book) == 2  # a straddle is stored as its call leg and its put leg
    assert {p.instrument.option_type for p in book.positions} == {"call", "put"}
    assert book.blotter_frame().shape[0] == 1  # but the blotter keeps one ticket


def test_closing_a_line_brings_it_back_to_flat() -> None:
    at = open_page()
    at.segmented_control(key="book_family").set_value("Stock").run()
    at.number_input(key="book_stock_qty").set_value(300.0).run()
    at.button(key="book_add").click().run()
    assert at.session_state.book.underlying_quantity() == 300.0

    at.button(key="book_close").click().run()

    assert not at.exception
    assert at.session_state.book.is_empty
    assert at.session_state.book.blotter_frame().shape[0] == 2


def test_zero_quantity_is_refused_with_a_warning() -> None:
    at = open_page()
    at.segmented_control(key="book_family").set_value("Vanilla option").run()
    at.number_input(key="book_vanilla_qty").set_value(0.0).run()
    at.button(key="book_add").click().run()

    assert not at.exception
    assert at.session_state.book.is_empty
    assert any("Quantity is zero" in warning.value for warning in at.warning)


def test_an_averaging_window_after_expiry_is_explained_not_crashed() -> None:
    """Rule of the page: a bad input shows a message, never a traceback."""
    at = open_page()
    at.segmented_control(key="book_family").set_value("Exotic").run()
    at.selectbox(key="book_exotic_key").select("asian").run()
    at.slider(key="book_asian_expiry").set_value(90).run()
    at.slider(key="book_asian_avg_start").set_value(200).run()

    assert not at.exception
    assert any("cannot be built" in warning.value for warning in at.warning)
    assert at.button(key="book_add").disabled

    # And the page recovers as soon as the window is valid again.
    at.slider(key="book_asian_avg_start").set_value(10).run()
    assert not at.button(key="book_add").disabled
    at.button(key="book_add").click().run()
    assert not at.exception
    assert len(at.session_state.book) == 1


# --------------------------------------------------------------------------- #
# Risk reports
# --------------------------------------------------------------------------- #
def test_every_risk_report_renders() -> None:
    at = with_sample_book()

    # The desk numbers above the reports are full repricings, not Greeks.
    assert metric_value(at, "P&L if the spot gaps -10%") != "0"
    assert metric_value(at, "Daily break-even move").endswith("%")

    for report in ("Risk profile", "Spot x vol grid", "Greeks by line", "P&L by horizon", "Stress tests"):
        at.segmented_control(key="book_report").set_value(report).run()
        assert not at.exception, f"{report} raised {at.exception}"
        assert at.get("plotly_chart"), f"{report} drew no chart"


def test_changing_the_report_controls_changes_the_chart() -> None:
    at = with_sample_book()
    at.segmented_control(key="book_report").set_value("Stress tests").run()
    before = at.dataframe[-1].value.shape[0]

    at.segmented_control(key="book_report").set_value("Risk profile").run()
    at.pills(key="book_profile_panels").set_value(["pnl"]).run()
    assert not at.exception
    assert at.get("plotly_chart")

    # Deselecting every panel is an input error, and it is explained.
    at.pills(key="book_profile_panels").set_value([]).run()
    assert not at.exception
    assert any("at least one panel" in warning.value for warning in at.warning)

    assert before > 0  # the stress table did hold the library's scenarios


# --------------------------------------------------------------------------- #
# Hedging
# --------------------------------------------------------------------------- #
def test_delta_hedge_button_flattens_the_delta() -> None:
    at = with_sample_book()
    # Move the market so the opening hedge goes stale, exactly as it would on a desk.
    at.sidebar.number_input(key="mkt_spot").set_value(105.0).run()
    book = at.session_state.book
    assert abs(book.greeks(market_of(at))["delta"]) > 1.0

    at.button(key="book_do_hedge").click().run()

    assert not at.exception
    book = at.session_state.book
    assert abs(book.greeks(market_of(at))["delta"]) < 1e-9
    assert book.underlying_quantity() != 0.0


def test_second_greek_hedge_kills_the_chosen_greek() -> None:
    at = with_sample_book()
    at.selectbox(key="book_hedge_greek").select("vega").run()
    before = at.session_state.book.greeks(market_of(at))
    assert abs(before["vega"]) > 1.0

    at.button(key="book_do_greek_hedge").click().run()

    assert not at.exception
    after = at.session_state.book.greeks(market_of(at))
    assert abs(after["vega"]) < 1e-6  # the option is sized to kill vega
    assert abs(after["delta"]) < 1e-6  # and the stock mops up the delta of both


# --------------------------------------------------------------------------- #
# Saving, loading and resetting
# --------------------------------------------------------------------------- #
def test_the_page_offers_a_book_that_can_be_rebuilt_from_its_json() -> None:
    at = with_sample_book()
    # The payload of a download button is served by URL, so AppTest cannot read it
    # back; what is asserted here is that the button is offered and that the exact
    # document it carries restores the same book.
    assert at.get("download_button")
    restored = Book.from_json(at.session_state.book.to_json())

    market = market_of(at)
    assert len(restored) == len(at.session_state.book)
    assert restored.value(market) == pytest.approx(at.session_state.book.value(market))


def test_reset_needs_a_confirmation_first() -> None:
    at = with_sample_book()
    assert at.button(key="book_reset").disabled

    at.checkbox(key="book_reset_confirm").check().run()
    at.number_input(key="book_reset_cash").set_value(50_000.0).run()
    at.button(key="book_reset").click().run()

    assert not at.exception
    book = at.session_state.book
    assert book.is_empty
    assert book.cash == 50_000.0
    assert at.session_state.sim is None


# --------------------------------------------------------------------------- #
# The page reacts to the shared market
# --------------------------------------------------------------------------- #
def test_the_page_reprices_with_the_sidebar_market() -> None:
    at = open_page()
    at.session_state.book = Book("My book", cash=100_000.0)
    at.session_state.book.trade(
        EuropeanOption("call", 100.0, 1.0), 100, Market(spot=100.0, vol=0.2, rate=0.03, div=0.01)
    )
    at.run()
    quiet = metric_value(at, "Net liquidation value")

    at.sidebar.slider(key="mkt_vol_pct").set_value(45.0).run()

    assert not at.exception
    assert metric_value(at, "Net liquidation value") != quiet  # long vega: a vol rise is a gain
    assert at.session_state.book.value(market_of(at)) > 100_000.0


def test_an_unsolvable_hedge_is_explained_instead_of_offered() -> None:
    """An option with no gamma cannot hedge gamma: the page says so and offers no trade."""
    at = with_sample_book()
    at.selectbox(key="book_hedge_greek").select("gamma").run()
    at.number_input(key="book_hedge_strike").set_value(1_000.0).run()
    at.slider(key="book_hedge_days").set_value(1).run()

    assert not at.exception
    assert Underlying().greeks(market_of(at))["gamma"] == 0.0
    assert EuropeanOption("call", 1_000.0, 1 / 365).greeks(market_of(at))["gamma"] == 0.0
    assert any("cannot be solved" in warning.value for warning in at.warning)
    with pytest.raises(KeyError):  # no button to execute an impossible hedge
        at.button(key="book_do_greek_hedge")
