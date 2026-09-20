"""Tests for the "Interview drills" page (``app_pages/interview.py``).

The page is driven through the real app shell with Streamlit's ``AppTest``, so
``st.navigation`` and the sidebar market behave exactly as they do in a browser.

What is covered:

* the page renders, with its five tabs and an empty scorecard;
* the quick-fire drill on both of its control paths — a numeric question marked
  against the library's own answer, and a multiple-choice comparison question
  that also draws its reveal chart;
* the buttons that change what is shown: check, new question, reset score, the
  topic control, the mental-maths reveal, the flashcard filter, the "show me"
  boxes of the question bank and the pitch selector;
* honesty: the numbers the page prints are re-derived here from the library
  (mental maths, the Asian comparison, the pitch summary);
* the guards of rule 11: checking with an empty answer box, deselecting every
  flashcard topic, and an extreme market.
"""

from __future__ import annotations

import math

import pytest
from streamlit.testing.v1 import AppTest

from app_lib import ui
from optionlab import STRATEGY_REGISTRY, AsianOption, EuropeanOption, Market
from optionlab import strategies as strat

APP = "../app.py"
PAGE = "app_pages/interview.py"

TABS = ("Quick-fire drill", "Mental maths", "Flashcards", "Question bank", "Story prompts")
DEFAULT_MARKET = Market(spot=100.0, vol=0.20, rate=0.03, div=0.01)
THREE_MONTHS = 91.0 / 365.0


def open_page() -> AppTest:
    """Start the app shell on the default market, then switch to the drills page."""
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.switch_page(PAGE).run()
    return at


def all_text(at: AppTest) -> str:
    """Every markdown and caption string on the page, concatenated."""
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def metric_values(at: AppTest, label: str) -> list[str]:
    return [m.value for m in at.metric if m.label == label]


@pytest.fixture(scope="module")
def page() -> AppTest:
    return open_page()


# --------------------------------------------------------------------------- #
# The page renders
# --------------------------------------------------------------------------- #
def test_page_runs(page: AppTest) -> None:
    assert not page.exception
    assert page.title[0].value == "Interview drills"


def test_the_five_tabs_are_rendered(page: AppTest) -> None:
    assert tuple(tab.label for tab in page.get("tab")) == TABS


def test_the_scorecard_starts_empty(page: AppTest) -> None:
    assert metric_values(page, "Asked") == ["0"]
    assert metric_values(page, "Correct") == ["0"]
    assert metric_values(page, "Hit rate") == ["-"]
    assert page.session_state["interview_seed"] == 1


def test_the_drill_asks_a_question_in_its_own_market(page: AppTest) -> None:
    question = page.session_state["interview_question"]
    assert question["prompt"]
    assert question["why"]
    assert question["kind"] in {"number", "choice"}
    # The drill invents its own world, which is shown next to the question and is
    # independent of the sidebar.
    spot, vol, rate, div = question["market"]
    assert spot > 0 and vol > 0
    assert metric_values(page, "Spot") == [ui.money(spot)]


def test_the_teaching_boxes_are_there(page: AppTest) -> None:
    labels = [status.label for status in page.status]
    assert "How this drill works" in labels
    assert labels.count("How to read this") >= 1
    text = all_text(page)
    assert text.count("Interview angle") >= 2
    assert "What to notice" in text


# --------------------------------------------------------------------------- #
# The quick-fire drill: numeric questions
# --------------------------------------------------------------------------- #
def test_a_correct_number_scores_and_reveals_the_reasoning() -> None:
    at = open_page()
    at.segmented_control(key="interview_topic").set_value("Pricing").run()
    question = at.session_state["interview_question"]
    assert question["kind"] == "number"

    key = f"interview_answer_{question['key']}"
    at.number_input(key=key).set_value(float(question["answer"])).run()
    at.button(key="interview_check").click().run()

    assert not at.exception
    assert at.success and question["answer_label"] in at.success[0].value
    assert at.session_state["interview_asked"] == 1
    assert at.session_state["interview_right"] == 1
    assert metric_values(at, "Hit rate") == ["100%"]
    # The reveal carries the desk explanation and the numbers behind it.
    assert question["why"] in all_text(at)

    # A new question moves the seed on and hides the answer again.
    at.button(key="interview_new").click().run()
    assert not at.exception
    assert at.session_state["interview_seed"] == 2
    assert not at.success
    assert at.session_state["interview_asked"] == 1

    # Resetting clears the score as well.
    at.button(key="interview_reset").click().run()
    assert at.session_state["interview_asked"] == 0
    assert at.session_state["interview_right"] == 0
    assert metric_values(at, "Hit rate") == ["-"]


def test_a_wrong_number_is_marked_wrong_and_still_explains() -> None:
    at = open_page()
    at.segmented_control(key="interview_topic").set_value("Greeks").run()
    question = at.session_state["interview_question"]

    key = f"interview_answer_{question['key']}"
    miss = float(question["answer"]) + 50.0 * float(question["tolerance"]) + 1.0
    at.number_input(key=key).set_value(miss).run()
    at.button(key="interview_check").click().run()

    assert not at.exception
    assert at.error and question["answer_label"] in at.error[0].value
    assert at.session_state["interview_asked"] == 1
    assert at.session_state["interview_right"] == 0
    # Being wrong still shows the reasoning: that is the point of the drill.
    assert question["why"] in all_text(at)


def test_checking_an_empty_box_warns_instead_of_scoring() -> None:
    at = open_page()
    at.button(key="interview_check").click().run()

    assert not at.exception
    assert at.warning and "answer in the box" in at.warning[0].value
    assert at.session_state["interview_asked"] == 0
    assert not at.session_state["interview_revealed"]


# --------------------------------------------------------------------------- #
# The quick-fire drill: comparison questions
# --------------------------------------------------------------------------- #
def test_a_comparison_question_is_multiple_choice_and_draws_its_reveal() -> None:
    at = open_page()
    charts_before = len(at.get("plotly_chart"))

    at.segmented_control(key="interview_topic").set_value("Comparisons").run()
    question = at.session_state["interview_question"]
    assert question["kind"] == "choice"
    assert len(question["options"]) == 3
    assert question["answer_label"] in question["options"]

    at.segmented_control(key=f"interview_choice_{question['key']}").set_value(
        question["answer_label"]
    ).run()
    at.button(key="interview_check").click().run()

    assert not at.exception
    assert at.success and question["answer_label"] in at.success[0].value
    assert at.session_state["interview_right"] == 1
    # Comparison questions add the overlay chart that settles the argument.
    assert len(at.get("plotly_chart")) == charts_before + 1


# --------------------------------------------------------------------------- #
# Mental maths: the numbers must be the library's
# --------------------------------------------------------------------------- #
def test_mental_maths_hides_the_exact_column_until_asked(page: AppTest) -> None:
    table = page.dataframe[0].value
    assert list(table.columns) == ["Rule of thumb", "Quoted in", "Approximation"]
    assert len(table) == 6


def test_revealing_mental_maths_shows_the_library_values() -> None:
    at = open_page()
    at.toggle(key="interview_maths_reveal").set_value(True).run()
    assert not at.exception

    table = at.dataframe[0].value
    assert list(table.columns) == [
        "Rule of thumb",
        "Quoted in",
        "Approximation",
        "Exact (library)",
        "Error",
    ]

    call = EuropeanOption("call", 100.0, THREE_MONTHS)
    put = EuropeanOption("put", 100.0, THREE_MONTHS)
    greeks = {name: float(value) for name, value in call.greeks(DEFAULT_MARKET).items()}

    exact = list(table["Exact (library)"])
    assert exact[0] == ui.number(greeks["price"], decimals=2)
    assert exact[1] == ui.number(greeks["delta"], decimals=3)
    assert exact[4] == ui.number(greeks["theta"] / 365.0, decimals=3)
    assert exact[5] == ui.number(greeks["price"] - float(put.price(DEFAULT_MARKET)), decimals=4)

    # Put-call parity is an identity, not an approximation: the error is zero.
    assert abs(float(table["Error"].iloc[5])) < 1e-9
    # The 0.4 rule is close but not exact, and it is stated as an approximation.
    assert 0.0 < abs(float(table["Error"].iloc[0])) < 0.2


def test_changing_the_maturity_reprices_the_mental_maths_option() -> None:
    at = open_page()
    three_month = metric_values(at, "Price")[0]

    at.segmented_control(key="interview_maths_tenor").set_value("1 year").run()
    assert not at.exception

    one_year = metric_values(at, "Price")[0]
    expected = EuropeanOption("call", 100.0, 1.0).price(DEFAULT_MARKET)
    assert one_year == ui.money(float(expected))
    assert float(one_year) > float(three_month)


# --------------------------------------------------------------------------- #
# Flashcards
# --------------------------------------------------------------------------- #
def test_the_flashcard_filter_narrows_the_deck_and_survives_an_empty_pick() -> None:
    at = open_page()
    assert "cards in the deck" in all_text(at)

    at.pills(key="interview_card_topics").set_value(["Greeks"]).run()
    assert not at.exception
    # Eight Greek cards, one per entry the page takes from GREEK_INFO.
    assert "8 cards in the deck." in all_text(at)

    at.pills(key="interview_card_topics").set_value([]).run()
    assert not at.exception
    assert any("No topic selected" in info.value for info in at.info)


# --------------------------------------------------------------------------- #
# Question bank
# --------------------------------------------------------------------------- #
def test_the_show_me_boxes_print_the_library_numbers() -> None:
    at = open_page()
    for box in at.checkbox:
        box.check()
    at.run()
    assert not at.exception

    frames = {frozenset(df.value.columns): df.value for df in at.dataframe}
    by_first_column = {next(iter(columns & {"Structure", "What is priced", "Maturity",
                                            "Spot move"}), None): frame
                       for columns, frame in frames.items()}
    assert set(by_first_column) >= {"Structure", "What is priced", "Maturity", "Spot move"}

    # The Jensen question: the average of the calls must be dearer than the call
    # on the average, and both must be the library's own prices.
    asian_frame = by_first_column["Structure"].set_index("Structure")
    asian = AsianOption("call", 100.0, 1.0, averaging="arithmetic")
    expected_asian = float(asian.price(DEFAULT_MARKET))
    fixings = [(i + 1) / 12.0 for i in range(12)]
    expected_average = sum(
        float(EuropeanOption("call", 100.0, fixing).price(DEFAULT_MARKET)) for fixing in fixings
    ) / len(fixings)

    assert asian_frame.loc["Call on the average (Asian)", "Price"] == pytest.approx(expected_asian)
    assert asian_frame.loc["Average of the calls", "Price"] == pytest.approx(expected_average)
    assert expected_average > expected_asian


def test_the_theme_filter_reduces_the_question_list() -> None:
    at = open_page()
    at.segmented_control(key="interview_bank_theme").set_value("Brainteasers").run()
    assert not at.exception
    assert "3 questions shown." in all_text(at)
    # Only the brainteasers keep their "show me" boxes on screen.
    assert len(at.checkbox) == 2


# --------------------------------------------------------------------------- #
# Story prompts: the pitch is built from the library's own summary
# --------------------------------------------------------------------------- #
def test_the_pitch_quotes_the_strategy_summary() -> None:
    at = open_page()
    spread = STRATEGY_REGISTRY["bull_call_spread"].build_default(spot=100.0, expiry=THREE_MONTHS)
    report = strat.summary(spread, DEFAULT_MARKET)

    assert metric_values(at, "Entry cost") == [ui.money(abs(report["net_premium"]))]
    assert metric_values(at, "Max gain") == [ui.money(report["max_profit"])]
    assert metric_values(at, "Max loss") == [ui.money(abs(report["max_loss"]))]
    assert ui.money(report["breakevens"][0]) in all_text(at)

    # A credit structure flips the label and reports an unlimited loss honestly.
    at.selectbox(key="interview_pitch_strategy").set_value("short_strangle").run()
    assert not at.exception

    strangle = STRATEGY_REGISTRY["short_strangle"].build_default(spot=100.0, expiry=THREE_MONTHS)
    credit = strat.summary(strangle, DEFAULT_MARKET)
    assert math.isinf(credit["max_loss"])
    assert metric_values(at, "Premium collected") == [ui.money(abs(credit["net_premium"]))]
    assert metric_values(at, "Max loss") == ["unlimited"]


def test_the_pitch_follows_the_trader_units_toggle() -> None:
    """The spoken pitch and the metric row must never quote two different units."""
    at = open_page()
    spread = STRATEGY_REGISTRY["bull_call_spread"].build_default(spot=100.0, expiry=THREE_MONTHS)
    report = strat.summary(spread, DEFAULT_MARKET)

    desk_vega = metric_values(at, "Vega")[-1]
    assert desk_vega == ui.greek_value("vega", report["greeks_trader"]["vega"])
    assert "Trader units" in all_text(at)

    at.sidebar.toggle(key="trader_units").set_value(False).run()
    assert not at.exception

    raw_vega = metric_values(at, "Vega")[-1]
    assert raw_vega == ui.greek_value("vega", report["greeks"]["vega"])
    assert raw_vega != desk_vega
    # The sentence quotes the same number, with the library's own unit next to it.
    from optionlab import GREEK_INFO

    assert f"{raw_vega} vega ({GREEK_INFO['vega']['raw_unit']})" in all_text(at)
    assert "Raw derivatives" in all_text(at)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def test_an_extreme_market_still_renders() -> None:
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.sidebar.number_input(key="mkt_spot").set_value(1.0).run()
    at.sidebar.slider(key="mkt_vol_pct").set_value(150.0).run()
    at.sidebar.slider(key="mkt_div_pct").set_value(10.0).run()
    at.switch_page(PAGE).run()

    assert not at.exception
    assert not at.error
    # The drill has its own market, so it is unaffected by the sidebar.
    assert at.session_state["interview_question"]["market"][0] > 1.0
    # The pitch still prices, on a spot of 1.
    assert metric_values(at, "Entry cost") or metric_values(at, "Premium collected")
