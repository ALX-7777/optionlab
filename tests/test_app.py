"""Tests for the app shell: navigation, the shared market and the session state.

Each page has its own test file (``tests/test_page_*.py``). These tests cover
what ``app.py`` and ``app_lib`` are responsible for, plus the invariants that
hold *across* pages — one widget-key namespace, one set of formatters, one
piece of page furniture, one shared book. They run headless with Streamlit's
AppTest: no browser, no server.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app_lib import ui
from optionlab import Book, Market

APP = "../app.py"
PAGES = [
    "app_pages/home.py",
    "app_pages/greeks.py",
    "app_pages/strategies.py",
    "app_pages/exotics.py",
    "app_pages/book.py",
    "app_pages/simulator.py",
    "app_pages/interview.py",
]

#: The page directory, resolved from this file so the tests do not depend on cwd.
PAGE_DIR = Path(__file__).resolve().parent.parent / "app_pages"

#: ``key="..."``, ``key=f"..."`` and ``key_prefix="..."``. For an f-string only
#: the literal head matters, which is what carries the page prefix.
KEY_PATTERN = re.compile(r"""\bkey(?:_prefix)?\s*=\s*f?["']([^"'{]*)""")

#: ``ui.param_controls(spec, spot, "book", ...)`` passes its prefix positionally.
PREFIX_PATTERN = re.compile(r"""param_controls\([^)]*?["']([a-z_]+)["']""", re.S)


def page_source(page: str) -> str:
    return (PAGE_DIR / f"{page}.py").read_text(encoding="utf-8")


def declared_keys(page: str) -> set[str]:
    """Every widget/chart key literal the page declares."""
    source = page_source(page)
    return {match for match in KEY_PATTERN.findall(source) if match} | set(
        PREFIX_PATTERN.findall(source)
    )


@pytest.fixture(scope="module")
def app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=60).run()


@pytest.fixture(scope="module")
def pages() -> dict[str, AppTest]:
    """One booted app per page, so the cross-page tests pay the import cost once."""
    booted: dict[str, AppTest] = {}
    for page in PAGES:
        at = AppTest.from_file(APP, default_timeout=90).run()
        at.switch_page(page).run()
        assert not at.exception, f"{page} raised {at.exception}"
        booted[Path(page).stem] = at
    return booted


def test_app_starts_without_error(app: AppTest) -> None:
    assert not app.exception


def test_session_is_initialised(app: AppTest) -> None:
    assert isinstance(app.session_state.book, Book)
    assert app.session_state.sim is None
    assert app.session_state.trader_units is True


def test_sidebar_holds_the_market_controls(app: AppTest) -> None:
    assert app.session_state.mkt_spot == 100.0
    assert app.session_state.mkt_vol_pct == 20.0
    assert app.sidebar.number_input(key="mkt_spot") is not None
    for key in ("mkt_vol_pct", "mkt_rate_pct", "mkt_div_pct"):
        assert app.sidebar.slider(key=key) is not None
    assert app.sidebar.toggle(key="trader_units") is not None


def test_market_controls_change_the_shared_market() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.number_input(key="mkt_spot").set_value(250.0).run()
    at.sidebar.slider(key="mkt_vol_pct").set_value(35.0).run()
    assert not at.exception

    from app_lib import state  # imported here so the script has run first

    assert at.session_state.mkt_spot == 250.0
    assert at.session_state.mkt_vol_pct == 35.0
    # The page-facing accessor converts percent inputs into library decimals.
    market = Market(spot=250.0, vol=0.35, rate=0.03, div=0.01)
    assert state.current_market.__doc__  # the accessor is documented
    assert market.spot == at.session_state.mkt_spot


def test_reset_market_button_restores_defaults() -> None:
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.number_input(key="mkt_spot").set_value(321.0).run()
    assert at.session_state.mkt_spot == 321.0
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert at.session_state.mkt_spot == 100.0


@pytest.mark.parametrize("page", PAGES)
def test_every_page_runs(page: str) -> None:
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.switch_page(page).run()
    assert not at.exception, f"{page} raised {at.exception}"


# --------------------------------------------------------------------------- #
# One widget-key namespace
#
# Session state is shared by every page, so two pages using the same key would
# silently drive each other's controls. Every key must therefore carry its own
# page's name, which also makes collisions impossible by construction.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("page", [Path(name).stem for name in PAGES])
def test_every_widget_key_carries_its_page_name(page: str) -> None:
    # A bare page name is the prefix handed to ui.param_controls, which appends
    # the spec and parameter names to it.
    offenders = sorted(
        key for key in declared_keys(page) if key != page and not key.startswith(f"{page}_")
    )
    assert not offenders, f"{page}.py declares un-prefixed keys: {offenders}"


def test_no_widget_key_is_claimed_by_two_pages() -> None:
    seen: dict[str, str] = {}
    clashes: list[str] = []
    for name in PAGES:
        page = Path(name).stem
        for key in declared_keys(page):
            if key in seen and seen[key] != page:
                clashes.append(f"{key!r} in both {seen[key]} and {page}")
            seen[key] = page
    assert not clashes, clashes


def test_pages_never_reach_around_the_component_library() -> None:
    """The three rules of ``app_lib.ui``, checked as text so they cannot drift."""
    for name in PAGES:
        source = page_source(Path(name).stem)
        assert "st.plotly_chart" not in source, f"{name} bypasses ui.chart"
        assert "use_container_width" not in source, f"{name} uses a deprecated width"
        assert "unsafe_allow_html" not in source, f"{name} injects raw HTML"
        assert "st.set_page_config" not in source, f"{name} re-configures the page"


# --------------------------------------------------------------------------- #
# One set of formatters
# --------------------------------------------------------------------------- #
def test_the_formatters_name_the_non_finite_cases_apart() -> None:
    """A NaN is not an infinity, and a missing number is neither."""
    assert ui.money(None) == "-"
    assert ui.money(float("nan")) == "n/a"
    assert ui.money(float("inf")) == "unlimited"
    assert ui.money(float("-inf")) == "-unlimited"
    assert ui.money(1234.5, signed=True) == "+1,234.50"

    assert ui.number(None) == "-"
    assert ui.number(float("nan")) == "n/a"
    assert ui.number(float("-inf")) == "-∞"

    assert ui.percent(float("nan")) == "n/a"
    assert ui.percent(0.2, decimals=1) == "20.0%"

    # A P&L bound is the one place where an infinity is a real answer.
    assert ui.extreme(float("inf")) == "unlimited"
    assert ui.extreme(float("-inf")) == "unlimited"
    assert ui.extreme(float("nan")) == "n/a"
    assert ui.extreme(-15.69) == "-15.69"
    assert ui.extreme(-15.69, absolute=True) == "15.69"


def test_moneyness_reads_correctly_from_both_sides() -> None:
    assert ui.moneyness_caption(105.0, 100.0, "call") == "5% out of the money"
    assert ui.moneyness_caption(105.0, 100.0, "put") == "5% in the money"
    assert ui.moneyness_caption(95.0, 100.0, "call") == "5% in the money"
    assert ui.moneyness_caption(95.0, 100.0, "put") == "5% out of the money"
    assert ui.moneyness_caption(100.0, 100.0, "put") == "at the money"
    # With no side given the caption describes the strike instead of guessing.
    assert ui.moneyness_caption(105.0, 100.0) == "5% above the spot"
    assert ui.moneyness_caption(95.0, 100.0) == "5% below the spot"
    assert ui.moneyness_caption(105.0, 0.0) == ""


def test_greeks_are_always_quoted_with_their_units() -> None:
    """``greek_help`` carries the library's own unit for the selected toggle."""
    from optionlab import GREEK_INFO

    for name in ("delta", "gamma", "vega", "theta", "rho"):
        assert GREEK_INFO[name]["trader_unit"] in ui.greek_help(name, trader_units=True)
        assert GREEK_INFO[name]["raw_unit"] in ui.greek_help(name, trader_units=False)
        assert GREEK_INFO[name]["description"] in ui.greek_help(name)
    assert ui.greek_value("gamma", 0.012345) == "0.0123"
    assert ui.greek_value("delta", float("nan")) == "n/a"


# --------------------------------------------------------------------------- #
# One piece of page furniture
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("page", [Path(name).stem for name in PAGES])
def test_every_page_opens_with_the_same_orientation_box(page: str, pages) -> None:
    at = pages[page]
    assert at.title, f"{page} has no ui.page_header"
    # ui.explain renders an expander carrying an icon, which AppTest calls a status.
    labels = [status.label for status in at.status]
    assert labels and labels[0] == "How to use this page", labels[:3]


@pytest.mark.parametrize("page", [Path(name).stem for name in PAGES])
def test_every_page_carries_at_least_two_interview_boxes(page: str, pages) -> None:
    text = "\n".join(element.value for element in pages[page].markdown)
    assert text.count("Interview angle") >= 2, f"{page} shows too few interview boxes"


@pytest.mark.parametrize("page", [Path(name).stem for name in PAGES])
def test_every_chart_is_keyed_to_its_page(page: str, pages) -> None:
    keys = [element.key for element in pages[page].get("plotly_chart")]
    assert all(key and key.startswith(f"{page}_") for key in keys), keys


# --------------------------------------------------------------------------- #
# One shared book
# --------------------------------------------------------------------------- #
def test_a_trade_made_on_one_page_is_visible_on_the_others() -> None:
    at = AppTest.from_file(APP, default_timeout=90).run()
    assert at.session_state.book.is_empty

    at.switch_page("app_pages/strategies.py").run()
    at.button(key="strategies_trade_add").click().run()
    assert not at.exception
    assert len(at.session_state.book.positions) == 2  # the two legs of a straddle

    at.switch_page("app_pages/exotics.py").run()
    at.button(key="exotics_add").click().run()
    assert not at.exception
    assert len(at.session_state.book.positions) == 3

    # The book page sees all three lines, and the simulator will start from them.
    at.switch_page("app_pages/book.py").run()
    assert not at.exception
    assert [m.value for m in at.metric if m.label == "Open lines"] == ["3"]

    at.switch_page("app_pages/simulator.py").run()
    assert not at.exception
    assert at.button(key="simulator_start").disabled is False


def test_resetting_the_book_also_clears_a_running_simulation() -> None:
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.switch_page("app_pages/simulator.py").run()
    at.button(key="simulator_add_starter").click().run()
    at.slider(key="simulator_horizon").set_value(10).run()
    at.slider(key="simulator_steps").set_value(10).run()
    at.button(key="simulator_start").click().run()
    assert at.session_state.sim is not None

    at.switch_page("app_pages/book.py").run()
    at.checkbox(key="book_reset_confirm").check().run()
    at.button(key="book_reset").click().run()

    assert not at.exception
    assert at.session_state.book.is_empty
    assert at.session_state.sim is None

    at.switch_page("app_pages/simulator.py").run()
    assert not at.exception
    assert at.button(key="simulator_start").disabled is True


# --------------------------------------------------------------------------- #
# Edge cases the whole app has to survive
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "spot", "vol", "rate", "div"),
    [
        ("penny stock, panic vol", 1.0, 150.0, 10.0, 10.0),
        ("index level, dead vol", 100_000.0, 1.0, 0.0, 0.0),
        ("dividend above the rate", 100.0, 45.0, 0.0, 10.0),
    ],
)
def test_an_extreme_market_never_breaks_a_page(
    label: str, spot: float, vol: float, rate: float, div: float
) -> None:
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.sidebar.number_input(key="mkt_spot").set_value(spot).run()
    at.sidebar.slider(key="mkt_vol_pct").set_value(vol).run()
    at.sidebar.slider(key="mkt_rate_pct").set_value(rate).run()
    at.sidebar.slider(key="mkt_div_pct").set_value(div).run()
    for page in PAGES:
        at.switch_page(page).run()
        assert not at.exception, f"{page} raised on {label}: {at.exception}"


def test_a_book_holding_an_expired_line_is_explained_not_crashed() -> None:
    """A line can outlive its expiry: a saved book, or a simulation that ran past it."""
    at = AppTest.from_file(APP, default_timeout=90).run()
    from optionlab import EuropeanOption

    book = Book("My book", cash=100_000.0)
    market = Market(spot=100.0, vol=0.20, rate=0.03, div=0.01)
    book.trade(EuropeanOption("call", 100.0, 0.5), 10, market)
    for position in book.positions:
        object.__setattr__(position.instrument, "expiry", -0.5)
        assert position.is_expired(market)
    at.session_state.book = book
    at.run()

    at.switch_page("app_pages/book.py").run()
    assert not at.exception

    at.switch_page("app_pages/simulator.py").run()
    assert not at.exception
    # Nothing left to simulate, and the page says so rather than offering a run.
    assert any("expired" in warning.value for warning in at.warning)


def test_the_trader_units_toggle_is_honoured_by_every_page() -> None:
    """Raw units are a different quote, so the number and the unit line must both move."""
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.switch_page("app_pages/home.py").run()
    desk_vega = [m.value for m in at.metric if m.label == "Vega"][0]

    at.sidebar.toggle(key="trader_units").set_value(False).run()
    assert at.session_state.trader_units is False

    raw_vega = [m.value for m in at.metric if m.label == "Vega"][0]
    # Raw vega is per 1.00 of volatility: a hundred times the per-point number.
    assert float(raw_vega.replace(",", "")) == pytest.approx(
        float(desk_vega.replace(",", "")) * 100.0, rel=1e-3
    )

    # Give the book a position so the pages that only show Greeks when there is
    # risk to show (the book, the simulator) also have to state their units.
    at.switch_page("app_pages/strategies.py").run()
    at.button(key="strategies_trade_add").click().run()

    for page in PAGES:
        at.switch_page(page).run()
        assert not at.exception, f"{page} raised in raw units: {at.exception}"
        captions = "\n".join(element.value for element in at.caption)
        assert "Raw derivatives" in captions, f"{page} does not state its units"
