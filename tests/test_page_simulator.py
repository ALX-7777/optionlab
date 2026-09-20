"""Tests for the Trading simulator page (``app_pages/simulator.py``).

Every test drives the real app through ``AppTest``: start the entry point, switch
to the page, then click the widgets a user would click. The page owns the keys
prefixed with ``simulator_``; the simulation itself lives in ``session_state.sim``.
"""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

APP = "../app.py"
PAGE = "app_pages/simulator.py"


def open_page() -> AppTest:
    """Start the app and navigate to the simulator page."""
    at = AppTest.from_file(APP, default_timeout=90).run()
    at.switch_page(PAGE).run()
    assert not at.exception
    return at


def start_simulation(at: AppTest) -> AppTest:
    """Open the starter position and start a short simulation."""
    at.button(key="simulator_add_starter").click().run()
    assert not at.exception
    at.slider(key="simulator_horizon").set_value(10).run()
    at.slider(key="simulator_steps").set_value(10).run()
    at.button(key="simulator_start").click().run()
    assert not at.exception
    return at


# --------------------------------------------------------------------------- #
# The page itself
# --------------------------------------------------------------------------- #
def test_page_runs_with_an_empty_book() -> None:
    at = open_page()
    assert at.session_state.sim is None
    # With no position there is nothing to simulate, so the page offers starters.
    assert at.session_state.book.is_empty
    assert at.segmented_control(key="simulator_starter") is not None
    assert at.button(key="simulator_add_starter") is not None


def test_empty_book_cannot_start_a_simulation() -> None:
    """Edge case: the start button is disabled while the book holds nothing."""
    at = open_page()
    assert at.session_state.book.is_empty
    assert at.button(key="simulator_start").disabled is True
    assert at.session_state.sim is None


def test_starter_position_opens_a_position_in_the_book() -> None:
    at = open_page()
    at.segmented_control(key="simulator_starter").set_value("Long 20 calls").run()
    at.button(key="simulator_add_starter").click().run()

    assert not at.exception
    book = at.session_state.book
    assert not book.is_empty
    assert book.underlying_quantity() == 0.0
    # The book summary replaces the starter picker once there is a position.
    assert at.button(key="simulator_start").disabled is False


def test_hedging_rule_choice_changes_the_policy_used() -> None:
    at = open_page()
    at.button(key="simulator_add_starter").click().run()
    at.segmented_control(key="simulator_policy").set_value("No hedge").run()
    assert not at.exception
    # The band parameter only exists for the band policy.
    at.segmented_control(key="simulator_policy").set_value("Delta band").run()
    assert at.slider(key="simulator_band") is not None

    at.segmented_control(key="simulator_policy").set_value("No hedge").run()
    at.button(key="simulator_start").click().run()

    assert not at.exception
    assert at.session_state.sim.policy.label == "no hedge"


# --------------------------------------------------------------------------- #
# Running the simulation
# --------------------------------------------------------------------------- #
def test_simulation_starts_and_steps_forward() -> None:
    at = start_simulation(at=open_page())
    sim = at.session_state.sim

    assert sim is not None
    assert sim.step_index == 0
    assert sim.scenario.n_steps == 10
    # Nothing to attribute yet: the page says so instead of drawing empty charts.
    assert at.info

    at.button(key="simulator_next").click().run()
    assert not at.exception
    assert at.session_state.sim.step_index == 1

    at.button(key="simulator_next5").click().run()
    assert at.session_state.sim.step_index == 6
    assert len(at.session_state.sim.history) == 7


def test_run_to_the_end_finishes_and_disables_the_controls() -> None:
    at = start_simulation(at=open_page())
    at.button(key="simulator_run_all").click().run()

    sim = at.session_state.sim
    assert not at.exception
    assert sim.is_finished
    assert sim.step_index == sim.scenario.n_steps
    assert at.success  # "the scenario is finished"
    assert at.button(key="simulator_next").disabled is True

    at.button(key="simulator_reset").click().run()
    assert not at.exception
    assert at.session_state.sim.step_index == 0
    assert at.button(key="simulator_next").disabled is False


def test_new_setup_clears_the_simulation() -> None:
    at = start_simulation(at=open_page())
    at.button(key="simulator_clear").click().run()

    assert not at.exception
    assert at.session_state.sim is None
    assert at.button(key="simulator_start") is not None


def test_manual_hedge_flattens_the_delta() -> None:
    at = open_page()
    at.button(key="simulator_add_starter").click().run()
    at.segmented_control(key="simulator_policy").set_value("No hedge").run()
    at.slider(key="simulator_horizon").set_value(10).run()
    at.slider(key="simulator_steps").set_value(10).run()
    at.button(key="simulator_start").click().run()

    at.button(key="simulator_next5").click().run()
    sim = at.session_state.sim
    assert abs(sim.book.greeks(sim.current_market)["delta"]) > 1e-6

    at.button(key="simulator_hedge_now").click().run()
    assert not at.exception
    sim = at.session_state.sim
    assert abs(sim.book.greeks(sim.current_market)["delta"]) < 1e-6
    assert sim.book.underlying_quantity() != 0.0


def test_a_position_expiring_mid_run_settles_without_breaking_the_page() -> None:
    """Edge case: the starter straddle expires long before a 250-day horizon ends."""
    at = open_page()
    at.button(key="simulator_add_starter").click().run()  # a 60-day straddle
    at.slider(key="simulator_horizon").set_value(250).run()
    at.slider(key="simulator_steps").set_value(50).run()
    at.button(key="simulator_start").click().run()
    at.button(key="simulator_run_all").click().run()

    assert not at.exception
    sim = at.session_state.sim
    assert sim.is_finished
    assert len(sim.book.settlements) >= 2  # the call and the put cash-settled
    assert sim.history["n_settled"].sum() == len(sim.book.settlements)
    assert abs(sim.book.underlying_quantity()) < 1e-9  # the hedge was unwound too

    for view in ("P&L explain", "Gamma vs theta", "Dashboard"):
        at.segmented_control(key="simulator_view").set_value(view).run()
        assert not at.exception, f"{view} broke after the position settled"


# --------------------------------------------------------------------------- #
# The four analysis views
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("view", ["P&L explain", "Gamma vs theta", "Dashboard", "History"])
def test_every_analysis_view_renders(view: str) -> None:
    at = start_simulation(at=open_page())
    at.button(key="simulator_run_all").click().run()
    at.segmented_control(key="simulator_view").set_value(view).run()

    assert not at.exception
    assert at.segmented_control(key="simulator_view").value == view


def test_history_view_shows_one_row_per_step_and_offers_a_download() -> None:
    at = start_simulation(at=open_page())
    at.button(key="simulator_run_all").click().run()
    at.segmented_control(key="simulator_view").set_value("History").run()
    at.slider(key="simulator_history_rows").set_value(11).run()

    assert not at.exception
    table = at.dataframe[0].value
    assert len(table) == 11  # 10 steps plus the starting row
    assert list(table["Step"]) == list(range(10, -1, -1))
    assert at.download_button(key="simulator_download_history") is not None


def test_pnl_explain_can_switch_to_per_step_bars() -> None:
    at = start_simulation(at=open_page())
    at.button(key="simulator_run_all").click().run()
    at.segmented_control(key="simulator_view").set_value("P&L explain").run()
    assert at.toggle(key="simulator_attrib_steps").value is False

    at.toggle(key="simulator_attrib_steps").set_value(True).run()
    assert not at.exception
    assert at.toggle(key="simulator_attrib_steps").value is True


# --------------------------------------------------------------------------- #
# The hedging laboratory
# --------------------------------------------------------------------------- #
def test_hedging_experiment_runs_and_reports_the_statistics() -> None:
    at = open_page()
    at.slider(key="simulator_exp_paths").set_value(200).run()
    at.multiselect(key="simulator_exp_frequencies").set_value([4, 52]).run()
    at.button(key="simulator_run_experiment").click().run()

    assert not at.exception
    params = at.session_state.simulator_experiment_params
    assert params["n_paths"] == 200
    assert params["frequencies"] == (4, 52)

    summary = at.dataframe[0].value
    assert list(summary["Hedges"]) == [4, 52]
    # More hedges, tighter distribution: that is the 1 / sqrt(N) law.
    assert summary["Std of P&L"].iloc[1] < summary["Std of P&L"].iloc[0]


def test_hedging_experiment_needs_two_frequencies() -> None:
    """Edge case: comparing frequencies needs at least two of them."""
    at = open_page()
    at.slider(key="simulator_exp_paths").set_value(200).run()
    at.multiselect(key="simulator_exp_frequencies").set_value([13]).run()
    at.button(key="simulator_run_experiment").click().run()

    assert not at.exception
    assert at.warning
    assert "at least two" in at.warning[0].value
