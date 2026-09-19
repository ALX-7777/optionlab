"""Smoke tests for the Streamlit app, run headless with Streamlit's AppTest."""

from streamlit.testing.v1 import AppTest


def test_app_runs_without_error():
    at = AppTest.from_file("../app.py", default_timeout=30).run()
    assert not at.exception


def test_trading_adds_a_position_to_the_book():
    at = AppTest.from_file("../app.py", default_timeout=30).run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.session_state.book.positions) == 1
