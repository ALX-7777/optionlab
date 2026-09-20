"""OptionLab — an interactive lab for option Greeks, strategies and trading practice.

Entry point of the Streamlit app. It sets the page configuration, initialises the
session (market inputs, book, simulator), declares the navigation and renders the
market controls that every page shares.

Run it with::

    uv run streamlit run app.py
"""

import streamlit as st

from app_lib import state

st.set_page_config(
    page_title="OptionLab",
    page_icon=":material/candlestick_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

state.init_session()

page = st.navigation(
    {
        "Learn": [
            st.Page("app_pages/home.py", title="Start here", icon=":material/flag:", default=True),
            st.Page("app_pages/greeks.py", title="Greeks explorer", icon=":material/show_chart:"),
            st.Page("app_pages/strategies.py", title="Strategies", icon=":material/account_tree:"),
            st.Page("app_pages/exotics.py", title="Exotics", icon=":material/diamond:"),
        ],
        "Practice": [
            st.Page("app_pages/book.py", title="My book", icon=":material/inventory_2:"),
            st.Page("app_pages/simulator.py", title="Trading simulator", icon=":material/speed:"),
            st.Page("app_pages/interview.py", title="Interview drills", icon=":material/psychology:"),
        ],
    },
    position="sidebar",
)

state.sidebar_market()

page.run()
