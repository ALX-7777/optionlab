"""OptionLab: a Streamlit front end for the optionlab library."""

import streamlit as st

from optionlab import Book, EuropeanOption, Market
from optionlab.plotting import profiles, strategy_plots
from optionlab.strategies import STRATEGY_REGISTRY, summary

st.set_page_config(page_title="OptionLab", page_icon=":material/show_chart:", layout="wide")
st.title("OptionLab")

# --- Sidebar: the market, shared by every tab --------------------------------
with st.sidebar:
    st.header("Market")
    spot = st.number_input("Spot", min_value=1.0, value=100.0, step=1.0)
    vol = st.slider("Volatility", 0.05, 1.0, 0.20, step=0.01)
    rate = st.slider("Interest rate", 0.0, 0.10, 0.03, step=0.005)
    div = st.slider("Dividend yield", 0.0, 0.10, 0.0, step=0.005)
mkt = Market(spot=spot, vol=vol, rate=rate, div=div)

# The book lives in session_state so it survives reruns (every widget click reruns the script)
if "book" not in st.session_state:
    st.session_state.book = Book(cash=100_000)

vanilla_tab, strategy_tab, book_tab = st.tabs(["Vanilla Greeks", "Strategies", "My book"])

with vanilla_tab:
    option_type = st.segmented_control("Type", ["call", "put"], default="call")
    strike = st.number_input("Strike", min_value=1.0, value=100.0, step=1.0)
    expiry = st.slider("Expiry (years)", 0.05, 2.0, 0.5, step=0.05)
    option = EuropeanOption(option_type, strike, expiry)
    st.plotly_chart(profiles.greek_dashboard(option, mkt), theme=None)
    greek = st.selectbox("Greek to watch through time", ["delta", "gamma", "vega", "theta"])
    st.plotly_chart(profiles.greek_evolution(option, mkt, greek=greek), theme=None)

with strategy_tab:
    name = st.selectbox("Strategy", list(STRATEGY_REGISTRY))
    strategy = STRATEGY_REGISTRY[name].build_default(spot=spot, expiry=0.5)
    info = summary(strategy, mkt)
    st.caption(info["description"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Net premium", f"{info['net_premium']:.2f}")
    col2.metric("Max profit", f"{info['max_profit']:.2f}")
    col3.metric("Max loss", f"{info['max_loss']:.2f}")
    st.plotly_chart(strategy_plots.payoff_diagram(strategy, mkt), theme=None)

with book_tab:
    book = st.session_state.book
    quantity = st.number_input("Quantity (negative = sell)", value=10, step=1)
    if st.button("Trade the option from the Vanilla tab"):
        book.trade(option, quantity, mkt)
    st.metric("Book P&L", f"{book.pnl(mkt):,.2f}")
    st.dataframe(book.positions_frame(mkt))
