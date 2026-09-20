"""The component library every page is built from.

Six rules keep the app coherent. ``tests/test_app.py`` checks all of them, so a
page that drifts away from one fails the suite rather than the eye.

1. **One way to show a figure.** Build the figure with ``mode=ui.CHART_MODE`` and
   render it with :func:`chart`, never with ``st.plotly_chart`` directly.
2. **Every chart is explained.** :func:`explain` for how to read it,
   :func:`takeaways` for what to notice, :func:`interview_note` for how the same
   idea shows up in an interview. Every page carries at least two of the last.
3. **Controls are generated, never hand-written.** :func:`param_controls` builds
   the inputs for any strategy or exotic from its registry specification, so the
   labels, defaults and help text come from the library.
4. **One piece of page furniture.** :func:`page_header` once at the top, then
   :func:`how_to_use` directly under it, then :func:`section` for each block.
5. **One set of formatters.** :func:`money`, :func:`percent`, :func:`number`,
   :func:`extreme`, :func:`greek_value` and :func:`moneyness_caption`. They name
   the awkward cases (a missing number, a NaN, an infinity) the same way
   everywhere, so no page has to invent its own wording for them.
6. **One widget-key namespace.** Session state is shared by every page, so every
   key — including the ``key_prefix`` handed to :func:`param_controls` and the
   key handed to :func:`chart` — starts with the name of the page that owns it.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from optionlab import GREEK_INFO, to_trader_units

#: The app's theme is dark (see .streamlit/config.toml), so figures are built
#: with full-contrast dark ink. Pass this to every optionlab plotting function.
CHART_MODE = "dark"

#: Greeks shown by default in a metric row.
HEADLINE_GREEKS = ("price", "delta", "gamma", "vega", "theta")

_GREEK_DECIMALS = {
    "price": 2,
    "delta": 3,
    "gamma": 4,
    "vega": 3,
    "theta": 3,
    "rho": 3,
    "vanna": 4,
    "volga": 4,
    "charm": 4,
    "speed": 5,
    "color": 5,
    "zomma": 5,
}


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #
def page_header(title: str, subtitle: str, *, icon: str | None = None) -> None:
    """Title plus a one-line promise of what the page is for."""
    st.title(title, anchor=False, icon=icon, width="stretch")
    st.caption(subtitle)


def section(title: str, caption: str | None = None, *, icon: str | None = None) -> None:
    """A labelled block inside a page."""
    st.subheader(title, anchor=False, icon=icon, divider=False)
    if caption:
        st.caption(caption)


def explain(title: str, body: str, *, icon: str = ":material/help:", expanded: bool = False) -> None:
    """A collapsed box that teaches: how to read a chart, what a Greek means."""
    with st.expander(title, icon=icon, expanded=expanded):
        st.markdown(body)


def how_to_use(body: str, *, expanded: bool = False) -> None:
    """The orientation box every page carries directly under its header.

    One box, one title, one icon, in the same place on all seven pages, so a
    reader who opens it once knows where to find it again.
    """
    explain("How to use this page", body, icon=":material/help:", expanded=expanded)


def takeaways(items: Sequence[str], *, title: str = "What to notice") -> None:
    """The two or three things a reader should take away from the chart above."""
    with st.container(border=True):
        st.markdown(f":blue-badge[:material/visibility: {title}]")
        for item in items:
            st.markdown(f"- {item}")


def interview_note(body: str, *, question: str | None = None) -> None:
    """How this idea is asked about in a sales and trading interview."""
    with st.container(border=True):
        st.markdown(":violet-badge[:material/psychology: Interview angle]")
        if question:
            st.markdown(f"**{question}**")
        st.markdown(body)


def empty_state(message: str, *, icon: str = ":material/info:") -> None:
    """Shown instead of a chart when there is nothing to display yet."""
    st.info(message, icon=icon)


# --------------------------------------------------------------------------- #
# Charts and tables
# --------------------------------------------------------------------------- #
def chart(fig: go.Figure, *, key: str | None = None) -> None:
    """Render an optionlab figure. Build it with ``mode=ui.CHART_MODE``.

    ``theme=None`` keeps the library's own look instead of letting Streamlit
    restyle the figure.
    """
    st.plotly_chart(fig, theme=None, key=key)


def table(df: pd.DataFrame, *, hide_index: bool = True, height: int | None = None, **kwargs: Any) -> None:
    """Render a dataframe with the app's defaults."""
    if height is not None:
        kwargs["height"] = height
    st.dataframe(df, hide_index=hide_index, **kwargs)


# --------------------------------------------------------------------------- #
# Numbers
# --------------------------------------------------------------------------- #
def money(value: float | None, *, decimals: int = 2, signed: bool = False) -> str:
    """Format a currency amount, e.g. ``1,234.56`` or ``+1,234.56``.

    Non-finite inputs are named rather than printed: ``None`` is a missing
    number (``-``), a NaN is an undefined one (``n/a``) and an infinity is an
    unbounded one (``unlimited`` / ``-unlimited``). A NaN is not an infinity, so
    the three must not collapse into one word — a break-even move that does not
    exist would otherwise read as an unlimited amount of money.
    """
    if value is None:
        return "-"
    number_value = float(value)
    if math.isnan(number_value):
        return "n/a"
    if math.isinf(number_value):
        return "unlimited" if number_value > 0 else "-unlimited"
    sign = "+" if signed and number_value > 0 else ""
    return f"{sign}{number_value:,.{decimals}f}"


def percent(value: float | None, *, decimals: int = 2) -> str:
    """Format a decimal fraction as a percentage: ``0.2 -> 20.00%``."""
    if value is None:
        return "-"
    number_value = float(value)
    if math.isnan(number_value):
        return "n/a"
    if math.isinf(number_value):
        return "unlimited" if number_value > 0 else "-unlimited"
    return f"{number_value * 100:.{decimals}f}%"


def number(value: float | None, *, decimals: int = 3) -> str:
    """Format a plain number, naming the non-finite cases like :func:`money`."""
    if value is None:
        return "-"
    number_value = float(value)
    if math.isnan(number_value):
        return "n/a"
    if math.isinf(number_value):
        return "∞" if number_value > 0 else "-∞"
    return f"{number_value:,.{decimals}f}"


def extreme(value: float | None, *, signed: bool = True, absolute: bool = False) -> str:
    """A P&L bound: ``"unlimited"`` instead of an infinity, money otherwise.

    Max profit and max loss are the two places in the app where an infinity is a
    real answer rather than an error, and every page that shows them needs the
    same wording. ``absolute=True`` quotes the magnitude, for the places where
    the label already carries the direction ("Max loss: 4.50").
    """
    if value is None or math.isnan(float(value)):
        return "n/a"
    if math.isinf(float(value)):
        return "unlimited"
    return money(abs(float(value))) if absolute else money(float(value), signed=signed)


def greek_value(name: str, value: float) -> str:
    """Format one Greek with a sensible number of decimals for its size."""
    return number(value, decimals=_GREEK_DECIMALS.get(name, 3))


def greek_help(name: str, *, trader_units: bool = True) -> str:
    """Tooltip text for a Greek: its unit and one sentence of intuition."""
    info = GREEK_INFO.get(name, {})
    unit = info.get("trader_unit" if trader_units else "raw_unit", "")
    description = info.get("description", "")
    return f"**{unit}** — {description}" if unit else description


# --------------------------------------------------------------------------- #
# Metric rows
# --------------------------------------------------------------------------- #
def metric_row(items: Sequence[tuple], *, border: bool = True, per_row: int = 5) -> None:
    """A row of metric cards.

    Each item is ``(label, value)``, ``(label, value, help)`` or
    ``(label, value, help, delta)``; the fourth element is the small signed
    change ``st.metric`` prints under the number, used by the simulator's live
    panel. More than ``per_row`` metrics wrap onto a second row rather than
    being squeezed, so a caller never has to count columns.
    """
    if not items:
        return
    entries = list(items)
    for start in range(0, len(entries), per_row):
        chunk = entries[start : start + per_row]
        columns = st.columns(len(chunk), border=border)
        for column, item in zip(columns, chunk):
            label, value = item[0], item[1]
            help_text = item[2] if len(item) > 2 else None
            delta = item[3] if len(item) > 3 else None
            column.metric(label, value, delta=delta, help=help_text, width="stretch")


def greek_metrics(
    greeks: Mapping[str, float],
    *,
    keys: Iterable[str] = HEADLINE_GREEKS,
    trader_units: bool = True,
    quantity: float = 1.0,
) -> None:
    """Show the headline Greeks of an instrument as a row of metrics.

    ``greeks`` is a raw dict from ``instrument.greeks(mkt)``; conversion to
    trader units happens here so that every page quotes the same numbers.
    """
    values = to_trader_units(dict(greeks)) if trader_units else dict(greeks)
    items = []
    for key in keys:
        if key not in values:
            continue
        info = GREEK_INFO.get(key, {})
        label = info.get("label", key.capitalize())
        items.append(
            (label, greek_value(key, float(values[key]) * quantity), greek_help(key, trader_units=trader_units))
        )
    metric_row(items)


def units_caption(trader_units: bool = True) -> None:
    """One line reminding the reader which units the Greeks are quoted in."""
    if trader_units:
        st.caption(
            "Trader units: vega per 1 volatility point, theta per calendar day, rho per 1%. "
            "Switch to raw derivatives with the sidebar toggle."
        )
    else:
        st.caption(
            "Raw derivatives: vega per 1.00 of volatility, theta per year, rho per 1.00 of rate. "
            "Switch to trader units with the sidebar toggle."
        )


# --------------------------------------------------------------------------- #
# Controls generated from the registries
# --------------------------------------------------------------------------- #
def param_controls(
    spec: Any,
    spot: float,
    key_prefix: str,
    *,
    columns: int = 3,
    container: Any = None,
) -> dict[str, Any]:
    """Build the inputs for a strategy or an exotic from its registry spec.

    Every entry of ``spec.params`` carries a name, a kind, a default and a
    description. Strike-like defaults are relative to the spot (1.05 means 5%
    out of the money) and are turned into absolute prices here; the returned
    dict can be passed straight to ``build_strategy`` / ``build_exotic``.

    Parameters
    ----------
    spec:
        A ``StrategySpec`` or ``ExoticSpec``.
    spot:
        Current spot, used to place the strike defaults.
    key_prefix:
        Prefix for the widget keys, unique per page (e.g. ``"strategy"``).
    columns:
        How many controls per row.
    container:
        Optional Streamlit container to render into (defaults to the page).
    """
    target = container if container is not None else st
    params = list(spec.params)
    values: dict[str, Any] = {}
    step = max(0.5, round(spot / 100.0, 2))

    for start in range(0, len(params), columns):
        row = params[start : start + columns]
        cols = target.columns(len(row))
        for col, param in zip(cols, row):
            key = f"{key_prefix}_{spec.name}_{param.name}"
            label = param.name.replace("_", " ").capitalize()
            help_text = getattr(param, "description", "") or None
            kind = param.kind
            default = param.default

            if kind in {"strike", "barrier"}:
                values[param.name] = col.number_input(
                    f"{label}",
                    min_value=0.01,
                    value=float(round(float(default) * spot, 2)),
                    step=step,
                    key=key,
                    help=help_text,
                )
            elif kind == "expiry":
                days = col.slider(
                    f"{label} (days)",
                    min_value=1,
                    max_value=1095,
                    value=int(round(float(default) * 365)),
                    step=1,
                    key=key,
                    help=help_text,
                )
                values[param.name] = days / 365.0
            elif kind == "start_time":
                days = col.slider(
                    f"{label} (days from today)",
                    min_value=0,
                    max_value=730,
                    value=int(round(float(default) * 365)),
                    step=1,
                    key=key,
                    help=help_text,
                )
                values[param.name] = days / 365.0
            elif kind == "quantity":
                values[param.name] = col.number_input(
                    label,
                    min_value=1.0,
                    value=float(default),
                    step=1.0,
                    key=key,
                    help=help_text,
                )
            elif kind == "amount":
                values[param.name] = col.number_input(
                    label,
                    min_value=0.0,
                    value=float(default),
                    step=1.0,
                    key=key,
                    help=help_text,
                )
            elif kind == "option_type":
                choice = col.segmented_control(
                    label, ["call", "put"], default=str(default), key=key, help=help_text
                )
                values[param.name] = choice or str(default)
            elif kind == "choice":
                options = list(getattr(param, "choices", ()) or [str(default)])
                choice = col.segmented_control(
                    label, options, default=str(default), key=key, help=help_text
                )
                values[param.name] = choice or str(default)
            else:  # a kind added to the library later: fall back to a number box
                values[param.name] = col.number_input(
                    label, value=float(default), key=key, help=help_text
                )

    return values


def market_key(market: Any) -> tuple[float, float, float, float, float]:
    """The shared market as a hashable tuple, for a ``@st.cache_data`` argument.

    Cached helpers take primitives and rebuild their instruments inside, so every
    page that caches per-market work needs the same five numbers in the same
    order: ``(spot, vol, rate, div, t)``.
    """
    return (
        float(market.spot),
        float(market.vol),
        float(market.rate),
        float(market.div),
        float(market.t),
    )


def moneyness_caption(strike: float, spot: float, option_type: str | None = None) -> str:
    """"5% out of the money" and friends, for a caption under a strike input.

    Moneyness has no meaning without a side, so pass ``option_type`` whenever
    the page knows it and the caption reads correctly for a put as well as for
    a call. With no side given the caption stays honest by describing the
    strike's position rather than guessing: "5% above the spot".
    """
    if spot <= 0:
        return ""
    gap = strike / spot - 1.0
    if abs(gap) < 0.005:
        return "at the money"
    distance = f"{abs(gap) * 100:.0f}%"
    if option_type is None:
        return f"{distance} {'above' if gap > 0 else 'below'} the spot"
    out_of_the_money = gap > 0 if str(option_type).lower() == "call" else gap < 0
    return f"{distance} {'out of' if out_of_the_money else 'in'} the money"
