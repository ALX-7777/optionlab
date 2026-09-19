"""Plotly figures for optionlab.

Every function in this sub-package RETURNS a ``plotly.graph_objects.Figure``
(nothing is ever shown or written to disk by the library) and every figure is
styled by :func:`optionlab.plotting.theme.apply_theme`. All figure functions
take ``mode="auto" | "light" | "dark"``; in Streamlit use
``st.plotly_chart(fig, theme=None)`` to keep the optionlab look.

Modules
-------
* :mod:`~optionlab.plotting.theme` -- template, palette, colourscales, helpers.
* :mod:`~optionlab.plotting.profiles` -- Greeks and P&L of ANY instrument
  against spot, vol, time, rates (profiles, dashboards, surfaces, comparisons).
* :mod:`~optionlab.plotting.strategy_plots` -- payoff diagrams and leg-by-leg
  dashboards of option strategies.
* :mod:`~optionlab.plotting.exotic_plots` -- exotic vs vanilla, barrier paths,
  digital replication, Asian averaging.
* :mod:`~optionlab.plotting.book_plots` -- risk of a whole book.
* :mod:`~optionlab.plotting.sim_plots` -- simulation dashboards, P&L
  attribution, hedging experiments.

This package is the only part of optionlab that imports plotly; ``import
optionlab`` alone never does.
"""

from . import book_plots, exotic_plots, profiles, sim_plots, strategy_plots, theme
from .book_plots import (
    book_greeks_breakdown,
    book_pnl_by_horizon,
    book_risk_profile,
    book_scenario_heatmap,
    stress_test_chart,
)
from .exotic_plots import (
    asian_averaging_figure,
    barrier_paths_figure,
    digital_replication_figure,
    exotic_vs_vanilla,
    vanilla_benchmark,
)
from .profiles import (
    call_put_comparison,
    compare_instruments,
    evaluate_on_grid,
    gamma_theta_tradeoff,
    greek_dashboard,
    greek_evolution,
    greek_profile,
    greek_surface,
    greek_vs_time,
    pnl_profile,
    taylor_pnl_explain,
)
from .sim_plots import (
    gamma_theta_chart,
    hedging_error_histogram,
    hedging_error_vs_frequency,
    pnl_attribution_chart,
    pnl_attribution_waterfall,
    simulation_dashboard,
)
from .strategy_plots import (
    compare_strategies,
    payoff_diagram,
    strategy_greeks_dashboard,
    strategy_time_decay,
    strategy_vol_sensitivity,
)
from .theme import COLORS, PALETTE, apply_theme

__all__ = [
    # modules
    "theme",
    "profiles",
    "strategy_plots",
    "exotic_plots",
    "book_plots",
    "sim_plots",
    # theme
    "PALETTE",
    "COLORS",
    "apply_theme",
    # any instrument
    "evaluate_on_grid",
    "greek_profile",
    "greek_dashboard",
    "greek_evolution",
    "greek_vs_time",
    "greek_surface",
    "compare_instruments",
    "call_put_comparison",
    "pnl_profile",
    "taylor_pnl_explain",
    "gamma_theta_tradeoff",
    # strategies
    "payoff_diagram",
    "strategy_greeks_dashboard",
    "strategy_time_decay",
    "strategy_vol_sensitivity",
    "compare_strategies",
    # exotics
    "vanilla_benchmark",
    "exotic_vs_vanilla",
    "barrier_paths_figure",
    "digital_replication_figure",
    "asian_averaging_figure",
    # book
    "book_risk_profile",
    "book_scenario_heatmap",
    "book_greeks_breakdown",
    "book_pnl_by_horizon",
    "stress_test_chart",
    # simulation
    "simulation_dashboard",
    "pnl_attribution_chart",
    "pnl_attribution_waterfall",
    "gamma_theta_chart",
    "hedging_error_histogram",
    "hedging_error_vs_frequency",
]
