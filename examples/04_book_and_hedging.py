"""Manage a small option book: risk report, hedged vs unhedged simulation, hedging experiment.

Run from the project root::

    python examples/04_book_and_hedging.py
    python examples/04_book_and_hedging.py --seed 7 --realized-vol 0.30 --paths 5000

The script

1. builds a book that is SHORT an at-the-money straddle and LONG a strangle as
   wings (an iron butterfly: short gamma, long theta, limited tail risk) and
   prints its positions, Greeks, dollar Greeks and stress tests;
2. writes the risk-profile, scenario-heatmap, breakdown and stress-test figures;
3. runs the SAME market scenario twice -- once delta-hedged every day, once not
   hedged at all -- and explains both P&Ls Greek by Greek;
4. runs the classic hedging-frequency experiment (hedging error ~ 1/sqrt(N)).

Every figure is written as interactive HTML into ``--outdir``. Nothing is
opened in a browser unless ``--show`` is passed.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optionlab.book import Book, TransactionCosts  # noqa: E402
from optionlab.instruments import EuropeanOption  # noqa: E402
from optionlab.market import Market  # noqa: E402
from optionlab.plotting import book_plots, sim_plots  # noqa: E402
from optionlab.simulator import (  # noqa: E402
    ATTRIBUTION_TERMS,
    DeltaHedgeEveryN,
    NoHedge,
    TradingSimulator,
    attribution_totals,
    delta_hedging_experiment,
    gamma_scalping_summary,
    gbm_scenario,
    hedging_experiment_summary,
)
from optionlab.strategies import Strategy, long_strangle, short_straddle  # noqa: E402

EXPIRY = 0.25   # three months
N_STEPS = 63    # trading days in three months


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=14, help="seed of the market scenario and of the experiment")
    parser.add_argument("--spot", type=float, default=100.0)
    parser.add_argument("--vol", type=float, default=0.20, help="implied volatility (0.20 = 20%%)")
    parser.add_argument("--realized-vol", type=float, default=0.30,
                        help="volatility the spot really moves at in the simulation")
    parser.add_argument("--rate", type=float, default=0.03)
    parser.add_argument("--div", type=float, default=0.01)
    parser.add_argument("--contracts", type=float, default=100.0, help="number of straddles sold")
    parser.add_argument("--paths", type=int, default=3000, help="paths of the hedging experiment")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--show", action="store_true", help="open the figures in the browser")
    return parser.parse_args(argv)


def section(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def build_packages(spot: float) -> tuple[Strategy, Strategy]:
    """The short straddle and its long wings, from the strategy factories."""
    low, high = round(0.88 * spot, 2), round(1.12 * spot, 2)
    return short_straddle(spot, EXPIRY), long_strangle(low, high, EXPIRY)


def build_book(mkt: Market, contracts: float) -> Book:
    costs = TransactionCosts(stock_bps=1.0, option_vol_spread=0.0025)
    book = Book("Iron butterfly book", cash=0.0, costs=costs)  # a desk funds itself: no idle capital
    straddle, wings = build_packages(float(mkt.spot))
    book.trade(straddle, contracts, mkt, note="sell the ATM straddle")
    book.trade(wings, contracts, mkt, note="buy the wings")
    return book


def print_risk_report(book: Book, mkt: Market) -> None:
    section(f"1. The book at inception (spot {mkt.spot:g}, implied vol {mkt.vol:.1%})")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:,.3f}".format):
        columns = ["label", "quantity", "unit_price", "value", "delta", "gamma", "vega", "theta"]
        print("Positions (Greeks in trader units: vega per vol point, theta per day)")
        print(book.positions_frame(mkt, include_cash=True)[columns].to_string(index=False))
        print(f"\nFees paid to open the book: {sum(t.fees for t in book.trades):,.2f}")

        print("\nDollar Greeks")
        explanations = {
            "delta_cash": "stock-equivalent exposure",
            "gamma_cash": "change of delta cash for +1% spot",
            "vega_cash": "P&L for +1 vol point",
            "theta_cash": "P&L of one calendar day",
            "rho_cash": "P&L for +1% in rates",
        }
        for name, value in book.dollar_greeks(mkt).items():
            print(f"  {name:<11} {value:>12,.2f}   ({explanations[name]})")
        print(f"  one-day breakeven move: {book.breakeven_move(mkt):.2%} "
              "(short gamma: the book earns theta as long as the spot moves LESS than this)")

        print("\nStress tests (full repricing)")
        print(book.stress_tests(mkt)[["spot", "vol", "pnl"]].to_string())


def describe_run(name: str, history: pd.DataFrame) -> None:
    totals = attribution_totals(history)
    print(f"\n{name}: final P&L {totals['actual']:+,.2f} "
          f"(lowest point on the way {history['pnl_cum'].min():+,.2f}, "
          f"{int(history['hedge_trades'].sum())} hedge trades)")
    print("  " + " | ".join(f"{term} {totals[term]:+,.1f}" for term in ATTRIBUTION_TERMS))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    mkt0 = Market(spot=args.spot, vol=args.vol, rate=args.rate, div=args.div)
    figures: dict[str, object] = {}

    # 1-2. The book and its risk ------------------------------------------------
    book = build_book(mkt0, args.contracts)
    print_risk_report(book, mkt0)
    figures["04_01_risk_profile"] = book_plots.book_risk_profile(book, mkt0)
    figures["04_02_scenario_heatmap"] = book_plots.book_scenario_heatmap(book, mkt0)
    figures["04_03_scenario_heatmap_1m"] = book_plots.book_scenario_heatmap(book, mkt0, horizon_days=30)
    figures["04_04_gamma_breakdown"] = book_plots.book_greeks_breakdown(book, mkt0, greek="gamma")
    figures["04_05_stress_tests"] = book_plots.stress_test_chart(book, mkt0)

    # 3. One scenario, two ways of running the book ---------------------------------
    scenario = gbm_scenario(
        mkt0, horizon=EXPIRY, n_steps=N_STEPS, realized_vol=args.realized_vol,
        implied_vol_model="spot_correlated", seed=args.seed,
    )
    section(f"2. Simulation: {scenario.name}")
    print(f"Spot path realised {scenario.realized_vol():.1%} (the book sold {args.vol:.1%}); "
          f"spot ends at {scenario.spot[-1]:.2f}, implied vol at {scenario.vol[-1]:.1%}.")

    hedged = TradingSimulator(book.copy("hedged daily"), scenario, DeltaHedgeEveryN(1)).run()
    unhedged = TradingSimulator(book.copy("not hedged"), scenario, NoHedge()).run()
    describe_run("Delta-hedged every day", hedged)
    describe_run("Not hedged", unhedged)

    scalping = gamma_scalping_summary(hedged)
    print(f"\nHedged book: gamma P&L {scalping['gamma_pnl']:+,.1f} vs theta P&L {scalping['theta_pnl']:+,.1f} "
          f"-> gamma + theta = {scalping['gamma_plus_theta']:+,.1f}")
    if scalping["gamma_plus_theta"] < 0:
        verdict = "the moves cost more than the theta collected"
    else:
        verdict = "the theta collected more than paid for the moves"
    print(f"  realised vol {scalping['realized_vol']:.1%} vs average implied {scalping['mean_implied_vol']:.1%}: "
          f"{verdict} (a SHORT-gamma book wants realised < implied).")

    figures["04_06_dashboard_hedged"] = sim_plots.simulation_dashboard(
        hedged, title="Delta-hedged every day: simulation dashboard")
    figures["04_07_dashboard_unhedged"] = sim_plots.simulation_dashboard(
        unhedged, title="Not hedged: simulation dashboard")
    figures["04_08_attribution_hedged"] = sim_plots.pnl_attribution_chart(
        hedged, title="Delta-hedged every day: cumulative P&L attribution (currency)")
    figures["04_09_attribution_unhedged"] = sim_plots.pnl_attribution_chart(
        unhedged, title="Not hedged: cumulative P&L attribution (currency)")
    figures["04_10_attribution_steps_hedged"] = sim_plots.pnl_attribution_chart(hedged, cumulative=False, x="step")
    figures["04_11_waterfall_hedged"] = sim_plots.pnl_attribution_waterfall(hedged)
    figures["04_12_gamma_vs_theta_hedged"] = sim_plots.gamma_theta_chart(hedged)

    # 4. Hedging-frequency experiment ---------------------------------------------------
    option = EuropeanOption("call", args.spot, 1.0)
    section(f"3. Hedging experiment: buy {option.label}, delta hedge to expiry ({args.paths} paths)")
    with pd.option_context("display.width", 200, "display.float_format", "{:,.4f}".format):
        fair = delta_hedging_experiment(option, mkt0, n_paths=args.paths, seed=args.seed)
        print(f"Realised = implied = {args.vol:.0%}: the mean is ~0 and std * sqrt(N) is ~constant "
              f"(premium {fair.attrs['premium']:.3f})")
        print(hedging_experiment_summary(fair).to_string())

        rich = delta_hedging_experiment(option, mkt0, n_paths=args.paths, seed=args.seed,
                                        realized_vol=args.realized_vol)
        print(f"\nRealised {args.realized_vol:.0%} vs implied {args.vol:.0%}: the option buyer earns the "
              "vol edge on average ('theory_mean' is 0.5 * Gamma * S^2 * (realised^2 - implied^2) dt)")
        print(hedging_experiment_summary(rich)[["mean", "stderr", "std", "theory_mean"]].to_string())

    figures["04_13_hedging_error_histogram"] = sim_plots.hedging_error_histogram(fair)
    figures["04_14_hedging_error_vs_frequency"] = sim_plots.hedging_error_vs_frequency(fair)
    figures["04_15_hedging_error_histogram_vol_edge"] = sim_plots.hedging_error_histogram(rich)

    section(f"Figures written to {args.outdir.resolve()}")
    for name, fig in figures.items():
        path = args.outdir / f"{name}.html"
        # "directory": one shared plotly.min.js next to the pages (small files, works offline).
        fig.write_html(path, include_plotlyjs="directory")
        print(f"  {path.name}")
        if args.show:
            webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    main()
