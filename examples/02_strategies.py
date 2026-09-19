"""Predefined option strategies: summary tables, payoff diagrams and Greeks dashboards.

Run from the project root::

    python examples/02_strategies.py --list
    python examples/02_strategies.py --strategy iron_condor calendar_spread
    python examples/02_strategies.py --all --spot 250 --vol 0.35 --expiry 0.5

For every chosen strategy the script prints a summary (premium, breakevens,
maximum profit / loss, Greeks, legs) and writes two interactive HTML figures
into ``--outdir`` (``02_<strategy>_payoff.html`` and ``02_<strategy>_greeks.html``):
the payoff diagram and the leg-by-leg Greeks dashboard. Nothing is opened in a
browser unless ``--show`` is passed.
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optionlab.market import Market  # noqa: E402
from optionlab.plotting.strategy_plots import payoff_diagram, strategy_greeks_dashboard  # noqa: E402
from optionlab.strategies import STRATEGY_REGISTRY, get_strategy_spec, legs_table, summary  # noqa: E402

DEFAULT_SELECTION = ("bull_call_spread", "long_straddle", "iron_condor", "calendar_spread")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--strategy", nargs="+", metavar="NAME", help="one or more registry names")
    choice.add_argument("--all", action="store_true", help="every predefined strategy")
    choice.add_argument("--list", action="store_true", help="list the available strategies and exit")
    parser.add_argument("--spot", type=float, default=100.0)
    parser.add_argument("--vol", type=float, default=0.20, help="flat volatility (0.20 = 20%%)")
    parser.add_argument("--rate", type=float, default=0.03)
    parser.add_argument("--div", type=float, default=0.0)
    parser.add_argument("--expiry", type=float, default=0.5, help="(front) expiry in years")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--show", action="store_true", help="open the figures in the browser")
    return parser.parse_args(argv)


def print_catalogue() -> None:
    category = None
    for name, spec in STRATEGY_REGISTRY.items():
        if spec.category != category:
            category = spec.category
            print(f"\n{category}")
        print(f"  {name:<22} {spec.title:<22} [{', '.join(spec.view)}]")


def format_pnl(value: float) -> str:
    return "unlimited" if math.isinf(value) else f"{value:+.4f}"


def print_summary(report: dict, legs) -> None:
    print("=" * 78)
    print(f"{report['name']}   [{', '.join(report['view'])}]")
    print(textwrap.fill(report["description"], width=78))
    print("-" * 78)
    print(f"net premium      : {report['net_premium']:+.4f} ({report['premium_type']})")
    horizon = "front expiry" if report["is_multi_expiry"] else "expiry"
    print(f"analysed at      : {horizon} T={report['horizon']:.2f}")
    print(f"breakevens       : {', '.join(f'{b:.2f}' for b in report['breakevens']) or 'none'}")
    print(f"max profit       : {format_pnl(report['max_profit'])}")
    print(f"max loss         : {format_pnl(report['max_loss'])}")
    print(f"prob. of profit  : {report['probability_of_profit']:.1%} (risk-neutral)")
    trader = report["greeks_trader"]
    print(
        "Greeks (trader)  : "
        f"delta {trader['delta']:+.4f} | gamma {trader['gamma']:+.5f} | "
        f"vega/pt {trader['vega']:+.4f} | theta/day {trader['theta']:+.4f} | rho/1% {trader['rho']:+.4f}"
    )
    columns = ["leg", "unit_price", "value", "delta", "gamma", "vega", "theta"]
    print(legs[columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.list:
        print_catalogue()
        return
    if args.all:
        names = list(STRATEGY_REGISTRY)
    else:
        try:
            names = [get_strategy_spec(name).name for name in (args.strategy or DEFAULT_SELECTION)]
        except ValueError as exc:
            raise SystemExit(f"error: {exc} (use --list to see them with their market view)")

    mkt = Market(spot=args.spot, vol=args.vol, rate=args.rate, div=args.div)
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name in names:
        strategy = STRATEGY_REGISTRY[name].build_default(args.spot, args.expiry)
        print_summary(summary(strategy, mkt), legs_table(strategy, mkt, trader_units=True))
        figures = {  # the HTML pages are white: use full-contrast ink
            "payoff": payoff_diagram(strategy, mkt, mode="light"),
            "greeks": strategy_greeks_dashboard(strategy, mkt, mode="light"),
        }
        for kind, fig in figures.items():
            path = args.outdir / f"02_{name}_{kind}.html"
            # "directory": one shared plotly.min.js next to the pages (small files, works offline).
            fig.write_html(path, include_plotlyjs="directory")
            print(f"wrote {path}")
            if args.show:
                webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    main()
