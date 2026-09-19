"""Guided tour of the Greeks of a vanilla call and put.

Run from the project root::

    python examples/01_vanilla_greeks.py
    python examples/01_vanilla_greeks.py --strike 110 --vol 0.35 --expiry 0.5 --show

The script prints the price and Greeks of the two options (raw and trader
units) and writes one interactive HTML figure per step of the tour into
``outputs/`` (``01_*.html``). Nothing is opened unless ``--show`` is passed.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optionlab import black_scholes as bs  # noqa: E402
from optionlab.instruments import EuropeanOption, Position  # noqa: E402
from optionlab.market import Market  # noqa: E402
from optionlab.plotting import profiles  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spot", type=float, default=100.0, help="current spot (default 100)")
    parser.add_argument("--strike", type=float, default=100.0, help="strike of both options (default 100)")
    parser.add_argument("--vol", type=float, default=0.20, help="volatility, 0.20 = 20%% (default 0.20)")
    parser.add_argument("--rate", type=float, default=0.05, help="continuous interest rate (default 0.05)")
    parser.add_argument("--div", type=float, default=0.02, help="continuous dividend yield (default 0.02)")
    parser.add_argument("--expiry", type=float, default=1.0, help="expiry in years (default 1.0)")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="where the HTML files go")
    parser.add_argument("--show", action="store_true", help="open every figure in the browser")
    return parser.parse_args(argv)


def greeks_table(call: EuropeanOption, put: EuropeanOption, mkt: Market) -> str:
    """Price and Greeks of the two options, raw and in trader units, as plain text."""
    columns = []
    for option in (call, put):
        raw = option.greeks(mkt)
        columns += [raw, bs.to_trader_units(raw)]
    header = f"{'':<8}{'call raw':>14}{'call trader':>14}{'put raw':>14}{'put trader':>14}   trader unit"
    lines = [header, "-" * len(header)]
    for key in columns[0]:
        cells = "".join(f"{column[key]:>14.6g}" for column in columns)
        lines.append(f"{key:<8}{cells}   {bs.GREEK_INFO[key]['trader_unit']}")
    return "\n".join(lines)


def build_figures(call: EuropeanOption, put: EuropeanOption, mkt: Market) -> dict[str, object]:
    """The tour, in reading order: ``{file stem: figure}``."""
    strike, expiry = call.strike, call.expiry
    light = dict(mode="light")  # the HTML pages are white: use full-contrast ink
    return {
        # 1-2. What does each option look like? Six Greeks against spot, today's market marked.
        "01_01_call_dashboard": profiles.greek_dashboard(call, mkt, **light),
        "01_02_put_dashboard": profiles.greek_dashboard(put, mkt, **light),
        # 3. Put-call parity in pictures: same gamma and vega, deltas one forward apart.
        "01_03_call_vs_put": profiles.call_put_comparison(strike, expiry, mkt, **light),
        # 4-5. THE classic: delta becomes a step and gamma a spike as expiry approaches.
        "01_04_delta_evolution": profiles.greek_evolution(call, mkt, "delta", **light),
        "01_05_gamma_evolution": profiles.greek_evolution(call, mkt, "gamma", **light),
        # 6. Less vol does to vega-per-spot what less time does: optionality concentrates at the strike.
        "01_06_vega_by_vol_level": profiles.greek_evolution(call, mkt, "vega", vary="vol", **light),
        # 7. Time decay is not linear: at-the-money theta accelerates into expiry.
        "01_07_theta_through_time": profiles.greek_vs_time(put, mkt, "theta", **light),
        # 8-9. Two dimensions at once.
        "01_08_gamma_surface": profiles.greek_surface(call, mkt, "gamma", **light),
        "01_09_vanna_heatmap": profiles.greek_surface(call, mkt, "vanna", y="vol", kind="heatmap", **light),
        # 10. From today's value curve to the hockey stick: the gap is time value.
        "01_10_call_pnl_profile": profiles.pnl_profile(call, mkt, **light),
        # 11. What the Greeks are for: explaining the P&L of a move (here with +2 vol points over a week).
        "01_11_taylor_pnl_explain": profiles.taylor_pnl_explain(call, mkt, dvol=0.02, dt=7 / 365, **light),
        # 12. Long gamma pays rent (theta); short gamma collects it and fears the move.
        "01_12_gamma_theta_long_call": profiles.gamma_theta_tradeoff(call, mkt, **light),
        "01_13_gamma_theta_short_put": profiles.gamma_theta_tradeoff(Position(put, -1.0), mkt, **light),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mkt = Market(spot=args.spot, vol=args.vol, rate=args.rate, div=args.div)
    call = EuropeanOption("call", args.strike, args.expiry)
    put = EuropeanOption("put", args.strike, args.expiry)

    print(f"Market: spot={mkt.spot:g} vol={mkt.vol:.1%} rate={mkt.rate:.2%} div={mkt.div:.2%}")
    print(f"Options: strike={args.strike:g} expiry={args.expiry:g}y\n")
    print(greeks_table(call, put, mkt))

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting figures to {args.outdir.resolve()}")
    for stem, fig in build_figures(call, put, mkt).items():
        path = args.outdir / f"{stem}.html"
        # "directory": one shared plotly.min.js next to the pages (small files, works offline).
        fig.write_html(path, include_plotlyjs="directory")
        print(f"  {path.name}")
        if args.show:
            webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    main()
