"""Exotic options: prices, Greeks, closed form vs Monte Carlo, and the pictures that explain them.

Run from the project root::

    python examples/03_exotics.py
    python examples/03_exotics.py --spot 250 --vol 0.35 --expiry 1.0 --paths 100000 --seed 7

The script

1. builds one of each exotic from ``EXOTIC_REGISTRY`` (digitals, the four
   barrier types, Asians, lookbacks) and prints price and Greeks next to the
   vanilla benchmark;
2. checks every closed form against a Monte Carlo price (with its standard
   error), using the right simulation for each product: plain paths, the
   Broadie-Glasserman-Kou correction for discretely watched barriers, a control
   variate for the Asian, Brownian-bridge extremes for the lookback;
3. writes the teaching figures into ``--outdir`` (``03_*.html``).

Nothing is opened in a browser unless ``--show`` is passed.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optionlab import black_scholes as bs  # noqa: E402
from optionlab.exotics import (  # noqa: E402
    EXOTIC_REGISTRY,
    AsianOption,
    BarrierOption,
    DigitalOption,
    LookbackOption,
    mc_price_brownian_bridge,
    mc_price_with_control_variate,
)
from optionlab.instruments import Instrument  # noqa: E402
from optionlab.market import Market  # noqa: E402
from optionlab.monte_carlo import mc_price  # noqa: E402
from optionlab.plotting import exotic_plots, profiles  # noqa: E402

TABLE_GREEKS = ("delta", "gamma", "vega", "theta")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spot", type=float, default=100.0)
    parser.add_argument("--vol", type=float, default=0.20, help="flat volatility (0.20 = 20%%)")
    # Not r - q = vol**2 / 2: there the at-the-money strike is exactly the median of S_T, d2 = 0,
    # and several numbers of the tour (digital MC error, asset-digital gamma) vanish by coincidence.
    parser.add_argument("--rate", type=float, default=0.04)
    parser.add_argument("--div", type=float, default=0.01)
    parser.add_argument("--expiry", type=float, default=0.5, help="expiry of every product, in years")
    parser.add_argument("--paths", type=int, default=40_000, help="Monte Carlo paths per product")
    parser.add_argument("--seed", type=int, default=11, help="seed of every simulation")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"), help="where the HTML files go")
    parser.add_argument("--show", action="store_true", help="open every figure in the browser")
    return parser.parse_args(argv)


def build_exotics(spot: float, expiry: float) -> dict[str, Instrument]:
    """One product per registry entry, at its default (spot-relative) parameters."""
    return {name: spec.build_default(spot, expiry) for name, spec in EXOTIC_REGISTRY.items()}


def vanilla_of(exotic: Instrument, mkt: Market) -> Instrument:
    return exotic_plots.vanilla_benchmark(exotic, mkt)


def greeks_table(exotics: dict[str, Instrument], mkt: Market) -> str:
    """Price and trader-unit Greeks of every exotic, next to its vanilla benchmark's price."""
    header = f"{'product':<26}{'price':>10}{'vanilla':>10}" + "".join(f"{g:>12}" for g in TABLE_GREEKS)
    lines = [header, "-" * len(header)]
    for exotic in exotics.values():
        greeks = bs.to_trader_units(exotic.greeks(mkt))
        cells = "".join(f"{greeks[g]:>12.5f}" for g in TABLE_GREEKS)
        vanilla = float(vanilla_of(exotic, mkt).price(mkt))
        lines.append(f"{exotic.label:<26}{greeks['price']:>10.4f}{vanilla:>10.4f}{cells}")
    lines.append("(vega per vol point, theta per calendar day)")
    return "\n".join(lines)


def monte_carlo_rows(
    exotic: Instrument, mkt: Market, n_paths: int, seed: int
) -> list[tuple[str, float, float, float]]:
    """``(method, reference closed form, MC price, stderr)`` rows for one product."""
    closed_form = float(exotic.price(mkt))
    tau = float(exotic.expiry - mkt.t)
    n_steps = max(int(round(252 * tau)), 1)  # daily monitoring
    if isinstance(exotic, DigitalOption):
        return [("terminal payoff, 1 step", closed_form, *mc_price(exotic, mkt, n_paths, 1, seed))]
    if isinstance(exotic, BarrierOption):
        discrete = float(exotic.discrete_barrier_adjusted(mkt.vol, dt=tau / n_steps).price(mkt))
        price, stderr = mc_price(exotic, mkt, n_paths, n_steps, seed)
        return [
            ("daily monitoring vs continuous formula", closed_form, price, stderr),
            ("daily monitoring vs BGK-shifted formula", discrete, price, stderr),
        ]
    if isinstance(exotic, AsianOption):
        price, stderr = mc_price_with_control_variate(exotic, mkt, n_paths, n_steps, seed)
        return [("control variate (geometric twin)", closed_form, price, stderr)]
    if isinstance(exotic, LookbackOption):
        return [
            ("daily monitoring (misses extremes)", closed_form, *mc_price(exotic, mkt, n_paths, n_steps, seed)),
            ("Brownian-bridge extremes (continuous)", closed_form,
             *mc_price_brownian_bridge(exotic, mkt, n_paths, 16, seed)),
        ]
    return [("plain Monte Carlo", closed_form, *mc_price(exotic, mkt, n_paths, n_steps, seed))]


def monte_carlo_table(exotics: dict[str, Instrument], mkt: Market, n_paths: int, seed: int) -> str:
    header = f"{'product':<26}{'method':<42}{'formula':>10}{'MC':>10}{'stderr':>9}{'gap/stderr':>12}"
    lines = [header, "-" * len(header)]
    for exotic in exotics.values():
        for method, reference, price, stderr in monte_carlo_rows(exotic, mkt, n_paths, seed):
            gap = (price - reference) / stderr if stderr > 0 else 0.0
            lines.append(f"{exotic.label:<26}{method:<42}{reference:>10.4f}{price:>10.4f}{stderr:>9.4f}{gap:>+12.1f}")
    lines.append(
        "A |gap| below ~3 stderr means 'same price'. Large gaps are the point of two rows: daily\n"
        "monitoring of a barrier or a lookback is NOT the continuous contract of the closed form\n"
        "(the BGK shift and the Brownian bridge repair it), and the arithmetic Asian formula is a\n"
        "lognormal approximation (a few stderr of the very precise control-variate estimate)."
    )
    return "\n".join(lines)


def build_figures(exotics: dict[str, Instrument], mkt: Market, seed: int) -> dict[str, object]:
    """The tour, in reading order: ``{file stem: figure}``."""
    light = dict(mode="light")  # the HTML pages are white: use full-contrast ink
    knock_out, knock_in = exotics["up_and_out_barrier"], exotics["down_and_in_barrier"]
    digital, asian = exotics["cash_digital"], exotics["asian"]
    # One week before expiry the step payoff shows its teeth: that is when replication gets hard.
    last_week = Market(mkt.spot, mkt.vol, mkt.rate, mkt.div, t=digital.expiry - 7 / 365)
    return {
        # 1-2. Barriers: the knock-out is short gamma and short vol near the barrier; the knock-in explodes there.
        "03_01_up_and_out_vs_vanilla": exotic_plots.exotic_vs_vanilla(knock_out, mkt, **light),
        "03_02_down_and_in_put_vs_vanilla": exotic_plots.exotic_vs_vanilla(knock_in, mkt, **light),
        # 3-4. A barrier option is a bet on the PATH.
        "03_03_up_and_out_paths": exotic_plots.barrier_paths_figure(knock_out, mkt, 80, seed, **light),
        "03_04_down_and_in_paths": exotic_plots.barrier_paths_figure(knock_in, mkt, 80, seed, **light),
        # 5. The generic plots work on exotics too: delta of the knock-out as expiry approaches.
        "03_05_up_and_out_delta_evolution": profiles.greek_evolution(knock_out, mkt, "delta", **light),
        # 6-7. Digital: delta is a bump that becomes a spike; desks hedge it as a call spread.
        "03_06_digital_vs_vanilla": exotic_plots.exotic_vs_vanilla(digital, mkt, **light),
        "03_07_digital_replication_last_week": exotic_plots.digital_replication_figure(digital, last_week, **light),
        # 8-9. Asian: same shapes as the vanilla, damped; the average calms down as it fills in.
        "03_08_asian_vs_vanilla": exotic_plots.exotic_vs_vanilla(asian, mkt, **light),
        "03_09_asian_averaging": exotic_plots.asian_averaging_figure(asian, mkt, seed, **light),
        # 10. Lookback: hindsight costs about twice the at-the-money vanilla.
        "03_10_lookback_vs_vanilla": exotic_plots.exotic_vs_vanilla(exotics["floating_lookback"], mkt, **light),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mkt = Market(spot=args.spot, vol=args.vol, rate=args.rate, div=args.div)
    exotics = build_exotics(args.spot, args.expiry)

    print(f"Market: spot={mkt.spot:g} vol={mkt.vol:.1%} rate={mkt.rate:.2%} div={mkt.div:.2%}; "
          f"every product expires at T={args.expiry:g}y\n")
    print("1. Prices and Greeks (closed forms; Greeks of path-dependent products by bump-and-reprice)")
    print(greeks_table(exotics, mkt))
    print(f"\n2. Closed form vs Monte Carlo ({args.paths} paths, seed {args.seed})")
    print(monte_carlo_table(exotics, mkt, args.paths, args.seed))

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"\n3. Writing figures to {args.outdir.resolve()}")
    for stem, fig in build_figures(exotics, mkt, args.seed).items():
        path = args.outdir / f"{stem}.html"
        # "directory": one shared plotly.min.js next to the pages (small files, works offline).
        fig.write_html(path, include_plotlyjs="directory")
        print(f"  {path.name}")
        if args.show:
            webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    main()
