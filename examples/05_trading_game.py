"""A terminal trading game: run an option book through a market you do not control.

Run from the project root::

    python examples/05_trading_game.py                 # interactive
    python examples/05_trading_game.py --demo          # plays a scripted game (no typing)
    python examples/05_trading_game.py --seed 3 --script "buy 10 straddle 100 42d; hedge; next 42"

You get some capital and a market: a spot, ONE implied volatility (flat
surface) and a clock that advances one trading day per ``next``. Options are
always quoted at the current implied vol -- but the vol at which the spot will
REALLY move is hidden until the end. That is the game: decide whether options
are cheap or rich, take the position, manage its Greeks, and read the P&L
attribution to see whether you were paid for the risk you meant to take.

Commands (type ``help`` in the game)::

    buy 10 call 105 0.25     sell 5 put 95 21d      buy 3 straddle 100 42d
    stock -300               hedge                  autohedge on
    next 5                   book                   risk
    quote put 95 21d         chain 42d              explain
    close 2                  close all              blotter        quit

An expiry is an ABSOLUTE time in years on the game clock (``0.25``) or ``Nd``,
meaning "N trading days from today" (``21d``), which always lands exactly on a
game date. At the end you get a P&L attribution report, and interactive HTML
figures are written into ``--outdir``. Nothing is opened in a browser unless
``--show`` is passed.
"""

from __future__ import annotations

import argparse
import math
import sys
import webbrowser
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optionlab.black_scholes import to_trader_units  # noqa: E402
from optionlab.book import Book, TransactionCosts  # noqa: E402
from optionlab.instruments import CompositeInstrument, EuropeanOption, Instrument, Underlying  # noqa: E402
from optionlab.market import Market  # noqa: E402
from optionlab.plotting import sim_plots  # noqa: E402
from optionlab.simulator import (  # noqa: E402
    DeltaHedgeEveryN,
    NoHedge,
    TradingSimulator,
    attribution_totals,
    gamma_scalping_summary,
    gbm_scenario,
)

TRADING_DAYS = 252

HELP = """\
Commands   (expiry = absolute time in years, e.g. 0.25, or Nd = N trading days from today, e.g. 21d)
  buy  <qty> call|put|straddle <strike> <expiry>     buy 10 call 105 0.25
  sell <qty> call|put|straddle <strike> <expiry>     sell 5 put 95 21d
  buy|sell <qty> stock                               sell 300 stock
  stock <signed qty>                                 stock -300      (sells 300 shares)
  hedge [target]           trade the stock so that the book delta is 0 (or <target> shares)
  autohedge on|off         delta hedge automatically at the end of every day
  close <line#>|all        close one line of the 'book' table, or everything
  quote call|put|straddle <strike> <expiry>          price and Greeks of one unit
  chain [expiry]           calls and puts around the money (default expiry: the last game date)
  next [n]                 let n trading days pass (default 1)
  book      positions and their Greeks        risk     dollar Greeks, spot ladder, stress tests
  explain   P&L attribution so far            blotter  your trades
  help                                        quit     stop now and get the final report"""

DEMO_SCRIPT = (
    "help; chain 42d; buy 300 straddle 100 42d; hedge; book; next 5; risk; "
    "sell 150 call 110 42d; autohedge on; next 10; quote put 95 42d; buy 100 put 95 42d; "
    "buy lots of calls; sell 5 call -20 42d; frobnicate; close 9; "
    "next 15; explain; close 1; next 100"
)

#: Seed of the scripted games when none is given (a market where the spot stays near the strikes).
DEMO_SEED = 9

#: Display groups of the attribution terms used in the day-by-day report.
REPORT_GROUPS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("delta", ("delta",), True),
    ("gamma", ("gamma",), True),
    ("theta", ("theta",), True),
    ("vega", ("vega",), True),
    ("vanna+volga", ("vanna", "volga"), False),
    ("carry+rho", ("carry", "rho"), False),
    ("fees+edge", ("fees", "trading"), False),
    ("unexplained", ("unexplained",), True),
)

OPTION_KINDS = {"call": "call", "calls": "call", "c": "call", "put": "put", "puts": "put", "p": "put"}
STOCK_WORDS = {"stock", "stocks", "share", "shares", "underlying"}


# ---------------------------------------------------------------------- #
# Parsing helpers (every failure is a ValueError with a message for the player)
# ---------------------------------------------------------------------- #
def parse_number(token: str, what: str) -> float:
    try:
        value = float(token)
    except ValueError:
        raise ValueError(f"{what} must be a number, got {token!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{what} must be finite, got {token!r}")
    return value


def parse_positive(token: str, what: str) -> float:
    value = parse_number(token, what)
    if value <= 0:
        raise ValueError(f"{what} must be positive, got {token!r}")
    return value


def nice_step(spot: float) -> float:
    """Strike spacing of the option chain: a round number close to 5% of the spot."""
    target = 0.05 * spot
    magnitude = 10.0 ** math.floor(math.log10(target))
    return min((m * magnitude for m in (1.0, 2.0, 2.5, 5.0, 10.0)), key=lambda step: abs(step - target))


# ---------------------------------------------------------------------- #
# The game
# ---------------------------------------------------------------------- #
class TradingGame:
    """Command interpreter around a :class:`~optionlab.simulator.TradingSimulator`.

    ``execute`` never raises on bad input: it prints what was wrong and the
    game goes on. ``running`` turns False on ``quit``, at the end of the
    scenario, or when the net liquidation value is wiped out.
    """

    def __init__(self, sim: TradingSimulator, say: Callable[[str], None] = print) -> None:
        self.sim = sim
        self.say = say
        self.running = True
        self.commands: dict[str, Callable[[list[str]], None]] = {
            "buy": lambda args: self._order(args, +1.0),
            "sell": lambda args: self._order(args, -1.0),
            "stock": self._stock,
            "hedge": self._hedge,
            "autohedge": self._autohedge,
            "close": self._close,
            "quote": self._quote,
            "chain": self._chain,
            "next": self._next,
            "n": self._next,
            "book": self._book,
            "risk": self._risk,
            "explain": self._explain,
            "blotter": self._blotter,
            "help": lambda args: self.say(HELP),
            "?": lambda args: self.say(HELP),
            "quit": self._quit,
            "exit": self._quit,
            "q": self._quit,
        }

    # --- shortcuts ----------------------------------------------------------------
    @property
    def book(self) -> Book:
        return self.sim.book

    @property
    def mkt(self) -> Market:
        return self.sim.current_market

    @property
    def prompt(self) -> str:
        return f"day {self.sim.step_index} > "

    # --- interpreter ----------------------------------------------------------------
    def execute(self, line: str) -> None:
        tokens = line.strip().lower().split()
        if not tokens:
            return
        handler = self.commands.get(tokens[0])
        if handler is None:
            self.say(f"  ! unknown command {tokens[0]!r} (type 'help')")
            return
        try:
            handler(tokens[1:])
        except Exception as exc:  # noqa: BLE001 - a typo must never end the game
            self.say(f"  ! {exc}")

    # --- instruments ------------------------------------------------------------------
    def _expiry(self, token: str) -> float:
        scenario, mkt = self.sim.scenario, self.mkt
        if token.endswith("d"):
            days = parse_positive(token[:-1], "the number of days")
            if days != int(days):
                raise ValueError(f"use a whole number of trading days, got {token!r}")
            index = self.sim.step_index + int(days)
            if index <= scenario.n_steps:
                return float(scenario.times[index])
            day_length = scenario.horizon / scenario.n_steps
            return float(scenario.times[-1]) + (index - scenario.n_steps) * day_length
        expiry = parse_number(token, "expiry")
        if expiry <= mkt.t + 1e-9:
            raise ValueError(
                f"expiry {expiry:g} is not in the future: the clock is at t = {mkt.t:.4f} "
                "(an expiry is an absolute time in years, or Nd for N trading days from today)"
            )
        return expiry

    def _option(self, kind: str, strike_token: str, expiry_token: str) -> Instrument:
        strike = parse_positive(strike_token, "strike")
        expiry = self._expiry(expiry_token)
        if kind == "straddle":
            legs = (EuropeanOption("call", strike, expiry), EuropeanOption("put", strike, expiry))
            return CompositeInstrument(f"Straddle {strike:g} T={expiry:.2f}", legs)
        if kind not in OPTION_KINDS:
            raise ValueError(f"unknown product {kind!r}: use call, put, straddle or stock")
        return EuropeanOption(OPTION_KINDS[kind], strike, expiry)

    def _trade(self, instrument: Instrument, quantity: float, note: str) -> None:
        trade = self.sim.trade(instrument, quantity, note=note)
        verb = "BOUGHT" if quantity > 0 else "SOLD"
        self.say(
            f"  {verb} {abs(quantity):g} x {instrument.label} @ {trade.price:,.4f}"
            f"   cash {trade.cash_flow:+,.2f} (of which fees {trade.fees:,.2f})"
        )
        self._greeks_line()

    # --- trading commands -----------------------------------------------------------------
    def _order(self, args: list[str], sign: float) -> None:
        usage = "usage: buy|sell <qty> call|put|straddle <strike> <expiry>   or   buy|sell <qty> stock"
        if len(args) == 2 and args[1] in STOCK_WORDS:
            self._trade(Underlying(), sign * parse_positive(args[0], "quantity"), "player")
        elif len(args) == 4:
            quantity = parse_positive(args[0], "quantity")
            self._trade(self._option(args[1], args[2], args[3]), sign * quantity, "player")
        else:
            raise ValueError(usage)

    def _stock(self, args: list[str]) -> None:
        if len(args) != 1:
            raise ValueError("usage: stock <signed qty>   (stock -300 sells 300 shares)")
        quantity = parse_number(args[0], "quantity")
        if quantity == 0:
            raise ValueError("quantity must be non-zero")
        self._trade(Underlying(), quantity, "player")

    def _hedge(self, args: list[str]) -> None:
        if len(args) > 1:
            raise ValueError("usage: hedge [target delta in shares]")
        target = parse_number(args[0], "target delta") if args else 0.0
        trade = self.book.hedge_delta(self.mkt, target, note="player delta hedge")
        if trade is None:
            self.say(f"  the book delta is already {target:g}: nothing to do")
            return
        verb = "BOUGHT" if trade.quantity > 0 else "SOLD"
        self.say(f"  {verb} {abs(trade.quantity):,.2f} shares @ {trade.price:,.2f} (fees {trade.fees:,.2f})")
        self._greeks_line()

    def _autohedge(self, args: list[str]) -> None:
        if args not in (["on"], ["off"]):
            raise ValueError("usage: autohedge on|off")
        if args == ["on"]:
            self.sim.policy = DeltaHedgeEveryN(1)
            self.say("  autohedge ON: the book delta is brought back to 0 at the end of every day")
            if not self.book.is_empty:
                self._hedge([])
        else:
            self.sim.policy = NoHedge()
            self.say("  autohedge OFF: the delta is yours to manage")

    def _close(self, args: list[str]) -> None:
        if len(args) != 1:
            raise ValueError("usage: close <line#>|all   (line numbers are in the 'book' table)")
        if self.book.is_empty:
            raise ValueError("the book is empty")
        if args[0] == "all":
            trades = self.book.liquidate(self.mkt, note="player close")
        else:
            number = parse_number(args[0], "line number")
            positions = self.book.positions
            if number != int(number) or not 1 <= number <= len(positions):
                raise ValueError(f"line number must be between 1 and {len(positions)}")
            instrument = positions[int(number) - 1].instrument
            trades = [self.book.close_position(instrument, self.mkt, note="player close")]
        for trade in trades:
            verb = "BOUGHT" if trade.quantity > 0 else "SOLD"
            self.say(f"  {verb} {abs(trade.quantity):g} x {trade.instrument.label} @ {trade.price:,.4f} "
                     f"(fees {trade.fees:,.2f})")
        self._greeks_line()

    # --- information commands ---------------------------------------------------------------
    def _quote(self, args: list[str]) -> None:
        if len(args) != 3:
            raise ValueError("usage: quote call|put|straddle <strike> <expiry>")
        instrument = self._option(*args)
        greeks = to_trader_units(instrument.greeks(self.mkt))
        self.say(
            f"  {instrument.label}: price {greeks['price']:,.4f} | delta {greeks['delta']:+.3f} | "
            f"gamma {greeks['gamma']:.4f} | vega/pt {greeks['vega']:.4f} | theta/day {greeks['theta']:+.4f}"
            f"   (implied vol {self.mkt.vol:.1%}, {instrument.expiry - self.mkt.t:.3f} y to expiry)"
        )

    def _chain(self, args: list[str]) -> None:
        if len(args) > 1:
            raise ValueError("usage: chain [expiry]")
        expiry = self._expiry(args[0]) if args else float(self.sim.scenario.times[-1])
        if expiry <= self.mkt.t + 1e-9:
            raise ValueError("the game is on its last date: give an expiry, e.g. chain 21d")
        mkt, step = self.mkt, nice_step(float(self.mkt.spot))
        centre = round(float(mkt.spot) / step) * step
        rows = []
        for strike in (centre + k * step for k in range(-3, 4)):
            if strike <= 0:
                continue
            call = EuropeanOption("call", strike, expiry).greeks(mkt)
            put = EuropeanOption("put", strike, expiry).greeks(mkt)
            rows.append({"call delta": call["delta"], "call": call["price"], "STRIKE": strike,
                         "put": put["price"], "put delta": put["delta"]})
        self.say(f"  Expiry t = {expiry:.4f} ({expiry - mkt.t:.3f} y away), spot {mkt.spot:,.2f}, "
                 f"implied vol {mkt.vol:.1%}")
        self._table(pd.DataFrame(rows), "{:,.3f}")

    def _book(self, args: list[str]) -> None:
        columns = ["label", "quantity", "unit_price", "value", "delta", "gamma", "vega", "theta"]
        frame = self.book.positions_frame(self.mkt, include_cash=True)[columns]
        frame.insert(0, "#", [str(i + 1) for i in range(len(self.book))] + ["", ""])
        self.say("  Position Greeks in trader units: vega per vol point, theta per calendar day")
        self._table(frame, "{:,.3f}")

    def _risk(self, args: list[str]) -> None:
        if self.book.is_empty:
            self.say("  the book is empty: no market risk (cash only)")
            return
        mkt = self.mkt
        dollars = self.book.dollar_greeks(mkt)
        self.say(f"  delta cash {dollars['delta_cash']:+,.0f} | gamma cash per +1% {dollars['gamma_cash']:+,.0f} | "
                 f"vega per vol pt {dollars['vega_cash']:+,.1f} | theta per day {dollars['theta_cash']:+,.1f}")
        breakeven = self.book.breakeven_move(mkt)
        if math.isfinite(breakeven):
            side = "MORE" if dollars["gamma_cash"] > 0 else "LESS"
            self.say(f"  one-day breakeven move {breakeven:.2%}: you need the spot to move {side} than that")
        ladder = self.book.spot_ladder(mkt, shocks=(-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10))
        ladder.index = [f"{shock:+.0%}" for shock in ladder.index]
        self.say("  Spot ladder (full repricing; Greeks in trader units)")
        self._table(ladder[["spot", "pnl", "delta", "gamma", "vega", "theta"]], "{:,.2f}", index=True)
        self.say("  Stress tests")
        self._table(self.book.stress_tests(mkt)[["pnl"]], "{:+,.2f}", index=True)

    def _explain(self, args: list[str]) -> None:
        history = self.sim.history
        if len(history) < 2:
            self.say("  nothing to explain yet: let at least one day pass ('next')")
            return
        self._attribution_lines(attribution_totals(history))

    def _blotter(self, args: list[str]) -> None:
        if not self.book.trades:
            self.say("  no trades yet")
            return
        self._table(self.book.blotter_frame(), "{:,.4f}")

    def _quit(self, args: list[str]) -> None:
        self.running = False

    # --- the clock ------------------------------------------------------------------------------
    def _next(self, args: list[str]) -> None:
        if len(args) > 1:
            raise ValueError("usage: next [number of days]")
        days = parse_positive(args[0], "the number of days") if args else 1.0
        if days != int(days):
            raise ValueError("the number of days must be a whole number")
        start = self.mkt
        n_settled = len(self.book.settlements)
        rows = []
        for _ in range(min(int(days), self.sim.steps_remaining)):
            rows.append(self.sim.step())
            if self.book.value(self.mkt) <= 0:
                break
        end = self.mkt
        self.say(
            f"  {len(rows)} day(s) pass: spot {start.spot:,.2f} -> {end.spot:,.2f} "
            f"({end.spot / start.spot - 1.0:+.2%}), implied vol {start.vol:.1%} -> {end.vol:.1%} "
            f"({100.0 * (end.vol - start.vol):+.1f} pts)"
        )
        parts = []
        for name, terms, always in REPORT_GROUPS:
            amount = sum(row[f"pnl_{term}"] for row in rows for term in terms)
            if always or abs(amount) >= 0.005:
                parts.append(f"{name} {amount:+,.2f}")
        self.say(f"  P&L {sum(row['pnl_step'] for row in rows):+,.2f}  =  " + " | ".join(parts))
        hedge_shares = sum(row["hedge_shares"] for row in rows)
        if any(row["hedge_trades"] for row in rows):
            self.say(f"  autohedge traded {hedge_shares:+,.2f} shares net over the period")
        for record in self.book.settlements[n_settled:]:
            self.say(f"  EXPIRED {record.quantity:+g} x {record.instrument.label} with the spot at "
                     f"{record.spot:,.2f}: cash {record.cash_flow:+,.2f}")
        self.status()
        if self.book.value(end) <= 0:
            self.say("  MARGIN CALL: your net liquidation value is wiped out. Game over.")
            self.running = False
        elif self.sim.is_finished:
            self.say("  That was the last trading day.")
            self.running = False

    # --- display ----------------------------------------------------------------------------------
    def _table(self, frame: pd.DataFrame, number_format: str, index: bool = False) -> None:
        text = frame.to_string(index=index, float_format=number_format.format, na_rep="")
        self.say("\n".join("    " + line for line in text.splitlines()))

    def _greeks_line(self) -> None:
        if self.book.is_empty:
            self.say("  book is flat (cash only)")
            return
        dollars = self.book.dollar_greeks(self.mkt)
        delta = self.book.greeks(self.mkt)["delta"]
        self.say(
            f"  Greeks: delta {delta:+,.1f} sh | gamma cash per +1% {dollars['gamma_cash']:+,.0f} | "
            f"vega per vol pt {dollars['vega_cash']:+,.1f} | theta per day {dollars['theta_cash']:+,.1f}"
        )

    def status(self) -> None:
        mkt, scenario = self.mkt, self.sim.scenario
        self.say(
            f"\nDay {self.sim.step_index}/{scenario.n_steps} | t = {mkt.t:.4f} y (last date {scenario.times[-1]:.4f}) "
            f"| spot {mkt.spot:,.2f} | implied vol {mkt.vol:.1%}"
        )
        self.say(
            f"  NLV {self.book.value(mkt):,.2f} | P&L {self.book.pnl(mkt):+,.2f} | cash {self.book.cash:,.2f} "
            f"| {len(self.book)} line(s) | autohedge {'off' if isinstance(self.sim.policy, NoHedge) else 'on'}"
        )
        self._greeks_line()

    def _attribution_lines(self, totals: dict[str, float]) -> None:
        scale = max(abs(value) for value in totals.values()) or 1.0
        for name, value in totals.items():
            if name == "actual":
                self.say(f"    {'-' * 58}")
            bar = "#" * int(round(24 * abs(value) / scale))
            self.say(f"    {name:<12} {value:>14,.2f}  {'+' if value >= 0 else '-'}{bar}")

    # --- end of game ---------------------------------------------------------------------------------
    def final_report(self, realized_vol: float) -> None:
        history = self.sim.history
        mkt = self.mkt
        capital = self.book.initial_cash
        pnl = float(self.book.pnl(mkt))
        self.say(f"\n{'=' * 88}\nFINAL REPORT after {self.sim.step_index} trading day(s)\n{'=' * 88}")
        ratio = f" ({pnl / capital:+.2%} of your capital)" if capital > 0 else ""
        self.say(f"  NLV {self.book.value(mkt):,.2f}  |  P&L {pnl:+,.2f}{ratio}")
        if len(history) < 2:
            self.say("  No day passed, so there is no P&L to explain.")
            return
        drawdown = float((history["pnl_cum"].cummax() - history["pnl_cum"]).max())
        fees = float(history["fees_paid"].sum())
        self.say(f"  worst drawdown {drawdown:,.2f} | {len(self.book.trades)} trade(s) | fees paid {fees:,.2f}")
        self.say("\n  Where the P&L came from (the terms add up to the actual P&L)")
        totals = attribution_totals(history)
        self._attribution_lines(totals)

        summary = gamma_scalping_summary(history)
        self.say(f"\n  The hidden number: the spot was simulated with a volatility of {realized_vol:.1%}; "
                 f"this path realised {summary['realized_vol']:.1%}.")
        self.say(f"  Options were quoted at {summary['mean_implied_vol']:.1%} implied on average "
                 f"({history['implied_vol'].iloc[0]:.1%} at the start, {history['implied_vol'].iloc[-1]:.1%} at the end).")
        for line in coach(totals, summary):
            self.say(f"  * {line}")


def coach(totals: dict[str, float], summary: dict[str, float]) -> list[str]:
    """A few sentences reading the attribution the way a desk head would."""
    lines = []
    market_terms = ("delta", "gamma", "theta", "vega")
    gross = sum(abs(totals[term]) for term in market_terms)
    if gross < 1e-9:
        return ["You carried no market risk: no risk, no P&L, nothing to learn. Put a position on next time."]
    if abs(totals["delta"]) > 0.5 * gross:
        lines.append(
            f"Most of your P&L was DIRECTIONAL (delta {totals['delta']:+,.0f}). That is a bet on the spot, not on "
            "volatility: 'hedge' or 'autohedge on' removes it and leaves the pure vol position."
        )
    scalping = summary["gamma_plus_theta"]
    long_gamma = totals["gamma"] > 0 and totals["theta"] < 0
    short_gamma = totals["gamma"] < 0 and totals["theta"] > 0
    if long_gamma or short_gamma:
        if long_gamma:
            story = (f"You were LONG gamma: the moves earned {totals['gamma']:+,.0f} and theta cost "
                     f"{totals['theta']:+,.0f} (net {scalping:+,.0f}).")
        else:
            story = (f"You were SHORT gamma: theta paid you {totals['theta']:+,.0f} and the moves cost "
                     f"{totals['gamma']:+,.0f} (net {scalping:+,.0f}).")
        gap = (f"realised vol {summary['realized_vol']:.1%} vs {summary['mean_implied_vol']:.1%} implied "
               "on average over the whole game")
        market_helped = (summary["vol_edge"] > 0) == long_gamma
        if market_helped == (scalping > 0):
            story += f" That is what {gap} should do to a {'long' if long_gamma else 'short'}-gamma book."
        else:
            story += (f" That goes against {gap}: gamma P&L is path dependent -- what counts is how much the "
                      "spot moved WHILE you held the gamma, and how close to your strikes it was.")
        lines.append(story)
    if abs(totals["vega"]) > 0.15 * gross:
        lines.append(
            f"Vega P&L {totals['vega']:+,.0f}: the re-marking of your options as implied vol moved. "
            "Long options gain when implied rises (typically when the spot falls)."
        )
    if abs(totals["carry"]) > 0.25 * max(abs(totals["actual"]), 1e-9):
        lines.append(
            f"Carry {totals['carry']:+,.0f} is mostly the interest on your cash, earned whatever you trade: "
            f"your P&L from taking risk was {totals['actual'] - totals['carry']:+,.0f}."
        )
    if -totals["fees"] > 0.1 * max(abs(totals["actual"]), 1e-9):
        lines.append(
            f"Fees cost {-totals['fees']:,.0f}: every hedge crosses a spread. Hedging less often (or inside a "
            "band) keeps more of the edge, at the price of a noisier P&L."
        )
    if abs(totals["unexplained"]) > 0.2 * gross:
        lines.append(
            f"A large unexplained term ({totals['unexplained']:+,.0f}) means start-of-day Greeks were not enough: "
            "big gaps, an expiry close to the strike, or Greeks that change fast (short-dated options)."
        )
    return lines


# ---------------------------------------------------------------------- #
# Set-up and main loop
# ---------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=None, help="replay a game (default: a new random market)")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--spot", type=float, default=100.0)
    parser.add_argument("--vol", type=float, default=0.20, help="starting implied volatility (0.20 = 20%%)")
    parser.add_argument("--realized-vol", type=float, default=None,
                        help="vol the spot really moves at (default: hidden, between 0.6x and 1.5x the implied)")
    parser.add_argument("--drift", type=float, default=0.05, help="real-world expected return of the spot")
    parser.add_argument("--rate", type=float, default=0.02)
    parser.add_argument("--div", type=float, default=0.0)
    parser.add_argument("--days", type=int, default=63, help="length of the game in trading days")
    parser.add_argument("--vol-model", default="spot_correlated",
                        choices=["constant", "mean_reverting", "spot_correlated"],
                        help="dynamics of the implied vol")
    parser.add_argument("--jumps", type=float, default=0.0, help="expected number of spot jumps per year")
    parser.add_argument("--no-costs", action="store_true", help="trade at the model price, no spread")
    parser.add_argument("--script", default=None, help='commands separated by ";" (non-interactive)')
    parser.add_argument("--demo", action="store_true", help="play a built-in script (non-interactive)")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--show", action="store_true", help="open the figures in the browser at the end")
    return parser.parse_args(argv)


def build_game(args: argparse.Namespace) -> tuple[TradingGame, float, int]:
    """Create the market scenario, the book and the game; returns ``(game, hidden vol, seed)``."""
    if args.days < 1:
        raise ValueError("--days must be at least 1")
    seed = args.seed
    if seed is None:
        scripted = args.demo or args.script is not None
        seed = DEMO_SEED if scripted else int(np.random.default_rng().integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    hidden_vol = args.realized_vol if args.realized_vol is not None else args.vol * float(rng.uniform(0.6, 1.5))
    mkt0 = Market(spot=args.spot, vol=args.vol, rate=args.rate, div=args.div)
    scenario = gbm_scenario(
        mkt0, horizon=args.days / TRADING_DAYS, n_steps=args.days, realized_vol=hidden_vol, drift=args.drift,
        implied_vol_model=args.vol_model, jump_intensity=args.jumps, seed=rng, name="trading game",
    )
    costs = None if args.no_costs else TransactionCosts(stock_bps=1.0, option_vol_spread=0.0025)
    book = Book("Player", cash=args.capital, costs=costs)
    return TradingGame(TradingSimulator(book, scenario, NoHedge())), hidden_vol, seed


def write_outputs(game: TradingGame, outdir: Path, show: bool) -> None:
    history = game.sim.history
    if len(history) < 2:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    figures = {
        "05_01_dashboard": sim_plots.simulation_dashboard(history, title="Trading game: dashboard"),
        "05_02_attribution": sim_plots.pnl_attribution_chart(
            history, title="Trading game: cumulative P&L attribution (currency)"),
        "05_03_waterfall": sim_plots.pnl_attribution_waterfall(history),
        "05_04_gamma_vs_theta": sim_plots.gamma_theta_chart(history),
    }
    print(f"\nFigures and data written to {outdir.resolve()}")
    for name, fig in figures.items():
        path = outdir / f"{name}.html"
        # "directory": one shared plotly.min.js next to the pages (small files, works offline).
        fig.write_html(path, include_plotlyjs="directory")
        print(f"  {path.name}")
        if show:
            webbrowser.open(path.resolve().as_uri())
    history.to_csv(outdir / "05_history.csv")
    game.book.blotter_frame().to_csv(outdir / "05_blotter.csv", index=False)
    print("  05_history.csv\n  05_blotter.csv")


def main(argv: list[str] | None = None) -> None:
    """Play the game; ``argv`` defaults to the command line (tests pass their own list)."""
    args = parse_args(argv)
    game, hidden_vol, seed = build_game(args)
    script = DEMO_SCRIPT if args.demo and args.script is None else args.script
    costs = "no transaction costs" if args.no_costs else "costs: 1 bp on stock, 0.25 vol point on options"

    print(f"OPTION TRADING GAME (seed {seed}: pass --seed {seed} to replay this market)")
    print(f"You have {args.capital:,.0f} of capital and {args.days} trading days; {costs}.")
    print("Options trade at the implied vol you see. The vol the spot REALLY moves at is hidden until the end.")
    print("Type 'help' for the commands.")
    game.status()

    if script is not None:
        for command in script.split(";"):
            if not game.running:
                break
            print(f"\n{game.prompt}{command.strip()}")
            game.execute(command)
    else:
        while game.running:
            try:
                line = input(f"\n{game.prompt}")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            game.execute(line)

    game.final_report(hidden_vol)
    write_outputs(game, args.outdir, args.show)


if __name__ == "__main__":
    main()
