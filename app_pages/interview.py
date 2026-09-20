"""Interview drills — rehearse the questions a sales and trading desk actually asks.

Five ways to practise, and every number on the page is computed by the library
at the moment it is shown:

* **Quick-fire drill** — a seeded question with its own small market, an answer
  box, and a reveal that gives the Black-Scholes answer, whether the estimate
  was close enough, and the one line a desk would say. A score is kept for the
  session.
* **Mental maths** — the approximations traders quote from memory, each next to
  the exact value and the error it carries.
* **Flashcards** — the Greeks in the library's own words plus the concepts that
  come up in every round, hidden until you ask for them.
* **Question bank** — classic questions grouped by theme, each with a model
  answer and, where a number settles the argument, the computed illustration.
* **Story prompts** — the non-technical half: a structure to follow and a pitch
  built from one of the app's own strategies.

Two design notes. The quick-fire drill deliberately invents its **own** market
from the seed, so the numbers are new on every question and the same seed always
gives the same question; every other tab prices in the market set in the sidebar.
And nothing here uses Monte Carlo: the whole page is analytic, cached on the
plain market numbers, so the tabs can be rendered eagerly and still feel instant.
"""

from __future__ import annotations

import math
import random
from typing import Any

import pandas as pd
import streamlit as st

from app_lib import state, ui
from optionlab import (
    GREEK_INFO,
    STRATEGY_REGISTRY,
    AsianOption,
    EuropeanOption,
    Market,
    build_exotic,
    strategies as strat,
    to_trader_units,
)
from optionlab.plotting import profiles

# --------------------------------------------------------------------------- #
# Page constants
# --------------------------------------------------------------------------- #
DRILL_TOPICS = ("Mixed", "Pricing", "Greeks", "P&L", "Comparisons")
DEFAULT_TOPIC = "Mixed"

TENORS: dict[str, int] = {
    "1 week": 7,
    "1 month": 30,
    "3 months": 91,
    "6 months": 182,
    "1 year": 365,
}
DEFAULT_TENOR = "3 months"

#: Worlds the quick-fire drill draws from. Round, realistic, easy to say out loud.
_SPOTS = (25.0, 40.0, 50.0, 60.0, 75.0, 80.0, 100.0, 120.0, 150.0, 200.0, 250.0)
_VOLS = (0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40)
_RATES = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05)
_DIVS = (0.0, 0.0, 0.01, 0.02, 0.03)

_TENOR_WORDS = {7: "1-week", 14: "2-week", 30: "1-month", 60: "2-month", 91: "3-month",
                182: "6-month", 365: "1-year"}

#: Cards that are not about a single Greek. Topic drives the filter pills.
CONCEPT_CARDS: tuple[dict[str, str], ...] = (
    {
        "topic": "Volatility",
        "front": "Implied volatility against realised volatility",
        "back": "Implied is the volatility the option is quoted at: the market's price of future "
        "movement, quoted in a comparable unit. Realised is what the underlying actually did, "
        "measured after the fact from returns. A delta-hedged long option makes money when "
        "realised comes in above implied and bleeds when it comes in below, because gamma pays "
        "for the moves and theta charges the rent. So 'is vol cheap?' always means 'is implied "
        "below what I think will be realised?', never 'is 15 a low number?'.",
    },
    {
        "topic": "Volatility",
        "front": "Skew and smile: why are they there?",
        "back": "Skew is implied volatility changing with strike at one maturity. Equity index "
        "skew slopes down: low strikes trade at higher implied vol than high strikes. Three "
        "reasons a desk gives — crashes are real, so returns are not lognormal and the left tail "
        "is fatter; volatility rises when the market falls, which is a negative spot/vol "
        "correlation; and there is a structural bid for downside protection from investors who "
        "are long the market. Single stocks smile more symmetrically, because they can be taken "
        "over as well as blow up.",
    },
    {
        "topic": "Volatility",
        "front": "Term structure of volatility",
        "back": "Implied vol as a function of maturity at a fixed moneyness. It is usually upward "
        "sloping in a quiet market (the future is more uncertain than next week) and inverts in a "
        "panic, when the front end prices an event that is happening now. Term structure is what "
        "makes a calendar spread a trade rather than an accident: you are long one maturity's vol "
        "and short another's, so you care about the shape, not the level.",
    },
    {
        "topic": "Volatility",
        "front": "Why is vega not additive across maturities?",
        "back": "Vega assumes a parallel shift in implied vol, and implied vol never shifts in "
        "parallel. A 1-month vega and a 1-year vega respond to different shocks: the front end "
        "moves several points on an earnings date while the back end barely twitches. Desks "
        "therefore quote vega by bucket and weight the buckets, often by 1/sqrt(T), because "
        "short-dated implied vol is far more volatile than long-dated. Adding raw vega across the "
        "curve tells you the number but not the risk.",
    },
    {
        "topic": "Hedging",
        "front": "Delta hedging and gamma scalping",
        "back": "Delta hedging means holding minus delta shares against the option so that small "
        "spot moves do not change the value of the package. Because gamma keeps changing the "
        "delta, re-hedging a long option forces you to sell shares after a rally and buy them "
        "back after a fall: you are systematically buying low and selling high, and those "
        "round trips are the gamma scalp. Over one period the delta-hedged P&L is roughly "
        "half gamma times the squared move plus theta times the time — a bet on realised "
        "volatility against implied.",
    },
    {
        "topic": "Hedging",
        "front": "How often should you re-hedge?",
        "back": "More often means a smaller hedging error but more crossing of the bid-offer and "
        "more commission; less often means a cheaper hedge with a much fatter distribution of "
        "outcomes. With continuous hedging and the model true, the P&L converges to the value of "
        "the option; in practice the hedging error falls roughly like the square root of the "
        "number of hedges, so going from daily to hourly cuts the noise by about five while "
        "multiplying the costs. Most desks hedge on a delta band rather than a clock, which is "
        "the same trade-off expressed in risk units.",
    },
    {
        "topic": "Hedging",
        "front": "Why can you not hedge vega with the underlying?",
        "back": "The stock has a delta of one and no other Greek at all: no gamma, no vega, no "
        "theta. That is exactly why it is the natural delta hedge, and exactly why it cannot "
        "touch vega. Vega can only be hedged with another option — the same maturity if you want "
        "to kill the risk, a different maturity or strike if you are happy to keep a view on the "
        "shape of the surface.",
    },
    {
        "topic": "Products",
        "front": "Pin risk",
        "back": "On expiry day, with the spot sitting on the strike, a vanilla option's delta is "
        "somewhere between 0 and 1 and nobody knows where it will settle. You do not know how "
        "much stock you will be left with after exercise, so you cannot hedge; a one-cent move "
        "flips the answer. Gamma is effectively infinite at that point. The practical answer is "
        "to reduce the position into the expiry rather than to hedge it, and to expect the "
        "assignment decision to be a coin flip.",
    },
    {
        "topic": "Products",
        "front": "Why are barrier options hard to risk-manage?",
        "back": "A knock-out dies the moment the spot touches the barrier, so its value is "
        "discontinuous in the spot. Just inside the barrier the option still carries intrinsic "
        "value; just outside it is worth the rebate, which is usually nothing. The hedge ratio "
        "that bridges that jump can exceed one and can flip sign, and gamma goes sharply "
        "negative near the barrier for a reverse knock-out. Desks quote a wider price, hedge "
        "with a barrier shift, and expect to lose money on the days the barrier is in play.",
    },
    {
        "topic": "Products",
        "front": "Digital risk and the call-spread hedge",
        "back": "A cash digital pays a fixed amount if it finishes in the money. Its price is the "
        "discounted probability of that, which makes it a clean bet on where the spot ends. The "
        "risk is the step: near expiry and near the strike, delta spikes and gamma flips sign, so "
        "the theoretical hedge is untradable. Desks book it as a tight call spread, wide enough "
        "to be hedgeable; the width is the bid-offer, and a seller places the spread so that it "
        "always pays at least as much as the digital.",
    },
    {
        "topic": "Products",
        "front": "Why is an option priced off the forward, not the spot?",
        "back": "Because the hedge is a position in the stock held to expiry, and carrying it "
        "costs the financing rate and earns the dividend. The forward is the spot carried at the "
        "rate and leaked by the dividend, and it is the level at which the payoff is centred. "
        "That is why an 'at-the-money' option struck at the spot is not quite symmetric, and why "
        "the call and the put at the same strike differ by exactly the discounted distance from "
        "the forward to the strike.",
    },
    {
        "topic": "Desk life",
        "front": "Market making and the bid-offer",
        "back": "A market maker quotes two-way and earns the spread, not a view. The spread has "
        "to pay for three things: the inventory risk of the position you are left with, the cost "
        "of hedging it (and re-hedging it, which is where gamma and bid-offer meet), and the "
        "chance that the client knows something you do not. That last one is why the price you "
        "show moves with the size and with who is asking, and why an illiquid strike is quoted "
        "wider than the screen.",
    },
    {
        "topic": "Desk life",
        "front": "What does a desk's risk report look like?",
        "back": "One line per underlying, with the aggregate Greeks: delta in shares or in "
        "currency, gamma per 1% move, vega by maturity bucket, theta per day, and rho. Under it "
        "sits a scenario grid — P&L for spot down 10% to up 10% against vol down 5 points to up "
        "5 points — plus the expiry ladder showing what rolls off and when. Traders read the "
        "grid first, because the Greeks are a local approximation and the grid is the answer to "
        "'what actually happens if'.",
    },
    {
        "topic": "Desk life",
        "front": "Someone hands you a position. What do you say first?",
        "back": "Name the biggest risk, quantify it, then say how you would flatten it. 'I am "
        "short 40,000 gamma and long 12,000 vega in the front month, so a gap hurts me today and "
        "a vol rally helps me by expiry; I would buy back some of the short strike into the "
        "close.' Numbers, direction, action. What loses marks is describing the trade leg by leg "
        "instead of aggregating it.",
    },
)

#: Classic interview questions with a model answer, grouped by theme. ``show``
#: names the computed illustration rendered under the answer; ``link`` points at
#: the page of this app that draws the same thing.
BANK: tuple[dict[str, Any], ...] = (
    {
        "theme": "Options and Greeks",
        "question": "Explain delta three ways.",
        "answer": "One: it is the derivative of the price with respect to the spot, so it is the "
        "P&L for a one-point move. Two: it is the hedge ratio — sell delta shares per option and "
        "you are flat for small moves. Three: it is a rough probability of finishing in the "
        "money. The third one is a shortcut: delta is N(d1) and the actual risk-neutral "
        "probability is N(d2), which is lower for a call by the half-variance term, so delta "
        "overstates the chance, and the gap widens with volatility and maturity.",
        "show": "tenors",
        "link": ("app_pages/greeks.py", "See it drawn on the Greeks explorer", ":material/show_chart:"),
    },
    {
        "theme": "Options and Greeks",
        "question": "Why is gamma largest at the money and near expiry?",
        "answer": "Gamma measures how fast delta changes, and delta only changes where the "
        "outcome is still in doubt. Far from the strike the option is nearly stock or nearly "
        "nothing, so delta is pinned at 1 or 0 and gamma is small. As expiry approaches the delta "
        "curve steepens into a step at the strike, so at-the-money gamma grows without bound "
        "while gamma away from the strike dies. That is why a front-month book is re-hedged all "
        "day and a long-dated book is not.",
        "show": "tenors",
        "link": None,
    },
    {
        "theme": "Options and Greeks",
        "question": "What is the delta of an at-the-money option, exactly?",
        "answer": "A little above 0.5 for a call, not exactly 0.5. 'At the money' usually means "
        "struck at the spot, while the option is priced off the forward, which sits above the "
        "spot when the rate exceeds the dividend yield. Delta is N(d1), and d1 carries both the "
        "carry term and half the variance, so it is positive even when the forward equals the "
        "strike. Say 'about 0.5, a touch higher for a call, and the gap grows with vol, time and "
        "rates' and you have answered the real question.",
        "show": "tenors",
        "link": None,
    },
    {
        "theme": "Options and Greeks",
        "question": "You are long an option. Which Greeks are you long and which are you short?",
        "answer": "Long a vanilla, you are long gamma and long vega, and you are short theta — "
        "you own convexity and you pay rent for it every day. Delta depends on the type and the "
        "strike: long for a call, short for a put. Rho is long for a call and short for a put, "
        "and matters only for long maturities. Shorting the option flips every sign: short "
        "gamma, short vega, long theta. The pairing is the whole point — nobody gets gamma for "
        "free.",
        "show": None,
        "link": None,
    },
    {
        "theme": "Volatility",
        "question": "Implied vol is 20 and the stock has been realising 12. What do you do?",
        "answer": "Sell the volatility and delta-hedge it: short a straddle or a strangle, hedge "
        "the delta, and collect the difference between the rent you receive and the moves you "
        "have to pay for. The position makes money if realised stays below implied over the life. "
        "What kills it: a gap, because short gamma loses on any large move and the loss grows "
        "with the square of it; an event you did not price, such as earnings or a bid; and "
        "realised vol arriving all at once rather than smoothly. Size it so that one bad day does "
        "not end the trade.",
        "show": "straddle",
        "link": ("app_pages/simulator.py", "Run it in the trading simulator", ":material/speed:"),
    },
    {
        "theme": "Volatility",
        "question": "Is vega additive across maturities?",
        "answer": "Not usefully. Adding a 1-month vega to a 1-year vega assumes the whole surface "
        "moves in parallel, and it does not: the front end is several times more volatile than "
        "the back. A desk buckets vega by maturity and often weights it by 1/sqrt(T) to make the "
        "buckets comparable. The same warning applies across strikes, where skew means a vol "
        "shock is not flat either.",
        "show": "tenors",
        "link": None,
    },
    {
        "theme": "Volatility",
        "question": "What moves equity index volatility?",
        "answer": "Direction first: vol rises when the market falls, because protection is bought "
        "in a fall and because leverage rises as equity value drops. Then the calendar — "
        "earnings, central bank meetings, elections — which lifts the maturities that span the "
        "event and leaves the others alone. Then flows: structured product issuance leaves dealers "
        "systematically long or short vega at particular strikes, and their hedging pushes the "
        "surface around. Finally realised volatility itself, because implied cannot sit far below "
        "what the market is actually doing for long.",
        "show": None,
        "link": None,
    },
    {
        "theme": "Hedging",
        "question": "You delta-hedge a long at-the-money call daily. When do you make money?",
        "answer": "When the underlying moves more than the option was priced for. Over one day "
        "the delta-hedged P&L is about half gamma times the squared move, minus the day's theta. "
        "Those two balance when the move is about one standard deviation, the implied daily move; "
        "bigger, and the gamma pays for the rent; smaller, and you bleed. Over the life it is the "
        "same statement in aggregate: realised volatility above implied pays, below it does not.",
        "show": "straddle",
        "link": ("app_pages/simulator.py", "Watch the daily P&L build up", ":material/speed:"),
    },
    {
        "theme": "Hedging",
        "question": "You are short 50 at-the-money straddles and the market gaps 5%. Talk me "
        "through the next five minutes.",
        "answer": "First, say the direction of the risk out loud: short straddles is short gamma, "
        "so the gap has already cost money and the delta is now against me — short the spot after "
        "a rally, long it after a fall. Second, re-hedge the delta, because the position keeps "
        "losing as the move extends. Third, decide whether to reduce the gamma itself by buying "
        "back part of the short strike, which is expensive right after a gap but caps the damage "
        "if the move continues. Fourth, tell the desk head the number. The order matters: hedge, "
        "then reduce, then explain.",
        "show": "straddle",
        "link": ("app_pages/book.py", "Build the position in the book", ":material/inventory_2:"),
    },
    {
        "theme": "Products and structuring",
        "question": "A client wants upside exposure but does not want to pay a premium. What do "
        "you show them?",
        "answer": "A risk reversal or a collar: buy the out-of-the-money call and sell an "
        "out-of-the-money put (or, against a stock position, sell the call and buy the put) so "
        "that the premiums net to about zero. Be explicit about what they have given up — the "
        "downside below the short strike is now theirs, so 'zero cost' means 'paid for in risk, "
        "not in cash'. Then show the strike combination that makes the trade actually zero cost "
        "at today's skew, because equity skew usually makes the put you sell dearer than the call "
        "you buy.",
        "show": None,
        "link": ("app_pages/strategies.py", "Price a risk reversal", ":material/account_tree:"),
    },
    {
        "theme": "Products and structuring",
        "question": "Why would a hedger buy an Asian option instead of a vanilla?",
        "answer": "Because their exposure is an average, not a closing price. An importer buying "
        "oil every week is exposed to the average price over the quarter, so an option on that "
        "average matches the risk exactly and removes the timing luck of a single fixing. It is "
        "also cheaper: an average moves less than its last point, roughly vol over root three for "
        "a full window, so the option carries less volatility and less premium. The trade-off is "
        "that the Greeks fade as the average fills in, which is the opposite of a vanilla.",
        "show": "asian",
        "link": ("app_pages/exotics.py", "Compare it with the vanilla", ":material/diamond:"),
    },
    {
        "theme": "Products and structuring",
        "question": "Why can a knock-out call have a delta greater than one, or even negative?",
        "answer": "Because the payoff is discontinuous at the barrier. For an up-and-out call the "
        "barrier sits above the strike, so the option is worth real intrinsic value just below "
        "the barrier and nothing just above it. The value therefore falls as the spot rises near "
        "the barrier, which is a negative delta, and the slope bridging that jump can be far "
        "larger than one in absolute terms. Gamma is large and negative there too, which is why "
        "desks widen the price and shift the barrier in their hedging model.",
        "show": None,
        "link": ("app_pages/exotics.py", "See the barrier profile", ":material/diamond:"),
    },
    {
        "theme": "Brainteasers",
        "question": "Price me a coin-flip payoff: I pay you a fixed amount if the stock is above "
        "a level in three months.",
        "answer": "That is a cash-or-nothing digital, and its value is the payout times the "
        "discounted risk-neutral probability of finishing above the level — N(d2), not a real "
        "world probability. Quote it, then immediately say how you would hedge it: as a tight "
        "call spread around the strike, because the digital's own delta is a spike you cannot "
        "trade. The width of that spread is your bid-offer, and a seller places it so the spread "
        "pays at least as much as the digital everywhere.",
        "show": "digital",
        "link": ("app_pages/exotics.py", "See the replication", ":material/diamond:"),
    },
    {
        "theme": "Brainteasers",
        "question": "What is worth more: a call on the average, or the average of the calls?",
        "answer": "The average of the calls. The payoff is a convex function of the price, so by "
        "Jensen's inequality averaging inside the maximum is worth less than averaging outside "
        "it: averaging first destroys the optionality of every individual fixing. The same logic "
        "explains why an option on a basket is worth less than a basket of options, with "
        "correlation playing the part of the averaging. The numbers below are both priced by the "
        "library on the market in the sidebar.",
        "show": "asian",
        "link": None,
    },
    {
        "theme": "Brainteasers",
        "question": "A stock is at 100 and will be either 90 or 110 at expiry, each equally "
        "likely. What is the 100 call worth?",
        "answer": "In a two-state world with no rates, the call pays 10 in the up state and 0 in "
        "the down state, so half of 10 is 5. The point of the question is what comes next: the "
        "probabilities that matter are the risk-neutral ones, which are set by the requirement "
        "that the hedged portfolio earns the risk-free rate, not by your view. Here the stock is "
        "a martingale and the real and risk-neutral probabilities happen to coincide; change the "
        "up and down moves and they will not. Note this is arithmetic in a two-state model, not "
        "a Black-Scholes price.",
        "show": None,
        "link": None,
    },
    {
        "theme": "Markets and behaviour",
        "question": "How would you explain an option to a client who has never traded one?",
        "answer": "Insurance. You pay a premium today for the right, not the obligation, to buy "
        "or sell at a fixed price later. The most you can lose as a buyer is the premium; the "
        "seller collects it and takes the risk. Then one concrete example with their own "
        "exposure in it — 'you hold the stock at 100, for 3 you can lock in the right to sell at "
        "95 for three months' — and one sentence on what makes the premium move: how far the "
        "strike is, how long it runs, and how much the market expects the stock to move.",
        "show": None,
        "link": None,
    },
    {
        "theme": "Markets and behaviour",
        "question": "A large client calls for a price in size. Walk me through what happens.",
        "answer": "Sales takes the enquiry and passes the axe and the context to the trader. The "
        "trader prices the mid from the curve, then adjusts for what the trade does to the book — "
        "does it add to a risk we already have or offset it — for the cost of hedging it, and for "
        "the size relative to what trades on screen. Sales shows the price, the client deals or "
        "does not, and the trader hedges the delta immediately and the vega over the session. "
        "Then it is booked, the risk report updates, and someone checks the P&L explains.",
        "show": None,
        "link": None,
    },
)

#: Structure prompts for the non-technical half of the interview.
STORY_PROMPTS: tuple[dict[str, Any], ...] = (
    {
        "icon": ":material/record_voice_over:",
        "prompt": "What did the market do yesterday, and why do you think that happened?",
        "structure": (
            "**What** — one number per market: index, rates, the dollar, oil. Direction and size.",
            "**Why** — one candidate cause, not four. Name the data print, the meeting or the flow.",
            "**So what** — what it implies for positioning, and what would confirm or kill the story.",
            "**Honesty** — say which part is your read and which part is the consensus explanation.",
        ),
        "trap": "Listing five numbers with no narrative, or inventing a cause you cannot defend. "
        "One market, one story, one consequence beats a market recap every time.",
    },
    {
        "icon": ":material/psychology_alt:",
        "prompt": "Tell me about a risk you took.",
        "structure": (
            "**Situation** — what was at stake, in one sentence, with the size of the decision.",
            "**View** — what you believed and why, and what you did to test it before committing.",
            "**Risk** — what you could lose, and the limit you set in advance.",
            "**Outcome and learning** — what happened, and what you would change. A loss you "
            "managed well is a better answer than a win you got lucky on.",
        ),
        "trap": "A story with no downside in it. If nothing could have gone wrong, no risk was "
        "taken, and the interviewer will say so.",
    },
    {
        "icon": ":material/campaign:",
        "prompt": "Sell me this option.",
        "structure": (
            "**Their problem** — start from the client's exposure, not from the product.",
            "**The fit** — why this structure matches that exposure, in one sentence.",
            "**The cost** — the premium in cash and as a percentage, and what it buys.",
            "**The catch** — what they give up, said before they ask. Credibility is the product.",
        ),
        "trap": "Pitching features (barriers, averaging, autocall) before establishing the "
        "exposure. Sales is translation, not description.",
    },
    {
        "icon": ":material/work:",
        "prompt": "Why sales and trading, and why this desk?",
        "structure": (
            "**The work** — what the job actually is day to day, said in your own words.",
            "**The evidence** — something you have done that looks like it: a trade, a model, a "
            "competition, this app.",
            "**The fit** — why this product and this desk rather than the one next door.",
            "**The question back** — one specific thing you want to know about the desk.",
        ),
        "trap": "'I like fast-paced environments.' Replace every adjective with an example.",
    },
)

#: Strategies offered by the pitch builder, in menu order.
PITCH_STRATEGIES = (
    "bull_call_spread",
    "long_straddle",
    "short_strangle",
    "iron_condor",
    "risk_reversal",
    "collar",
    "calendar_spread",
    "put_backspread",
)

#: What would make each market view wrong. Keyed on the library's own view tags.
WRONG_BY_VIEW = {
    "bullish": "the underlying drifts sideways or lower and the premium decays against me",
    "bearish": "the underlying grinds higher and I pay for protection I never use",
    "neutral": "a trend in either direction that walks the spot away from my strikes",
    "long vol": "the market goes quiet: realised volatility comes in under implied and I pay "
    "theta every day for a move that never arrives",
    "short vol": "a gap through my short strike, or an event that was not in the price",
    "income": "the one tail that costs more than a year of collected premium",
    "hedge": "paying for insurance I never need, which is a real cost and not a free option",
    "arbitrage": "the financing, the borrow or the dividend assumption moving against me",
}


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #
def _tenor_word(days: int) -> str:
    """``91 -> "3-month"``, used inside the drill prompts."""
    return _TENOR_WORDS.get(int(days), f"{int(days)}-day")


def _tick(spot: float) -> float:
    """Strike increment a listed market would quote at this spot."""
    if spot >= 200:
        return 5.0
    if spot >= 80:
        return 2.5
    if spot >= 30:
        return 1.0
    return 0.5


def _strike_at(spot: float, moneyness: float) -> float:
    """A tradable strike near ``moneyness * spot``."""
    tick = _tick(spot)
    return max(tick, round(spot * moneyness / tick) * tick)


def _signed_percent(value: float, *, decimals: int = 1) -> str:
    return f"{value * 100:+.{decimals}f}%"


def _moneyness_phrase(option_type: str, strike: float, spot: float) -> str:
    """"7% out of the money", read from the point of view of the option asked about.

    The shared helper does the work so that the wording matches every other page;
    a spot of zero has no moneyness at all, which the drill treats as at the money.
    """
    return ui.moneyness_caption(strike, spot, option_type) or "at the money"


def _relative_error(approximation: float, exact: float) -> float | None:
    """Signed relative error, or ``None`` when the exact value is ~0."""
    if abs(exact) < 1e-12:
        return None
    return approximation / exact - 1.0


def _pick_larger(labels: tuple[str, str], values: tuple[float, float], tie: str) -> str:
    """Which of two computed numbers is bigger, with an honest tie case."""
    first, second = values
    scale = max(abs(first), abs(second), 1e-12)
    if abs(first - second) <= 1e-9 * scale:
        return tie
    return labels[0] if first > second else labels[1]


# --------------------------------------------------------------------------- #
# Quick-fire drill: one question generator per template
# --------------------------------------------------------------------------- #
def _drill_market(rng: random.Random) -> tuple[float, float, float, float]:
    return rng.choice(_SPOTS), rng.choice(_VOLS), rng.choice(_RATES), rng.choice(_DIVS)


def _q_atm_price(rng: random.Random) -> dict[str, Any]:
    spot, vol, rate, div = _drill_market(rng)
    days = rng.choice((30, 60, 91, 182, 365))
    option_type = rng.choice(("call", "put"))
    expiry = days / 365.0
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    option = EuropeanOption(option_type, spot, expiry)
    price = float(option.price(mkt))
    rule = 0.4 * spot * vol * math.sqrt(expiry)
    forward = float(mkt.forward(expiry))
    return {
        "topic": "Pricing",
        "kind": "number",
        "market": (spot, vol, rate, div),
        "prompt": f"What is the {_tenor_word(days)} at-the-money {option_type} worth? "
        f"The strike is the spot, {spot:g}.",
        "unit": "currency, for one option",
        "answer": price,
        "answer_label": ui.money(price),
        "tolerance": max(0.10 * price, 0.01),
        "step": max(round(price / 20.0, 2), 0.01),
        "format": "%.2f",
        "why": f"The desk shortcut is 0.4 x spot x vol x sqrt(T): 0.4 x {spot:g} x "
        f"{ui.percent(vol, decimals=0)} x sqrt({expiry:.2f}) = {ui.money(rule)}. Black-Scholes "
        f"says {ui.money(price)}, a gap of {ui.money(price - rule, signed=True)}. The shortcut "
        f"prices the option off the spot; the model prices it off the forward at "
        f"{ui.money(forward)}, and that is where most of the difference comes from.",
        "facts": (
            ("Black-Scholes", ui.money(price)),
            ("0.4 x S x vol x sqrt(T)", ui.money(rule)),
            ("Forward", ui.money(forward)),
            ("One-sigma move to expiry", ui.money(spot * vol * math.sqrt(expiry))),
        ),
    }


def _q_parity(rng: random.Random) -> dict[str, Any]:
    spot, vol, rate, div = _drill_market(rng)
    days = rng.choice((30, 91, 182, 365))
    strike = _strike_at(spot, rng.choice((0.95, 1.0, 1.05)))
    expiry = days / 365.0
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    call = float(EuropeanOption("call", strike, expiry).price(mkt))
    put = float(EuropeanOption("put", strike, expiry).price(mkt))
    carried_spot = spot * math.exp(-div * expiry)
    discounted_strike = strike * math.exp(-rate * expiry)
    return {
        "topic": "Pricing",
        "kind": "number",
        "market": (spot, vol, rate, div),
        "prompt": f"The {_tenor_word(days)} {strike:g} call is worth {ui.money(call)}. "
        "Where is the put with the same strike and maturity?",
        "unit": "currency, for one option",
        "answer": put,
        "answer_label": ui.money(put),
        "tolerance": max(0.10 * put, 0.05),
        "step": max(round(put / 20.0, 2), 0.01),
        "format": "%.2f",
        "why": "Put-call parity: put = call - spot x exp(-div x T) + strike x exp(-rate x T) = "
        f"{ui.money(call)} - {ui.money(carried_spot)} + {ui.money(discounted_strike)} = "
        f"{ui.money(put)}. It is an identity enforced by arbitrage, not a model output: it holds "
        "at any volatility, which is why a desk quotes one side and derives the other.",
        "facts": (
            ("Call", ui.money(call)),
            ("Put", ui.money(put)),
            ("Call minus put", ui.money(call - put, signed=True)),
            ("Discounted forward minus strike", ui.money(carried_spot - discounted_strike, signed=True)),
        ),
    }


def _q_probability_itm(rng: random.Random) -> dict[str, Any]:
    spot, vol, rate, div = _drill_market(rng)
    days = rng.choice((30, 91, 182))
    option_type = rng.choice(("call", "put"))
    moneyness = rng.choice((0.9, 0.95, 1.0, 1.05, 1.1))
    strike = _strike_at(spot, moneyness)
    expiry = days / 365.0
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    digital = build_exotic(
        "cash_digital", option_type=option_type, strike=strike, expiry=expiry, payout=1.0
    )
    probability = float(digital.probability_itm(mkt)) * 100.0
    delta = float(EuropeanOption(option_type, strike, expiry).greeks(mkt)["delta"])
    return {
        "topic": "Pricing",
        "kind": "number",
        "market": (spot, vol, rate, div),
        "prompt": f"What is the risk-neutral chance that the {_tenor_word(days)} {strike:g} "
        f"{option_type} finishes in the money? Answer in percent.",
        "unit": "percent",
        "answer": probability,
        "answer_label": f"{probability:.1f}%",
        "tolerance": 4.0,
        "step": 1.0,
        "format": "%.1f",
        "why": f"It is N(d2), here {probability:.1f}%. The delta of the same option is "
        f"{ui.greek_value('delta', delta)}, which traders use as a rough probability; the two "
        f"differ by {abs(abs(delta) * 100.0 - probability):.1f} points because delta is N(d1) and "
        "carries the extra half-variance term. The gap widens with volatility and with maturity.",
        "facts": (
            ("N(d2), the probability", f"{probability:.1f}%"),
            ("Delta as a proxy", f"{abs(delta) * 100.0:.1f}%"),
            ("Strike", ui.money(strike)),
            ("Moneyness", _moneyness_phrase(option_type, strike, spot)),
        ),
    }


def _q_delta(rng: random.Random) -> dict[str, Any]:
    spot, vol, rate, div = _drill_market(rng)
    days = rng.choice((14, 30, 91, 182, 365))
    option_type = rng.choice(("call", "put"))
    strike = _strike_at(spot, rng.choice((0.9, 0.95, 1.0, 1.05, 1.1)))
    expiry = days / 365.0
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    greeks = {k: float(v) for k, v in EuropeanOption(option_type, strike, expiry).greeks(mkt).items()}
    delta = greeks["delta"]
    return {
        "topic": "Greeks",
        "kind": "number",
        "market": (spot, vol, rate, div),
        "prompt": f"What is the delta of the {_tenor_word(days)} {strike:g} {option_type}? "
        "Quote it per share, so between -1 and +1.",
        "unit": "delta per 1.00 of spot",
        "answer": delta,
        "answer_label": ui.greek_value("delta", delta),
        "tolerance": 0.03,
        "step": 0.01,
        "format": "%.3f",
        "why": f"Delta is {ui.greek_value('delta', delta)}. The option is "
        f"{_moneyness_phrase(option_type, strike, spot)}, and delta is the hedge ratio, so you "
        "would trade that many shares against one option. Read it as a rough chance of finishing "
        "in the money, and remember that an at-the-money call sits a little above 0.5 because the "
        "option is priced off the forward, not off the spot.",
        "facts": (
            ("Delta", ui.greek_value("delta", delta)),
            ("Price", ui.money(greeks["price"])),
            ("Gamma", ui.greek_value("gamma", greeks["gamma"])),
            ("Strike", ui.money(strike)),
        ),
    }


def _q_vega(rng: random.Random) -> dict[str, Any]:
    spot, vol, rate, div = _drill_market(rng)
    days = rng.choice((30, 91, 182, 365))
    option_type = rng.choice(("call", "put"))
    expiry = days / 365.0
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    greeks = {k: float(v) for k, v in EuropeanOption(option_type, spot, expiry).greeks(mkt).items()}
    vega_point = greeks["vega"] / 100.0
    rule = 0.004 * spot * math.sqrt(expiry)
    return {
        "topic": "Greeks",
        "kind": "number",
        "market": (spot, vol, rate, div),
        "prompt": f"What is the vega of the {_tenor_word(days)} at-the-money {option_type}, "
        "per one volatility point, for one option?",
        "unit": "currency per 1 vol point",
        "answer": vega_point,
        "answer_label": ui.greek_value("vega", vega_point),
        "tolerance": max(0.15 * abs(vega_point), 0.005),
        "step": max(round(abs(vega_point) / 20.0, 3), 0.001),
        "format": "%.3f",
        "why": f"Vega per point is {ui.greek_value('vega', vega_point)}. The at-the-money "
        f"shortcut is 0.004 x spot x sqrt(T) = {ui.number(rule, decimals=3)}, because the normal "
        "density at the money is about 0.4 and a point is one hundredth of a vol. Vega grows with "
        "the square root of time, so four times the maturity is twice the vega — long-dated "
        "options are the volatility instrument, short-dated ones are the gamma instrument.",
        "facts": (
            ("Vega per point", ui.greek_value("vega", vega_point)),
            ("0.004 x S x sqrt(T)", ui.number(rule, decimals=3)),
            ("Price", ui.money(greeks["price"])),
            ("Gamma", ui.greek_value("gamma", greeks["gamma"])),
        ),
    }


def _q_theta(rng: random.Random) -> dict[str, Any]:
    spot, vol, rate, div = _drill_market(rng)
    days = rng.choice((14, 30, 91, 182))
    option_type = rng.choice(("call", "put"))
    expiry = days / 365.0
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    greeks = {k: float(v) for k, v in EuropeanOption(option_type, spot, expiry).greeks(mkt).items()}
    theta_day = greeks["theta"] / 365.0
    gamma_rent = -0.5 * greeks["gamma"] * (vol * spot) ** 2 / 365.0
    return {
        "topic": "Greeks",
        "kind": "number",
        "market": (spot, vol, rate, div),
        "prompt": f"What is the theta of the {_tenor_word(days)} at-the-money {option_type}, "
        "per calendar day, for one option? Mind the sign.",
        "unit": "currency per calendar day",
        "answer": theta_day,
        "answer_label": ui.greek_value("theta", theta_day),
        "tolerance": max(0.20 * abs(theta_day), 0.002),
        "step": max(round(abs(theta_day) / 20.0, 3), 0.001),
        "format": "%.3f",
        "why": "For a delta-hedged option the daily rent is about -0.5 x gamma x (vol x spot)^2 / "
        f"365 = {ui.number(gamma_rent, decimals=3)}. The model's theta is "
        f"{ui.greek_value('theta', theta_day)}; the difference is the carry — interest on the "
        "strike and the dividend on the stock you are replicating. Theta is the price of gamma: "
        "quote the two together or the answer is only half of one.",
        "facts": (
            ("Theta per day", ui.greek_value("theta", theta_day)),
            ("-0.5 x gamma x (vol x S)^2 / 365", ui.number(gamma_rent, decimals=3)),
            ("Gamma", ui.greek_value("gamma", greeks["gamma"])),
            ("Implied daily move", ui.money(spot * vol / math.sqrt(252.0))),
        ),
    }


def _q_short_straddle_gap(rng: random.Random) -> dict[str, Any]:
    spot, vol, rate, div = _drill_market(rng)
    days = rng.choice((14, 30, 60, 91))
    quantity = rng.choice((10, 20, 25, 50))
    move = rng.choice((-6.0, -5.0, -4.0, -3.0, 3.0, 4.0, 5.0, 6.0))
    expiry = days / 365.0
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    straddle = strat.build_strategy("long_straddle", strike=spot, expiry=expiry, quantity=1.0)
    greeks = {k: float(v) for k, v in straddle.greeks(mkt).items()}
    new_spot = spot * (1.0 + move / 100.0)
    change = float(straddle.price(mkt.bumped(spot=new_spot))) - greeks["price"]
    pnl = -quantity * change
    quadratic = -quantity * 0.5 * greeks["gamma"] * (new_spot - spot) ** 2
    return {
        "topic": "P&L",
        "kind": "number",
        "market": (spot, vol, rate, div),
        "prompt": f"You are short {quantity} of the {_tenor_word(days)} at-the-money straddle. "
        f"The spot gaps {move:+.0f}% in a second, implied vol unchanged. What is your P&L?",
        "unit": "currency, whole position",
        "answer": pnl,
        "answer_label": ui.money(pnl, signed=True),
        "tolerance": max(0.15 * abs(pnl), 0.05 * quantity * spot / 1000.0),
        "step": max(round(abs(pnl) / 20.0, 2), 0.01),
        "format": "%.2f",
        "why": f"Short a straddle is short gamma, so any move hurts. One straddle has gamma "
        f"{ui.greek_value('gamma', greeks['gamma'])}, and -0.5 x gamma x dS^2 x {quantity} = "
        f"{ui.money(quadratic, signed=True)}. Repricing the whole package gives "
        f"{ui.money(pnl, signed=True)}: the quadratic term is "
        f"{'too large' if abs(quadratic) > abs(pnl) else 'too small'} by "
        f"{ui.money(abs(quadratic - pnl))}, because gamma itself changes as the spot travels. "
        "Say the sign and the order of magnitude first; the second decimal is not the point.",
        "facts": (
            ("Full reprice", ui.money(pnl, signed=True)),
            ("-0.5 x gamma x dS^2", ui.money(quadratic, signed=True)),
            ("Straddle before", ui.money(greeks["price"])),
            ("Spot after the gap", ui.money(new_spot)),
        ),
    }


def _q_hedged_overnight(rng: random.Random) -> dict[str, Any]:
    spot, vol, rate, div = _drill_market(rng)
    days = rng.choice((14, 30, 60))
    quantity = rng.choice((10, 25, 50, 100))
    move = rng.choice((-3.0, -2.0, -1.5, 1.5, 2.0, 3.0))
    expiry = days / 365.0
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    call = EuropeanOption("call", spot, expiry)
    greeks = {k: float(v) for k, v in call.greeks(mkt).items()}
    new_spot = spot * (1.0 + move / 100.0)
    tomorrow = mkt.bumped(spot=new_spot, t=mkt.t + 1.0 / 365.0)
    change = float(call.price(tomorrow)) - greeks["price"] - greeks["delta"] * (new_spot - spot)
    pnl = quantity * change
    gamma_earn = quantity * 0.5 * greeks["gamma"] * (new_spot - spot) ** 2
    theta_pay = quantity * greeks["theta"] / 365.0
    sigma_move = spot * vol / math.sqrt(252.0)
    return {
        "topic": "P&L",
        "kind": "number",
        "market": (spot, vol, rate, div),
        "prompt": f"You are long {quantity} of the {_tenor_word(days)} at-the-money call, "
        f"delta-hedged at the close. Overnight the spot moves {move:+.1f}% and implied vol is "
        "unchanged. What did the hedged position make?",
        "unit": "currency, whole position",
        "answer": pnl,
        "answer_label": ui.money(pnl, signed=True),
        "tolerance": max(0.30 * abs(pnl), 0.05 * quantity * spot / 1000.0),
        "step": max(round(abs(pnl) / 20.0, 2), 0.01),
        "format": "%.2f",
        "why": f"Gamma earns 0.5 x gamma x dS^2 = {ui.money(gamma_earn, signed=True)} and theta "
        f"charges {ui.money(theta_pay, signed=True)} for the day, so the sum is about "
        f"{ui.money(gamma_earn + theta_pay, signed=True)}; the full reprice says "
        f"{ui.money(pnl, signed=True)}. The two sides balance when the move is about one standard "
        f"deviation, {ui.money(sigma_move)} here against the {ui.money(abs(new_spot - spot))} that "
        "actually happened. That comparison is the whole trade.",
        "facts": (
            ("Full reprice", ui.money(pnl, signed=True)),
            ("Gamma earns", ui.money(gamma_earn, signed=True)),
            ("Theta pays", ui.money(theta_pay, signed=True)),
            ("One-sigma daily move", ui.money(sigma_move)),
        ),
    }


def _q_vol_drop(rng: random.Random) -> dict[str, Any]:
    spot, vol, rate, div = _drill_market(rng)
    days = rng.choice((30, 91, 182, 365))
    quantity = rng.choice((10, 25, 50, 100))
    points = rng.choice((-5.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 5.0))
    new_vol = max(vol + points / 100.0, 0.01)
    expiry = days / 365.0
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    call = EuropeanOption("call", spot, expiry)
    greeks = {k: float(v) for k, v in call.greeks(mkt).items()}
    pnl = quantity * (float(call.price(mkt.bumped(vol=new_vol))) - greeks["price"])
    linear = quantity * greeks["vega"] / 100.0 * points
    direction = "rises" if points > 0 else "falls"
    point_word = "point" if abs(points) == 1.0 else "points"
    return {
        "topic": "P&L",
        "kind": "number",
        "market": (spot, vol, rate, div),
        "prompt": f"You are long {quantity} of the {_tenor_word(days)} at-the-money call. "
        f"Implied vol {direction} {abs(points):.0f} {point_word}, spot unchanged. What is your "
        "P&L?",
        "unit": "currency, whole position",
        "answer": pnl,
        "answer_label": ui.money(pnl, signed=True),
        "tolerance": max(0.15 * abs(pnl), 0.05 * quantity * spot / 1000.0),
        "step": max(round(abs(pnl) / 20.0, 2), 0.01),
        "format": "%.2f",
        "why": f"Vega per point is {ui.greek_value('vega', greeks['vega'] / 100.0)} for one "
        f"option, so {quantity} of them times {abs(points):.0f} {point_word} is about "
        f"{ui.money(linear, signed=True)}. The exact reprice is {ui.money(pnl, signed=True)}; the "
        f"{ui.money(abs(pnl - linear))} difference is volga, the convexity of the price in "
        "volatility, which is small at the money and grows in the wings.",
        "facts": (
            ("Full reprice", ui.money(pnl, signed=True)),
            ("Vega x points", ui.money(linear, signed=True)),
            ("Vega per point, one option", ui.greek_value("vega", greeks["vega"] / 100.0)),
            ("Vol after the move", ui.percent(new_vol, decimals=1)),
        ),
    }


def _q_gamma_tenor(rng: random.Random) -> dict[str, Any]:
    spot, vol, rate, div = _drill_market(rng)
    short_days, long_days = rng.choice(((7, 365), (30, 365), (7, 91), (30, 182), (14, 91)))
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    short = EuropeanOption("call", spot, short_days / 365.0)
    long = EuropeanOption("call", spot, long_days / 365.0)
    labels = (f"The {_tenor_word(short_days)} call", f"The {_tenor_word(long_days)} call")
    values = (float(short.greeks(mkt)["gamma"]), float(long.greeks(mkt)["gamma"]))
    tie = "They are the same"
    return {
        "topic": "Comparisons",
        "kind": "choice",
        "market": (spot, vol, rate, div),
        "prompt": "Both struck at the spot, same underlying. Which has more gamma: the "
        f"{_tenor_word(short_days)} call or the {_tenor_word(long_days)} call?",
        "options": (labels[0], labels[1], tie),
        "answer_label": _pick_larger(labels, values, tie),
        "why": f"At the money gamma is about 0.4 / (spot x vol x sqrt(T)), so it grows as maturity "
        f"shrinks: {ui.greek_value('gamma', values[0])} against "
        f"{ui.greek_value('gamma', values[1])} here, a factor of "
        f"{max(values) / min(values):.1f}. Short-dated gamma explodes into expiry, which is why "
        "the front-month book is re-hedged all day and why nobody wants to be short it over an "
        "event.",
        "facts": (
            (labels[0], ui.greek_value("gamma", values[0])),
            (labels[1], ui.greek_value("gamma", values[1])),
            ("Ratio", f"{max(values) / min(values):.1f}x"),
        ),
        "compare": {
            "greek": "gamma",
            "legs": ((labels[0], "call", spot, short_days / 365.0),
                     (labels[1], "call", spot, long_days / 365.0)),
        },
    }


def _q_vega_tenor(rng: random.Random) -> dict[str, Any]:
    spot, vol, rate, div = _drill_market(rng)
    short_days, long_days = rng.choice(((30, 365), (91, 365), (30, 182), (7, 91)))
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    labels = (f"The {_tenor_word(short_days)} call", f"The {_tenor_word(long_days)} call")
    values = tuple(
        float(EuropeanOption("call", spot, days / 365.0).greeks(mkt)["vega"]) / 100.0
        for days in (short_days, long_days)
    )
    tie = "They are the same"
    return {
        "topic": "Comparisons",
        "kind": "choice",
        "market": (spot, vol, rate, div),
        "prompt": "Both struck at the spot, same underlying. Which has more vega: the "
        f"{_tenor_word(short_days)} call or the {_tenor_word(long_days)} call?",
        "options": (labels[0], labels[1], tie),
        "answer_label": _pick_larger(labels, values, tie),
        "why": f"Vega grows with the square root of maturity: "
        f"{ui.greek_value('vega', values[0])} against {ui.greek_value('vega', values[1])} per vol "
        "point. Gamma and vega pull in opposite directions along the curve — the front end is a "
        "gamma instrument, the back end a vega one — which is exactly the trade a calendar "
        "spread expresses.",
        "facts": (
            (labels[0], ui.greek_value("vega", values[0])),
            (labels[1], ui.greek_value("vega", values[1])),
            ("Ratio", f"{max(values) / max(min(values), 1e-12):.1f}x"),
        ),
        "compare": {
            "greek": "vega",
            "legs": ((labels[0], "call", spot, short_days / 365.0),
                     (labels[1], "call", spot, long_days / 365.0)),
        },
    }


def _q_call_put_vega(rng: random.Random) -> dict[str, Any]:
    spot, vol, rate, div = _drill_market(rng)
    days = rng.choice((30, 91, 182, 365))
    strike = _strike_at(spot, rng.choice((0.95, 1.0, 1.05)))
    expiry = days / 365.0
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    call = float(EuropeanOption("call", strike, expiry).greeks(mkt)["vega"]) / 100.0
    put = float(EuropeanOption("put", strike, expiry).greeks(mkt)["vega"]) / 100.0
    labels = ("The call", "The put")
    tie = "They are the same"
    return {
        "topic": "Comparisons",
        "kind": "choice",
        "market": (spot, vol, rate, div),
        "prompt": f"Same strike ({strike:g}), same {_tenor_word(days)} maturity. Which has more "
        "vega, the call or the put?",
        "options": (labels[0], labels[1], tie),
        "answer_label": _pick_larger(labels, (call, put), tie),
        "why": f"Both are {ui.greek_value('vega', call)} per point: the vega formula has no option "
        "type in it. The reason is parity — a call minus a put is a forward, and a forward has no "
        "vega, no gamma and no theta from optionality. The same argument makes call and put gamma "
        "identical too. If an interviewer asks this, they are testing whether you know parity or "
        "just the formulas.",
        "facts": (
            ("Call vega per point", ui.greek_value("vega", call)),
            ("Put vega per point", ui.greek_value("vega", put)),
            ("Difference", ui.number(call - put, decimals=6)),
        ),
        "compare": {
            "greek": "vega",
            "legs": (("Call", "call", strike, expiry), ("Put", "put", strike, expiry)),
        },
    }


def _q_call_put_price(rng: random.Random) -> dict[str, Any]:
    spot, vol, rate, div = _drill_market(rng)
    days = rng.choice((91, 182, 365))
    expiry = days / 365.0
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    call = float(EuropeanOption("call", spot, expiry).price(mkt))
    put = float(EuropeanOption("put", spot, expiry).price(mkt))
    forward = float(mkt.forward(expiry))
    labels = ("The call", "The put")
    tie = "They are worth exactly the same"
    carry = (
        "here the rate and the dividend cancel exactly, so the forward is the spot and the two "
        "are worth the same to the last decimal"
        if abs(rate - div) < 1e-12
        else f"here the forward sits {ui.money(abs(forward - spot))} "
        f"{'above' if forward > spot else 'below'} the strike"
    )
    return {
        "topic": "Comparisons",
        "kind": "choice",
        "market": (spot, vol, rate, div),
        "prompt": f"Strike equals spot at {spot:g}, {_tenor_word(days)} maturity, rate "
        f"{ui.percent(rate, decimals=0)} and dividend {ui.percent(div, decimals=0)}. Which is "
        "dearer, the call or the put?",
        "options": (labels[0], labels[1], tie),
        "answer_label": _pick_larger(labels, (call, put), tie),
        "why": "The option is struck on the spot but priced off the forward, and "
        f"{carry}. Call minus put is the discounted distance between forward and strike, "
        f"{ui.money(call - put, signed=True)} — that is put-call parity, and it says nothing "
        "about direction. Swap the rate and the dividend and the answer swaps with them.",
        "facts": (
            ("Call", ui.money(call)),
            ("Put", ui.money(put)),
            ("Forward", ui.money(forward)),
            ("Call minus put", ui.money(call - put, signed=True)),
        ),
        "compare": {
            "greek": "price",
            "legs": (("Call", "call", spot, expiry), ("Put", "put", spot, expiry)),
        },
    }


def _q_gamma_moneyness(rng: random.Random) -> dict[str, Any]:
    spot, vol, rate, div = _drill_market(rng)
    days = rng.choice((30, 91, 182))
    expiry = days / 365.0
    away = _strike_at(spot, 1.10)
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    atm = float(EuropeanOption("call", spot, expiry).greeks(mkt)["gamma"])
    otm = float(EuropeanOption("call", away, expiry).greeks(mkt)["gamma"])
    labels = (f"The {spot:g} call", f"The {away:g} call")
    tie = "They are the same"
    return {
        "topic": "Comparisons",
        "kind": "choice",
        "market": (spot, vol, rate, div),
        "prompt": f"Both {_tenor_word(days)} calls on a spot of {spot:g}. Which has more gamma: "
        f"the one struck at {spot:g} or the one struck at {away:g}?",
        "options": (labels[0], labels[1], tie),
        "answer_label": _pick_larger(labels, (atm, otm), tie),
        "why": f"Gamma peaks where the outcome is still in doubt, which is near the strike: "
        f"{ui.greek_value('gamma', atm)} at the money against {ui.greek_value('gamma', otm)} "
        f"{(away / spot - 1.0) * 100:.0f}% away. Away from the money the delta is already close "
        "to 0 or 1, so there is "
        "little left for it to do. Careful with the wording though: a long-dated wing option can "
        "carry more gamma than a short-dated one at the same distance, because its delta is still "
        "moving.",
        "facts": (
            (labels[0], ui.greek_value("gamma", atm)),
            (labels[1], ui.greek_value("gamma", otm)),
            ("Ratio", f"{atm / max(otm, 1e-12):.1f}x"),
        ),
        "compare": {
            "greek": "gamma",
            "legs": ((labels[0], "call", spot, expiry), (labels[1], "call", away, expiry)),
        },
    }


#: Question templates by topic. "Mixed" draws from all of them.
_TEMPLATES: dict[str, tuple] = {
    "Pricing": (_q_atm_price, _q_parity, _q_probability_itm),
    "Greeks": (_q_delta, _q_vega, _q_theta),
    "P&L": (_q_short_straddle_gap, _q_hedged_overnight, _q_vol_drop),
    "Comparisons": (
        _q_gamma_tenor,
        _q_vega_tenor,
        _q_call_put_vega,
        _q_call_put_price,
        _q_gamma_moneyness,
    ),
}


# --------------------------------------------------------------------------- #
# Cached computations (primitive arguments in, plain data out)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=512)
def drill_question(seed: int, topic: str) -> dict[str, Any]:
    """One quick-fire question, fully determined by ``(seed, topic)``.

    The randomness is a ``random.Random`` seeded with the string ``topic:seed``,
    which is stable across processes, so the same seed always produces the same
    question. Every number in the returned dict is computed by the library here,
    never at display time.
    """
    pool = (
        [template for group in _TEMPLATES.values() for template in group]
        if topic not in _TEMPLATES
        else list(_TEMPLATES[topic])
    )
    rng = random.Random(f"{topic}:{seed}")
    question = dict(rng.choice(pool)(rng))
    question["seed"] = int(seed)
    # Suffix of the answer widget's key: a new question always gets a new widget,
    # so the box starts empty instead of carrying the previous answer over.
    question["key"] = f"{topic}_{seed}"
    return question


@st.cache_data(show_spinner=False, max_entries=64)
def atm_snapshot(spot: float, vol: float, rate: float, div: float, expiry: float) -> dict[str, float]:
    """Raw Greeks of the at-the-money call, plus the levels the cards quote."""
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    call = EuropeanOption("call", spot, expiry)
    out = {name: float(value) for name, value in call.greeks(mkt).items()}
    out["forward"] = float(mkt.forward(expiry))
    out["put_price"] = float(EuropeanOption("put", spot, expiry).price(mkt))
    out["daily_move"] = spot * vol / math.sqrt(252.0)
    return out


@st.cache_data(show_spinner=False, max_entries=64)
def mental_maths(
    spot: float, vol: float, rate: float, div: float, days: int, move_pct: float
) -> pd.DataFrame:
    """The desk approximations against the library's exact numbers.

    One row per rule of thumb, with the approximation, the exact value and the
    relative error. Values are pre-formatted because the rows carry different
    units; only the error stays numeric so the table can colour it.
    """
    expiry = days / 365.0
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    call = EuropeanOption("call", spot, expiry)
    put = EuropeanOption("put", spot, expiry)
    greeks = {name: float(value) for name, value in call.greeks(mkt).items()}
    move = spot * move_pct / 100.0

    rows: list[dict[str, Any]] = []

    def add(rule: str, unit: str, approximation: float, exact: float, decimals: int = 3) -> None:
        rows.append(
            {
                "Rule of thumb": rule,
                "Quoted in": unit,
                "Approximation": ui.number(approximation, decimals=decimals),
                "Exact (library)": ui.number(exact, decimals=decimals),
                "Error": _relative_error(approximation, exact),
            }
        )

    add(
        "ATM option price ~ 0.4 x S x vol x sqrt(T)",
        "currency",
        0.4 * spot * vol * math.sqrt(expiry),
        greeks["price"],
        decimals=2,
    )
    add("ATM call delta ~ 0.50", "per 1.00 of spot", 0.5, greeks["delta"])
    add(
        "One-day move ~ vol / 16 of the spot",
        "currency",
        spot * vol / 16.0,
        spot * vol / math.sqrt(252.0),
        decimals=2,
    )
    add(
        f"Gamma P&L on a {move_pct:+.1f}% move ~ 0.5 x gamma x dS^2",
        "currency",
        0.5 * greeks["gamma"] * move**2,
        float(call.price(mkt.bumped(spot=spot + move))) - greeks["price"] - greeks["delta"] * move,
        decimals=3,
    )
    add(
        "Theta per day ~ -0.5 x gamma x (vol x S)^2 / 365",
        "currency per day",
        -0.5 * greeks["gamma"] * (vol * spot) ** 2 / 365.0,
        greeks["theta"] / 365.0,
    )
    add(
        "Put-call parity: C - P = exp(-rT) x (F - K)",
        "currency",
        float(mkt.discount_factor(expiry)) * (float(mkt.forward(expiry)) - spot),
        greeks["price"] - float(put.price(mkt)),
        decimals=4,
    )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=64)
def tenor_table(spot: float, vol: float, rate: float, div: float) -> pd.DataFrame:
    """The at-the-money call across maturities: the answer to half the bank."""
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    rows = []
    for label, days in TENORS.items():
        expiry = days / 365.0
        option = EuropeanOption("call", spot, expiry)
        greeks = to_trader_units({k: float(v) for k, v in option.greeks(mkt).items()})
        digital = build_exotic(
            "cash_digital", option_type="call", strike=spot, expiry=expiry, payout=1.0
        )
        rows.append(
            {
                "Maturity": label,
                "Price": greeks["price"],
                "Delta": greeks["delta"],
                "Gamma": greeks["gamma"],
                "Vega (per point)": greeks["vega"],
                "Theta (per day)": greeks["theta"],
                "N(d2) chance ITM": float(digital.probability_itm(mkt)),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=64)
def straddle_table(
    spot: float, vol: float, rate: float, div: float, days: int, quantity: int
) -> pd.DataFrame:
    """P&L of a short at-the-money straddle for a set of instant spot moves."""
    expiry = days / 365.0
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    straddle = strat.build_strategy("long_straddle", strike=spot, expiry=expiry, quantity=1.0)
    base = float(straddle.price(mkt))
    rows = []
    for move in (-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0):
        new_spot = spot * (1.0 + move / 100.0)
        value = float(straddle.price(mkt.bumped(spot=new_spot)))
        rows.append(
            {
                "Spot move": move / 100.0,
                "Spot after": new_spot,
                "Straddle value": value,
                f"P&L short {quantity}": -quantity * (value - base),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=64)
def asian_table(spot: float, vol: float, rate: float, div: float, years: float) -> pd.DataFrame:
    """Call on the average, average of the calls, and the vanilla — all priced."""
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    asian = AsianOption("call", spot, years, averaging="arithmetic")
    vanilla = asian.vanilla_equivalent()
    fixings = [years * (i + 1) / 12.0 for i in range(12)]
    average_of_calls = sum(
        float(EuropeanOption("call", spot, fixing).price(mkt)) for fixing in fixings
    ) / len(fixings)
    return pd.DataFrame(
        [
            {
                "Structure": "Call on the average (Asian)",
                "Price": float(asian.price(mkt)),
                "What it pays": "max(average of the fixings - strike, 0), paid at expiry",
            },
            {
                "Structure": "Average of the calls",
                "Price": average_of_calls,
                "What it pays": "the mean of twelve vanilla calls, one per monthly fixing",
            },
            {
                "Structure": "Vanilla call at expiry",
                "Price": float(vanilla.price(mkt)),
                "What it pays": "max(final spot - strike, 0)",
            },
        ]
    )


@st.cache_data(show_spinner=False, max_entries=64)
def digital_table(
    spot: float, vol: float, rate: float, div: float, days: int, strike: float, payout: float
) -> pd.DataFrame:
    """The digital and the call spreads a desk would actually book against it."""
    expiry = days / 365.0
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    digital = build_exotic(
        "cash_digital", option_type="call", strike=strike, expiry=expiry, payout=payout
    )
    rows = [
        {
            "What is priced": "The digital itself (model)",
            "Price": float(digital.price(mkt)),
            "Note": f"payout {payout:g} x discounted N(d2) = "
            f"{float(digital.probability_itm(mkt)) * 100.0:.1f}% chance",
        }
    ]
    for width in (0.02 * spot, 0.05 * spot):
        spread = digital.replicating_call_spread(width, placement="conservative")
        rows.append(
            {
                "What is priced": f"Conservative call spread, width {width:g}",
                "Price": float(spread.price(mkt)),
                "Note": "pays at least as much as the digital everywhere, so it is what a seller "
                "charges",
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=64)
def pitch_numbers(
    name: str, spot: float, vol: float, rate: float, div: float, days: int
) -> dict[str, Any]:
    """Everything the 60-second pitch quotes, straight from ``strategies.summary``."""
    spec = STRATEGY_REGISTRY[name]
    mkt = Market(spot=spot, vol=vol, rate=rate, div=div)
    built = spec.build_default(spot=spot, expiry=days / 365.0)
    report = strat.summary(built, mkt)
    return {
        "title": spec.title,
        "label": report["name"],
        "category": spec.category,
        "description": spec.description,
        "view": list(spec.view),
        "net_premium": float(report["net_premium"]),
        "options_premium": float(report["options_premium"]),
        "premium_type": report["premium_type"],
        "breakevens": [float(level) for level in report["breakevens"]],
        "max_profit": float(report["max_profit"]),
        "max_loss": float(report["max_loss"]),
        "probability_of_profit": report["probability_of_profit"],
        "greeks": {k: float(v) for k, v in report["greeks"].items()},
        "greeks_trader": {k: float(v) for k, v in report["greeks_trader"].items()},
    }


# --------------------------------------------------------------------------- #
# Session state owned by this page, and the callbacks that change it
# --------------------------------------------------------------------------- #
st.session_state.setdefault("interview_seed", 1)
st.session_state.setdefault("interview_asked", 0)
st.session_state.setdefault("interview_right", 0)
st.session_state.setdefault("interview_revealed", False)
st.session_state.setdefault("interview_was_right", None)
st.session_state.setdefault("interview_missing_answer", False)
st.session_state.setdefault("interview_question", None)


def _next_question() -> None:
    """Move to the next seed and hide the previous answer."""
    st.session_state.interview_seed = int(st.session_state.interview_seed) + 1
    st.session_state.interview_revealed = False
    st.session_state.interview_was_right = None
    st.session_state.interview_missing_answer = False


def _reset_drill() -> None:
    """Fresh score, fresh question."""
    st.session_state.interview_asked = 0
    st.session_state.interview_right = 0
    _next_question()


def _topic_changed() -> None:
    """A new topic means a new question, so the old reveal must go."""
    st.session_state.interview_revealed = False
    st.session_state.interview_was_right = None
    st.session_state.interview_missing_answer = False


def _check_answer() -> None:
    """Mark the answer the user is looking at and update the score."""
    question = st.session_state.get("interview_question")
    if not question:
        return
    suffix = question["key"]
    if question["kind"] == "number":
        given = st.session_state.get(f"interview_answer_{suffix}")
        if given is None:
            st.session_state.interview_missing_answer = True
            return
        correct = abs(float(given) - float(question["answer"])) <= float(question["tolerance"])
    else:
        given = st.session_state.get(f"interview_choice_{suffix}")
        if given is None:
            st.session_state.interview_missing_answer = True
            return
        correct = str(given) == str(question["answer_label"])

    st.session_state.interview_missing_answer = False
    st.session_state.interview_revealed = True
    st.session_state.interview_was_right = bool(correct)
    st.session_state.interview_asked = int(st.session_state.interview_asked) + 1
    st.session_state.interview_right = int(st.session_state.interview_right) + int(correct)


# --------------------------------------------------------------------------- #
# The shared market
# --------------------------------------------------------------------------- #
mkt = state.current_market()
spot = state.spot()
trader_units = state.trader_units()

ui.page_header(
    "Interview drills",
    "Practise out loud, then check yourself against the library's own numbers.",
)

st.markdown(
    "Everything here is a rehearsal for the same forty minutes: a few quick numbers to see "
    "whether you think in the right units, a few concepts to see whether you understand the "
    "risk, and a few stories to see whether anyone would put you in front of a client. No number "
    "on this page is typed in by hand — each one is priced by the library when it is shown."
)

ui.how_to_use(
    """
Five tabs, in the order of a real interview round.

1. **Quick-fire drill** — one generated question at a time, marked against the library
   inside a stated tolerance. It invents its own small market so the numbers are new
   every time; every other tab uses the sidebar market.
2. **Mental maths** — the six shortcuts a trader quotes from memory, each next to the
   exact value and its error. Do the left-hand column in your head before you reveal the
   right-hand one.
3. **Flashcards** — the Greeks in the library's own words, plus the concepts that come up
   again and again, each with a live number attached.
4. **Question bank** — the classics, grouped by theme, with a model answer and, where a
   number settles the argument, a table that computes it.
5. **Story prompts** — the half of the interview with no numbers in it, plus a pitch
   builder that writes a 60-second pitch around a structure you pick.

Say every answer out loud. Method, then number, then the direction of the bias.
"""
)

quick_tab, maths_tab, cards_tab, bank_tab, story_tab = st.tabs(
    ["Quick-fire drill", "Mental maths", "Flashcards", "Question bank", "Story prompts"]
)


# --------------------------------------------------------------------------- #
# 1. Quick-fire drill
# --------------------------------------------------------------------------- #
with quick_tab:
    ui.section(
        "One question at a time",
        "Each question comes with its own small market. Estimate first, then check — the point "
        "is the order of magnitude and the reasoning, not the second decimal.",
    )

    asked = int(st.session_state.interview_asked)
    right = int(st.session_state.interview_right)
    ui.metric_row(
        [
            ("Asked", f"{asked}", "Questions you have checked this session."),
            ("Correct", f"{right}", "Answers inside the tolerance shown with each question."),
            (
                "Hit rate",
                f"{right / asked * 100:.0f}%" if asked else "-",
                "A desk would be happy with anything above 70% on the number questions.",
            ),
        ]
    )

    topic_choice = st.segmented_control(
        "Topic",
        list(DRILL_TOPICS),
        default=DEFAULT_TOPIC,
        key="interview_topic",
        on_change=_topic_changed,
        help="Mixed draws from all four families. Pick one to drill a weakness.",
    )
    topic = topic_choice or DEFAULT_TOPIC

    question = drill_question(int(st.session_state.interview_seed), topic)
    st.session_state.interview_question = question
    q_spot, q_vol, q_rate, q_div = question["market"]

    with st.container(border=True):
        st.markdown(
            f":blue-badge[:material/quiz: {question['topic']}] "
            f":gray-badge[:material/tag: seed {question['seed']}]"
        )
        st.caption("The world for this question, invented from the seed:")
        ui.metric_row(
            [
                ("Spot", ui.money(q_spot)),
                ("Implied volatility", ui.percent(q_vol, decimals=1)),
                ("Interest rate", ui.percent(q_rate, decimals=1)),
                ("Dividend yield", ui.percent(q_div, decimals=1)),
            ],
            border=False,
        )
        st.markdown(f"#### {question['prompt']}")

        if question["kind"] == "number":
            st.number_input(
                f"Your answer ({question['unit']})",
                value=None,
                step=float(question["step"]),
                format=question["format"],
                key=f"interview_answer_{question['key']}",
                placeholder="Type your estimate",
                help=f"Marked correct within {ui.number(question['tolerance'], decimals=3)} of the "
                "library's value.",
            )
        else:
            st.segmented_control(
                "Your answer",
                list(question["options"]),
                default=None,
                key=f"interview_choice_{question['key']}",
                help="One of the three is right. Ties are a real answer here, not a trick.",
            )

        with st.container(horizontal=True):
            st.button(
                "Check",
                icon=":material/check:",
                type="primary",
                key="interview_check",
                on_click=_check_answer,
                disabled=bool(st.session_state.interview_revealed),
                help="Marks your answer and adds it to the score.",
            )
            st.button(
                "New question",
                icon=":material/refresh:",
                key="interview_new",
                on_click=_next_question,
                help="Moves to the next seed. The score is kept.",
            )
            st.button(
                "Reset score",
                icon=":material/restart_alt:",
                key="interview_reset",
                on_click=_reset_drill,
                help="Back to nought out of nought.",
            )

    if st.session_state.interview_missing_answer:
        st.warning(
            "Put an answer in the box first — a wrong estimate you can defend is worth more than "
            "a blank one.",
            icon=":material/edit:",
        )

    if st.session_state.interview_revealed:
        if st.session_state.interview_was_right:
            st.success(
                f"Correct. The library says {question['answer_label']}.",
                icon=":material/check_circle:",
            )
        else:
            st.error(
                f"Not this time. The library says {question['answer_label']}.",
                icon=":material/cancel:",
            )

        with st.container(border=True):
            st.markdown(":violet-badge[:material/lightbulb: What a desk would say]")
            st.markdown(question["why"])
            ui.metric_row([(label, value) for label, value in question["facts"]], border=False)

        compare = question.get("compare")
        if compare:
            instruments = {
                label: EuropeanOption(option_type, strike, expiry)
                for label, option_type, strike, expiry in compare["legs"]
            }
            try:
                fig = profiles.compare_instruments(
                    instruments,
                    Market(spot=q_spot, vol=q_vol, rate=q_rate, div=q_div),
                    greek=compare["greek"],
                    trader_units=True,
                    mode=ui.CHART_MODE,
                )
            except ValueError as exc:
                st.warning(f"That comparison cannot be drawn: {exc}", icon=":material/warning:")
            else:
                ui.chart(fig, key="interview_drill_compare")
                ui.explain(
                    "How to read this",
                    "The two instruments of the question, overlaid on the same axes, with the "
                    "spot running along the bottom and everything else frozen at the market of "
                    "the question. The dot marks where the spot is now, which is the single point "
                    "the question asked about — the rest of the curve tells you whether the "
                    "answer would survive a move.\n\n"
                    "Watch the shape, not just the height: which curve is taller *here* can be "
                    "the opposite of which curve is taller two standard deviations away, and that "
                    "is exactly the follow-up question an interviewer asks next. Greeks are "
                    "quoted in desk units throughout this drill, whatever the sidebar toggle says.",
                    icon=":material/help:",
                )

    ui.explain(
        "How this drill works",
        "Every question is generated from a seed, so the same seed always gives the same "
        "question and the same market — useful if you want to come back to one you got wrong. "
        "The market shown in the card is **not** the sidebar market: the drill invents its own so "
        "that the numbers are new each time. Every other tab on this page uses the sidebar.\n\n"
        "Number questions are marked inside a tolerance, shown on the tooltip of the answer box, "
        "because an interviewer wants the right order of magnitude fast, not four decimals slowly. "
        "Comparison questions have a third option — *they are the same* — and it is sometimes the "
        "right one.",
        icon=":material/help:",
    )

    ui.interview_note(
        "Say the method, then the number, then the correction. 'At the money, so roughly 0.4 "
        "times spot times vol times root time' — do the arithmetic out loud — 'and the forward "
        "is above the strike, so the call is a touch more than that.' In one breath you have "
        "given an answer, a method and the direction of its bias, and if the arithmetic slips "
        "the interviewer can see exactly where. Going silent for twenty seconds and then saying "
        "one number scores worse, even when the number is right.",
        question="How should I actually answer a mental pricing question out loud?",
    )


# --------------------------------------------------------------------------- #
# 2. Mental maths
# --------------------------------------------------------------------------- #
with maths_tab:
    ui.section(
        "The approximations traders quote from memory",
        "All of them on the sidebar market, each against the exact value. Try the column on the "
        "left in your head before you reveal the one on the right.",
    )

    controls = st.columns([3, 2], vertical_alignment="bottom")
    tenor_choice = controls[0].segmented_control(
        "Maturity",
        list(TENORS),
        default=DEFAULT_TENOR,
        key="interview_maths_tenor",
        help="Maturity of the at-the-money option every rule is applied to.",
    )
    move_pct = controls[1].slider(
        "Spot move for the gamma rule",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.5,
        format="%.1f%%",
        key="interview_maths_move",
        help="Instantaneous move used by the gamma P&L row and the chart below.",
    )
    days = TENORS[tenor_choice or DEFAULT_TENOR]

    if spot <= 0 or mkt.vol <= 0:
        ui.empty_state(
            "These rules need a positive spot and a positive volatility. Raise them in the "
            "sidebar and the table will fill in.",
            icon=":material/warning:",
        )
    else:
        snapshot = atm_snapshot(spot, float(mkt.vol), float(mkt.rate), float(mkt.div), days / 365.0)
        ui.greek_metrics(snapshot, trader_units=trader_units)
        ui.units_caption(trader_units)

        reveal = st.toggle(
            "Reveal the exact numbers",
            value=False,
            key="interview_maths_reveal",
            help="Off by default so you can do the arithmetic first, which is the drill.",
        )

        table = mental_maths(
            spot, float(mkt.vol), float(mkt.rate), float(mkt.div), days, float(move_pct)
        )
        columns = ["Rule of thumb", "Quoted in", "Approximation"]
        config: dict[str, Any] = {
            "Rule of thumb": st.column_config.TextColumn("Rule of thumb", width="large"),
            "Quoted in": st.column_config.TextColumn("Quoted in", width=150),
            "Approximation": st.column_config.TextColumn("Approximation", width=140),
        }
        if reveal:
            columns += ["Exact (library)", "Error"]
            config["Exact (library)"] = st.column_config.TextColumn("Exact (library)", width=140)
            config["Error"] = st.column_config.NumberColumn(
                "Error",
                format="percent",
                width=110,
                help="Approximation divided by the exact value, minus one.",
            )
        ui.table(
            table[columns],
            column_config=config,
            height=min(44 * len(table) + 44, 420),
        )
        if not reveal:
            st.caption(
                "The exact column is hidden. Work each row out in your head, then flip the "
                "toggle."
            )

        ui.explain(
            "How to read this",
            "Each row is one shortcut a trader says out loud, applied to the at-the-money option "
            "of the maturity you picked, on the market in the sidebar.\n\n"
            "- **The price rule** comes from the density at the money being about 0.4. It prices "
            "the option off the spot, so it drifts once rates, dividends or maturity get large.\n"
            "- **The delta rule** is deliberately crude: at the money a call is a little above "
            "0.5, because d1 carries the carry and half the variance.\n"
            "- **The daily move** uses 16 because there are about 256 trading days in a year and "
            "the square root of 256 is 16. The exact column uses 252.\n"
            "- **The gamma and theta rules** are the two halves of the same equation, which is "
            "why the chart below plots them together.\n"
            "- **Put-call parity is not an approximation at all**: the error is zero to the last "
            "decimal, and that is the point of including it.",
            icon=":material/help:",
        )

        call = EuropeanOption("call", spot, days / 365.0)
        try:
            fig = profiles.gamma_theta_tradeoff(
                call, mkt, dt=1.0 / 365.0, trader_units=trader_units, mode=ui.CHART_MODE
            )
        except ValueError as exc:
            st.warning(f"The chart cannot be drawn on this market: {exc}", icon=":material/warning:")
        else:
            ui.chart(fig, key="interview_gamma_theta")
            ui.explain(
                "How to read this",
                "One day in the life of a delta-hedged long call. The horizontal axis is where "
                "the spot ends up tomorrow; the vertical axis is what the hedged package made. "
                "The solid line is the full repricing, the dashed one is the approximation "
                "`0.5 x gamma x dS^2 + theta x dt` from the table above — the two sit almost on "
                "top of each other, which is why the shortcut is worth memorising.\n\n"
                "The parabola dips below zero for small moves (theta wins) and lifts above it for "
                "large ones (gamma wins). The crossing points are marked: they are the move you "
                "need just to pay the rent.",
                icon=":material/help:",
            )

            gamma = snapshot["gamma"]
            theta_day = snapshot["theta"] / 365.0
            sigma_move = snapshot["daily_move"]
            breakeven_phrase = (
                f"is {ui.money(math.sqrt(-2.0 * theta_day / gamma))}"
                if gamma > 0 and theta_day < 0
                else "does not exist on this market, because gamma and theta do not disagree"
            )
            ui.takeaways(
                [
                    f"The implied daily move is {ui.money(sigma_move)}, about "
                    f"{_signed_percent(sigma_move / spot)} of the spot. That is the number the "
                    "option is priced for, and the one you compare every actual day against.",
                    f"Gamma is {ui.greek_value('gamma', gamma)} and theta "
                    f"{ui.greek_value('theta', theta_day)} a day. The move that just pays for the "
                    f"day {breakeven_phrase}, against the {ui.money(sigma_move)} implied — "
                    "close by construction, and not exactly equal because of carry.",
                    f"The rule of thumb 0.4 x S x vol x sqrt(T) gives "
                    f"{ui.money(0.4 * spot * float(mkt.vol) * math.sqrt(days / 365.0))} for this "
                    f"option against the model's {ui.money(snapshot['price'])}. Quote the "
                    "shortcut, then say which way it is biased.",
                ]
            )


# --------------------------------------------------------------------------- #
# 3. Flashcards
# --------------------------------------------------------------------------- #
with cards_tab:
    ui.section(
        "Flashcards",
        "Concept on the front, answer on the back. The Greek cards use the library's own "
        "definitions and quote a live number from the market in the sidebar.",
    )

    card_topics = ("Greeks", "Volatility", "Hedging", "Products", "Desk life")
    chosen_topics = st.pills(
        "Topics",
        list(card_topics),
        selection_mode="multi",
        default=list(card_topics),
        key="interview_card_topics",
        help="Narrow the deck to the family you are weakest on.",
    )
    open_all = st.toggle(
        "Open every card",
        value=False,
        key="interview_cards_open",
        help="Off for revision, on for a quick read-through.",
    )

    selected = list(chosen_topics or ())
    if not selected:
        ui.empty_state(
            "No topic selected, so the deck is empty. Pick at least one above.",
            icon=":material/filter_list:",
        )
    else:
        card_snapshot = (
            atm_snapshot(spot, float(mkt.vol), float(mkt.rate), float(mkt.div), 0.25)
            if spot > 0
            else None
        )
        cards: list[dict[str, str]] = []

        if "Greeks" in selected:
            greek_values = (
                to_trader_units(dict(card_snapshot)) if (card_snapshot and trader_units)
                else dict(card_snapshot or {})
            )
            for name in ("delta", "gamma", "vega", "theta", "rho", "vanna", "volga", "charm"):
                info = GREEK_INFO[name]
                back = info["description"]
                if card_snapshot is not None:
                    unit = info["trader_unit" if trader_units else "raw_unit"]
                    back += (
                        f"\n\nOn the market in the sidebar, the 3-month at-the-money call has "
                        f"**{info['label'].lower()} "
                        f"{ui.greek_value(name, float(greek_values[name]))}** ({unit})."
                    )
                cards.append(
                    {
                        "topic": "Greeks",
                        "front": f"{info['label']} — what is it, and what does it do to your P&L?",
                        "back": back,
                    }
                )

        cards += [card for card in CONCEPT_CARDS if card["topic"] in selected]

        st.caption(f"{len(cards)} cards in the deck.")
        for card in cards:
            with st.expander(card["front"], expanded=open_all, icon=":material/style:"):
                st.markdown(f":gray-badge[:material/label: {card['topic']}]")
                st.markdown(card["back"])

        ui.takeaways(
            [
                "Answer a flashcard out loud in three sentences: what it is, what it does to your "
                "P&L, and one thing that surprises people about it. That is the shape of a good "
                "interview answer.",
                "The Greek cards quote today's numbers on purpose. 'Vega is the sensitivity to "
                "implied vol' is a definition; 'I am long about 0.2 per point on this option, so "
                "a one-point rally pays me 0.2' is an answer.",
            ]
        )


# --------------------------------------------------------------------------- #
# 4. Question bank
# --------------------------------------------------------------------------- #
with bank_tab:
    ui.section(
        "Question bank",
        "Classic questions grouped by theme, each with a model answer. Where a number settles "
        "the argument, it is computed on the market in the sidebar.",
    )

    themes = list(dict.fromkeys(entry["theme"] for entry in BANK))
    theme_choice = st.segmented_control(
        "Theme",
        ["All", *themes],
        default="All",
        key="interview_bank_theme",
        help="One theme at a time is a good way to rehearse a round.",
    )
    theme_filter = theme_choice or "All"

    shown = [
        entry for entry in BANK if theme_filter == "All" or entry["theme"] == theme_filter
    ]
    st.caption(f"{len(shown)} questions shown.")

    for index, entry in enumerate(BANK):
        if theme_filter != "All" and entry["theme"] != theme_filter:
            continue
        with st.expander(entry["question"], icon=":material/help_center:"):
            st.markdown(f":blue-badge[:material/category: {entry['theme']}]")
            st.markdown(entry["answer"])

            if entry["show"] and spot > 0:
                if st.checkbox(
                    "Show me the numbers",
                    key=f"interview_show_{index}",
                    help="Priced by the library on the market in the sidebar.",
                ):
                    kind = entry["show"]
                    if kind == "tenors":
                        ui.table(
                            tenor_table(spot, float(mkt.vol), float(mkt.rate), float(mkt.div)),
                            column_config={
                                "Price": st.column_config.NumberColumn(format="%.3f"),
                                "Delta": st.column_config.NumberColumn(format="%.3f"),
                                "Gamma": st.column_config.NumberColumn(format="%.4f"),
                                "Vega (per point)": st.column_config.NumberColumn(format="%.3f"),
                                "Theta (per day)": st.column_config.NumberColumn(format="%.3f"),
                                "N(d2) chance ITM": st.column_config.NumberColumn(
                                    format="percent",
                                    help="Risk-neutral probability of finishing in the money.",
                                ),
                            },
                        )
                        st.caption(
                            "The at-the-money call across maturities, in desk units. Delta stays "
                            "near 0.5 while gamma and vega move in opposite directions along the "
                            "curve."
                        )
                    elif kind == "straddle":
                        quantity = 50
                        table = straddle_table(
                            spot, float(mkt.vol), float(mkt.rate), float(mkt.div), 30, quantity
                        )
                        ui.table(
                            table,
                            column_config={
                                "Spot move": st.column_config.NumberColumn(format="percent"),
                                "Spot after": st.column_config.NumberColumn(format="%.2f"),
                                "Straddle value": st.column_config.NumberColumn(format="%.3f"),
                                f"P&L short {quantity}": st.column_config.NumberColumn(
                                    format="%.2f"
                                ),
                            },
                        )
                        st.caption(
                            f"Short {quantity} of the 1-month at-the-money straddle, instant "
                            "moves, implied vol unchanged. The loss grows with the square of the "
                            "move: that is short gamma."
                        )
                    elif kind == "asian":
                        table = asian_table(
                            spot, float(mkt.vol), float(mkt.rate), float(mkt.div), 1.0
                        )
                        ui.table(
                            table,
                            column_config={
                                "Structure": st.column_config.TextColumn(width=220),
                                "Price": st.column_config.NumberColumn(format="%.3f"),
                                "What it pays": st.column_config.TextColumn(width="large"),
                            },
                        )
                        gap = float(table.loc[1, "Price"]) - float(table.loc[0, "Price"])
                        st.caption(
                            f"One-year structures struck at the spot. The average of the calls is "
                            f"{ui.money(gap)} dearer than the call on the average — Jensen's "
                            "inequality, priced."
                        )
                    elif kind == "digital":
                        strike = _strike_at(spot, 1.05)
                        table = digital_table(
                            spot, float(mkt.vol), float(mkt.rate), float(mkt.div), 91,
                            strike, 10.0,
                        )
                        ui.table(
                            table,
                            column_config={
                                "What is priced": st.column_config.TextColumn(width=260),
                                "Price": st.column_config.NumberColumn(format="%.3f"),
                                "Note": st.column_config.TextColumn(width="large"),
                            },
                        )
                        st.caption(
                            f"Pays 10 if the spot finishes above {strike:g} in three months. The "
                            "spreads are what a seller would actually book; the difference "
                            "between them and the model price is the price of not being short a "
                            "step."
                        )

            if entry["link"]:
                page, label, icon = entry["link"]
                st.page_link(page, label=label, icon=icon)

    ui.interview_note(
        "Answer in layers. First the one-sentence answer, then the mechanism, then the caveat "
        "that shows you have actually traded the idea in your head. For 'why is gamma biggest at "
        "the money': the delta only changes where the outcome is in doubt; it is the curvature of "
        "the price, so it peaks at the strike and grows as expiry approaches; and the caveat is "
        "that a long-dated wing option can still carry more gamma than a short-dated one at the "
        "same distance, because its delta is still moving. Three layers, twenty seconds.",
        question="How much detail is the right amount?",
    )


# --------------------------------------------------------------------------- #
# 5. Story prompts
# --------------------------------------------------------------------------- #
with story_tab:
    ui.section(
        "The half of the interview with no numbers in it",
        "Same rule as the technical half: structure first, evidence second, honesty throughout.",
    )

    for row_start in (0, 2):
        for column, prompt in zip(st.columns(2), STORY_PROMPTS[row_start : row_start + 2]):
            with column.container(border=True, height="stretch"):
                st.markdown(f"{prompt['icon']} **{prompt['prompt']}**")
                for step in prompt["structure"]:
                    st.markdown(f"- {step}")
                st.caption(f"Trap: {prompt['trap']}")

    ui.section(
        "Pitch me a trade in 60 seconds",
        "Pick a structure and the app builds the pitch around real numbers: view, trade, risk, "
        "and what would make you wrong.",
    )

    pitch_controls = st.columns([2, 3], vertical_alignment="bottom")
    pitch_name = pitch_controls[0].selectbox(
        "Structure",
        list(PITCH_STRATEGIES),
        format_func=lambda key: STRATEGY_REGISTRY[key].title,
        key="interview_pitch_strategy",
        help="Eight structures that cover the usual views: directional, volatility, income and "
        "protection.",
    )
    pitch_tenor = pitch_controls[1].segmented_control(
        "Maturity",
        list(TENORS),
        default=DEFAULT_TENOR,
        key="interview_pitch_tenor",
        help="Reference maturity; a calendar spread places its two legs around it.",
    )

    pitch_days = TENORS[pitch_tenor or DEFAULT_TENOR]

    if spot <= 0:
        ui.empty_state(
            "A pitch needs a positive spot. Set one in the sidebar.", icon=":material/warning:"
        )
    else:
        try:
            pitch = pitch_numbers(
                str(pitch_name), spot, float(mkt.vol), float(mkt.rate), float(mkt.div), pitch_days
            )
        except (ValueError, KeyError) as exc:
            pitch = None
            st.warning(
                f"That structure cannot be built on this market: {exc}. Try another maturity.",
                icon=":material/warning:",
            )

        if pitch is not None:
            # The spoken pitch follows the sidebar toggle like every other page, so
            # the sentence and the metric row can never quote two different units.
            greeks = pitch["greeks_trader"] if trader_units else pitch["greeks"]
            unit_key = "trader_unit" if trader_units else "raw_unit"
            premium = pitch["net_premium"]
            paid = premium > 0
            max_profit = pitch["max_profit"]
            max_loss = pitch["max_loss"]
            breakevens = pitch["breakevens"]
            probability = pitch["probability_of_profit"]

            profit_label = ui.extreme(max_profit, absolute=True)
            loss_label = ui.extreme(max_loss, absolute=True)
            breakeven_label = (
                ", ".join(ui.money(level) for level in breakevens) if breakevens else "none"
            )
            profit_phrase = (
                "an unlimited gain" if math.isinf(max_profit) else f"a gain of {ui.money(max_profit)}"
            )
            if math.isinf(max_loss):
                loss_phrase = "an unlimited loss"
            elif max_loss >= 0:
                loss_phrase = f"no loss at all — the worst case is still {ui.money(max_loss)}"
            else:
                loss_phrase = f"a loss of {ui.money(abs(max_loss))}"
            # The "is theta material?" test is always made on the desk number (per
            # calendar day), so the wording does not change with the units toggle.
            theta_per_day = float(pitch["greeks_trader"]["theta"])
            theta_phrase = (
                "time barely matters here, because the legs cancel"
                if abs(theta_per_day) < 0.001 * max(abs(premium), 1.0)
                else "time is against me" if theta_per_day < 0 else "time is on my side"
            )

            ui.metric_row(
                [
                    (
                        "Entry cost" if paid else "Premium collected",
                        ui.money(abs(premium)),
                        "Positive means you pay a debit; negative means you receive a credit. "
                        "Stock legs are included.",
                    ),
                    (
                        "Max gain",
                        profit_label,
                        "Best case at the analysis horizon, as a positive amount.",
                    ),
                    (
                        "Max loss",
                        loss_label,
                        "Worst case at the analysis horizon, as a positive amount. Read it next "
                        "to the chance of profit, never on its own.",
                    ),
                    (
                        "Model chance of profit",
                        ui.percent(probability, decimals=0) if probability is not None else "-",
                        "Risk-neutral, not a forecast. Read it next to the maximum loss.",
                    ),
                ]
            )
            ui.greek_metrics(
                pitch["greeks"],
                keys=("delta", "gamma", "vega", "theta"),
                trader_units=trader_units,
            )
            ui.units_caption(trader_units)
            st.caption(f"Breakeven at expiry: {breakeven_label}.")

            wrong = [
                WRONG_BY_VIEW[tag] for tag in pitch["view"] if tag in WRONG_BY_VIEW
            ] or ["the market doing nothing I had a view on, which costs me the premium"]
            first_wrong = wrong[0][0].upper() + wrong[0][1:]

            options_premium = pitch["options_premium"]
            stock_note = (
                f" That is the whole package including the stock at {ui.money(spot)}; the options "
                f"on their own are a {'debit' if options_premium > 0 else 'credit'} of "
                f"{ui.money(abs(options_premium))}."
                if abs(premium - options_premium) > 1e-9
                else ""
            )

            with st.container(border=True):
                st.markdown(
                    f":violet-badge[:material/mic: 60-second pitch] "
                    f":gray-badge[:material/category: {pitch['category']}]"
                )
                st.markdown(
                    f"**View.** {pitch['description'].split('. ')[0]}. In one line: this is a "
                    f"{', '.join(pitch['view'])} trade.\n\n"
                    f"**Trade.** The {pitch['label']}, for a "
                    f"{'debit' if paid else 'credit'} of {ui.money(abs(premium))} — "
                    f"{abs(premium) / spot * 100:.1f}% of spot.{stock_note}\n\n"
                    f"**Risk.** Worst case is {loss_phrase}, best case {profit_phrase}, breakeven "
                    f"at {breakeven_label}. Today I am "
                    f"{ui.greek_value('delta', greeks['delta'])} delta, "
                    f"{ui.greek_value('vega', greeks['vega'])} vega "
                    f"({GREEK_INFO['vega'][unit_key]}) and "
                    f"{ui.greek_value('theta', greeks['theta'])} theta "
                    f"({GREEK_INFO['theta'][unit_key]}), so {theta_phrase}.\n\n"
                    f"**What would make me wrong.** {first_wrong}"
                    + (f"; also {wrong[1]}." if len(wrong) > 1 else ".")
                )
            st.caption(
                "Every number in the pitch is priced by the library on the sidebar market and the "
                "maturity above. Change either and the pitch changes with them."
            )

            ui.explain(
                "Why this is the right shape",
                "Four beats, in this order, and nothing else:\n\n"
                "1. **View** — what you think, in one sentence. No preamble.\n"
                "2. **Trade** — the structure and what it costs, as cash and as a percentage of "
                "spot, because that is how a client hears it.\n"
                "3. **Risk** — worst case, best case, breakeven, and the Greeks you are running "
                "today. This is the part candidates skip and interviewers are waiting for.\n"
                "4. **What would make you wrong** — the single scenario that kills the trade. "
                "Saying it yourself is what makes the rest credible.\n\n"
                "If you only have thirty seconds, drop the best case, never the worst.",
                icon=":material/help:",
            )

    ui.interview_note(
        "Pick a trade you can defend, not the cleverest one. A three-month call spread with a "
        "clear view, a known maximum loss and one sentence on what would kill it beats a "
        "structure you half understand. And finish on the risk, not on the upside — the last "
        "thing you say is the thing they remember, and a trader who volunteers the downside is a "
        "trader who can be left alone with a position.",
        question="Pitch me a trade.",
    )
