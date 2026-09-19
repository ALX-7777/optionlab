# OptionLab

[![CI](https://github.com/ALX-7777/optionlab/actions/workflows/ci.yml/badge.svg)](https://github.com/ALX-7777/optionlab/actions/workflows/ci.yml)
![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![uv](https://img.shields.io/badge/managed%20with-uv-purple)
![Streamlit](https://img.shields.io/badge/app-Streamlit-red)
![Docker](https://img.shields.io/badge/runs%20in-Docker-2496ED)

**An interactive lab to understand option Greeks and practise managing an option book.**

OptionLab prices European options with Black-Scholes and plots every Greek against the spot, the volatility and the time to expiry. It applies the same tools to 37 classic option strategies and to exotic options. A virtual book and a market simulator let you trade, hedge, and see exactly where your P&L comes from.

---

## Features

**Vanilla options**
- Black-Scholes prices with a dividend yield, and implied volatility
- 11 Greeks: delta, gamma, vega, theta, rho, vanna, volga, charm, speed, color and zomma, in raw or trader units
- Charts:
  - each Greek against spot, volatility, time or rate
  - how a Greek changes as expiry approaches
  - 3D surfaces
  - call vs put
  - the Taylor P&L approximation
  - gamma vs theta

**Strategies.** 37 predefined strategies, each with a description of the market view it expresses, a payoff diagram, breakevens, maximum profit and loss, and combined Greeks:

| Family | Strategies |
|---|---|
| Single leg | long / short call, long / short put |
| Stock + option | covered call, protective put, collar |
| Vertical spreads | bull call, bear put, bull put, bear call |
| Volatility | long / short straddle, long / short strangle, strip, strap |
| Butterflies & condors | call / put butterfly, iron butterfly, call condor, iron condor |
| Ratio & backspreads | call / put ratio spreads, call / put backspreads |
| Directional combos | risk reversal, seagull, jade lizard |
| Synthetics & arbitrage | synthetic long / short, conversion, reversal, box spread |
| Time spreads | calendar, diagonal, double calendar |

**Exotic options.** Closed-form prices, each checked against Monte Carlo:
- Digital options: cash-or-nothing and asset-or-nothing
- Barrier options: up / down, knock-in / knock-out, with rebates
- Asian options: geometric (exact) and arithmetic (Turnbull-Wakeman approximation)
- Lookback options: floating and fixed strike

**Book management**
- Trades, cash and positions, with aggregated Greeks and dollar Greeks
- Risk ladders, spot × volatility scenario grids and stress tests
- Hedging: delta hedging, neutralising any Greek with a chosen instrument, or several Greaks at once (e.g. delta-gamma neutral)
- Save and load a book as JSON

**Trading simulator**
- Market scenarios where realised volatility can differ from implied volatility, with optional jumps and a moving implied volatility
- Hedging policies: no hedge, every N steps, delta bands, or hedging at a volatility you choose
- P&L attribution by Greek: delta, gamma, theta, vega, vanna, volga, rho and carry
- An experiment on hedging frequency, and a trading game in the terminal

---

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) and Git. Docker is optional.

```bash
git clone https://github.com/ALX-7777/optionlab.git
cd optionlab
uv sync                          # creates .venv and installs the exact versions from uv.lock
uv run streamlit run app.py      # opens the app at http://localhost:8501
```

<details><summary>Without uv (pip)</summary>

```bash
python -m venv .venv
source .venv/Scripts/activate    # Windows (Git Bash); on macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
</details>

## The Streamlit app

| Tab | What you can do |
|---|---|
| **Vanilla Greeks** | Choose a call or a put, its strike and expiry, see every Greek against the spot, and watch one Greek change as expiry approaches |
| **Strategies** | Pick any of the 37 strategies and see its description, net premium, maximum profit and loss, and payoff diagram |
| **My book** | Trade the option from the first tab and follow your positions, Greeks and P&L |

You set the market (spot, volatility, interest rate, dividend yield) in the sidebar, and every tab uses it.

## Example scripts

Each script prints a report in the terminal and saves interactive charts as HTML files in `outputs/`. Add `--show` to open them in your browser.

| Script | What it shows |
|---|---|
| `examples/01_vanilla_greeks.py` | A guided tour of the Greeks of a call and a put |
| `examples/02_strategies.py` | Summary, payoff diagram and Greeks of any strategy (`--list` shows them all) |
| `examples/03_exotics.py` | Each exotic against its vanilla, closed form against Monte Carlo, barrier paths, digital replication |
| `examples/04_book_and_hedging.py` | A book risk report, a hedged vs unhedged simulation, and the hedging-frequency experiment |
| `examples/05_trading_game.py` | An interactive trading game in the terminal (`--demo` plays a scripted game) |

```bash
uv run python examples/01_vanilla_greeks.py
uv run python examples/02_strategies.py --strategy iron_condor long_straddle
uv run python examples/05_trading_game.py
```

## Using the library

```python
from optionlab import Book, EuropeanOption, Market, to_trader_units
from optionlab.exotics import BarrierOption
from optionlab.plotting import profiles

mkt = Market(spot=100, vol=0.20, rate=0.03, div=0.01)
call = EuropeanOption("call", strike=100, expiry=0.5)

call.price(mkt)                           # 6.09
to_trader_units(call.greeks(mkt))         # delta 0.55, vega 0.28 per vol point, theta -0.018 per day, ...

fig = profiles.greek_evolution(call, mkt, greek="gamma")   # a Plotly figure
fig.write_html("gamma.html")

barrier = BarrierOption(option_type="call", strike=100, barrier=120,
                        expiry=0.5, barrier_type="up-and-out")
book = Book(cash=100_000)
book.trade(call, 10, mkt)
book.trade(barrier, -10, mkt)
book.positions_frame(mkt)                 # positions, values and Greeks, with a TOTAL row
```

Conventions:
- Times are in years, and rates and volatilities are decimals, so 0.20 means 20%.
- `greeks()` returns raw derivatives.
- `to_trader_units()` converts them to trader units: vega per volatility point, theta per calendar day, rho per 1%.

## Development

| Task | Command |
|---|---|
| Run the tests | `uv run pytest` |
| Tests with coverage | `uv run pytest --cov=optionlab --cov-report=term-missing` |
| Lint | `uv run ruff check .` |
| Add a dependency | `uv add <package>`, then regenerate `requirements.txt` with `uv export --format requirements.txt --no-dev --no-hashes -o requirements.txt` |

The test suite has **1,314 tests** and covers **98%** of the library.

## Docker

```bash
docker build -t optionlab .
docker run --rm -p 8501:8501 optionlab
```

Then open http://localhost:8501.

The image:
- is based on `python:3.13-slim`
- installs the dependencies with uv from `uv.lock`, without the development tools
- starts the Streamlit app

## Continuous integration

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push to `main` and on every pull request. It:

1. installs the exact environment from `uv.lock` (`uv sync --locked`)
2. lints the code with ruff
3. runs the test suite, and fails if coverage drops below 80%

## How the maths is checked

- Black-Scholes prices match textbook values: call 10.4506 and put 5.5735 for S = K = 100, T = 1, r = 5%, σ = 20%.
- Every analytic Greek is compared with finite differences of the price.
- Put-call parity holds, and for barriers, knock-in + knock-out = vanilla.
- Digital, barrier, Asian and lookback prices match reference values from Haug, *The Complete Guide to Option Pricing Formulas*.
- Every closed-form exotic price is compared with a Monte Carlo price.
- Delta hedging behaves as theory predicts:
  - When realised volatility equals implied volatility, the average hedged P&L is about zero.
  - A long-gamma book makes money when realised volatility is above implied volatility.

## Project structure

```
.
├── app.py                  # Streamlit app
├── optionlab/              # the library
│   ├── market.py           # market state: spot, volatility, rate, dividend, time
│   ├── black_scholes.py    # prices, Greeks, implied volatility
│   ├── instruments.py      # options, positions, multi-leg instruments
│   ├── numerical.py        # Greeks by bump-and-reprice (used for exotics)
│   ├── monte_carlo.py      # path simulation and Monte Carlo pricing
│   ├── strategies.py       # the 37 strategies and their analytics
│   ├── exotics/            # digital, barrier, Asian and lookback options
│   ├── book.py             # positions, cash, risk reports, hedging
│   ├── simulator.py        # market scenarios, trading simulator, P&L attribution
│   └── plotting/           # all the Plotly charts
├── examples/               # 5 runnable scripts
├── tests/                  # pytest suite
├── pyproject.toml          # project metadata, dependencies, tool settings
├── uv.lock                 # exact versions of every dependency
├── requirements.txt        # the same dependencies, in pip format
├── Dockerfile
├── .dockerignore
└── .github/workflows/ci.yml
```

## Limitations

OptionLab is an educational tool, not a trading system. Its model makes simplifying assumptions:
- a flat volatility, with no smile
- constant interest rates and dividend yield
- continuous barrier monitoring
- lognormal prices

The arithmetic Asian price is an approximation, about 0.15% away from the Monte Carlo price.

## Author

Alix Bernal — final project for the *Tooling for Data Scientists* course, X-HEC, 2026.
