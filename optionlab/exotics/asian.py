"""Asian (average-price) options with continuous averaging.

An Asian option pays on the **average** of the spot over a window
``[avg_start, expiry]`` instead of the spot on the last day::

    call = max(A - K, 0)        put = max(K - A, 0)

    arithmetic:  A = 1/L * integral of S_u du          (L = expiry - avg_start)
    geometric:   A = exp(1/L * integral of ln S_u du)

Why they exist, and why they are cheap
--------------------------------------
Averaging smooths the path: one wild closing print cannot make or break the
payoff, which is why Asians are the standard product for commodity and FX
hedgers whose real exposure is an average purchase price. An average moves
less than its last point, so the option is worth less than a vanilla: for a
window covering the whole life, the volatility of the average is about
``sigma / sqrt(3)`` (the variance of the time-average of a Brownian motion is
``T / 3`` instead of ``T``), and its forward only picks up half of the carry.

How the Greeks die
------------------
Every day a slice of the average gets *fixed*. Once a fraction of the window
is known, only the remaining weight ``w = remaining / L`` of the payoff still
reacts to the spot: delta, gamma and vega all shrink roughly like ``w`` and
the option ends its life as an almost deterministic cash amount. This is the
opposite of a vanilla, whose at-the-money gamma explodes into expiry, and it
makes Asians much easier to hedge on their last days.

Pricing
-------
* **Geometric** averaging has an exact closed form (Kemna and Vorst, 1990,
  here with cost of carry ``b = rate - div``) because a geometric average of
  lognormals is lognormal.
* **Arithmetic** averaging has no closed form. The Turnbull-Wakeman (1991) /
  Levy approach matches the first two moments of the average with a lognormal
  and prices it with the Black formula. It is fast, vectorised and smooth
  (hence good Greeks). At the money it is within about 1% of the true price
  for volatilities up to 40% and maturities up to a year (0.2% at 10% vol),
  and within 0.2% of the *spot* at every strike. The true average is less
  skewed than a lognormal, so the fit overprices low strikes and underprices
  high ones: on a cheap out-of-the-money option the same absolute error is a
  larger percentage, and the bias grows with ``vol**2 * T``.
  :func:`mc_price_with_control_variate` gives the unbiased benchmark.

Both share one picture: *an Asian is a vanilla on the average*, priced with
Black's formula on the forward and the variance of what is still random.

State and discretisation
------------------------
The part of the average that is already known lives in the instrument fields
(``running_average`` over ``[avg_start, observed_until]`` and ``last_spot``).
:meth:`AsianOption.observe` and :meth:`AsianOption.path_payoff` both use the
**trapezoid rule** on the observation dates (on the logarithm of the spot for
geometric averaging), so walking a path with ``observe`` and feeding the same
path to ``path_payoff`` give the same average. The trapezoid average of a path
sampled with step ``dt`` converges to the continuous average with an error of
order ``dt**2``: a daily grid is indistinguishable from continuous averaging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import ndtr

from .. import black_scholes as bs
from ..instruments import (
    EXPIRY_TOL,
    EuropeanOption,
    Instrument,
    register_instrument,
    slice_paths_to_expiry,
)
from ..market import Market
from ..monte_carlo import Seed, simulate_gbm_paths_on_grid

__all__ = ["AVERAGING_TYPES", "AsianOption", "mc_price_with_control_variate"]

#: Accepted values of ``AsianOption.averaging``.
AVERAGING_TYPES: tuple[str, ...] = ("arithmetic", "geometric")

# Gauss-Legendre rule on [0, 1] used for the second moment of the arithmetic average.
_LEGENDRE_NODES, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(32)  # rule on [-1, 1]
_GL_NODES = 0.5 * (_LEGENDRE_NODES + 1.0)
_GL_WEIGHTS = 0.5 * _LEGENDRE_WEIGHTS
_GL_NODES.flags.writeable = False
_GL_WEIGHTS.flags.writeable = False


# ---------------------------------------------------------------------- #
# Numerical building blocks
# ---------------------------------------------------------------------- #
def _g1(z) -> np.ndarray:
    """``(exp(z) - 1) / z - 1``, accurate for every ``z`` including ``z -> 0``.

    ``1 + _g1(b * T)`` is the ratio between the expected average of a spot
    growing at rate ``b`` over a window of length ``T`` and its starting value.
    """
    z = np.asarray(z, dtype=float)
    small = np.abs(z) < 1e-2
    z_safe = np.where(small, 1.0, z)
    series = z / 2 * (1 + z / 3 * (1 + z / 4 * (1 + z / 5 * (1 + z / 6 * (1 + z / 7)))))
    return np.where(small, series, (np.expm1(z_safe) - z_safe) / z_safe)


def _black(forward, strike, total_var, omega: float) -> np.ndarray:
    """Undiscounted Black value ``E[max(omega * (X - strike), 0)]``.

    ``X`` is lognormal with mean ``forward`` and log-variance ``total_var``.
    Degenerate inputs collapse to ``max(omega * (forward - strike), 0)``: no
    variance left, or a non-positive strike (a call that is certain to be
    exercised, a put that never is).
    """
    forward, strike, total_var = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (forward, strike, total_var))
    )
    regular = (total_var > 0) & (strike > 0) & (forward > 0)
    sd = np.sqrt(np.where(regular, total_var, 1.0))
    log_moneyness = np.log(np.where(regular, forward, 1.0) / np.where(regular, strike, 1.0))
    d1 = log_moneyness / sd + 0.5 * sd
    d2 = log_moneyness / sd - 0.5 * sd  # not d1 - sd: stays -inf (not nan) for an infinite variance
    value = omega * (forward * ndtr(omega * d1) - strike * ndtr(omega * d2))
    return np.where(regular, value, np.maximum(omega * (forward - strike), 0.0))


def _trapezoid_weights(times: np.ndarray, start: float, end: float) -> np.ndarray:
    """Weights ``w`` such that ``values @ w`` integrates a sampled path over ``[start, end]``.

    The path is the piecewise-linear interpolation of ``(times, values)``; if
    ``start`` falls inside a step the left end is interpolated, and if the
    samples stop before ``end`` the last value is held flat.
    """
    weights = np.zeros_like(times)
    if times.size > 1:
        dt = np.diff(times)
        left = np.maximum(times[:-1], start)
        width = np.maximum(times[1:] - left, 0.0)
        lam = (left - times[:-1]) / dt
        weights[:-1] += 0.5 * width * (1.0 - lam)
        weights[1:] += 0.5 * width * (1.0 + lam)
    weights[-1] += max(end - max(times[-1], start), 0.0)
    return weights


def _optional_positive(name: str, value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"AsianOption.{name} must be a number or None") from exc
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"AsianOption.{name} must be finite and strictly positive, got {value!r}")
    return number


# ---------------------------------------------------------------------- #
# The instrument
# ---------------------------------------------------------------------- #
@register_instrument
@dataclass(frozen=True)
class AsianOption(Instrument):
    """Average-price (fixed-strike) Asian option, continuous averaging.

    Parameters
    ----------
    option_type : {"call", "put"}
    strike : float
        Strike compared with the average (> 0).
    expiry : float
        ABSOLUTE expiry in years; also the end of the averaging window.
    averaging : {"arithmetic", "geometric"}, default "arithmetic"
        Arithmetic is what trades; geometric is the exactly solvable cousin
        (always a little cheaper for calls, since a geometric mean never
        exceeds the arithmetic mean).
    avg_start : float, default 0.0
        ABSOLUTE start of the averaging window (must be before ``expiry``).
        While ``mkt.t < avg_start`` the option is *forward-starting*.
    running_average : float, optional
        Average already accumulated over ``[avg_start, observed_until]``
        (arithmetic or geometric, matching ``averaging``). ``None`` when
        nothing has been fixed yet.
    observed_until : float, optional
        ABSOLUTE time of the last observation (``None`` if never observed).
        Required with ``running_average``; at most ``expiry``.
    last_spot : float, optional
        Spot seen at ``observed_until``: the left end of the next trapezoid
        in :meth:`observe`. It never enters the price.

    Notes
    -----
    * **Observe first, then price.** Pricing right after :meth:`observe` is
      the normal workflow. If the market clock is nevertheless past the last
      observation, ``price`` assumes the spot has been sitting at its current
      level since then ("the spot jumped, then time passed"). This is what
      makes bumped Greeks right on a whole spot ladder: theta is the P&L of
      time passing *with the average filling in at the ladder spot*, i.e. the
      quantity that balances gamma and carry in the pricing equation.
    * Seasoned arithmetic Asians use the classic strike adjustment: with a
      fraction of the window fixed at ``A_run`` and a weight ``w`` still
      random, ``max(A - K, 0) = w * max(A_future - K*, 0)`` where
      ``K* = (K - (1 - w) * A_run) / w``. When ``K* <= 0`` the call is certain
      to be exercised and is worth the discounted ``E[A] - K`` (the put is
      worth 0).
    * Greeks come from the numerical engine; they are smooth because the
      price is a closed-form expression.

    Examples
    --------
    >>> from optionlab.market import Market
    >>> put = AsianOption("put", 85.0, 0.25, averaging="geometric")
    >>> round(put.price(Market(spot=80.0, vol=0.2, rate=0.05, div=-0.03)), 4)
    4.6922
    """

    option_type: str
    strike: float
    expiry: float
    averaging: str = "arithmetic"
    avg_start: float = 0.0
    running_average: float | None = None
    observed_until: float | None = None
    last_spot: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_type", bs.normalize_option_type(self.option_type))
        averaging = self.averaging.strip().lower() if isinstance(self.averaging, str) else self.averaging
        if averaging not in AVERAGING_TYPES:
            raise ValueError(f"averaging must be one of {AVERAGING_TYPES}, got {self.averaging!r}")
        object.__setattr__(self, "averaging", averaging)
        try:
            strike, expiry, avg_start = float(self.strike), float(self.expiry), float(self.avg_start)
            observed = None if self.observed_until is None else float(self.observed_until)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "AsianOption strike, expiry, avg_start and observed_until must be numbers"
            ) from exc
        if not (math.isfinite(strike) and strike > 0):
            raise ValueError(f"strike must be a finite positive number, got {self.strike!r}")
        if not (math.isfinite(expiry) and math.isfinite(avg_start)):
            raise ValueError("expiry and avg_start must be finite")
        if not avg_start < expiry:
            raise ValueError(f"avg_start ({avg_start}) must be strictly before expiry ({expiry})")
        running = _optional_positive("running_average", self.running_average)
        last = _optional_positive("last_spot", self.last_spot)
        if observed is not None:
            if not math.isfinite(observed):
                raise ValueError("observed_until must be finite")
            if observed > expiry + EXPIRY_TOL:
                raise ValueError(f"observed_until ({observed}) cannot be after expiry ({expiry})")
            observed = min(observed, expiry)
        if running is not None and (observed is None or observed < avg_start):
            raise ValueError("running_average needs observed_until inside [avg_start, expiry]")
        if running is None and observed is not None and observed > avg_start + EXPIRY_TOL:
            raise ValueError("observed_until is inside the averaging window: running_average is required")
        if last is not None and observed is None:
            raise ValueError("last_spot is the spot seen at observed_until, which is missing")
        for name, value in (
            ("strike", strike), ("expiry", expiry), ("avg_start", avg_start),
            ("running_average", running), ("observed_until", observed), ("last_spot", last),
        ):
            object.__setattr__(self, name, value)

    # --- small helpers ------------------------------------------------------
    @property
    def is_call(self) -> bool:
        return self.option_type == "call"

    @property
    def label(self) -> str:
        kind = "arith" if self.averaging == "arithmetic" else "geo"
        return f"Asian {kind} {'C' if self.is_call else 'P'} {self.strike:g} T={self.expiry:.2f}"

    def vanilla_equivalent(self) -> EuropeanOption:
        """The vanilla option with the same type, strike and expiry.

        It is the natural benchmark: the Asian is cheaper and has smaller
        Greeks, because an average moves less than its last point.
        """
        return EuropeanOption(self.option_type, self.strike, self.expiry)

    @property
    def _omega(self) -> float:
        return 1.0 if self.is_call else -1.0

    @property
    def _window_length(self) -> float:
        return self.expiry - self.avg_start

    @property
    def _fixed_until(self) -> float:
        """End of the stretch of the window covered by ``running_average``."""
        if self.observed_until is None:
            return self.avg_start
        return max(self.observed_until, self.avg_start)

    def _scale(self, value):
        """Quantity that is averaged: the spot, or its logarithm (geometric)."""
        if self.averaging == "arithmetic":
            return np.asarray(value, dtype=float)
        value = np.asarray(value, dtype=float)
        return np.log(np.where(value > 0, value, 1.0))

    def _known_average(self, spot, until, trapezoid: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Contribution of ``[avg_start, until]`` to the final average.

        Returns ``(known, gap)``: ``known`` is the integral of the (log-)spot
        over the stretch divided by the FULL window length, i.e. already
        weighted; ``gap`` is the length bridged between the last observation
        and ``until`` using ``spot``: held flat at ``spot`` (pricing), or a
        ``trapezoid`` from ``last_spot`` (recording a new observation).
        """
        start = self._fixed_until
        until = np.asarray(until, dtype=float)
        gap = np.maximum(until - start, 0.0)
        x = self._scale(spot)
        if trapezoid and self.last_spot is not None:
            # If the last observation predates the window, the left end is first interpolated
            # at avg_start (exactly what path_payoff does for a grid step straddling it).
            x_last = self._scale(self.last_spot)
            since_last = until - self.observed_until
            skipped = (start - self.observed_until) / np.where(since_last > 0, since_last, 1.0)
            bridge = 0.5 * (x_last + (x - x_last) * skipped + x)
        else:
            bridge = x
        state = 0.0 if self.running_average is None else (
            float(self._scale(self.running_average)) * (start - self.avg_start)
        )
        return (state + bridge * gap) / self._window_length, gap

    def _window(self, t) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(a, ell, tau1)``: start and length of the still-random stretch, and the wait before it."""
        t = np.asarray(t, dtype=float)
        a = np.minimum(np.maximum(t, self._fixed_until), self.expiry)
        return a, self.expiry - a, np.maximum(a - t, 0.0)

    def _random_average(self, spot, vol, carry, tau1, ell) -> tuple[np.ndarray, np.ndarray]:
        """Forward and log-variance of the average over the still-random stretch.

        The stretch starts ``tau1`` from now and lasts ``ell``. With
        ``x = b * ell`` and ``y = (b + vol**2) * ell`` the arithmetic moments are::

            E[A]   = S e^{b tau1} (e^x - 1) / x
            E[A^2] = S^2 e^{(2b + vol^2) tau1} * 2 * int_0^1 s e^{x s} (e^{y s} - 1) / (y s) ds

        The integral has the textbook closed form (Turnbull-Wakeman), but that
        form is 0/0 at ``b = 0``, ``b = -vol**2`` and ``2b = -vol**2`` and loses
        digits nearby, so it is evaluated with a 32-point Gauss-Legendre rule
        (exact to machine precision for these smooth integrands), written in
        terms of ``moment - 1`` to keep full accuracy for short windows.
        """
        if self.averaging == "geometric":
            var = vol**2 * (tau1 + ell / 3.0)
            forward = spot * np.exp((carry - 0.5 * vol**2) * (tau1 + 0.5 * ell) + 0.5 * var)
            return forward, var
        x = np.asarray(carry * ell, dtype=float)
        y = np.asarray((carry + vol**2) * ell, dtype=float)
        g1_x = _g1(x)
        xs = x[..., None] * _GL_NODES
        with np.errstate(over="ignore"):  # absurd vol**2 * T: the variance is +inf, which _black handles
            # e^{xs} g(ys) - 1, with g = 1 + g1, regrouped so that nothing cancels for short windows
            integrand = np.exp(xs) * _g1(y[..., None] * _GL_NODES) + np.expm1(xs)
            second_minus_1 = 2.0 * np.sum(_GL_WEIGHTS * _GL_NODES * integrand, axis=-1)
        forward = spot * np.exp(carry * tau1) * (1.0 + g1_x)
        var = vol**2 * tau1 + np.log1p(second_minus_1) - 2.0 * np.log1p(g1_x)
        return forward, np.where(vol > 0, np.maximum(var, 0.0), 0.0)

    # --- valuation ------------------------------------------------------------
    def price(self, mkt: Market):
        """Model value, vectorised over the market fields.

        Geometric averaging is exact; arithmetic averaging is the
        Turnbull-Wakeman / Levy lognormal approximation. Handles
        forward-starting windows (``mkt.t < avg_start``), seasoned averages
        and, at or after expiry, the settlement value.
        """
        spot = np.asarray(mkt.spot, dtype=float)
        vol = np.asarray(mkt.vol, dtype=float)
        rate = np.asarray(mkt.rate, dtype=float)
        carry = rate - np.asarray(mkt.div, dtype=float)
        a, ell, tau1 = self._window(mkt.t)
        weight = ell / self._window_length
        known, gap = self._known_average(spot, a)
        forward, var = self._random_average(spot, vol, carry, tau1, ell)

        if self.averaging == "arithmetic":
            # max(A - K, 0) = w * max(A_future - K*, 0); K* <= 0 means "certain exercise".
            k_star = (self.strike - known) * self._window_length / np.where(ell > 0, ell, 1.0)
            value = np.where(
                ell > 0,
                weight * _black(forward, k_star, var, self._omega),
                np.maximum(self._omega * (known - self.strike), 0.0),
            )
        else:
            # The final average exp(known) * G_future**w is exactly lognormal.
            total_var = weight**2 * var
            log_forward = np.log(np.where(forward > 0, forward, 1.0))
            total_forward = np.exp(known + weight * (log_forward - 0.5 * var) + 0.5 * total_var)
            total_forward = np.where((spot <= 0) & ((ell > 0) | (gap > 0)), 0.0, total_forward)
            value = _black(total_forward, self.strike, total_var, self._omega)

        tau = np.maximum(self.expiry - np.asarray(mkt.t, dtype=float), 0.0)
        out = np.exp(-rate * tau) * value + 0.0
        return float(out) if mkt.is_scalar else out

    def payoff(self, spot_T):
        """Settlement value as a function of the terminal spot.

        The part of the window that has not been observed is assumed to have
        been spent at ``spot_T``, exactly as :meth:`price` does at expiry, so
        ``payoff(s) == price(Market(spot=s, ..., t=expiry))``. A fresh Asian
        therefore shows the vanilla hockey stick ("the spot sat at ``spot_T``
        for the whole window"), which flattens as the average fills in and is
        a constant once the window is fully observed.
        """
        spot_T = np.asarray(spot_T, dtype=float)
        known, gap = self._known_average(spot_T, self.expiry)
        if self.averaging == "geometric":
            known = np.where((spot_T <= 0) & (gap > 0), 0.0, np.exp(known))
        return np.maximum(self._omega * (known - self.strike), 0.0)

    def remaining_weight(self, mkt: Market):
        """Fraction of the averaging window that is still random (1 = nothing fixed yet).

        This is the number that scales the Greeks down as the average fills in.
        """
        _, ell, _ = self._window(mkt.t)
        out = ell / self._window_length + np.zeros(mkt.shape)
        return float(out) if mkt.is_scalar else out

    def effective_vol(self, mkt: Market):
        """Black volatility of the still-random part of the average, annualised to expiry.

        About ``vol / sqrt(3)`` for a fresh Asian: this is the "why is it
        cheaper than a vanilla" number. Returns 0 at or after expiry.
        """
        vol = np.asarray(mkt.vol, dtype=float)
        carry = np.asarray(mkt.rate, dtype=float) - np.asarray(mkt.div, dtype=float)
        _, ell, tau1 = self._window(mkt.t)
        _, var = self._random_average(np.asarray(mkt.spot, dtype=float), vol, carry, tau1, ell)
        tau = self.expiry - np.asarray(mkt.t, dtype=float)
        out = np.sqrt(np.where(tau > 0, var / np.where(tau > 0, tau, 1.0), 0.0)) + np.zeros(mkt.shape)
        return float(out) if mkt.is_scalar else out

    # --- path dependence --------------------------------------------------------
    def observe(self, spot: float, t: float) -> "AsianOption":
        """Record the spot seen at time ``t`` and return the updated option.

        The stretch since the previous observation is added to the running
        average with the trapezoid rule (on log-spots for geometric
        averaging). Observations before ``avg_start`` only remember the spot
        (the step that straddles ``avg_start`` is interpolated linearly, as
        :meth:`path_payoff` does); observations that are not later than the
        previous one, or made once the window is complete, return ``self``.
        """
        spot = _optional_positive("observe(spot)", spot)
        if spot is None:
            raise ValueError("observe needs a spot")
        t = min(float(t), self.expiry)
        if self.observed_until is not None and t <= self.observed_until + EXPIRY_TOL:
            return self
        if t <= self.avg_start:
            return replace(self, observed_until=t, last_spot=spot)
        known, _ = self._known_average(spot, t, trapezoid=True)
        average = float(known) * self._window_length / (t - self.avg_start)
        if self.averaging == "geometric":
            average = math.exp(average)
        return replace(self, running_average=average, observed_until=t, last_spot=spot)

    def path_payoff(self, paths: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Payoff per simulated path: stored average combined with the path average.

        The path average is the trapezoid rule on the grid ``times`` (on
        log-spots for geometric averaging), restricted to the averaging
        window. It converges to the continuous average as the grid is refined
        (error of order ``dt**2``). A path starting after the last observation
        is treated as in :meth:`price`: the stretch in between is held at the
        first path value.
        """
        paths, times = slice_paths_to_expiry(paths, times, self.expiry)
        start = min(max(float(times[0]), self._fixed_until), self.expiry)
        known, _ = self._known_average(paths[:, 0], start)
        weights = _trapezoid_weights(times, start, self.expiry) / self._window_length
        average = known + self._scale(paths) @ weights
        if self.averaging == "geometric":
            average = np.exp(average)
        return np.maximum(self._omega * (average - self.strike), 0.0)

    def _grid_price(self, mkt: Market, times: np.ndarray) -> float:
        """Exact price of the GEOMETRIC Asian whose average is the trapezoid rule on ``times``.

        A weighted sum of log-spots is Gaussian, so the discretely averaged
        geometric Asian is still a Black formula; this is what makes it a
        bias-free control variate on any grid.
        """
        start = min(max(float(times[0]), self._fixed_until), self.expiry)
        known, _ = self._known_average(mkt.spot, start)
        weights = _trapezoid_weights(times, start, self.expiry) / self._window_length
        elapsed = times - times[0]
        drift = mkt.rate - mkt.div - 0.5 * mkt.vol**2
        mean_log = float(known) + float(weights @ (math.log(mkt.spot) + drift * elapsed))
        var = mkt.vol**2 * float(weights @ np.minimum.outer(elapsed, elapsed) @ weights)
        value = _black(math.exp(mean_log + 0.5 * var), self.strike, var, self._omega)
        return math.exp(-mkt.rate * (self.expiry - mkt.t)) * float(value)


# ---------------------------------------------------------------------- #
# Monte Carlo benchmark
# ---------------------------------------------------------------------- #
def mc_price_with_control_variate(
    asian: AsianOption,
    mkt: Market,
    n_paths: int = 20_000,
    n_steps: int = 252,
    seed: Seed = None,
) -> tuple[float, float]:
    """Monte Carlo price of an Asian, using its geometric twin as control variate.

    The arithmetic and geometric averages of the same path are almost
    perfectly correlated, and the geometric one has an exact price. So instead
    of estimating the arithmetic price directly we estimate the small
    *difference* between the two and add the known geometric price back::

        price = mean(X) - beta * (mean(Y) - E[Y])        beta = cov(X, Y) / var(Y)

    which typically divides the standard error by 20 to 50. ``E[Y]`` is the
    exact price of the geometric Asian *averaged on the simulation grid*, so
    the estimator has no control-variate bias; the only discretisation error
    left is the trapezoid rule (order ``dt**2``).

    Parameters
    ----------
    asian : AsianOption
        Fresh, forward-starting or seasoned (the stored state is honoured).
    mkt : Market
        Scalar market with ``spot > 0``.
    n_paths : int, default 20000
        Rounded up to an even number (antithetic pairs are always used).
    n_steps : int, default 252
        Uniform steps from ``mkt.t`` to expiry (``avg_start`` is added to the
        grid when the window has not started yet).
    seed : int, numpy Generator or None

    Returns
    -------
    price, stderr : float
        Estimate and one standard error. For a geometric Asian the control is
        the product itself, so the result is the grid price with ~0 error.
    """
    if not isinstance(asian, AsianOption):
        raise ValueError("mc_price_with_control_variate needs an AsianOption")
    if not mkt.is_scalar:
        raise ValueError("mc_price_with_control_variate needs a scalar Market (no array fields)")
    n_paths, n_steps = int(n_paths), int(n_steps)
    if n_paths < 4:
        raise ValueError("n_paths must be at least 4")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    if asian.is_expired(mkt):
        return float(asian.price(mkt)), 0.0

    times = np.linspace(mkt.t, asian.expiry, n_steps + 1)
    times[-1] = asian.expiry
    if mkt.t < asian.avg_start and np.min(np.abs(times - asian.avg_start)) > 1e-12:
        times = np.insert(times, int(np.searchsorted(times, asian.avg_start)), asian.avg_start)
    n_paths += n_paths % 2
    paths = simulate_gbm_paths_on_grid(
        mkt.spot, mkt.vol, mkt.rate, mkt.div, times, n_paths, seed, antithetic=True
    )

    twin = replace(asian, averaging="geometric")
    discount = math.exp(-mkt.rate * (asian.expiry - mkt.t))
    half = n_paths // 2
    target = discount * asian.path_payoff(paths, times)
    control = discount * twin.path_payoff(paths, times)
    target = 0.5 * (target[:half] + target[half:])
    control = 0.5 * (control[:half] + control[half:])

    control_var = float(control.var(ddof=1))
    beta = float(np.cov(target, control)[0, 1]) / control_var if control_var > 0 else 0.0
    samples = target - beta * (control - twin._grid_price(mkt, times))
    return float(samples.mean()), float(samples.std(ddof=1) / math.sqrt(half))
