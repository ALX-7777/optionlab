"""Market snapshot used by every pricer in the library.

A :class:`Market` is an immutable description of "the world right now": the
spot price, a flat implied volatility, a continuously compounded interest rate,
a continuous dividend yield and the current time ``t`` (in years).

Instruments carry an *absolute* expiry, so the time to maturity is always
``tau = expiry - mkt.t``. Moving the clock forward is therefore just
``mkt.bumped(t=mkt.t + dt)`` -- nothing has to be changed on the instruments.

Every field may be a float or a numpy array. Arrays are what make the plots
fast: pricing a whole spot ladder is a single vectorised call,
``inst.price(Market(spot=np.linspace(50, 150, 201), vol=0.2))``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Union

import numpy as np

__all__ = ["ArrayLike", "Market"]

ArrayLike = Union[float, np.ndarray]


def _coerce(name: str, value: Any) -> ArrayLike:
    """Convert a field value to a float or a read-only float array."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Market.{name} must be numeric, got {value!r}") from exc
    if arr.ndim == 0:
        return float(arr)
    view = arr.view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True, eq=False)
class Market:
    """Immutable market snapshot (Black-Scholes world).

    Parameters
    ----------
    spot : float or ndarray
        Price of the underlying. Must be non-negative.
    vol : float or ndarray
        Flat annualised volatility (0.20 means 20%). Must be non-negative.
    rate : float or ndarray, default 0.0
        Continuously compounded risk-free rate.
    div : float or ndarray, default 0.0
        Continuous dividend yield of the underlying.
    t : float or ndarray, default 0.0
        Current time in years. Instruments store an absolute ``expiry`` so the
        time to maturity is ``expiry - t``.

    Notes
    -----
    Fields may be arrays as long as they broadcast against each other. This is
    how scenario grids are built: a column vector of spots against a row
    vector of vols prices a full spot/vol matrix in one call.

    Examples
    --------
    >>> mkt = Market(spot=100.0, vol=0.2, rate=0.05)
    >>> mkt.bumped(spot=101.0).spot
    101.0
    >>> mkt.tau(1.0)
    1.0
    """

    spot: ArrayLike
    vol: ArrayLike
    rate: ArrayLike = 0.0
    div: ArrayLike = 0.0
    t: ArrayLike = 0.0

    def __post_init__(self) -> None:
        for f in fields(self):
            object.__setattr__(self, f.name, _coerce(f.name, getattr(self, f.name)))
        if np.any(np.asarray(self.spot) < 0):
            raise ValueError("Market.spot must be non-negative")
        if np.any(np.asarray(self.vol) < 0):
            raise ValueError("Market.vol must be non-negative")
        try:
            np.broadcast_shapes(*(np.shape(getattr(self, f.name)) for f in fields(self)))
        except ValueError as exc:
            raise ValueError("Market fields have shapes that do not broadcast together") from exc

    # ------------------------------------------------------------------ #
    # Equality (array-aware). Markets holding arrays are not hashable.
    # ------------------------------------------------------------------ #
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Market):
            return NotImplemented
        return all(
            np.array_equal(getattr(self, f.name), getattr(other, f.name)) for f in fields(self)
        )

    def __hash__(self) -> int:
        if not self.is_scalar:
            raise TypeError("a Market holding arrays is not hashable")
        return hash(tuple(getattr(self, f.name) for f in fields(self)))

    # ------------------------------------------------------------------ #
    # Shape helpers
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> tuple[int, ...]:
        """Broadcast shape of all fields (``()`` for an all-scalar market)."""
        return np.broadcast_shapes(*(np.shape(getattr(self, f.name)) for f in fields(self)))

    @property
    def is_scalar(self) -> bool:
        """True when every field is a plain float."""
        return self.shape == ()

    # ------------------------------------------------------------------ #
    # Scenario helpers
    # ------------------------------------------------------------------ #
    def bumped(self, **changes: Any) -> "Market":
        """Return a copy with some fields replaced.

        Parameters
        ----------
        **changes
            Any of ``spot``, ``vol``, ``rate``, ``div``, ``t`` with the new
            (absolute) value, float or array.

        Returns
        -------
        Market
            A new market; the original is never modified.

        Examples
        --------
        >>> Market(100.0, 0.2).bumped(vol=0.25, t=0.5)
        Market(spot=100.0, vol=0.25, rate=0.0, div=0.0, t=0.5)
        """
        valid = {f.name for f in fields(self)}
        unknown = set(changes) - valid
        if unknown:
            raise ValueError(
                f"Unknown Market field(s) {sorted(unknown)}; valid fields are {sorted(valid)}"
            )
        return replace(self, **changes)

    def tau(self, expiry: ArrayLike) -> ArrayLike:
        """Time to maturity ``expiry - t`` in years (negative once expired)."""
        return expiry - self.t

    def discount_factor(self, expiry: ArrayLike) -> ArrayLike:
        """Price today of 1 unit of cash paid at ``expiry``: ``exp(-rate * tau)``.

        For an already-passed expiry the factor is 1 (cash is cash).
        """
        return np.exp(-self.rate * np.maximum(self.tau(expiry), 0.0))

    def forward(self, expiry: ArrayLike) -> ArrayLike:
        """Forward price of the underlying for delivery at ``expiry``.

        ``F = S * exp((rate - div) * tau)``: the spot grows at the rate but
        leaks the dividend yield, which the forward buyer does not receive.
        """
        return self.spot * np.exp((self.rate - self.div) * np.maximum(self.tau(expiry), 0.0))

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict (arrays become nested lists)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = value.tolist() if isinstance(value, np.ndarray) else value
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Market":
        """Inverse of :meth:`to_dict`."""
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            raise ValueError(f"Unknown Market field(s) {sorted(unknown)}")
        if "spot" not in data or "vol" not in data:
            raise ValueError("Market.from_dict needs at least 'spot' and 'vol'")
        return cls(**data)
