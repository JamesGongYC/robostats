"""Two-sided confidence intervals for a binomial proportion.

Each function here estimates the same thing: the success probability ``p`` of the
Bernoulli process that generated ``successes`` out of ``n`` independent episodes.
They differ only in how the interval around that estimand is constructed, and so
in their coverage behaviour at small ``n`` and at proportions near 0 or 1.

The reported ``point`` is the sample proportion ``successes / n`` for every
method, including :func:`agresti_coull`, whose interval is deliberately not
centred on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "ConfidenceInterval",
    "agresti_coull",
    "clopper_pearson",
    "wilson",
]


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A two-sided confidence interval for a binomial proportion.

    Parameters
    ----------
    point : float
        The sample proportion ``successes / n``.
    lower : float
        Lower bound, in ``[0, 1]``.
    upper : float
        Upper bound, in ``[0, 1]``.
    confidence : float
        The nominal confidence level the interval was constructed at.
    method : str
        Name of the construction used.
    """

    point: float
    lower: float
    upper: float
    confidence: float
    method: str

    @property
    def width(self) -> float:
        """Width of the interval, ``upper - lower``."""
        return self.upper - self.lower


def _validate(successes: int, n: int, confidence: float) -> None:
    """Raise ``ValueError`` unless the inputs describe a well-posed binomial count."""
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if successes < 0:
        raise ValueError(f"successes must be non-negative, got {successes!r}")
    if successes > n:
        raise ValueError(f"successes must not exceed n, got successes={successes!r}, n={n!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie strictly inside (0, 1), got {confidence!r}")


def _clip(value: float) -> float:
    """Clip a bound into ``[0, 1]``."""
    return float(min(1.0, max(0.0, value)))


def wilson(successes: int, n: int, confidence: float = 0.95) -> ConfidenceInterval:
    """Wilson score interval for a binomial proportion, without continuity correction.

    The estimand is the success probability ``p``. The interval is the set of
    ``p`` not rejected by the score test, obtained by inverting
    ``(p_hat - p) / sqrt(p (1 - p) / n)``. Because the variance is evaluated
    under the null rather than at ``p_hat``, the interval stays inside ``[0, 1]``
    and remains non-degenerate at ``successes == 0`` and ``successes == n``.

    At those two endpoints the closed form reaches the boundary as a difference
    of two nearly equal quantities, which leaves a sub-1e-16 residual of the
    wrong sign, so the endpoints are taken analytically instead: the lower bound
    is exactly 0 at ``successes == 0`` and the upper bound is exactly 1 at
    ``successes == n``.

    Parameters
    ----------
    successes : int
        Number of successful episodes, ``0 <= successes <= n``.
    n : int
        Number of episodes, ``n > 0``.
    confidence : float, default 0.95
        Nominal two-sided confidence level, strictly inside ``(0, 1)``.

    Returns
    -------
    ConfidenceInterval
        Bounds clipped to ``[0, 1]``, with ``method="wilson"``.

    Raises
    ------
    ValueError
        If ``n <= 0``, ``successes < 0``, ``successes > n``, or ``confidence``
        lies outside ``(0, 1)``.

    References
    ----------
    Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. *JASA*, 22(158), 209-212.
    """
    _validate(successes, n, confidence)
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    z2 = z * z
    denominator = n + z2
    centre = (successes + z2 / 2.0) / denominator
    half_width = (z / denominator) * np.sqrt(successes * (n - successes) / n + z2 / 4.0)
    lower = 0.0 if successes == 0 else centre - half_width
    upper = 1.0 if successes == n else centre + half_width
    return ConfidenceInterval(
        point=successes / n,
        lower=_clip(lower),
        upper=_clip(upper),
        confidence=confidence,
        method="wilson",
    )


def clopper_pearson(successes: int, n: int, confidence: float = 0.95) -> ConfidenceInterval:
    """Clopper-Pearson exact interval for a binomial proportion.

    The estimand is the success probability ``p``. The interval is obtained by
    inverting the exact binomial test, which via the beta-binomial relation gives
    ``lower = Beta(alpha/2; x, n - x + 1)`` and
    ``upper = Beta(1 - alpha/2; x + 1, n - x)``. Coverage is guaranteed to be at
    least the nominal level, at the cost of being conservative.

    The two beta quantiles are undefined at the endpoints and are handled
    explicitly: at ``successes == 0`` the interval is ``[0, upper]``, and at
    ``successes == n`` it is ``[lower, 1]``.

    Parameters
    ----------
    successes : int
        Number of successful episodes, ``0 <= successes <= n``.
    n : int
        Number of episodes, ``n > 0``.
    confidence : float, default 0.95
        Nominal two-sided confidence level, strictly inside ``(0, 1)``.

    Returns
    -------
    ConfidenceInterval
        Bounds clipped to ``[0, 1]``, with ``method="clopper_pearson"``.

    Raises
    ------
    ValueError
        If ``n <= 0``, ``successes < 0``, ``successes > n``, or ``confidence``
        lies outside ``(0, 1)``.

    References
    ----------
    Clopper, C. J. and Pearson, E. S. (1934). The use of confidence or fiducial
    limits illustrated in the case of the binomial. *Biometrika*, 26(4), 404-413.
    """
    _validate(successes, n, confidence)
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else stats.beta.ppf(alpha / 2.0, successes, n - successes + 1)
    upper = 1.0 if successes == n else stats.beta.ppf(1.0 - alpha / 2.0, successes + 1, n - successes)
    return ConfidenceInterval(
        point=successes / n,
        lower=_clip(lower),
        upper=_clip(upper),
        confidence=confidence,
        method="clopper_pearson",
    )


def agresti_coull(successes: int, n: int, confidence: float = 0.95) -> ConfidenceInterval:
    """Agresti-Coull adjusted Wald interval for a binomial proportion.

    The estimand is the success probability ``p``. A Wald interval is formed
    after adding ``z**2 / 2`` successes and ``z**2 / 2`` failures, which at the
    95% level is the familiar "add two successes and two failures" adjustment.
    The centre is the adjusted proportion, not the sample proportion, so the
    returned ``point`` lies inside but generally off-centre of the interval.

    Parameters
    ----------
    successes : int
        Number of successful episodes, ``0 <= successes <= n``.
    n : int
        Number of episodes, ``n > 0``.
    confidence : float, default 0.95
        Nominal two-sided confidence level, strictly inside ``(0, 1)``.

    Returns
    -------
    ConfidenceInterval
        Bounds clipped to ``[0, 1]``, with ``method="agresti_coull"``. The
        unclipped construction can fall outside ``[0, 1]`` near the endpoints.

    Raises
    ------
    ValueError
        If ``n <= 0``, ``successes < 0``, ``successes > n``, or ``confidence``
        lies outside ``(0, 1)``.

    References
    ----------
    Agresti, A. and Coull, B. A. (1998). Approximate is better than "exact" for
    interval estimation of binomial proportions. *The American Statistician*,
    52(2), 119-126.
    """
    _validate(successes, n, confidence)
    z = stats.norm.ppf(0.5 + confidence / 2.0)
    z2 = z * z
    n_adjusted = n + z2
    p_adjusted = (successes + z2 / 2.0) / n_adjusted
    half_width = z * np.sqrt(p_adjusted * (1.0 - p_adjusted) / n_adjusted)
    return ConfidenceInterval(
        point=successes / n,
        lower=_clip(p_adjusted - half_width),
        upper=_clip(p_adjusted + half_width),
        confidence=confidence,
        method="agresti_coull",
    )
