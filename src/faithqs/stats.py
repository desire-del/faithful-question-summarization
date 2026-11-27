"""Statistics used by the protocol (docs/PROTOCOL.md sections 4-5). Pure Python, no SciPy.

- paired bootstrap over examples (10,000 resamples, seed 0) for means and differences;
- Kendall tau-b with ties, and its bootstrap CI;
- exact McNemar test for paired binary outcomes;
- mean +/- std across training seeds.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

N_RESAMPLES = 10_000
SEED = 0


@dataclass(frozen=True)
class Estimate:
    mean: float
    low: float
    high: float

    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0

    def as_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "ci95_low": self.low, "ci95_high": self.high}


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (same convention as numpy's default)."""
    if not sorted_values:
        raise ValueError("empty sample")
    pos = (len(sorted_values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def _resample_indices(n: int, n_resamples: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(n_resamples)]


def bootstrap_mean(
    values: Sequence[float], n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> Estimate:
    n = len(values)
    means = sorted(
        sum(values[i] for i in idx) / n for idx in _resample_indices(n, n_resamples, seed)
    )
    return Estimate(sum(values) / n, _percentile(means, 0.025), _percentile(means, 0.975))


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> Estimate:
    """CI of mean(a - b), resampling examples jointly (paired design)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    return bootstrap_mean([x - y for x, y in zip(a, b, strict=True)], n_resamples, seed)


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall tau-b (accounts for ties in both variables). O(n^2), fine for n <= a few 1000."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need two paired samples of length >= 2")
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denom if denom else 0.0


def bootstrap_kendall(
    x: Sequence[float],
    y: Sequence[float],
    n_resamples: int = 2_000,
    seed: int = SEED,
) -> Estimate:
    taus = sorted(
        kendall_tau_b([x[i] for i in idx], [y[i] for i in idx])
        for idx in _resample_indices(len(x), n_resamples, seed)
    )
    return Estimate(kendall_tau_b(x, y), _percentile(taus, 0.025), _percentile(taus, 0.975))


def paired_kendall_diff(
    x1: Sequence[float],
    x2: Sequence[float],
    y: Sequence[float],
    n_resamples: int = 2_000,
    seed: int = SEED,
) -> Estimate:
    """CI of tau(x1, y) - tau(x2, y) on the same resampled examples (H1)."""
    diffs = sorted(
        kendall_tau_b([x1[i] for i in idx], [y[i] for i in idx])
        - kendall_tau_b([x2[i] for i in idx], [y[i] for i in idx])
        for idx in _resample_indices(len(y), n_resamples, seed)
    )
    point = kendall_tau_b(x1, y) - kendall_tau_b(x2, y)
    return Estimate(point, _percentile(diffs, 0.025), _percentile(diffs, 0.975))


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def across_seeds(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n_seeds": len(values),
    }
