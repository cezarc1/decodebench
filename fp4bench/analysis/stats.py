"""Paired log-ratio CIs, t-CIs of means, the §8 decision table and the accuracy note's bootstrap."""

import math
import statistics
from collections.abc import Hashable, Mapping
from enum import StrEnum

import numpy as np
from scipy.stats import t as student_t

from fp4bench.core.types import Estimate, Verdict, is_finite

_K_LABELS = {Verdict.MXFP4_FASTER: Verdict.PINNED_FASTER, Verdict.NVFP4_FASTER: Verdict.ALT_FASTER}
_PHI_LABELS = {
    Verdict.MXFP4_FASTER: Verdict.FUSION_SLOWER,
    Verdict.NVFP4_FASTER: Verdict.FUSION_FASTER,
}


def paired_log_ratios(numerators: list[float], denominators: list[float]) -> list[float]:
    if len(numerators) != len(denominators):
        raise ValueError("unpaired samples")
    if len(numerators) < 2:
        raise ValueError("need at least 2 pairs")
    return [math.log(a) - math.log(b) for a, b in zip(numerators, denominators, strict=True)]


def ratio_ci(log_ratios: list[float], level: float) -> Estimate:
    """Geometric-mean ratio and its two-sided t CI (a 90% CI is TOST at alpha 0.05)."""
    n = len(log_ratios)
    m = statistics.mean(log_ratios)
    half = student_t.ppf(0.5 + level / 2, df=n - 1) * statistics.stdev(log_ratios) / math.sqrt(n)
    return Estimate(math.exp(m), math.exp(m - half), math.exp(m + half))


def mean_ci(values: list[float], level: float) -> tuple[float, float | None, float | None]:
    """Mean and two-sided t CI with n - 1 degrees of freedom; one value has no interval."""
    if not values:
        raise ValueError("no values")
    m = statistics.fmean(values)
    if len(values) < 2:
        return m, None, None
    half = float(
        student_t.ppf(0.5 + level / 2, df=len(values) - 1)
        * statistics.stdev(values)
        / math.sqrt(len(values))
    )
    return m, m - half, m + half


def classify(lo: float, hi: float, delta: float) -> Verdict:
    if lo >= 1 - delta and hi <= 1 + delta:
        return Verdict.EQUIVALENT
    if lo > 1 + delta:
        return Verdict.MXFP4_FASTER
    if hi < 1 - delta:
        return Verdict.NVFP4_FASTER
    if lo > 1 or hi < 1:
        return Verdict.DIFFERENT_SMALL
    return Verdict.INCONCLUSIVE


def relabel_for_k(verdict: Verdict) -> Verdict:
    return _K_LABELS.get(verdict, verdict)


def relabel_for_phi(verdict: Verdict) -> Verdict:
    return _PHI_LABELS.get(verdict, verdict)


def paired_bootstrap_ci(
    diffs: list[float], level: float = 0.95, resamples: int = 10_000, seed: int = 0
) -> Estimate:
    """Mean paired difference and its percentile bootstrap CI (legacy RandomState)."""
    if len(diffs) < 2:
        raise ValueError("need at least 2 pairs")
    if not all(map(is_finite, diffs)):
        raise ValueError("differences must all be finite numbers")
    values = np.asarray(diffs, dtype=float)
    picks = np.random.RandomState(seed).randint(0, len(values), size=(resamples, len(values)))
    means = np.sort(values[picks].mean(axis=1))
    k = round((1 - level) / 2 * resamples)
    return Estimate(float(values.mean()), float(means[k]), float(means[resamples - 1 - k]))


class OverallVerdict(StrEnum):
    VALIDATED = "validated"
    NULLIFIED_MXFP4_FASTER = "nullified_mxfp4_faster"
    NULLIFIED_NVFP4_FASTER = "nullified_nvfp4_faster"
    NULLIFIED_MIXED = "nullified_mixed"
    INCONCLUSIVE = "inconclusive"


def overall_verdict[K: Hashable](
    per_c: Mapping[K, Verdict], primary: tuple[K, ...]
) -> OverallVerdict:
    if not primary:
        raise ValueError("no primary endpoints")
    verdicts = [per_c[c] for c in primary]
    if all(v == Verdict.EQUIVALENT for v in verdicts):
        return OverallVerdict.VALIDATED
    faster = {v for v in verdicts if v in (Verdict.MXFP4_FASTER, Verdict.NVFP4_FASTER)}
    if faster == {Verdict.MXFP4_FASTER}:
        return OverallVerdict.NULLIFIED_MXFP4_FASTER
    if faster == {Verdict.NVFP4_FASTER}:
        return OverallVerdict.NULLIFIED_NVFP4_FASTER
    if faster:
        return OverallVerdict.NULLIFIED_MIXED
    return OverallVerdict.INCONCLUSIVE
