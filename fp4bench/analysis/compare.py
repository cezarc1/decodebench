"""Paired comparisons per cell: ratios, differences in ms, contrasts of differences, A/A rules."""

import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from fp4bench import settings
from fp4bench.analysis import stats
from fp4bench.analysis.cells import Values, paired_rounds, usable
from fp4bench.core.types import Cell, Treatment, ValueKey, Verdict

Relabel = Callable[[Verdict], Verdict]


@dataclass(frozen=True)
class RatioResult[K: (Cell, int)]:
    """numer / denom paired per round: with one round a point estimate (no CI, no verdict)."""

    n_rounds: int
    rounds: tuple[int, ...]
    ratio: float
    lo: float | None
    hi: float | None
    verdict: Verdict | None
    skipped: tuple[ValueKey[K], ...] = ()

    @property
    def ci(self) -> tuple[float, float]:
        if self.lo is None or self.hi is None:
            raise ValueError("a ratio of one round has no CI")
        return self.lo, self.hi


@dataclass(frozen=True)
class DeltaResult:
    """t_a - t_b in ms paired per round: the mean and its t-CI (none with one round)."""

    n_rounds: int
    rounds: tuple[int, ...]
    mean_ms: float
    lo_ms: float | None
    hi_ms: float | None
    per_round_ms: dict[int, float]


@dataclass(frozen=True)
class ContrastResult:
    """Δ(cells[0]) - Δ(cells[1]) in ms, paired per round over the rounds both cells have."""

    cells: tuple[Cell, Cell]
    n_rounds: int
    rounds: tuple[int, ...]
    missing_cells: tuple[Cell, ...]
    mean_ms: float | None
    lo_ms: float | None
    hi_ms: float | None
    classification: str | None
    per_round_ms: dict[int, float]


@dataclass(frozen=True)
class Decomposition:
    """R = F · Φ per cell: the product of F's and Φ's estimates next to R's."""

    r: float
    f: float
    phi: float
    f_times_phi: float
    f_times_phi_over_r: float
    n_rounds: dict[str, int]


def skipped_values[K: (Cell, int)](
    values: Values[K], treatments: tuple[Treatment, ...], key: K
) -> list[ValueKey[K]]:
    return sorted(
        k for k, v in values.items() if k.key == key and k.treatment in treatments and not usable(v)
    )


def ratio[K: (Cell, int)](
    values: Values[K], numer: Treatment, denom: Treatment, key: K, relabel: Relabel | None = None
) -> RatioResult[K] | None:
    rounds = paired_rounds(values, numer, denom, key)
    if not rounds:
        return None
    skipped = tuple(skipped_values(values, (numer, denom), key))
    nums = [values[ValueKey(r, numer, key)] for r in rounds]
    dens = [values[ValueKey(r, denom, key)] for r in rounds]
    if len(rounds) == 1:
        return RatioResult(1, tuple(rounds), nums[0] / dens[0], None, None, None, skipped)
    est, lo, hi = stats.ratio_ci(stats.paired_log_ratios(nums, dens), settings.CI_LEVEL)
    verdict = stats.classify(lo, hi, settings.DELTA)
    return RatioResult(
        len(rounds), tuple(rounds), est, lo, hi, relabel(verdict) if relabel else verdict, skipped
    )


def _key_label(key: Cell | int) -> str:
    return f"C={key.batch}, P={key.prompt_len}" if isinstance(key, Cell) else f"C={key}"


def ratio_table[K: (Cell, int)](
    values: Values[K],
    numer: Treatment,
    denom: Treatment,
    keys: Iterable[K],
    relabel: Relabel | None = None,
) -> dict[K, RatioResult[K]]:
    """`ratio` per key with a CI; every unusable value is skipped with a warning."""
    out = {}
    for key in keys:
        for skipped in skipped_values(values, (numer, denom), key):
            r, t, k = skipped
            print(
                f"warning: skipping round {r}, treatment {t}, {_key_label(k)}: unusable value "
                f"{values[skipped]!r} (needs a finite positive number)",
                file=sys.stderr,
            )
        result = ratio(values, numer, denom, key, relabel)
        if result is not None and result.n_rounds >= 2:
            out[key] = result
    return out


def round_diffs_ms[K: (Cell, int)](
    values: Values[K], a: Treatment, b: Treatment, key: K
) -> dict[int, float]:
    return {
        r: (values[ValueKey(r, a, key)] - values[ValueKey(r, b, key)]) * 1000
        for r in paired_rounds(values, a, b, key)
    }


def delta_ms[K: (Cell, int)](
    values: Values[K], a: Treatment, b: Treatment, key: K
) -> DeltaResult | None:
    per_round = round_diffs_ms(values, a, b, key)
    if not per_round:
        return None
    mean, lo, hi = stats.mean_ci(list(per_round.values()), settings.CI_LEVEL)
    return DeltaResult(len(per_round), tuple(sorted(per_round)), mean, lo, hi, per_round)


def contrast(
    values: Values[Cell],
    cells: tuple[Cell, Cell],
    a: Treatment,
    b: Treatment,
    classify: Callable[[float, float], str] | None = None,
) -> ContrastResult:
    plus, minus = (round_diffs_ms(values, a, b, cell) for cell in cells)
    rounds = sorted(set(plus) & set(minus))
    missing = tuple(c for c, d in zip(cells, (plus, minus), strict=True) if not d)
    if not rounds:
        return ContrastResult(cells, 0, (), missing, None, None, None, None, {})
    per_round = {r: plus[r] - minus[r] for r in rounds}
    mean, lo, hi = stats.mean_ci(list(per_round.values()), settings.CI_LEVEL)
    label = None if classify is None or lo is None or hi is None else classify(lo, hi)
    return ContrastResult(
        cells, len(rounds), tuple(rounds), missing, mean, lo, hi, label, per_round
    )


def aa_ratio_passes(aa: RatioResult | None) -> bool | None:
    """The A/A ratio rule: §8's verdict Equivalent and a CI that contains 1; None without a CI."""
    if aa is None or aa.lo is None or aa.hi is None:
        return None
    return aa.verdict == Verdict.EQUIVALENT and aa.lo <= 1 <= aa.hi


def aa_effect_passes(lo: float | None, hi: float | None, margin: float) -> bool:
    """The A/A rule for a contrast: a CI that contains 0 and lies inside ±margin."""
    return lo is not None and hi is not None and lo <= 0 <= hi and lo >= -margin and hi <= margin


def r_decomposition[K: (Cell, int)](
    r: Mapping[K, RatioResult[K]], f: Mapping[K, RatioResult[K]], phi: Mapping[K, RatioResult[K]]
) -> dict[K, Decomposition]:
    """R = F · Φ per key where all three have an estimate; reported, not gated."""
    out = {}
    for key in sorted(set(r) & set(f) & set(phi)):
        product = f[key].ratio * phi[key].ratio
        out[key] = Decomposition(
            r[key].ratio,
            f[key].ratio,
            phi[key].ratio,
            product,
            product / r[key].ratio,
            {"R": r[key].n_rounds, "F": f[key].n_rounds, "Phi": phi[key].n_rounds},
        )
    return out


def ratio_json(result: RatioResult, by_cell: bool) -> dict[str, object]:
    head = {"n_rounds": result.n_rounds}
    if by_cell:
        return {
            **head,
            "rounds": list(result.rounds),
            "ratio": result.ratio,
            "lo": result.lo,
            "hi": result.hi,
            "verdict": result.verdict,
        }
    return {
        **head,
        "ratio": result.ratio,
        "lo": result.lo,
        "hi": result.hi,
        "verdict": result.verdict,
        "skipped": [(r, t, k.batch if isinstance(k, Cell) else k) for r, t, k in result.skipped],
    }
