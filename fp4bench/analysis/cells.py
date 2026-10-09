"""Sessions and per (round, treatment, cell) values from typed rows."""

import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeGuard

from fp4bench.core.schema import M1Row, M2Row, ServerRow
from fp4bench.core.types import (
    Cell,
    SessionSlot,
    Treatment,
    ValueKey,
    WaveOrder,
    is_finite,
    is_number,
)

type Values[K: (Cell, int)] = Mapping[ValueKey[K], float]
Steps = dict[ValueKey[Cell], float]
M2Metric = Literal["aggregate_tps", "itl_p50_ms", "tps_per_user_p50"]
NAN = float("nan")


def is_pos_int(v: object) -> TypeGuard[int]:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def usable(v: object) -> bool:
    """A value a log-ratio can be taken of (NaN is stored as null)."""
    return is_finite(v) and v > 0


def is_round(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _latest_rows(servers: Iterable[ServerRow]) -> dict[SessionSlot, ServerRow]:
    out = {}
    for row in servers:
        out[SessionSlot(row.round, row.treatment)] = row
    return out


def latest_sessions(servers: Iterable[ServerRow]) -> dict[SessionSlot, ServerRow]:
    """The latest session per (round, treatment); a failed re-run replaces an earlier good one."""
    return {k: row for k, row in _latest_rows(servers).items() if row.failed is not True}


def failed_sessions(servers: Iterable[ServerRow]) -> list[ServerRow]:
    rows = _latest_rows(servers)
    return [rows[k] for k in sorted(rows) if rows[k].failed is True]


def session_ids(servers: Iterable[ServerRow]) -> set[str]:
    return {s.session_id for s in latest_sessions(servers).values()}


def stored_prompt_len(row: M1Row) -> int | None:
    return None if "prompt_len" in row.missing else row.prompt_len


def row_cell(row: M1Row, by_cell: bool = False) -> Cell | None:
    """(C, P), or None unless both are positive ints; without prompt_len, None in a cell study."""
    prompt_len = stored_prompt_len(row) if by_cell else row.prompt_len
    return row.cell if is_pos_int(row.c) and is_pos_int(prompt_len) else None


def m1_steps(rows: Iterable[M1Row], sids: set[str], by_cell: bool = False) -> Steps:
    """Median measured step per (round, treatment, cell); NaN if any rep is not finite."""
    steps = defaultdict(list)
    for row in rows:
        cell = row_cell(row, by_cell)
        if (
            row.session_id in sids
            and row.warmup is False
            and cell is not None
            and is_round(row.round)
        ):
            steps[ValueKey(row.round, row.treatment, cell)].append(row.step_s)
    return {
        k: statistics.median(v) if all(is_finite(x) for x in v) else NAN for k, v in steps.items()
    }


def by_batch(steps: Steps) -> dict[ValueKey[int], float]:
    return {ValueKey(r, t, cell.batch): v for (r, t, cell), v in steps.items()}


def m2_values(
    rows: Iterable[M2Row], sids: set[str], metric: M2Metric
) -> dict[ValueKey[int], float]:
    """Valid M2 cells only; a metric the bench did not report is NaN."""
    return {
        ValueKey(row.round, row.treatment, row.c): float(value)
        if is_number(value := getattr(row, metric))
        else NAN
        for row in rows
        if row.session_id in sids and row.valid
    }


def unusable[K: (Cell, int)](values: Values[K]) -> list[ValueKey[K]]:
    return sorted(k for k, v in values.items() if not usable(v))


def paired_rounds[K: (Cell, int)](
    values: Values[K], a: Treatment, b: Treatment, key: K
) -> list[int]:
    """Rounds in which both treatments have a usable value at `key`."""

    def usable_rounds(t: Treatment) -> set[int]:
        return {r for (r, tt, kk), v in values.items() if kk == key and tt == t and usable(v)}

    return sorted(usable_rounds(a) & usable_rounds(b))


def median_value[K: (Cell, int)](values: Values[K], treatment: Treatment, key: K) -> float | None:
    """Median over rounds of the usable values of (treatment, key)."""
    found = [v for (_, t, kk), v in values.items() if t == treatment and kk == key and usable(v)]
    return statistics.median(found) if found else None


def usable_rounds_of[K: (Cell, int)](values: Values[K], treatment: Treatment, key: K) -> int:
    return sum(1 for (_, t, kk), v in values.items() if t == treatment and kk == key and usable(v))


@dataclass(frozen=True)
class OrderEffect:
    n1: float | None
    n2: float | None
    count_n1: int
    count_n2: int
    n2_over_n1: float | None


def order_effect(rows: Iterable[M1Row], sids: set[str]) -> dict[Treatment, dict[Cell, OrderEffect]]:
    """Median measured step per (treatment, cell) split by which wave of the pair ran first."""
    groups = defaultdict(list)
    for row in rows:
        cell = row_cell(row)
        if (
            row.session_id in sids
            and row.warmup is False
            and cell is not None
            and row.first in (WaveOrder.N1, WaveOrder.N2)
            and usable(row.step_s)
        ):
            groups[(row.treatment, cell, row.first)].append(row.step_s)
    out: dict[Treatment, dict[Cell, OrderEffect]] = {}
    for t, cell in sorted({(t, cell) for t, cell, _ in groups}):
        n1, n2 = groups.get((t, cell, WaveOrder.N1), []), groups.get((t, cell, WaveOrder.N2), [])
        m_n1 = statistics.median(n1) if n1 else None
        m_n2 = statistics.median(n2) if n2 else None
        out.setdefault(t, {})[cell] = OrderEffect(
            m_n1, m_n2, len(n1), len(n2), m_n2 / m_n1 if m_n1 and m_n2 else None
        )
    return out


def sessions_of(servers: Iterable[ServerRow]) -> list[ServerRow]:
    return list(latest_sessions(servers).values())


def finite_mean(values: Iterable[float | None]) -> float:
    """NaN if there are no values or any is missing or not finite."""
    values = list(values)
    finite = [v for v in values if is_finite(v)]
    if not values or len(finite) != len(values):
        return NAN
    return statistics.mean(finite)


def none_if_nan(x: float | None) -> float | None:
    return x if is_finite(x) else None


def mean_nll(sessions: Iterable[ServerRow], treatment: Treatment) -> float | None:
    """The treatment's mean session NLL; None without a usable one in every session."""
    return none_if_nan(finite_mean(s.nll for s in sessions if s.treatment == treatment))
