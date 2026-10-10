"""What analyze_run returns, typed; `payload` is its JSON view."""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, override

from fp4bench.analysis.cells import OrderEffect, Values
from fp4bench.analysis.checks import AccuracyNote, Bandwidth, DesignCheck, NllCrosscheck
from fp4bench.analysis.compare import (
    ContrastResult,
    Decomposition,
    DeltaResult,
    RatioResult,
    ratio_json,
)
from fp4bench.analysis.gates import GateReport
from fp4bench.analysis.stats import OverallVerdict
from fp4bench.analysis.verdicts import Answer, Incomplete, KernelScan, NvxRatio, Reproduction
from fp4bench.core.schema import Record, Row, ServerRow
from fp4bench.core.types import (
    Arm,
    Cell,
    ContrastName,
    RatioName,
    RecordedKvDtype,
    SkippedMetric,
    Treatment,
    ValueKey,
)
from fp4bench.studies.base import Study
from fp4bench.studies.expc import (
    MAIN_RUN_DELTA_MS,
    PREDICTED_DELTA_MS,
    PREDICTED_EFFECT_MS,
    PREDICTED_R,
)

RatioTable = dict[int, RatioResult[int]]


class Kind(StrEnum):
    MAIN = "main"
    EXPB = "B"
    EXPC = "C"


BATCH_RATIOS = tuple(RatioName)
CELL_RATIOS = (RatioName.R, RatioName.AA)


def cell_treatments(study: Study) -> tuple[Treatment, ...]:
    """Whose median step a cell reports: R's denominator, numerator, then the A/A replica."""
    r, aa = study.ratio(RatioName.R), study.ratio(RatioName.AA)
    return tuple(dict.fromkeys((r.denom, r.numer, aa.numer, aa.denom)))


@dataclass(frozen=True)
class RunResult(ABC):
    """`study`: the study whose rules the run was analysed by (studies.registry.study_for)."""

    run_dir: Path
    study: Study
    kind: Kind
    verdict: OverallVerdict | Answer | Incomplete
    interpretable: bool
    gate_line: str
    gates: GateReport
    has_manifest: bool
    rounds: float | None
    failed: list[ServerRow]

    @abstractmethod
    def payload(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BatchRunResult(RunResult):
    """Main run, Experiment B and the smokes: ratios per batch at one prompt length."""

    nonblocking_failed: list[str]
    reading: str | None
    primary: tuple[int, ...]
    kv_cache_dtype: RecordedKvDtype
    kv_cache_dtype_note: str | None
    context: int
    bandwidth: dict[Treatment, dict[int, Bandwidth]]
    m1: dict[RatioName, RatioTable]
    r_f_phi: dict[int, Decomposition]
    m2_throughput: dict[RatioName, RatioTable]
    m2_itl: dict[RatioName, RatioTable]
    accuracy: AccuracyNote
    scan: KernelScan | None
    crosscheck: dict[int, NvxRatio] | None
    order: dict[Treatment, dict[int, OrderEffect]]
    skipped: dict[SkippedMetric, list[ValueKey[int]]]
    paired_at_primary: dict[int, int]
    tput: Values[int]
    per_user: Values[int]
    served_args: dict[Treatment, tuple[str, ...]]

    @override
    def payload(self) -> dict[str, Any]:
        def table(t: RatioTable) -> dict[int, Any]:
            return {c: ratio_json(v, by_cell=False) for c, v in t.items()}

        m1, tput, itl = self.m1, self.m2_throughput, self.m2_itl
        return {
            "verdict": self.verdict,
            "interpretable": self.interpretable,
            "gate_line": self.gate_line,
            "nonblocking_failed": self.nonblocking_failed,
            "experiment": self.kind,
            "hypothesis_reading": self.reading,
            "primary_concurrencies": self.primary,
            "kv_cache_dtype": str(self.kv_cache_dtype),
            "kv_cache_dtype_note": self.kv_cache_dtype_note,
            "effective_bandwidth": plain(self.bandwidth),
            "m1": table(m1[RatioName.R]),
            "m1_d": table(m1[RatioName.D]),
            "m1_k": table(m1[RatioName.K]),
            "m1_f": table(m1[RatioName.F]),
            "m1_phi": table(m1[RatioName.PHI]),
            "m1_r_f_phi": plain(self.r_f_phi),
            "aa": table(m1[RatioName.AA]),
            "m2_throughput": table(tput[RatioName.R]),
            "m2_d": table(tput[RatioName.D]),
            "m2_k": table(tput[RatioName.K]),
            "m2_f": table(tput[RatioName.F]),
            "m2_phi": table(tput[RatioName.PHI]),
            "m2_itl": table(itl[RatioName.R]),
            "m2_itl_d": table(itl[RatioName.D]),
            "m2_itl_k": table(itl[RatioName.K]),
            "m2_itl_f": table(itl[RatioName.F]),
            "m2_itl_phi": table(itl[RatioName.PHI]),
            "m2_aa": table(tput[RatioName.AA]),
            "accuracy": plain(self.accuracy),
            "failed_sessions": plain(self.failed),
            "kernel_scan": plain(self.scan),
            "nvx_crosscheck": plain(self.crosscheck),
            "order_effect": plain(self.order),
            "skipped": self.skipped,
            "gates": self.gates,
        }


@dataclass(frozen=True)
class CellResult:
    c: int
    p: int
    arms: list[Arm]
    kv_tokens_mean: int
    t_ms: dict[Treatment, float | None]
    r: RatioResult[Cell] | None
    delta_ms: DeltaResult | None
    aa: RatioResult[Cell] | None
    delta_aa_ms: DeltaResult | None
    aa_ratio_rule: bool | None


@dataclass(frozen=True)
class CellRunResult(RunResult):
    """Experiment C (§16): Δ and R per (batch, prompt length) cell and the contrasts of Δ."""

    effects: dict[ContrastName, ContrastResult]
    aa_effects: dict[str, ContrastResult]
    cells: dict[Cell, CellResult]
    reproduction: Reproduction
    design: DesignCheck
    nll: NllCrosscheck
    extension: str | None
    skipped: list[ValueKey[Cell]]
    paired: dict[Cell, int]

    @override
    def payload(self) -> dict[str, Any]:
        return jsonable(
            {
                "run": self.run_dir.name,
                "answer": self.verdict,
                "interpretable": self.interpretable,
                "gate_line": self.gate_line,
                "extension_note": self.extension,
                "registered_rounds": self.rounds,
                "effects": plain(self.effects, by_cell=True),
                "aa_effects": plain(self.aa_effects, by_cell=True),
                "nll_crosscheck": plain(self.nll),
                "cells": plain(list(self.cells.values()), by_cell=True),
                "config_reproduction": plain(self.reproduction),
                "design": plain(self.design),
                "predictions": {
                    "delta_ms": PREDICTED_DELTA_MS,
                    "r": PREDICTED_R,
                    "effects_ms": PREDICTED_EFFECT_MS,
                    "main_run_delta_ms": MAIN_RUN_DELTA_MS,
                },
                "failed_sessions": [s.session_id for s in self.failed],
                "skipped": [[r, t, list(c)] for r, t, c in self.skipped],
                "gates": self.gates,
            }
        )


def plain(obj: Any, by_cell: bool = False) -> Any:
    """Result dataclasses as dicts in field order; ratios as the summary shows them."""
    if isinstance(obj, RatioResult):
        return ratio_json(obj, by_cell)
    if isinstance(obj, Row):
        return obj.as_stored()
    if isinstance(obj, Record):
        return obj.to_json()
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: plain(getattr(obj, f.name), by_cell) for f in fields(obj)}
    if isinstance(obj, dict):
        return {k: plain(v, by_cell) for k, v in obj.items()}
    if isinstance(obj, list):
        return [plain(v, by_cell) for v in obj]
    if isinstance(obj, tuple) and not hasattr(obj, "_fields"):
        return tuple(plain(v, by_cell) for v in obj)
    return obj


def jsonable(obj: Any) -> Any:
    """Cell-keyed dicts with Cell.json_key strings, tuples as lists, non-finite floats as None."""
    if isinstance(obj, dict):
        return {k.json_key if isinstance(k, Cell) else k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
