"""A study: its treatments, cells, server settings, schedule and the rules its analysis applies."""

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from fp4bench import settings
from fp4bench.core.types import (
    Arm,
    Cell,
    ContrastName,
    KvDtype,
    RatioName,
    Treatment,
    VerdictLabels,
)
from fp4bench.studies import model
from fp4bench.studies.kernel_scan import Crosscheck, KernelScanSpec


class Prompts(StrEnum):
    M1 = "m1"
    CELLS = "cells"


class VerdictRule(StrEnum):
    H_EQ = "h_eq"
    H_B = "h_b"
    BATCH_VS_TOKENS = "batch_vs_tokens"


class G3Rule(StrEnum):
    AA_RATIO = "aa_ratio"
    AA_EFFECTS = "aa_effects"


@dataclass(frozen=True)
class Ratio:
    """t_numer / t_denom per cell: above 1 when denom is faster."""

    name: RatioName
    numer: Treatment
    denom: Treatment
    labels: VerdictLabels = VerdictLabels.FORMAT


@dataclass(frozen=True)
class Contrast:
    """Δ(cells[0]) − Δ(cells[1]), paired per round."""

    name: ContrastName
    cells: tuple[Cell, Cell]
    arm: Arm


@dataclass(frozen=True)
class ServerSettings:
    kv_dtype: KvDtype = KvDtype.BF16
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 4096
    hf_overrides: str = ""
    # Not in Study.to_protocol_dict, so neither a restart nor study matching sees it change;
    # recording it would change every manifest's protocol bytes. Every study serves 512.
    max_num_seqs: int = 512

    def args(self) -> tuple[str, ...]:
        """The `vllm serve` arguments every treatment shares (METHODOLOGY.md#server-args)."""
        utilization = f"{self.gpu_memory_utilization:.2f}"
        if float(utilization) != self.gpu_memory_utilization:
            raise ValueError(
                f"gpu_memory_utilization {self.gpu_memory_utilization!r} has more "
                f"than two decimals; it would be served as {utilization}"
            )
        for name, value in (
            ("max_model_len", self.max_model_len),
            ("max_num_seqs", self.max_num_seqs),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        args = (
            "--served-model-name",
            settings.SERVED_NAME,
            "--host",
            settings.HOST,
            "--port",
            str(settings.PORT),
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--max-num-batched-tokens",
            "16384",
            "--gpu-memory-utilization",
            utilization,
            "--no-enable-prefix-caching",
            "--seed",
            "0",
            "--api-server-count",
            "4",
            "--kv-cache-dtype",
            self.kv_dtype,
        )
        if self.hf_overrides:
            try:
                override = json.loads(self.hf_overrides)
            except ValueError as exc:
                raise ValueError(f"hf_overrides {self.hf_overrides!r} is not JSON: {exc}") from None
            if not (isinstance(override, dict) and override):
                raise ValueError(
                    f"hf_overrides {self.hf_overrides!r} must be a non-empty JSON object"
                )
            args += ("--hf-overrides", self.hf_overrides)
        return args


def cells_at(prompt_len: int, batches: tuple[int, ...]) -> tuple[Cell, ...]:
    return tuple(Cell(c, prompt_len) for c in batches)


def min_kv_tokens_for(c: int, prompt_len: int = settings.M1_INPUT_LEN) -> int:
    """KV tokens an M1 block holds at the end of its N2 wave (METHODOLOGY.md#kv-capacity)."""
    return c * (prompt_len + settings.M1_N2)


@dataclass(frozen=True)
class Study:
    """A study's design and analysis rules; a run's rounds, reps and cells are in its manifest."""

    name: str
    treatments: tuple[Treatment, ...]
    cells: tuple[Cell, ...]
    server: ServerSettings = ServerSettings()
    rounds: int = 5
    extension_rounds: int | None = None
    m1_reps: int = 5
    m2_duration_s: int = 30
    require_published: bool = True
    primary_batches: tuple[int, ...] = (8, 32, 128)
    prompts: Prompts = Prompts.M1
    ratios: tuple[Ratio, ...] = ()
    contrasts: tuple[Contrast, ...] = ()
    effect_margin_ms: float | None = None
    aa_effect_margin_ms: float | None = None
    g3: G3Rule = G3Rule.AA_RATIO
    verdict: VerdictRule = VerdictRule.H_EQ
    kernel_scan: KernelScanSpec | None = None
    crosscheck: Crosscheck | None = None
    smoke: bool = False

    def __post_init__(self) -> None:
        self._check_design()
        self._check_capacity()
        self._check_rules()
        self._check_scan()

    def _refuse(self, why: str) -> None:
        raise ValueError(f"study {self.name}: {why}")

    def _check_design(self) -> None:
        for what, items in (("treatments", self.treatments), ("cells", self.cells)):
            if not items or len(set(items)) != len(items):
                self._refuse(f"its {what} must be distinct and at least one: {items}")
        if unknown := [t for t in self.treatments if t not in model.TREATMENTS]:
            self._refuse(
                f"no treatment spec (studies.model.TREATMENTS) for {', '.join(map(str, unknown))}"
            )
        if self.prompts is Prompts.M1 and any(
            c.prompt_len != settings.M1_INPUT_LEN for c in self.cells
        ):
            self._refuse(
                f"cells {self.cells} served from the M1 prompts must be one "
                f"({settings.M1_INPUT_LEN}-token) cell per batch"
            )
        if not self.smoke and not set(self.primary_batches) <= set(self.batches):
            self._refuse(
                f"its verdict reads primary batches {self.primary_batches}, not all "
                f"among its batches {self.batches}"
            )
        if self.smoke and self.extension_rounds is not None:
            self._refuse("a smoke decides nothing, so it has no extension")

    def _check_capacity(self) -> None:
        if self.m1_reps + 1 > settings.M1_SETS:
            self._refuse(
                f"its {self.m1_reps} M1 reps and the warmup need {self.m1_reps + 1} prompt "
                f"sets; there are {settings.M1_SETS} (settings.M1_SETS)"
            )
        on_m1 = (c.batch for c in self.cells if c.prompt_len == settings.M1_INPUT_LEN)
        if (largest_on_m1 := max(on_m1, default=0)) > settings.M1_SET_SIZE:
            self._refuse(
                f"its largest batch on the M1 prompts ({largest_on_m1}) is above the "
                f"{settings.M1_SET_SIZE} prompts of an M1 prompt set (settings.M1_SET_SIZE)"
            )
        if (largest := max(self.batches)) > self.server.max_num_seqs:
            self._refuse(
                f"its largest batch {largest} is above its server's {self.server.max_num_seqs} "
                f"sequences (--max-num-seqs)"
            )
        longest = max(cell.prompt_len for cell in self.cells)
        if (needed := longest + settings.M1_N2) > self.server.max_model_len:
            self._refuse(
                f"its longest prompt ({longest} tokens) and the {settings.M1_N2}-token N2 wave "
                f"need a max_model_len of {needed}; its server's is {self.server.max_model_len}"
            )

    def _check_rules(self) -> None:
        names = [r.name for r in self.ratios]
        if len(set(names)) != len(names):
            self._refuse(f"two ratios share a name: {', '.join(names)}")
        margins = (self.effect_margin_ms, self.aa_effect_margin_ms)
        if self.verdict is not VerdictRule.BATCH_VS_TOKENS:
            if self.contrasts or margins != (None, None):
                self._refuse(
                    f"contrasts and their margins are the batch-vs-tokens rule's; its "
                    f"verdict is {self.verdict}"
                )
            return
        arms = [c.arm for c in self.contrasts]
        if sorted(arms) != sorted(Arm) or len({c.name for c in self.contrasts}) != len(arms):
            self._refuse(
                f"the batch-vs-tokens rule reads one contrast per arm "
                f"({', '.join(Arm)}), each named; it has "
                f"{', '.join(f'{c.name} ({c.arm})' for c in self.contrasts) or 'none'}"
            )
        for c in self.contrasts:
            if c.cells[0] == c.cells[1] or not set(c.cells) <= set(self.cells):
                self._refuse(
                    f"contrast {c.name} must pair two distinct cells the study "
                    f"measures, not {c.cells}"
                )
        if any(m is None or not m > 0 for m in margins):
            self._refuse(
                f"its contrasts have no margins (effect {margins[0]}, A/A {margins[1]} ms)"
            )

    def _check_scan(self) -> None:
        if (self.kernel_scan or self.crosscheck) and self.verdict is VerdictRule.BATCH_VS_TOKENS:
            self._refuse(
                "a kernel scan and a cross-check are read per batch; the batch-vs-tokens "
                "analysis has neither"
            )
        if (scan := self.kernel_scan) is not None:
            if missing := [
                t for t in (scan.reference, *scan.treatments) if t not in self.treatments
            ]:
                self._refuse(f"its kernel scan reads {', '.join(missing)}, which it does not serve")
            if missing_c := [
                c for c in (scan.selection_c, scan.tiebreak_c) if c not in self.batches
            ]:
                self._refuse(
                    f"its kernel scan decides at C={', '.join(map(str, missing_c))}, "
                    f"which it does not measure"
                )
        if (check := self.crosscheck) is not None and (
            missing := [t for t in (check.treatment, check.reference) if t not in self.treatments]
        ):
            self._refuse(f"its cross-check reads {', '.join(missing)}, which it does not serve")

    @property
    def batches(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(cell.batch for cell in self.cells))

    def ratio(self, name: RatioName) -> Ratio:
        for r in self.ratios:
            if r.name == name:
                return r
        raise KeyError(f"study {self.name} has no ratio {name}")

    def contrast_along(self, arm: Arm) -> Contrast:
        (contrast,) = (c for c in self.contrasts if c.arm is arm)
        return contrast

    def cell_list(self) -> tuple[Cell, ...]:
        return self.cells

    def m2_batches(self) -> tuple[int, ...]:
        return self.batches if self.m2_duration_s > 0 else ()

    def largest_wave(self) -> Cell:
        """The cell whose M1 block holds the most KV tokens."""
        return max(self.cells, key=lambda cell: min_kv_tokens_for(*cell))

    def min_kv_tokens(self) -> int:
        return min_kv_tokens_for(*self.largest_wave())

    def to_protocol_dict(self) -> dict[str, Any]:
        """The `protocol` a manifest line records (METHODOLOGY.md#study-matching)."""
        cells = (
            tuple((cell.batch, cell.prompt_len) for cell in self.cells)
            if self.prompts is Prompts.CELLS
            else ()
        )
        return {
            "rounds": self.rounds,
            "treatments": self.treatments,
            "concurrencies": self.batches,
            "m1_reps": self.m1_reps,
            "m2_duration_s": self.m2_duration_s,
            "require_published": self.require_published,
            "kv_cache_dtype": self.server.kv_dtype,
            "gpu_memory_utilization": self.server.gpu_memory_utilization,
            "primary_concurrencies": self.primary_batches,
            "cells": cells,
            "max_model_len": self.server.max_model_len,
            "hf_overrides": self.server.hf_overrides,
        }
