"""Every study by its name, and the study a run's manifest recorded."""

import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from fp4bench.core.schema import ManifestLine
from fp4bench.core.types import KvDtype
from fp4bench.studies.base import Study
from fp4bench.studies.expb import EXPB
from fp4bench.studies.expc import EXPC, SMOKE_C
from fp4bench.studies.main import FULL
from fp4bench.studies.smoke import SMOKE, SMOKE_NF

STUDIES: dict[str, Study] = {s.name: s for s in (SMOKE, SMOKE_NF, FULL, EXPB, SMOKE_C, EXPC)}

# METHODOLOGY.md#study-matching
LEGACY_DEFAULTS: dict[str, Any] = {
    "kv_cache_dtype": KvDtype.BF16,
    "gpu_memory_utilization": 0.9,
    "primary_concurrencies": [8, 32, 128],
    "cells": [],
    "max_model_len": 4096,
    "hf_overrides": "",
}
FP8_KV_CACHE_DTYPES = frozenset(dtype for dtype in KvDtype if dtype.is_fp8)


def study_to_run(name: str, rounds: int = 0) -> Study:
    """The registered study `name` with `rounds` rounds: 0 keeps its pre-registered count, and
    any other value must be that count or its registered extension's (EXPERIMENT.md §8, §13, §16).
    """
    study = STUDIES[name]
    extension = study.extension_rounds
    if rounds not in (0, study.rounds, extension):
        rule = (
            " and has no extension"
            if extension is None
            else f", and the only pre-registered change is extending (to {extension})"
        )
        raise ValueError(
            f"{rounds} is not a pre-registered round count of {name}: it runs {study.rounds}{rule}"
        )
    return replace(study, rounds=rounds) if rounds > 0 else study


def recorded_protocol(manifest: ManifestLine | None) -> dict[str, Any]:
    value = None if manifest is None else manifest.protocol
    return value if isinstance(value, dict) else {}


def _canonical(protocol: Mapping[str, Any]) -> str:
    return json.dumps({k: v for k, v in protocol.items() if k != "rounds"}, sort_keys=True)


def protocol_shape(study: Study) -> str:
    """What identifies a study's manifests: its recorded protocol but for `rounds`."""
    return _canonical(json.loads(json.dumps(study.to_protocol_dict())))


def check_distinct(studies: Mapping[str, Study]) -> None:
    seen: dict[str, str] = {}
    for name, study in studies.items():
        other = seen.setdefault(protocol_shape(study), name)
        if other != name:
            raise ValueError(
                f"studies {other} and {name} record the same protocol: a manifest "
                f"could not tell them apart"
            )


check_distinct(STUDIES)


def _with_recorded_rounds(study: Study, recorded: Mapping[str, Any]) -> Study:
    rounds = recorded.get("rounds")
    return replace(study, rounds=rounds) if type(rounds) is int and rounds >= 1 else study


def registered_study(manifest: ManifestLine | None) -> Study | None:
    """The study whose protocol the manifest recorded, but for `rounds` and LEGACY_DEFAULTS."""
    recorded = recorded_protocol(manifest)
    shape = _canonical({**LEGACY_DEFAULTS, **recorded})
    for study in STUDIES.values():
        if protocol_shape(study) == shape:
            return _with_recorded_rounds(study, recorded)
    return None


def fallback_study(protocol: Mapping[str, Any]) -> Study:
    """The study whose rules an unregistered protocol gets (METHODOLOGY.md#study-matching)."""
    if protocol.get("cells"):
        return EXPC
    kv_dtype = protocol.get("kv_cache_dtype")
    if (KvDtype.BF16 if kv_dtype is None else str(kv_dtype)) in FP8_KV_CACHE_DTYPES:
        return EXPB
    treatments = protocol.get("treatments")
    scanned = () if SMOKE.kernel_scan is None else SMOKE.kernel_scan.treatments
    if isinstance(treatments, list) and any(t in scanned for t in treatments):
        return SMOKE
    return FULL


def named_study(manifest: ManifestLine) -> Study:
    name = manifest.study
    if not isinstance(name, str) or name not in STUDIES:
        raise ValueError(
            f"the manifest names study {name!r}, which is not registered "
            f"(registered: {', '.join(STUDIES)})"
        )
    return _with_recorded_rounds(STUDIES[name], recorded_protocol(manifest))


def study_for(manifest: ManifestLine | None) -> Study:
    """The study a manifest names, else the one its protocol matches, else the fallback."""
    if manifest is not None and "study" not in manifest.missing:
        return named_study(manifest)
    return registered_study(manifest) or fallback_study(recorded_protocol(manifest))
