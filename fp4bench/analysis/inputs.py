"""What a run directory recorded: typed rows, the last manifest line, the checkpoint report."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fp4bench.analysis.cells import is_pos_int, session_ids
from fp4bench.core.schema import M1Row, M2Row, ManifestLine, ServerRow, load_rows
from fp4bench.core.types import (
    ActQuantFusion,
    Cell,
    JsonValue,
    KvDtype,
    LinearKernel,
    RecordedKvDtype,
    Treatment,
    UnknownKvDtype,
    is_finite,
    is_number,
)
from fp4bench.studies import model
from fp4bench.studies.main import FULL
from fp4bench.studies.registry import recorded_protocol as protocol


@dataclass(frozen=True)
class RunData:
    run_dir: Path
    servers: list[ServerRow]
    m1: list[M1Row]
    m2: list[M2Row]
    manifest: ManifestLine | None

    @property
    def checkpoint(self) -> tuple[dict | None, str | None]:
        return resolve_checkpoint(self.run_dir, self.manifest)


def load_run(run_dir: Path) -> RunData:
    """The run's rows; a row of an uncounted session that does not load is left out."""
    run_dir = Path(run_dir)
    manifests = load_rows(run_dir / "manifests.jsonl", ManifestLine)
    servers = load_rows(run_dir / "servers.jsonl", ServerRow)
    counted = session_ids(servers)

    def uncounted(row: JsonValue) -> bool:
        sid = row.get("session_id") if isinstance(row, dict) else None
        return not (isinstance(sid, str) and sid in counted)

    return RunData(
        run_dir=run_dir,
        servers=servers,
        m1=load_rows(run_dir / "m1.jsonl", M1Row, skip_if_bad=uncounted),
        m2=load_rows(run_dir / "m2.jsonl", M2Row, skip_if_bad=uncounted),
        manifest=manifests[-1] if manifests else None,
    )


def inputs(manifest: ManifestLine | None) -> dict[str, Any]:
    value = None if manifest is None else manifest.inputs
    return value if isinstance(value, dict) else {}


def recorded_server_args(manifest: ManifestLine | None) -> dict[Treatment, tuple[str, ...]]:
    """inputs.treatment_server_args: the args each treatment of the run was served with; none in
    a manifest from before the record."""
    recorded = inputs(manifest).get("treatment_server_args")
    if recorded is None:
        return {}
    if not (
        isinstance(recorded, dict)
        and all(t in Treatment for t in recorded)
        and all(
            isinstance(args, list) and all(isinstance(a, str) for a in args)
            for args in recorded.values()
        )
    ):
        raise ValueError(
            f"inputs.treatment_server_args of the last manifest line is not a list of strings "
            f"per treatment: {recorded!r}"
        )
    return {Treatment(t): tuple(args) for t, args in recorded.items()}


def expected_kernels(manifest: ManifestLine | None) -> dict[Treatment, LinearKernel]:
    """The kernel class each treatment's recorded server args select; model.TREATMENTS' for a
    treatment the manifest does not record."""
    recorded = recorded_server_args(manifest)
    return {
        t: model.expected_linear_kernel(t, recorded[t]) if t in recorded else spec.linear_kernel
        for t, spec in model.TREATMENTS.items()
    }


def expected_fusions(manifest: ManifestLine | None) -> dict[Treatment, bool]:
    """Whether each treatment's recorded server args fuse the activation quantization;
    model.TREATMENTS' for a treatment the manifest does not record."""
    recorded = recorded_server_args(manifest)
    return {
        t: model.expected_act_quant_fusion(t, recorded[t]) is ActQuantFusion.ON
        if t in recorded
        else spec.act_quant_fusion
        for t, spec in model.TREATMENTS.items()
    }


def registered_rounds(manifest: ManifestLine | None) -> float | None:
    rounds = protocol(manifest).get("rounds")
    return rounds if is_number(rounds) and rounds == int(rounds) and rounds >= 1 else None


def registered_reps(manifest: ManifestLine | None) -> int | None:
    reps = protocol(manifest).get("m1_reps")
    return reps if is_pos_int(reps) else None


def primary_batches(manifest: ManifestLine | None) -> tuple[int, ...]:
    """protocol.primary_concurrencies, else the main run's (METHODOLOGY.md#study-matching)."""
    value = protocol(manifest).get("primary_concurrencies")
    if isinstance(value, list) and value and all(is_pos_int(c) for c in value):
        return tuple(value)
    return FULL.primary_batches


def kv_cache_dtype(manifest: ManifestLine | None) -> RecordedKvDtype:
    value = protocol(manifest).get("kv_cache_dtype")
    if value is None:
        return KvDtype.BF16
    name = str(value)
    return KvDtype(name) if name in KvDtype else UnknownKvDtype(name)


def protocol_cells(manifest: ManifestLine | None) -> tuple[tuple[Cell, ...] | None, str | None]:
    """(protocol.cells, problem); a malformed or repeated entry gives no cells."""
    if manifest is None:
        return None, "no manifest"
    raw = protocol(manifest).get("cells")
    if not isinstance(raw, list) or not raw:
        return None, f"protocol.cells is not a non-empty list: {raw!r}"
    cells = []
    for item in raw:
        if not (
            isinstance(item, (list, tuple)) and len(item) == 2 and all(is_pos_int(x) for x in item)
        ):
            return None, f"protocol.cells has a malformed entry {item!r} (needs [C, P])"
        cells.append(Cell(item[0], item[1]))
    if len(set(cells)) != len(cells):
        return None, f"protocol.cells has duplicate entries: {raw!r}"
    return tuple(cells), None


def registered_batches(manifest: ManifestLine | None) -> set[int]:
    value = protocol(manifest).get("concurrencies")
    return set(value) if isinstance(value, list) and all(is_number(c) for c in value) else set()


def resolve_checkpoint(
    run_dir: Path, manifest: ManifestLine | None
) -> tuple[dict | None, str | None]:
    """(report, source): the manifest's, else the results directory's, labelled as such."""
    report = None if manifest is None else manifest.checkpoint_report
    if isinstance(report, dict):
        return report, "manifest (last line, computed by the run at its start)"
    if report is not None:
        return None, f"manifest (unusable: {report!r})"
    path = Path(run_dir).parent / "checkpoint_report.json"
    if not path.exists():
        return None, None
    label = f"fallback: {path} (not run-scoped, may be stale)"
    try:
        loaded = json.loads(path.read_text())
    except ValueError:
        return None, f"{label}, unreadable"
    return (loaded, label) if isinstance(loaded, dict) else (None, f"{label}, not a JSON object")


def bf16_reference(manifest: ManifestLine | None) -> tuple[dict | None, str | None]:
    """(inputs.bf16_reference_nll, problem): usable only with a finite mean."""
    if manifest is None:
        return None, "no manifest"
    ref = inputs(manifest).get("bf16_reference_nll")
    if ref is None:
        return None, "the manifest has no inputs.bf16_reference_nll"
    if not isinstance(ref, dict):
        return None, f"inputs.bf16_reference_nll is not usable: {ref!r}"
    if not is_finite(ref.get("mean")):
        return None, f"inputs.bf16_reference_nll has no finite mean: {ref.get('mean')!r}"
    return ref, None


def nll_prompts_sha(manifest: ManifestLine | None) -> str | None:
    sha = inputs(manifest).get("nll_prompts_sha256")
    return sha if isinstance(sha, str) else None
