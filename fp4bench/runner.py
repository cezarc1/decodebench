# METHODOLOGY.md#run-integrity
import json
import math
import shutil
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, NoReturn

from fp4bench import settings
from fp4bench.core.argv import flag_value
from fp4bench.core.files import file_sha256
from fp4bench.core.schema import (
    BlockTelemetry,
    M1Row,
    M2Row,
    ManifestLine,
    ServerRow,
    append_jsonl,
    append_row,
    load_rows,
)
from fp4bench.core.types import (
    Cell,
    CheckpointKind,
    GpuUuid,
    JsonObject,
    JsonValue,
    M2InvalidReason,
    ProbeError,
    PromptSet,
    Raised,
    RetryPolicy,
    SessionId,
    SessionSlot,
    Sha256,
    StartId,
    Treatment,
    attempt,
    error_as_text,
    is_finite,
    probe,
)
from fp4bench.decode_step import decode_block
from fp4bench.lib_bench import run_lib_decode
from fp4bench.manifest import CodeVersion, collect_manifest, gpu_info, verify_manifest
from fp4bench.metrics import READ, read_preemptions, read_preemptions_retrying
from fp4bench.prep import FETCH_FILE
from fp4bench.prompts import check_cell_prompt_sets
from fp4bench.sanity import checkpoint_report, content_fingerprint, layout_fingerprint, prompt_nlls
from fp4bench.schedule import round_order
from fp4bench.server import AUTOTUNE_CACHE_ENV, VllmServer, autotune_ran_fresh, parse_server_log
from fp4bench.studies import model
from fp4bench.studies.base import Prompts, Study
from fp4bench.studies.registry import LEGACY_DEFAULTS
from fp4bench.telemetry import GpuSampler, counter_delta, read_counters, throttle_summary
from fp4bench.wave import run_wave


def _round_is_complete(latest: dict[SessionSlot, ServerRow], r: int, study: Study) -> bool:
    if any(SessionSlot(r, t) not in latest for t in study.treatments):
        return False
    start_ids = {latest[SessionSlot(r, t)].start_id for t in study.treatments}
    return len(start_ids) == 1 and bool(next(iter(start_ids)))


def rounds_to_run(servers_rows: list[ServerRow], study: Study) -> list[int]:
    """Rounds without a complete set of sessions from one start."""
    latest = {SessionSlot(row.round, row.treatment): row for row in servers_rows}
    return [r for r in range(study.rounds) if not _round_is_complete(latest, r, study)]


def measured[T](fn: Callable[[], T]) -> tuple[T, BlockTelemetry]:
    before = read_counters()
    t0 = time.time()
    result = fn()
    window = time.time() - t0
    delta = counter_delta(before, read_counters())
    counters_delta = {event.value: n for event, n in delta.items()}
    return result, BlockTelemetry.from_json(
        {"window_s": window, "counters_delta": counters_delta, **throttle_summary(delta, window)}
    )


def _gpu_uuid() -> GpuUuid | ProbeError:
    return probe(lambda: GpuUuid(gpu_info()["uuid"]))


def _require_server_alive(server: VllmServer, session_id: str) -> None:
    if server.proc is None or server.proc.poll() is not None:
        code = None if server.proc is None else server.proc.returncode
        raise RuntimeError(
            f"vllm server died during session {session_id} (exit code {code}); "
            f"see {server.log_path}"
        )


SESSION_ID_ATTR = "fp4bench_session_id"


# METHODOLOGY.md#preemption-metric
SWEEP_READ = RetryPolicy(attempts=5, backoff_s=2.0, backoff_factor=2.0)


def _preemption_sleep(seconds: float) -> None:
    time.sleep(seconds)


def _read_preemptions(policy: RetryPolicy = READ) -> float:
    return read_preemptions_retrying(
        settings.BASE_URL, policy, sleep=_preemption_sleep, read=read_preemptions
    )


def _preemptions_in_sweep() -> float | None:
    read = attempt(lambda: _read_preemptions(SWEEP_READ))
    if isinstance(read, Raised):
        print(
            f"warning: could not read vLLM's preemption counter in "
            f"{SWEEP_READ.attempts} attempts, recorded as None: {read.exc!r}",
            file=sys.stderr,
        )
        return None
    return read


def _stop_for_unreadable_m1_counter(
    server: VllmServer, session_id: str, cell: Cell, when: str
) -> NoReturn:
    _require_server_alive(server, session_id)
    raise RuntimeError(
        f"vLLM's preemption counter could not be read {when} the M1 block at "
        f"C={cell.batch}, P={cell.prompt_len} of session "
        f"{session_id} ({SWEEP_READ.attempts} attempts): the block cannot pass G6a, "
        f"so the session stops here"
    )


def preemption_delta(before: float | None, after: float | None) -> float | None:
    """Preemptions between two reads; None if either read failed."""
    return None if before is None or after is None else after - before


def preemption_guard(cell: dict, delta: float | None) -> dict:
    """An M2 cell with its preemptions_delta; a moved or unread counter makes it invalid."""
    out = {**cell, "preemptions_delta": delta}
    if delta != 0:
        out["valid"] = False
        out["invalid_reasons"] = [*cell.get("invalid_reasons", []), M2InvalidReason.PREEMPTIONS]
    return out


def kv_capacity_problem(kv_cache_tokens: int | None, study: Study) -> str | None:
    """Why the KV pool cannot hold the largest M1 wave, else None (METHODOLOGY.md#kv-capacity)."""
    required = study.min_kv_tokens()
    c, p = study.largest_wave()
    where = f"C={c}, P={p}"
    if kv_cache_tokens is None:
        return (
            f'the server log has no "GPU KV cache size" line, so its KV cache cannot be '
            f"checked against the {required:,} tokens M1 holds at {where}"
        )
    if kv_cache_tokens < required:
        return (
            f"the KV cache holds {kv_cache_tokens:,} tokens, fewer than the {required:,} M1's "
            f"largest wave holds at {where} ({c} x ({p} + {settings.M1_N2})): "
            f"its blocks at that cell would preempt and fail G6a"
        )
    return None


def fresh_autotune_dir(run_dir: Path, session_id: str) -> Path:
    """A new, empty FlashInfer autotune cache directory for this start (METHODOLOGY.md#autotune)."""
    path = Path(run_dir) / "autotune" / session_id
    path.mkdir(parents=True)
    return path


def autotune_files(cache_dir: Path) -> dict[str, str]:
    """{path: sha256} of the files the server's autotune pass saved."""
    return {str(p): file_sha256(p) for p in sorted(Path(cache_dir).rglob("*")) if p.is_file()}


def _m1_blocks(
    study: Study, prompt_sets: list[PromptSet], cell_prompts: dict[Cell, list[PromptSet]] | None
) -> list[tuple[Cell, list[PromptSet]]]:
    blocks, missing = [], []
    for cell in study.cell_list():
        if cell_prompts is not None:
            sets = cell_prompts.get(cell)
        else:
            sets = prompt_sets if cell.prompt_len == settings.M1_INPUT_LEN else None
        if sets is None:
            missing.append(cell.prompt_file_key)
        else:
            blocks.append((cell, sets))
    if missing:
        raise ValueError(
            f"the protocol's cells {missing} have no prompt sets "
            f"(cell_prompts: {sorted(c.prompt_file_key for c in cell_prompts or {})})"
        )
    return blocks


def run_server_session(
    run_dir: Path,
    r: int,
    treatment: Treatment,
    study: Study,
    prompt_sets: list,
    nll_prompts: list,
    start_id: str,
    cell_prompts: dict | None = None,
) -> None:
    """NLL check, M1 blocks, M2 cells, servers row (METHODOLOGY.md#server-lifecycle)."""
    tag = f"r{r}_{treatment}"
    session_id = SessionId(f"{tag}-{uuid.uuid4().hex[:8]}")
    model_dir = model.TREATMENTS[treatment].model_dir
    base: dict[str, Any] = {"round": r, "treatment": treatment, "session_id": session_id}
    blocks = _m1_blocks(study, prompt_sets, cell_prompts)

    server_args = (*study.server.args(), *model.TREATMENTS[treatment].server_args)
    autotune_dir = fresh_autotune_dir(run_dir, session_id)
    server_env = {AUTOTUNE_CACHE_ENV: str(autotune_dir)}

    def wave(prompts, n):
        return run_wave(settings.BASE_URL, settings.SERVED_NAME, prompts, n)

    try:
        with (
            GpuSampler(run_dir / "telemetry" / f"{session_id}.csv"),
            VllmServer(
                model_dir,
                run_dir / "servers" / f"{session_id}.log",
                args=server_args,
                env=server_env,
            ) as server,
        ):
            argv = list(server.cmd)
            if problem := kv_capacity_problem(
                parse_server_log(server.log_text())["kv_cache_tokens"], study
            ):
                msg = f"session {session_id}: {problem}; see {server.log_path}"
                raise RuntimeError(msg)  # noqa: TRY301 - the except below tags it with the session
            _read_preemptions()
            nll_per_prompt = prompt_nlls(settings.BASE_URL, settings.SERVED_NAME, nll_prompts)
            nll = math.fsum(nll_per_prompt) / len(nll_per_prompt)
            for cell, sets in blocks:
                before = _preemptions_in_sweep()
                if before is None:
                    _stop_for_unreadable_m1_counter(server, session_id, cell, "before")
                rows, tel = measured(
                    partial(
                        decode_block,
                        wave,
                        sets,
                        cell.batch,
                        settings.M1_N1,
                        settings.M1_N2,
                        study.m1_reps,
                    )
                )
                delta = preemption_delta(before, _preemptions_in_sweep())
                for row in rows:
                    append_row(
                        run_dir / "m1.jsonl",
                        M1Row(
                            **base,
                            **row,
                            prompt_len=cell.prompt_len,
                            block_telemetry=tel,
                            preemptions_delta=delta,
                        ),
                    )
                if delta is None:
                    _stop_for_unreadable_m1_counter(server, session_id, cell, "after")
            for c in study.m2_batches():
                _require_server_alive(server, session_id)
                before = _preemptions_in_sweep()
                metrics, tel = measured(
                    partial(
                        run_lib_decode,
                        c,
                        study.m2_duration_s,
                        run_dir / "m2_raw" / f"{session_id}_c{c}.json",
                        run_dir / "lib_logs" / f"{session_id}.log",
                    )
                )
                metrics = preemption_guard(
                    metrics, preemption_delta(before, _preemptions_in_sweep())
                )
                append_row(run_dir / "m2.jsonl", M2Row(**base, c=c, **metrics, telemetry=tel))
            _require_server_alive(server, session_id)
            facts = parse_server_log(server.log_text())
            saved = autotune_files(autotune_dir)
    except Exception as exc:
        setattr(exc, SESSION_ID_ATTR, session_id)
        raise
    append_row(
        run_dir / "servers.jsonl",
        ServerRow(
            **base,
            start_id=start_id,
            gpu_uuid=error_as_text(_gpu_uuid()),
            server_argv=argv,
            nll=nll,
            nll_per_prompt=nll_per_prompt,
            **facts,
            server_env=server_env,
            autotune_cache_dir=str(autotune_dir),
            autotune_fresh=autotune_ran_fresh(facts, autotune_dir),
            autotune_saved_files=saved,
        ),
    )


# METHODOLOGY.md#run-integrity
PROTOCOL_FIELDS_ADDED_FOR_EXPC = ("cells", "max_model_len", "hf_overrides")


def check_protocol_unchanged(lines: Sequence[ManifestLine], study: Study) -> None:
    """On a restart, every protocol field but `rounds` must match, and `rounds` never falls
    below an earlier start's: an extended run stays extended. A count above the study's that is
    not its extension was never registered (study_to_run refuses it), so that run is a dead end."""
    if not lines:
        return
    new = json.loads(json.dumps(study.to_protocol_dict()))
    old = {
        **{k: LEGACY_DEFAULTS[k] for k in PROTOCOL_FIELDS_ADDED_FOR_EXPC},
        **(lines[0].protocol or {}),
    }
    errors = []
    differing = sorted(
        k for k in old.keys() | new.keys() if k != "rounds" and old.get(k) != new.get(k)
    )
    if differing:
        detail = "; ".join(
            f"{k}: first start {old.get(k)!r}, now {new.get(k)!r}" for k in differing
        )
        errors.append(
            f"protocol differs from the first start of this run in {differing} "
            f"({detail}); only `rounds` may change on a restart"
        )
    earlier = [
        r
        for r in ((line.protocol or {}).get("rounds") for line in lines)
        if isinstance(r, int) and not isinstance(r, bool)
    ]
    if earlier and study.rounds < (recorded := max(earlier)):
        lower = (
            f"rounds {study.rounds} is lower than the {recorded} registered by an earlier "
            f"start of this run"
        )
        errors.append(
            f"{lower}; an extended run resumes with its extension: pass --rounds {recorded}"
            if recorded == study.extension_rounds
            else f"{lower}, which is not a registered count of {study.name}: this run cannot "
            f"be resumed; start a new run id"
        )
    if errors:
        raise RuntimeError("; ".join(errors))


def _baseline(lines: Sequence[ManifestLine]) -> ManifestLine | None:
    """The first start that passed its checks: an aborted start ran no session."""
    return next((line for line in lines if not line.problems), None)


def check_code_unchanged(lines: Sequence[ManifestLine], code: CodeVersion) -> None:
    """On a restart, the code must be the baseline start's commit, both checkouts clean."""
    baseline = _baseline(lines)
    if baseline is None:
        return
    if "code_commit" in baseline.missing:
        print(
            f"warning: the first start of this run that passed its checks predates code_commit in "
            f"its manifest, so this start's code version ({code.describe()}) cannot be checked "
            "against it",
            file=sys.stderr,
        )
        return
    then = CodeVersion(baseline.code_commit, baseline.code_dirty)
    if not (then.is_clean and code == then):
        raise RuntimeError(
            f"this start's code ({code.describe()}) is not verifiably that of the first start "
            f"that passed its checks ({then.describe()}): a restart must run the same commit "
            "from a clean checkout, as uncommitted or unknown changes cannot be compared; a run "
            "is one code version; start a new run id"
        )


def _identity(inputs: dict) -> dict:
    out = dict(inputs)
    for kind in (CheckpointKind.MX, CheckpointKind.NV):
        if isinstance(out.get(f"{kind}_fetch"), dict):
            out[f"{kind}_fetch"] = {k: v for k, v in out[f"{kind}_fetch"].items() if k != "utc"}
    out.pop("local_staging", None)
    return out


def check_inputs_unchanged(lines: Sequence[ManifestLine], inputs: dict) -> None:
    """On a restart, the inputs must be those of the first start that passed its checks."""
    first = _baseline(lines)
    baseline = None if first is None else first.inputs
    if not isinstance(baseline, dict):
        return
    old, new = _identity(baseline), _identity(inputs)
    differing = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
    if differing:
        detail = "; ".join(
            f"{k}: first start {old.get(k)!r}, now {new.get(k)!r}" for k in differing
        )
        raise RuntimeError(
            f"inputs differ from the first start of this run that passed the environment check "
            f"in {differing} ({detail}); one run must use identical prompts and checkpoints: "
            f"restore the originals or start a new run id"
        )


def published_problem(kind: CheckpointKind) -> str:
    return (
        f"inputs.{kind}_fetch is missing: this protocol serves the {kind.upper()} checkpoint "
        "published on the HF Hub (EXPERIMENT.md §6), so run `fp4bench fetch "
        f"--kind {kind} --repo-id <repo> --revision <sha>` first"
    )


def _read_fetch(model_dir: str) -> dict | None:
    path = Path(model_dir) / FETCH_FILE
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if not (isinstance(record, dict) and record.get("repo_id") and record.get("revision")):
        raise ValueError(f"{path} does not name a repo_id and a revision: {record!r}")
    return record


def read_bf16_reference_nll(path) -> dict:
    """The BF16 reference's per_prompt and mean; raises unless it is over the run's NLL prompts."""
    ref = json.loads(Path(path).read_text())
    if not isinstance(ref, dict):
        raise ValueError(f"{path} is not a JSON object: {type(ref).__name__}")
    per_prompt, mean = ref.get("per_prompt"), ref.get("mean")
    if not (isinstance(per_prompt, list) and per_prompt and all(map(is_finite, per_prompt))):
        raise ValueError(
            f"{path}: per_prompt must be a non-empty list of finite numbers, got {per_prompt!r}"
        )
    if not is_finite(mean):
        raise ValueError(f"{path}: mean must be a finite number, got {mean!r}")
    try:
        prompts_sha = file_sha256(settings.NLL_PROMPTS_PATH)
    except OSError:
        prompts_sha = None
    if prompts_sha is not None and ref.get("nll_prompts_sha256") != prompts_sha:
        raise ValueError(
            f"{path}: nll_prompts_sha256 is {ref.get('nll_prompts_sha256')!r}, but the run's NLL "
            f"prompts file {settings.NLL_PROMPTS_PATH} hashes to {prompts_sha!r}: the BF16 "
            "reference was computed on other prompts; re-run quantize (or prepare --force, then "
            "quantize)"
        )
    return {"per_prompt": per_prompt, "mean": mean}


def load_cell_prompts(study: Study) -> dict[Cell, list[PromptSet]]:
    """Every cell of `study`'s prompt sets; raises unless the file serves them all."""
    path = settings.C_PROMPTS_PATH
    record = json.loads(Path(path).read_text())
    if not (isinstance(record, dict) and isinstance(record.get("cells"), dict)):
        raise ValueError(f'{path} has no "cells" object: {str(record)[:200]}')
    check_cell_prompt_sets(record["cells"], study.cells, study.m1_reps + 1)
    try:
        m1_sha = file_sha256(settings.M1_PROMPTS_PATH)
    except OSError:
        m1_sha = None
    if m1_sha is not None and record.get("m1_prompts_sha256") != m1_sha:
        raise ValueError(
            f"{path}: m1_prompts_sha256 is {record.get('m1_prompts_sha256')!r}, but the run's M1 "
            f"prompts file {settings.M1_PROMPTS_PATH} hashes to {m1_sha!r}: the cell prompts were "
            "built from other M1 prompts; re-run `fp4bench prepare --expc`"
        )
    return {cell: record["cells"][cell.prompt_file_key] for cell in study.cells}


def cell_prompts_sha256(study: Study) -> Sha256:
    """sha256 of the cell prompt file, once load_cell_prompts has shown it serves `study`."""
    load_cell_prompts(study)
    return file_sha256(settings.C_PROMPTS_PATH)


def collect_inputs(study: Study) -> dict[str, JsonValue | ProbeError]:
    """Identity of every input the sessions use."""
    inputs: dict[str, JsonValue | ProbeError] = {
        "m1_prompts_sha256": probe(lambda: file_sha256(settings.M1_PROMPTS_PATH)),
        "nll_prompts_sha256": probe(lambda: file_sha256(settings.NLL_PROMPTS_PATH)),
        "bf16_reference_nll": probe(lambda: read_bf16_reference_nll(settings.BF16_REF_NLL_PATH)),
        "mx_layout_fingerprint": probe(lambda: layout_fingerprint(model.MX_MODEL_DIR)),
        "nv_layout_fingerprint": probe(lambda: layout_fingerprint(model.NV_MODEL_DIR)),
        "mx_content_fingerprint": probe(lambda: content_fingerprint(model.MX_MODEL_DIR)),
        "nv_content_fingerprint": probe(lambda: content_fingerprint(model.NV_MODEL_DIR)),
    }
    if Treatment.NVX in study.treatments:
        inputs["nvx_content_fingerprint"] = probe(lambda: content_fingerprint(model.NVX_MODEL_DIR))
    if study.prompts is Prompts.CELLS:
        inputs["c_prompts_sha256"] = probe(lambda: cell_prompts_sha256(study))
    inputs["mx_fetch"] = probe(lambda: _read_fetch(model.MX_MODEL_DIR))
    inputs["nv_fetch"] = probe(lambda: _read_fetch(model.NV_MODEL_DIR))
    # METHODOLOGY.md#run-integrity: NVa's pin is part of the run's identity
    inputs["treatment_server_args"] = {
        str(t): list(model.TREATMENTS[t].server_args) for t in study.treatments
    }
    return inputs


def nva_pin_problems(study: Study) -> list[str]:
    """Problems unless NVa is pinned to a kernel other than NV's (METHODOLOGY.md#nv-alt)."""
    if Treatment.NVA not in study.treatments:
        return []
    nva = model.TREATMENTS[Treatment.NVA]
    args = nva.server_args
    if flag_value(args, "--linear-backend") is None:
        return [
            f"NVa has no --linear-backend in model.TREATMENTS (args {args!r}), so it "
            "would run vLLM's default kernel, the same one as NV: set NVa from the smoke "
            "kernel scan before a run that includes it"
        ]
    nv_kernel = model.TREATMENTS[Treatment.NV].linear_kernel
    if nva.linear_kernel == nv_kernel:
        return [
            f"NVa is pinned (args {args!r}) to the same kernel as NV "
            f"({nv_kernel}), so K and D would compare NV with itself: "
            "set NVa to the alternative kernel the smoke kernel scan selects"
        ]
    return []


STAGING_SUFFIX = ".copying"  # METHODOLOGY.md#staging


def stage_checkpoint(volume_dir, local_dir, volume_fingerprint: str) -> dict:
    """Make `local_dir` a copy of `volume_dir` (METHODOLOGY.md#staging); a mismatch is returned."""
    volume_dir, local_dir = Path(volume_dir), Path(local_dir)
    record = {
        "volume_dir": str(volume_dir),
        "local_dir": str(local_dir),
        "volume_content_fingerprint": volume_fingerprint,
        "local_content_fingerprint": None,
        "copied": False,
        "copy_s": 0.0,
    }
    if local_dir.exists():
        if probe(lambda: content_fingerprint(local_dir)) == volume_fingerprint:
            record["local_content_fingerprint"] = volume_fingerprint
            return record
        shutil.rmtree(local_dir)
    tmp = local_dir.with_name(local_dir.name + STAGING_SUFFIX)
    shutil.rmtree(tmp, ignore_errors=True)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        shutil.copytree(volume_dir, tmp)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    tmp.replace(local_dir)
    record["copy_s"] = time.monotonic() - started
    record["copied"] = True
    record["local_content_fingerprint"] = content_fingerprint(local_dir)
    return record


def _stage_kind(
    inputs: Mapping[str, JsonValue | ProbeError], kind: CheckpointKind, spec: model.TreatmentSpec
) -> dict:
    fingerprint = inputs[f"{kind}_content_fingerprint"]
    if not isinstance(fingerprint, str):
        raise ValueError(f"{kind}_content_fingerprint is not a fingerprint: {fingerprint!r}")
    return stage_checkpoint(spec.volume_dir, spec.model_dir, fingerprint)


def stage_checkpoints(
    study: Study, inputs: Mapping[str, JsonValue | ProbeError]
) -> tuple[dict, list[str]]:
    """Stage each volume dir the treatments serve, once; stops at the first failure."""
    treatment_of = {model.TREATMENTS[t].checkpoint: t for t in reversed(study.treatments)}
    records: dict[str, dict] = {}
    problems: list[str] = []
    for kind, treatment in sorted(treatment_of.items()):
        spec = model.TREATMENTS[treatment]
        volume_dir, local_dir = spec.volume_dir, spec.model_dir
        record = probe(partial(_stage_kind, inputs, kind, spec))
        if isinstance(record, ProbeError):
            records[kind] = {
                "volume_dir": volume_dir,
                "local_dir": local_dir,
                "error": record.detail,
            }
            problems.append(
                f"local staging of {kind.upper()} ({volume_dir} -> {local_dir}) "
                f"failed: {record.detail}"
            )
            break
        records[kind] = record
        if record["local_content_fingerprint"] != record["volume_content_fingerprint"]:
            problems.append(
                f"local copy of {kind.upper()} differs from the volume: content fingerprint "
                f"{record['local_content_fingerprint']} != {record['volume_content_fingerprint']} "
                f"({volume_dir} -> {local_dir})"
            )
            break
    return records, problems


def _record_error(
    run_dir: Path, session_hint: dict | None, exc: BaseException, nonfatal: bool = False
) -> None:
    row: JsonObject = {
        "utc": datetime.now(UTC).isoformat(),
        "session_hint": session_hint,
        "error": repr(exc),
        "traceback": traceback.format_exc(),
    }
    if nonfatal:
        row["nonfatal"] = True
    written = attempt(lambda: append_jsonl(run_dir / "errors.jsonl", row))
    if isinstance(written, Raised):
        print(f"warning: could not write errors.jsonl: {written.exc!r}", file=sys.stderr)


def _record_failed_session(
    run_dir: Path, r: int, treatment: Treatment, start_id: str, exc: Exception
) -> None:
    session_id = getattr(exc, SESSION_ID_ATTR, None) or SessionId(
        f"r{r}_{treatment}-{uuid.uuid4().hex[:8]}"
    )
    append_row(
        run_dir / "servers.jsonl",
        ServerRow(
            round=r,
            treatment=treatment,
            session_id=session_id,
            start_id=start_id,
            failed=True,
            error=repr(exc),
        ),
    )


def run_experiment(
    run_dir: Path, study: Study, commit: Callable[[], None], rerun_rounds: tuple[int, ...] = ()
) -> None:
    """Run every incomplete round, plus `rerun_rounds` (METHODOLOGY.md#run-integrity)."""
    run_dir = Path(run_dir)
    bad = sorted({r for r in rerun_rounds if not 0 <= r < study.rounds})
    if bad:
        raise ValueError(f"rerun_rounds {bad} outside the protocol's rounds 0..{study.rounds - 1}")
    run_dir.mkdir(parents=True, exist_ok=True)
    lines = load_rows(run_dir / "manifests.jsonl", ManifestLine)
    check_protocol_unchanged(lines, study)
    check_code_unchanged(lines, CodeVersion.from_env())
    manifest = collect_manifest()
    problems = verify_manifest(manifest)
    inputs = collect_inputs(study)
    input_errors = [
        f"input {name}: {value.to_json()}"
        for name, value in inputs.items()
        if isinstance(value, ProbeError)
    ]
    problems += input_errors
    problems += nva_pin_problems(study)
    if not input_errors:
        check_inputs_unchanged(lines, inputs)
    if study.require_published:
        problems += [
            published_problem(kind)
            for kind in (CheckpointKind.MX, CheckpointKind.NV)
            if inputs[f"{kind}_fetch"] is None
        ]
    ckpt_report = probe(lambda: checkpoint_report(model.MX_MODEL_DIR, model.NV_MODEL_DIR))
    if isinstance(ckpt_report, ProbeError):
        problems.append(f"checkpoint_report: {ckpt_report.to_json()}")
    else:
        for key in ("mx_problems", "nv_problems"):
            found = ckpt_report.get(key)
            if isinstance(found, list):
                problems += found
            else:
                problems.append(f"checkpoint_report has no {key} list")
    if not problems:
        inputs["local_staging"], staging_problems = stage_checkpoints(study, inputs)
        problems += staging_problems
    start_id = StartId(uuid.uuid4().hex)
    append_row(
        run_dir / "manifests.jsonl",
        ManifestLine(
            **manifest,
            study=study.name,
            start_id=start_id,
            inputs={name: error_as_text(value) for name, value in inputs.items()},
            checkpoint_report=error_as_text(ckpt_report),
            protocol=study.to_protocol_dict(),
            problems=problems,
        ),
    )
    commit()
    if problems:
        raise RuntimeError(f"environment check failed: {problems}")
    session_hint = None
    try:
        read_counters()
        prompt_sets = json.loads(Path(settings.M1_PROMPTS_PATH).read_text())
        nll_prompts = json.loads(Path(settings.NLL_PROMPTS_PATH).read_text())
        cell_prompts = load_cell_prompts(study) if study.prompts is Prompts.CELLS else None
        done = load_rows(run_dir / "servers.jsonl", ServerRow)
        for r in sorted(set(rounds_to_run(done, study)) | set(rerun_rounds)):
            for treatment in round_order(r, study.treatments):
                session_hint = {"round": r, "treatment": treatment}
                try:
                    run_server_session(
                        run_dir,
                        r,
                        treatment,
                        study,
                        prompt_sets,
                        nll_prompts,
                        start_id,
                        cell_prompts=cell_prompts,
                    )
                except Exception as exc:
                    if study.kernel_scan is None or treatment not in study.kernel_scan.treatments:
                        raise
                    _record_error(run_dir, session_hint, exc, nonfatal=True)
                    _record_failed_session(run_dir, r, treatment, start_id, exc)
                    print(
                        f"warning: scan treatment {treatment} failed in round {r} and is "
                        f"recorded as failed: {exc!r}",
                        file=sys.stderr,
                    )
                commit()
    except BaseException as exc:
        _record_error(run_dir, session_hint, exc)
        raise
    finally:
        commit()
