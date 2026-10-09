# METHODOLOGY.md#profiler
import gzip
import json
import math
import os
import re
import shutil
import statistics
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import NamedTuple

from fp4bench import settings
from fp4bench.core.files import write_text_atomic
from fp4bench.core.kernels import CategoryPattern, KernelCategory, KernelGroup, LinearLayer
from fp4bench.core.types import Treatment, error_as_object, probe
from fp4bench.studies import model
from fp4bench.studies.expb import EXPB
from fp4bench.studies.main import FULL

STEPS_RECORDED = 32
WARMUP_TOKENS = 64
PROFILED_TOKENS = 256
REFERENCE_TOKENS = (64, 256)


@dataclass(frozen=True)
class ProfileServer:
    """One `vllm serve` start, profiler attached; each C in `concurrencies` is profiled on it."""

    config: str
    treatment: Treatment
    concurrencies: tuple[int, ...]
    delay_iterations: int
    max_iterations: int = STEPS_RECORDED


CONFIG_STUDIES = {"A": FULL, "B": EXPB}
DEFAULT_SESSIONS = (
    ProfileServer("A", Treatment.MX, (1, 32), 16),
    ProfileServer("A", Treatment.NV, (1, 32), 16),
    ProfileServer("A", Treatment.NVNF, (1, 32), 16),
    ProfileServer("B", Treatment.MX, (512,), 48),
    ProfileServer("B", Treatment.NV, (512,), 48),
)


def profiler_config(trace_dir, delay: int, max_iters: int) -> str:
    """vLLM v0.31.0's --profiler-config for a window of STEPS_RECORDED steps after `delay`."""
    if not Path(trace_dir).is_absolute():
        raise ValueError(f"torch_profiler_dir must be absolute, got {trace_dir!r}")
    return json.dumps(
        {
            "profiler": "torch",
            "torch_profiler_dir": str(trace_dir),
            "ignore_frontend": True,
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": False,
            "delay_iterations": delay,
            "max_iterations": max_iters,
        }
    )


def server_args(spec: ProfileServer, trace_dir: Path) -> tuple[str, ...]:
    """What the runner serves, plus --profiler-config."""
    return (
        *CONFIG_STUDIES[spec.config].server.args(),
        *model.TREATMENTS[spec.treatment].server_args,
        "--profiler-config",
        profiler_config(trace_dir, spec.delay_iterations, spec.max_iterations),
    )


def session_name(config: str, treatment: str, c: int) -> str:
    return f"{config}_{treatment}_c{c}"


TRACE_SUFFIX = ".pt.trace.json.gz"
TRACE_GLOB = f"*{TRACE_SUFFIX}"


def trace_name(config: str, treatment: str, c: int) -> str:
    return session_name(config, treatment, c) + TRACE_SUFFIX


# METHODOLOGY.md#profiler: which kernels each pattern catches
CATEGORY_PATTERNS: tuple[CategoryPattern, ...] = (
    CategoryPattern(KernelCategory.ACT_QUANT, r"cvt_fp16_to_fp4|MXFP4Quantize|mxfp4_quantize"),
    CategoryPattern(KernelCategory.FP4_GEMM, r"BlockScaled|blockscaled_gemm"),
    CategoryPattern(KernelCategory.SILU_MUL, r"silu"),
    CategoryPattern(
        KernelCategory.ATTENTION, r"^fmha|reshape_and_cache|FillFunctor<unsigned char>"
    ),
    CategoryPattern(KernelCategory.BF16_GEMM, r"^nvjet|splitKreduce"),
    CategoryPattern(KernelCategory.NORM_ROPE_ELEMENTWISE, r"^triton_"),
    CategoryPattern(KernelCategory.SAMPLING, r"gumbel_sample|ArgMaxOps"),
)
LAYER_CATEGORIES: tuple[KernelCategory, ...] = (KernelCategory.FP4_GEMM, KernelCategory.ACT_QUANT)


class TreatmentPair(NamedTuple):
    """Two treatments, compared as a − b."""

    a: Treatment
    b: Treatment


PAIRS = (
    TreatmentPair(Treatment.NV, Treatment.MX),
    TreatmentPair(Treatment.NVNF, Treatment.MX),
    TreatmentPair(Treatment.NV, Treatment.NVNF),
)


class _CompiledPattern(NamedTuple):
    category: KernelCategory
    regex: re.Pattern[str]


@cache
def _compiled(patterns: tuple[CategoryPattern, ...]) -> tuple[_CompiledPattern, ...]:
    return tuple(_CompiledPattern(p.category, re.compile(p.regex)) for p in patterns)


def categorize(name: str, patterns: tuple[CategoryPattern, ...] | None = None) -> KernelCategory:
    for category, rx in _compiled(patterns or CATEGORY_PATTERNS):
        if rx.search(name):
            return category
    return KernelCategory.OTHER


KERNEL_CATS = {"kernel"}
MEMCPY_CATS = {"gpu_memcpy", "memcpy"}
MEMSET_CATS = {"gpu_memset", "memset"}
RUNTIME_CATS = {"cuda_runtime", "cuda_driver"}
GRAPH_LAUNCH_RE = re.compile(r"^cu(?:da)?GraphLaunch")
STEP_ANNOTATION_RE = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$")
FUSED_ACT_QUANT_RE = re.compile(r"silu_mul_cvt_fp16_to_fp4")


def load_trace(path) -> dict:
    """A Chrome-trace JSON, gzipped (as vLLM's torch profiler writes it) or plain."""
    path = Path(path)
    with path.open("rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    opener = gzip.open if gz else open
    with opener(path, "rt") as f:
        return json.load(f)


def _complete_events(trace: dict):
    for e in trace.get("traceEvents", []):
        if e.get("ph") == "X" and "ts" in e and "dur" in e:
            yield e


def gpu_events(trace: dict) -> dict[str, list[dict]]:
    """Kernels, memcpys and memsets (complete events); the category is compared lowercased."""
    out = {"kernel": [], "memcpy": [], "memset": []}
    for e in _complete_events(trace):
        cat = str(e.get("cat", "")).lower()
        if cat in KERNEL_CATS:
            out["kernel"].append(e)
        elif cat in MEMCPY_CATS:
            out["memcpy"].append(e)
        elif cat in MEMSET_CATS:
            out["memset"].append(e)
    return out


def union_us(intervals) -> float:
    """Total length of the union of [start, end) intervals."""
    total, cur_start, cur_end = 0.0, None, None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def kernel_table(
    kernels: list[dict], patterns: tuple[CategoryPattern, ...] | None = None
) -> list[dict]:
    """Per kernel name: category, count, total, mean and max µs; by total, descending."""
    durs: dict[str, list[float]] = defaultdict(list)
    for k in kernels:
        durs[k["name"]].append(float(k["dur"]))
    rows = [
        {
            "name": name,
            "category": categorize(name, patterns),
            "count": len(d),
            "total_us": math.fsum(d),
            "mean_us": math.fsum(d) / len(d),
            "max_us": max(d),
        }
        for name, d in durs.items()
    ]
    return sorted(rows, key=lambda r: (-r["total_us"], r["name"]))


def _stats(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def launch_groups(trace: dict, kernels: list[dict]) -> tuple[int, list[list[dict]], list[dict]]:
    """(graph launches, each launch's kernels in time order, eager kernels), by correlation id."""
    launches = {
        e["args"]["correlation"]
        for e in _complete_events(trace)
        if str(e.get("cat", "")).lower() in RUNTIME_CATS
        and GRAPH_LAUNCH_RE.match(str(e.get("name", "")))
        and "correlation" in e.get("args", {})
    }
    by_launch: dict[int, list[dict]] = defaultdict(list)
    eager = []
    for k in kernels:
        corr = k.get("args", {}).get("correlation")
        (by_launch[corr] if corr in launches else eager).append(k)
    groups = sorted(
        (sorted(g, key=lambda k: k["ts"]) for g in by_launch.values()), key=lambda g: g[0]["ts"]
    )
    return len(launches), groups, eager


def _span(group: list[dict]) -> float:
    return max(k["ts"] + k["dur"] for k in group) - min(k["ts"] for k in group)


def _replay_stats(n_launches: int, groups: list[list[dict]], eager: list[dict]) -> dict:
    counts = [len(g) for g in groups]
    spans = [_span(g) for g in groups]
    idle = [
        span - union_us((k["ts"], k["ts"] + k["dur"]) for k in g)
        for span, g in zip(spans, groups, strict=True)
    ]
    return {
        "launches": n_launches,
        "launches_with_kernels": len(groups),
        "kernels_per_launch": {"min": min(counts), "max": max(counts)} if counts else None,
        "gpu_span_us_per_launch": _stats(spans),
        "kernel_us_per_launch": _stats([math.fsum(k["dur"] for k in g) for g in groups]),
        "idle_us_per_launch": _stats(idle),
        "gpu_span_us_total": math.fsum(spans),
        "idle_us_total": math.fsum(idle),
        "graph_kernel_count": sum(counts),
        "graph_kernel_us": math.fsum(k["dur"] for g in groups for k in g),
        "eager_kernel_count": len(eager),
        "eager_kernel_us": math.fsum(k["dur"] for k in eager),
        "eager_kernel_names": sorted({k["name"] for k in eager}),
    }


LINEAR_LAYERS = tuple(LinearLayer)


def by_linear_layer(
    groups: list[list[dict]],
    category: KernelCategory,
    patterns: tuple[CategoryPattern, ...] | None = None,
) -> dict | None:
    """µs per step of `category`'s kernels per linear layer; None unless each replay has 4k."""
    seqs = [[k["dur"] for k in g if categorize(k["name"], patterns) == category] for g in groups]
    lengths = {len(q) for q in seqs}
    if not seqs or len(lengths) != 1 or not (n := lengths.pop()) or n % len(LINEAR_LAYERS):
        return None
    out = {}
    for i, layer in enumerate(LINEAR_LAYERS):
        per_replay = [math.fsum(q[i :: len(LINEAR_LAYERS)]) for q in seqs]
        out[layer] = {
            "us_per_step": math.fsum(per_replay) / len(per_replay),
            "mean_us_per_call": math.fsum(per_replay) / len(per_replay) / (n // len(LINEAR_LAYERS)),
            "calls_per_step": n // len(LINEAR_LAYERS),
        }
    return out


def step_annotations(trace: dict) -> dict:
    """The engine steps the annotations describe: count, prefill and decode requests, GPU time."""
    cpu, gpu = [], []
    for e in _complete_events(trace):
        m = STEP_ANNOTATION_RE.match(str(e.get("name", "")))
        if not m:
            continue
        cat = str(e.get("cat", "")).lower()
        if cat == "user_annotation":
            cpu.append(tuple(int(x) for x in m.groups()))
        elif cat == "gpu_user_annotation":
            gpu.append(float(e["dur"]))
    return {
        "steps": len(cpu),
        "context_requests": sorted({c[0] for c in cpu}),
        "context_tokens": sorted({c[1] for c in cpu}),
        "generation_requests": sorted({c[2] for c in cpu}),
        "gpu_us_per_step": _stats(gpu),
    }


def summarize_trace(
    trace: dict,
    n_steps: int = STEPS_RECORDED,
    patterns: tuple[CategoryPattern, ...] | None = None,
    top: int = 25,
) -> dict:
    """Per-step GPU time of one profiled window, by kernel and by category."""
    patterns = patterns or CATEGORY_PATTERNS
    ev = gpu_events(trace)
    kernels = ev["kernel"]
    if not kernels:
        raise ValueError("no GPU kernel events in the trace (CUPTI kernel activity missing?)")
    table = kernel_table(kernels, patterns)
    kernel_us = math.fsum(k["dur"] for k in kernels)
    categories = {c: {"count": 0, "total_us": 0.0} for c in KernelCategory}
    max_kernel_us = dict.fromkeys(KernelCategory, 0.0)
    for row in table:
        cat = categories[row["category"]]
        cat["count"] += row["count"]
        cat["total_us"] += row["total_us"]
        max_kernel_us[row["category"]] = max(max_kernel_us[row["category"]], row["max_us"])
    for cat in categories.values():
        cat["us_per_step"] = cat["total_us"] / n_steps
        cat["count_per_step"] = cat["count"] / n_steps
        cat["share_of_kernel_time"] = cat["total_us"] / kernel_us
    start = min(k["ts"] for k in kernels)
    end = max(k["ts"] + k["dur"] for k in kernels)
    busy = union_us((k["ts"], k["ts"] + k["dur"]) for k in kernels)
    n_launches, groups, eager = launch_groups(trace, kernels)
    graphs = _replay_stats(n_launches, groups, eager)
    annotations = Counter(
        e["name"]
        for e in _complete_events(trace)
        if str(e.get("cat", "")).lower() == "user_annotation"
    )
    per_launch = graphs["kernels_per_launch"]
    steps = step_annotations(trace)
    for row in table:
        row["us_per_step"] = row["total_us"] / n_steps
    constant = bool(per_launch and per_launch["min"] == per_launch["max"])
    fused = sum(row["count"] for row in table if FUSED_ACT_QUANT_RE.search(row["name"]))
    return {
        "n_steps": n_steps,
        "kernel_count": len(kernels),
        "kernel_us": kernel_us,
        "kernel_us_per_step": kernel_us / n_steps,
        "kernels_per_step": len(kernels) / n_steps,
        "categories": categories,
        "max_kernel_us": max_kernel_us,
        "span_us": end - start,
        "span_us_per_step": (end - start) / n_steps,
        "busy_us": busy,
        "busy_us_per_step": busy / n_steps,
        "idle_us_per_step": ((end - start) - busy) / n_steps,
        "idle_fraction": 1 - busy / (end - start) if end > start else 0.0,
        "memcpy": {
            "count": len(ev["memcpy"]),
            "total_us": math.fsum(e["dur"] for e in ev["memcpy"]),
        },
        "memset": {
            "count": len(ev["memset"]),
            "total_us": math.fsum(e["dur"] for e in ev["memset"]),
        },
        "graph_replays": graphs,
        "graph_span_us_per_step": graphs["gpu_span_us_total"] / n_steps,
        "idle_in_graph_us_per_step": graphs["idle_us_total"] / n_steps,
        "by_linear_layer": {
            cat: by_linear_layer(groups, cat, patterns) for cat in LAYER_CATEGORIES
        },
        "annotation_counts": dict(annotations.most_common(50)),
        "step_annotations": steps,
        "decode_check": {
            "recorded_steps": n_steps,
            "derived_steps": graphs["launches_with_kernels"],
            "steps_match": graphs["launches_with_kernels"] == n_steps,
            "kernels_per_launch_constant": constant,
            "annotated_steps": steps["steps"],
            "context_requests": steps["context_requests"],
            "generation_requests": steps["generation_requests"],
            "pure_decode": (
                graphs["launches_with_kernels"] == n_steps
                and constant
                and steps["context_requests"] in ([], [0])
                and steps["steps"] in (0, n_steps)
            ),
        },
        "fusion": {
            "fused_silu_mul_quant_per_step": fused / n_steps,
            "silu_mul_per_step": categories[KernelCategory.SILU_MUL]["count_per_step"],
        },
        "top_kernels": table[:top],
        "kernels": table,
    }


def _per_step(summary: dict, cat: str) -> float:
    return summary["categories"][cat]["us_per_step"]


def _layer_deltas(sa: dict, sb: dict) -> dict:
    out = {}
    for cat in LAYER_CATEGORIES:
        la = (sa.get("by_linear_layer") or {}).get(cat)
        lb = (sb.get("by_linear_layer") or {}).get(cat)
        if la and lb:
            out[cat] = {
                layer: la[layer]["us_per_step"] - lb[layer]["us_per_step"]
                for layer in LINEAR_LAYERS
            }
    return out


def pairwise_deltas(
    sessions: dict[str, dict],
    config: str = "A",
    concurrencies: tuple[int, ...] = (1, 32),
    pairs: tuple[TreatmentPair, ...] = PAIRS,
) -> list[dict]:
    """a − b in µs per step, per category, kernel group and total, for every pair that ran."""
    rows = []
    for c in concurrencies:
        for a, b in pairs:
            sa = sessions.get(session_name(config, a, c))
            sb = sessions.get(session_name(config, b, c))
            if sa is None or sb is None:
                continue
            cats = {cat: _per_step(sa, cat) - _per_step(sb, cat) for cat in KernelCategory}
            rows.append(
                {
                    "config": config,
                    "c": c,
                    "a": a,
                    "b": b,
                    "categories": cats,
                    "groups": {g: math.fsum(cats[cat] for cat in g.members) for g in KernelGroup},
                    **{
                        key: sa[key] - sb[key]
                        for key in (
                            "kernel_us_per_step",
                            "busy_us_per_step",
                            "idle_us_per_step",
                            "span_us_per_step",
                        )
                    },
                    **{
                        key: sa.get(key, 0.0) - sb.get(key, 0.0)
                        for key in ("idle_in_graph_us_per_step", "graph_span_us_per_step")
                    },
                    "by_linear_layer": _layer_deltas(sa, sb),
                }
            )
    return rows


def gemm_share(
    sessions: dict[str, dict],
    config: str = "B",
    c: int = 512,
    treatments=(Treatment.MX, Treatment.NV),
) -> dict:
    """§13's s: the FP4 GEMM share of the step, against the GPU span and the summed kernel time."""
    out = {}
    for t in treatments:
        s = sessions.get(session_name(config, t, c))
        if s is None:
            continue
        gemm, quant = _per_step(s, KernelCategory.FP4_GEMM), _per_step(s, KernelCategory.ACT_QUANT)
        out[t] = {
            "fp4_gemm_us_per_step": gemm,
            "span_us_per_step": s["span_us_per_step"],
            "kernel_us_per_step": s["kernel_us_per_step"],
            "s_span": gemm / s["span_us_per_step"],
            "s_kernel": gemm / s["kernel_us_per_step"],
            "act_quant_share_span": quant / s["span_us_per_step"],
        }
    return out


SESSIONS_FILE = "sessions.json"


def write_report(
    out_dir, patterns: tuple[CategoryPattern, ...] | None = None, top: int = 25
) -> dict:
    """Summarize every trace sessions.json lists into profiles.json and summary.md."""
    out_dir = Path(out_dir)
    patterns = patterns or CATEGORY_PATTERNS
    meta = json.loads((out_dir / SESSIONS_FILE).read_text())
    sessions, summaries = {}, {}
    for server in meta.get("servers", []):
        for rec in server.get("concurrencies", []):
            if not rec.get("trace_file") or not (out_dir / rec["trace_file"]).exists():
                continue
            name = session_name(server["config"], server["treatment"], rec["c"])
            summary = summarize_trace(
                load_trace(out_dir / rec["trace_file"]),
                n_steps=rec["max_iterations"],
                patterns=patterns,
                top=top,
            )
            summaries[name] = summary
            sessions[name] = {
                "config": server["config"],
                "treatment": server["treatment"],
                **{k: v for k, v in rec.items() if k != "c"},
                "c": rec["c"],
                "summary": summary,
            }
    report = {
        "spec": "EXPERIMENT.md §15",
        "generated_utc": datetime.now(UTC).isoformat(),
        "category_patterns": [list(p) for p in patterns],
        "categories": list(KernelCategory),
        "groups": {g: list(g.members) for g in KernelGroup},
        "environment": meta.get("environment"),
        "staging": meta.get("staging"),
        "run": meta.get("run"),
        "servers": [
            {k: v for k, v in s.items() if k != "concurrencies"}
            | {
                "sessions": [
                    session_name(s["config"], s["treatment"], r["c"])
                    for r in s.get("concurrencies", [])
                ]
            }
            for s in meta.get("servers", [])
        ],
        "sessions": sessions,
        "comparisons": {
            "pairwise": (
                pairwise_deltas(summaries)
                + pairwise_deltas(
                    summaries,
                    config="B",
                    concurrencies=(512,),
                    pairs=(TreatmentPair(Treatment.NV, Treatment.MX),),
                )
            ),
            "gemm_share_c512": gemm_share(summaries),
        },
        "errors": meta.get("errors", []),
    }
    (out_dir / "profiles.json").write_text(json.dumps(report, indent=1))
    (out_dir / "summary.md").write_text(render_summary_md(report))
    return report


def _f(x, digits=1) -> str:
    return "–" if x is None else f"{x:,.{digits}f}"


def _decode_check_row(name: str, sm: dict) -> str:
    g = sm["graph_replays"]
    kpl = g["kernels_per_launch"]
    kpl_text = "–" if kpl is None else f"{kpl['min']}–{kpl['max']}"
    span = g["gpu_span_us_per_launch"]
    span_text = "–" if span is None else "/".join(_f(span[k]) for k in ("min", "median", "max"))
    dc, fu = sm["decode_check"], sm["fusion"]
    return (
        f"| {name} | {dc['pure_decode']} | {dc['annotated_steps']} | "
        f"{dc['context_requests']} | {dc['generation_requests']} | "
        f"{g['launches']} ({g['launches_with_kernels']}) | {kpl_text} | "
        f"{span_text} | {g['eager_kernel_count']} ({_f(g['eager_kernel_us'])} µs) | "
        f"{_f(sm['max_kernel_us']['fp4_gemm'])} | {_f(fu['fused_silu_mul_quant_per_step'])} "
        f"| {_f(fu['silu_mul_per_step'])} | {sm['memcpy']['count']} "
        f"({_f(sm['memcpy']['total_us'])} µs) | {sm['memset']['count']} "
        f"({_f(sm['memset']['total_us'])} µs) |"
    )


def render_summary_md(report: dict) -> str:
    sessions = report["sessions"]
    lines = [
        "# Profiling sessions (EXPERIMENT.md §15)",
        "",
        "GPU kernel time per decode step (µs/step) from vLLM's torch profiler; "
        "`span` = first kernel start to last kernel end over the recorded steps, "
        "`busy` = union of kernel intervals, `idle` = span − busy (idle % = 1 − busy/span), "
        "`idle in graph` = the part of idle between the kernels of one CUDA-graph replay "
        "(the rest is between steps: eager-launch and CPU-scheduling gaps, which the "
        "profiler itself inflates), `graph span` = the GPU time of the forward pass's "
        "graph replay (mean per step; the eager lm_head and sampling come after it). "
        "The unprofiled step is (t₂₅₆ − t₆₄)/192 from one pair "
        "of unprofiled waves on the same server (M1-style, ms; a single pair, so a rough "
        "reference: M1 is the measurement).",
        "",
    ]
    head = [
        "session",
        "steps rec/derived",
        *KernelCategory,
        "Σ kernel",
        "busy",
        "idle",
        "idle in graph",
        "graph span",
        "span",
        "idle %",
        "kernels/step",
        "unprofiled step (ms)",
    ]
    lines += [
        "## Per session (µs/step)",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for name, s in sessions.items():
        sm = s["summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{sm['n_steps']}/{sm['decode_check']['derived_steps']}",
                    *(_f(sm["categories"][c]["us_per_step"]) for c in KernelCategory),
                    _f(sm["kernel_us_per_step"]),
                    _f(sm["busy_us_per_step"]),
                    _f(sm["idle_us_per_step"]),
                    _f(sm.get("idle_in_graph_us_per_step")),
                    _f(sm.get("graph_span_us_per_step")),
                    _f(sm["span_us_per_step"]),
                    f"{sm['idle_fraction']:.1%}",
                    _f(sm["kernels_per_step"]),
                    _f(s.get("unprofiled_step_ms"), 3),
                ]
            )
            + " |"
        )
    pair_head = [
        "config",
        "pair",
        "C",
        *KernelCategory,
        "Σ kernel",
        "idle",
        "idle in graph",
        "graph span",
        "span",
        "fp4_gemm",
        "act_quant+silu_mul",
        "everything else",
    ]
    lines += [
        "",
        "## Pairwise differences (µs/step, a − b)",
        "",
        "The last three columns split Σ kernel into the §15 groups. span − idle ≈ Σ kernel "
        "(they differ by kernel overlap).",
        "",
        "| " + " | ".join(pair_head) + " |",
        "|" + "---|" * len(pair_head),
    ]
    for row in report["comparisons"]["pairwise"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["config"],
                    f"{row['a']} − {row['b']}",
                    str(row["c"]),
                    *(_f(row["categories"][c]) for c in KernelCategory),
                    _f(row["kernel_us_per_step"]),
                    _f(row["idle_us_per_step"]),
                    _f(row.get("idle_in_graph_us_per_step")),
                    _f(row.get("graph_span_us_per_step")),
                    _f(row["span_us_per_step"]),
                    *(_f(row["groups"][g]) for g in KernelGroup),
                ]
            )
            + " |"
        )
    layer_head = [
        "config",
        "pair",
        "C",
        *(f"{cat} {layer}" for cat in LAYER_CATEGORIES for layer in LINEAR_LAYERS),
    ]
    lines += [
        "",
        "### The same differences per linear layer (µs/step, a − b)",
        "",
        "FP4 GEMM and activation quantization by the linear layer they feed (position in "
        "the graph replay). NV's down_proj act_quant is the fused SiLU-mul + quant; MX's "
        "and NV-nf's SiLU-mul is not in these columns (silu_mul above).",
        "",
        "| " + " | ".join(layer_head) + " |",
        "|" + "---|" * len(layer_head),
    ]
    for row in report["comparisons"]["pairwise"]:
        layers = row.get("by_linear_layer") or {}
        cells = [
            _f(layers[cat][layer]) if cat in layers else "–"
            for cat in LAYER_CATEGORIES
            for layer in LINEAR_LAYERS
        ]
        lines.append(
            f"| {row['config']} | {row['a']} − {row['b']} | {row['c']} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "## FP4 GEMM share s at C = 512 (§13)",
        "",
        "| treatment | fp4_gemm µs/step | span µs/step | Σ kernel µs/step | s (span) | "
        "s (Σ kernel) | act_quant share (span) |",
        "|---|---|---|---|---|---|---|",
    ]
    for t, g in report["comparisons"]["gemm_share_c512"].items():
        lines.append(
            f"| {t} | {_f(g['fp4_gemm_us_per_step'])} | {_f(g['span_us_per_step'])} | "
            f"{_f(g['kernel_us_per_step'])} | {g['s_span']:.3f} | "
            f"{g['s_kernel']:.3f} | {g['act_quant_share_span']:.3f} |"
        )
    layer_head = [
        "session",
        *(f"{cat} {layer}" for cat in LAYER_CATEGORIES for layer in LINEAR_LAYERS),
    ]
    lines += [
        "",
        "## FP4 GEMM and activation quantization per linear layer (µs/step)",
        "",
        "| " + " | ".join(layer_head) + " |",
        "|" + "---|" * len(layer_head),
    ]
    for name, s in sessions.items():
        layers = s["summary"].get("by_linear_layer") or {}
        cells = [
            _f(layers[cat][layer]["us_per_step"]) if layers.get(cat) else "–"
            for cat in LAYER_CATEGORIES
            for layer in LINEAR_LAYERS
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Decode-window checks",
        "",
        "One full CUDA-graph replay per decode step: the graph launches with kernels "
        "should equal the recorded steps, with the same kernel count each; vLLM's step "
        "annotations (`execute_context_<reqs>(<tokens>)_generation_<reqs>(<tokens>)`) give "
        "the prefill (context) and decode (generation) requests of every recorded step. "
        "`fused/step` counts `silu_mul_cvt_fp16_to_fp4` kernels (the SiLU-mul + quant "
        "fusion), `silu/step` the unfused SiLU-mul kernels.",
        "",
        "| session | pure decode | annotated steps | context reqs | generation reqs | "
        "graph launches (with kernels) | kernels/launch | GPU span/launch µs "
        "(min/median/max) | eager kernels | max fp4_gemm kernel µs | fused/step | "
        "silu/step | memcpy | memset |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [_decode_check_row(name, s["summary"]) for name, s in sessions.items()]
    lines += [
        "",
        "## Servers",
        "",
        "| config | treatment | linear kernels | fuse_act_quant | KV cache tokens | checks |",
        "|---|---|---|---|---|---|",
    ]
    for s in report["servers"]:
        facts = s.get("facts") or {}
        lines.append(
            f"| {s['config']} | {s['treatment']} | "
            f"{', '.join(facts.get('linear_kernels', [])) or '–'} | "
            f"{facts.get('fuse_act_quant')} | {facts.get('kv_cache_tokens')} | "
            f"{s.get('checks')} |"
        )
    lines += [
        "",
        "## Category patterns (ordered, first match wins)",
        "",
        "| category | regex |",
        "|---|---|",
    ]
    patterns = [CategoryPattern(KernelCategory(c), rx) for c, rx in report["category_patterns"]]
    lines += [f"| {p.category} | `{p.regex}` |" for p in patterns]
    lines += ["| other | (no match) |"]
    for name, s in sessions.items():
        sm = s["summary"]
        lines += [
            "",
            f"## Top kernels: {name}",
            "",
            "| # | kernel | category | count | total µs | µs/step | mean µs |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, k in enumerate(sm["top_kernels"], 1):
            kname = k["name"] if len(k["name"]) <= 110 else k["name"][:107] + "..."
            kname = kname.replace("|", "\\|")
            lines.append(
                f"| {i} | `{kname}` | {k['category']} | {k['count']} | "
                f"{_f(k['total_us'])} | {_f(k['us_per_step'], 2)} | "
                f"{_f(k['mean_us'], 2)} |"
            )
    if report.get("errors"):
        lines += ["", "## Errors", ""] + [
            f"- {e.get('where')}: {e.get('error')}" for e in report["errors"]
        ]
    return "\n".join(lines) + "\n"


def trace_files(trace_dir) -> set[Path]:
    return set(Path(trace_dir).glob(TRACE_GLOB))


def _gzip_complete(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 24):
                pass
    except (EOFError, OSError, zlib.error):
        return False
    return True


def wait_for_new_trace(
    trace_dir, known: set[Path], timeout_s: float = 900, poll_s: float = 2.0
) -> Path:
    """The one new trace file in `trace_dir`, once its size is stable and its gzip whole."""
    deadline = time.monotonic() + timeout_s
    last_size = None
    while True:
        new = sorted(trace_files(trace_dir) - set(known))
        if len(new) > 1:
            raise RuntimeError(
                f"{len(new)} new trace files in {trace_dir} (expected one, TP=1): "
                f"{[p.name for p in new]}"
            )
        if new:
            size = new[0].stat().st_size
            if size and size == last_size and _gzip_complete(new[0]):
                return new[0]
            last_size = size
        if time.monotonic() > deadline:
            raise TimeoutError(f"no complete new trace file in {trace_dir} after {timeout_s} s")
        time.sleep(poll_s)


def post(path: str, timeout_s: float = 900) -> dict:
    """POST to the server; the status and body are returned, an HTTP error status included."""
    req = urllib.request.Request(f"{settings.BASE_URL}{path}", data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return {"status": resp.status, "body": resp.read().decode(errors="replace")[:2000]}
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "body": exc.read().decode(errors="replace")[:2000]}


def _wave(prompts: list[list[int]], max_tokens: int) -> float:
    from fp4bench.wave import run_wave

    return run_wave(settings.BASE_URL, settings.SERVED_NAME, prompts, max_tokens)


def _copy_profiler_table(
    trace_dir: Path, dest: Path, since: float, timeout_s: float, poll_s: float
) -> str | None:
    deadline = time.monotonic() + timeout_s
    while True:
        fresh = [
            p
            for p in Path(trace_dir).glob("profiler_out_*.txt")
            if p.stat().st_mtime >= since and p.stat().st_size > 0
        ]
        if fresh:
            time.sleep(poll_s)
            shutil.copyfile(fresh[0], dest)
            return dest.name
        if time.monotonic() > deadline:
            return None
        time.sleep(poll_s)


def profile_concurrency(
    spec: ProfileServer,
    c: int,
    prompt_sets: list,
    trace_dir: Path,
    out_dir: Path,
    wave: Callable[[list, int], float] | None = None,
    post: Callable[[str], dict] = post,
    trace_timeout_s: float = 900,
    poll_s: float = 2.0,
    table_timeout_s: float = 0,
) -> dict:
    """One C on a running server: warmup, reference waves, the profiled wave and its trace."""
    wave = wave or _wave
    warm, prompts = prompt_sets[0][:c], prompt_sets[1][:c]
    if len(warm) != c or len(prompts) != c:
        raise ValueError(f"prompt sets 0 and 1 need {c} prompts")
    waves = {f"warmup_{WARMUP_TOKENS}": wave(warm, WARMUP_TOKENS)}
    for n in REFERENCE_TOKENS:
        waves[f"reference_{n}"] = wave(prompts, n)
    known = trace_files(trace_dir)
    since = time.time()
    start = post("/start_profile")
    if start["status"] != 200:
        raise RuntimeError(f"/start_profile returned {start}")
    waves[f"profiled_{PROFILED_TOKENS}"] = wave(prompts, PROFILED_TOKENS)
    stop = post("/stop_profile")
    found = wait_for_new_trace(trace_dir, known, timeout_s=trace_timeout_s, poll_s=poll_s)
    name = trace_name(spec.config, spec.treatment, c)
    shutil.move(str(found), out_dir / name)
    table = _copy_profiler_table(
        trace_dir,
        out_dir / (session_name(spec.config, spec.treatment, c) + ".profiler_out.txt"),
        since,
        table_timeout_s,
        poll_s,
    )
    lo, hi = REFERENCE_TOKENS
    return {
        "c": c,
        "delay_iterations": spec.delay_iterations,
        "max_iterations": spec.max_iterations,
        "trace_file": name,
        "trace_source_name": found.name,
        "trace_bytes": (out_dir / name).stat().st_size,
        "profiler_table_file": table,
        "start_profile": start,
        "stop_profile": stop,
        "waves_s": waves,
        "unprofiled_step_ms": 1000
        * (waves[f"reference_{hi}"] - waves[f"reference_{lo}"])
        / (hi - lo),
    }


PROFILER_LINE_RE = re.compile(r"[^\r\n]*[Pp]rofil[^\r\n]*")


def server_checks(spec: ProfileServer, facts: dict) -> dict:
    """The G1-style facts that make a profile interpretable: the expected FP4 kernel and fusion."""
    return {
        "linear_kernel_ok": facts.get("linear_kernels")
        == [model.TREATMENTS[spec.treatment].linear_kernel],
        "fuse_act_quant_ok": facts.get("fuse_act_quant")
        is model.TREATMENTS[spec.treatment].act_quant_fusion,
    }


def profile_server(spec: ProfileServer, out_dir: Path, prompt_sets: list, work_dir: Path) -> dict:
    """Start one server with the profiler attached, profile each C on it, stop it."""
    from fp4bench.runner import fresh_autotune_dir
    from fp4bench.server import AUTOTUNE_CACHE_ENV, VllmServer, parse_server_log

    session_id = f"{spec.config}_{spec.treatment}-{uuid.uuid4().hex[:8]}"
    trace_dir = Path(work_dir) / "traces" / session_id
    trace_dir.mkdir(parents=True)
    autotune_dir = fresh_autotune_dir(out_dir, session_id)
    env = {AUTOTUNE_CACHE_ENV: str(autotune_dir)}
    model_dir = model.TREATMENTS[spec.treatment].model_dir
    record = {
        **asdict(spec),
        "session_id": session_id,
        "model_dir": model_dir,
        "server_env": env,
        "trace_dir": str(trace_dir),
        "started_utc": datetime.now(UTC).isoformat(),
        "concurrencies": [],
    }
    log = out_dir / "servers" / f"{session_id}.log"
    t0 = time.monotonic()
    with VllmServer(model_dir, log, args=server_args(spec, trace_dir), env=env) as server:
        record["server_argv"] = list(server.cmd)
        record["startup_s"] = time.monotonic() - t0
        for c in spec.concurrencies:
            proc = server.proc
            if proc is None or proc.poll() is not None:
                code = None if proc is None else proc.returncode
                raise RuntimeError(f"vllm serve exited with {code}; see {log}")
            record["concurrencies"].append(
                profile_concurrency(spec, c, prompt_sets, trace_dir, out_dir, table_timeout_s=30)
            )
        text = server.log_text()
    facts = parse_server_log(text)
    record["facts"] = facts
    record["checks"] = server_checks(spec, facts)
    record["profiler_log_lines"] = sorted(set(PROFILER_LINE_RE.findall(text)))[:100]
    record["server_s"] = time.monotonic() - t0
    return record


def environment() -> dict:
    from fp4bench.manifest import cuda_version, gpu_info, package_versions

    gpu = probe(gpu_info)
    return {
        "utc": datetime.now(UTC).isoformat(),
        "image": settings.VLLM_IMAGE,
        "vllm_build_commit": os.environ.get("VLLM_BUILD_COMMIT"),
        "gpu": error_as_object(gpu),
        "packages": package_versions(),
        "cuda": cuda_version(),
    }


def stage_models(treatments) -> dict:
    """Stage each checkpoint the sessions serve (runner.stage_checkpoint)."""
    from fp4bench.prep import FETCH_FILE
    from fp4bench.runner import stage_checkpoint
    from fp4bench.sanity import content_fingerprint

    out = {}
    for t in sorted(set(treatments)):
        volume_dir, local_dir = model.TREATMENTS[t].volume_dir, model.TREATMENTS[t].model_dir
        if volume_dir in out:
            continue
        record = stage_checkpoint(volume_dir, local_dir, content_fingerprint(volume_dir))
        if record["local_content_fingerprint"] != record["volume_content_fingerprint"]:
            raise RuntimeError(f"local copy of {volume_dir} differs from the volume: {record}")
        fetch = Path(volume_dir) / FETCH_FILE
        record["fetch"] = json.loads(fetch.read_text()) if fetch.exists() else None
        out[volume_dir] = record
    return out


def run_profiles(
    out_dir: Path,
    sessions=DEFAULT_SESSIONS,
    commit: Callable[[], None] | None = None,
    work_dir: Path = Path(settings.LOCAL_DIR) / "profile",
) -> dict:
    """Every §15 session in this container, then the report; a failed server raises at the end."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = commit or (lambda: None)
    started = time.monotonic()
    meta = {
        "run": {
            "started_utc": datetime.now(UTC).isoformat(),
            "sessions": [asdict(s) for s in sessions],
        },
        "environment": environment(),
        "servers": [],
        "errors": [],
    }
    t0 = time.monotonic()
    meta["staging"] = stage_models(s.treatment for s in sessions)
    meta["run"]["staging_s"] = time.monotonic() - t0
    write_text_atomic(out_dir / SESSIONS_FILE, json.dumps(meta, indent=1))
    prompt_sets = json.loads(Path(settings.M1_PROMPTS_PATH).read_text())
    for spec in sessions:
        try:
            meta["servers"].append(profile_server(spec, out_dir, prompt_sets, work_dir))
        except Exception as exc:
            meta["errors"].append(
                {
                    "where": f"{spec.config}_{spec.treatment}",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"profile server {spec} failed: {exc!r}", file=sys.stderr)
        meta["run"]["elapsed_s"] = time.monotonic() - started
        write_text_atomic(out_dir / SESSIONS_FILE, json.dumps(meta, indent=1))
        commit()
    report = write_report(out_dir)
    meta["run"]["elapsed_s"] = time.monotonic() - started
    write_text_atomic(out_dir / SESSIONS_FILE, json.dumps(meta, indent=1))
    commit()
    if meta["errors"]:
        raise RuntimeError(
            f"{len(meta['errors'])} profile server(s) failed: "
            f"{[e['where'] + ': ' + e['error'] for e in meta['errors']]}"
        )
    return report


def main(argv: list[str]) -> None:
    if len(argv) != 2 or argv[0] != "report":
        raise SystemExit("usage: python -m fp4bench.profiling report <out_dir>")
    report = write_report(Path(argv[1]))
    print(f"wrote profiles.json and summary.md for {len(report['sessions'])} sessions")


if __name__ == "__main__":
    main(sys.argv[1:])
