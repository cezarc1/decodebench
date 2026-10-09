import copy
import importlib
import json
import math
import sys
import types

import numpy as np
import pytest

import fp4bench.gpu
from fp4bench import bytes_model as bm
from fp4bench import microbench as mb
from fp4bench.core.kernels import ActImpl, BenchKind, GemmImpl
from fp4bench.core.types import Format
from tests import REPO

L2 = 132_644_864
MX, NV = Format.MXFP4, Format.NVFP4


def test_shapes_are_qwen3_32b_linears():
    nk = mb.SHAPE_NK
    assert nk["qkv_proj"] == ((bm.N_Q_HEADS + 2 * bm.N_KV_HEADS) * bm.HEAD_DIM, bm.HIDDEN)
    assert nk["o_proj"] == (bm.HIDDEN, bm.N_Q_HEADS * bm.HEAD_DIM)
    assert nk["gate_up_proj"] == (2 * bm.INTERMEDIATE, bm.HIDDEN)
    assert nk["down_proj"] == (bm.HIDDEN, bm.INTERMEDIATE)
    assert nk["q_proj"] == (8192, 5120)
    assert nk["k_proj=v_proj"] == (1024, 5120)
    assert nk["gate_proj=up_proj"] == (25600, 5120)


def test_each_distinct_shape_is_timed_once():
    assert len(mb.SHAPES) == 7
    assert len({(n, k) for _, n, k, _ in mb.SHAPES}) == 7
    assert [kind for *_, kind in mb.SHAPES].count("merged") == 4


def test_pre_registered_m_values():
    assert mb.M_VALUES == (1, 8, 32, 64, 128, 256, 512)
    assert mb.ACT_QUANT_K == (5120, 8192, 25600)
    assert mb.N_SAMPLES == 7 and mb.MIN_CALLS_PER_SAMPLE == 200 and mb.L2_FACTOR == 2.0


def test_every_shape_meets_cute_dsl_alignment():
    for _, n, k, _ in mb.SHAPES:
        assert n % 8 == 0 and k % 32 == 0


def test_vllm_path_is_mx_tuned_nv_heuristic_as_deployed():
    assert mb.GEMM_IMPLS[GemmImpl.VLLM_CUTEDSL].autotuned == {MX: True, NV: False}
    assert mb.GEMM_IMPLS[GemmImpl.VLLM_CUTEDSL_NV_TUNED].autotuned == {NV: True}


def test_cells_cover_every_impl_format_once():
    declared = {(i, f) for i, spec in mb.GEMM_IMPLS.items() for f in spec.formats}
    assert set(mb.GEMM_CELLS) == declared
    assert len(mb.GEMM_CELLS) == len(declared) == 8


def test_comparisons_pair_an_mx_cell_with_an_nv_cell():
    for _, _, mx, nv in mb.COMPARISONS:
        assert mx[1] == MX and nv[1] == NV
        assert mx in mb.GEMM_CELLS and nv in mb.GEMM_CELLS


def test_pad_up():
    assert [mb.pad_up(x, 128) for x in (1, 128, 129, 512)] == [128, 128, 256, 512]
    assert mb.pad_up(0, 4) == 0


def test_scale_bytes_pad_rows_to_128_and_blocks_to_4():
    assert mb.scale_bytes(NV, 1, 5120) == 128 * 320
    assert mb.scale_bytes(MX, 1, 5120) == 128 * 160
    assert mb.scale_bytes(MX, 200, 8192) == 256 * 256
    assert mb.scale_bytes(NV, 1024, 48) == 1024 * 4


def test_gemm_bytes_gate_up_m1():
    nv = mb.gemm_bytes(NV, 1, 51200, 5120)
    assert nv == {
        "weights": 51200 * 2560,
        "weight_scales": 51200 * 320,
        "acts": 2560,
        "act_scales": 128 * 320,
        "output": 2 * 51200,
        "total": 51200 * 2560 + 51200 * 320 + 2560 + 128 * 320 + 2 * 51200,
    }
    mx = mb.gemm_bytes(MX, 1, 51200, 5120)
    assert mx["weights"] == nv["weights"]
    assert mx["weight_scales"] == nv["weight_scales"] // 2
    assert mx["act_scales"] == 128 * 160


def test_bytes_ratio_is_the_4_5_over_4_25_bits_when_weights_dominate():
    assert mb.bytes_ratio(1, 51200, 5120) == pytest.approx(4.5 / 4.25, abs=1e-3)
    assert mb.bytes_ratio(512, 51200, 5120) < mb.bytes_ratio(1, 51200, 5120)


def test_flops():
    assert mb.gemm_flops(512, 51200, 5120) == 2 * 512 * 51200 * 5120


def test_act_and_silu_bytes():
    assert mb.act_quant_bytes(NV, 32, 5120) == {
        "input": 2 * 32 * 5120,
        "packed": 32 * 2560,
        "scales": 128 * 320,
        "total": 2 * 32 * 5120 + 32 * 2560 + 128 * 320,
    }
    silu = mb.silu_quant_bytes(MX, 8, 25600)
    assert silu["input"] == 2 * 8 * 51200 and silu["packed"] == 8 * 12800
    assert silu["scales"] == 128 * 800


def test_rates():
    r = mb.rates(8_000_000, 2_000_000, 1.0)
    assert r["gbps"] == pytest.approx(8000.0)
    assert r["frac_hbm"] == pytest.approx(1.0)
    assert r["tflops"] == pytest.approx(2.0)


def test_rotation_count_formula():
    assert mb.rotation_count(30, 100) == math.ceil(200 / 30) + 1 == 8


@pytest.mark.parametrize("per_copy", [10_240, 2_800_000, 28_000_000, 140_000_000, 10 * L2])
def test_rotation_leaves_two_l2_between_reuses(per_copy):
    r = mb.rotation_count(per_copy, L2)
    assert r >= 2
    assert (r - 1) * per_copy >= 2 * L2 or r == 2
    assert (r - 2) * per_copy < 2 * L2 or r == 2


def test_rotation_count_is_capped_and_validated():
    assert mb.rotation_count(1, L2) == mb.MAX_COPIES
    with pytest.raises(ValueError, match="must be positive"):
        mb.rotation_count(0, L2)


@pytest.mark.parametrize(
    "copies,expected", [(1, 200), (3, 201), (11, 209), (200, 200), (25_909, 25_909)]
)
def test_calls_per_sample_is_whole_passes_of_at_least_200(copies, expected):
    n = mb.calls_per_sample(copies)
    assert n == expected and n % copies == 0 and n >= 200


def test_aggregate():
    assert mb.aggregate([3.0, 1.0, 2.0, 5.0, 4.0, 7.0, 6.0]) == {
        "median_us": 4.0,
        "min_us": 1.0,
        "max_us": 7.0,
    }


def test_practical_band_is_inclusive():
    assert not mb.practically_different(0.98)
    assert not mb.practically_different(1.02)
    assert mb.practically_different(0.979)
    assert mb.practically_different(1.0588)


def test_make_cell_success_and_failure():
    ok = mb.make_cell(
        mb.GemmCase(GemmImpl.CUDNN, MX, "o_proj", 1, 5120, 8192, 8_000_000, 2_000_000),
        [1.0, 1.0, 2.0],
        rel_err=1e-3,
    )
    assert ok["median_us"] == 1.0 and ok["frac_hbm"] == pytest.approx(1.0)
    assert ok["rel_err"] == 1e-3 and ok["error"] is None
    bad = mb.make_cell(
        mb.GemmCase(GemmImpl.CUDNN, MX, "o_proj", 1, 5120, 8192, 1, 1),
        error="RuntimeError: unsupported",
    )
    assert bad["median_us"] is None and bad["frac_hbm"] is None
    assert bad["error"] == "RuntimeError: unsupported"


def test_an_activation_cell_has_no_n_and_no_flops():
    cell = mb.make_cell(mb.ActCase(ActImpl.NV_QUANT_VLLM, NV, "K=5120", 8, 5120, 99), [1.0])
    assert (cell["kind"], cell["N"], cell["K"], cell["bytes"], cell["flops"]) == (
        "act",
        0,
        5120,
        99,
        0,
    )
    assert cell["tflops"] == 0.0


def test_bytes_regime():
    assert mb.bytes_regime(0.85, 0.81).startswith("both >= 80%")
    assert mb.bytes_regime(0.59, 0.95).startswith("a format < 60%")
    assert mb.bytes_regime(0.7, 0.75).startswith("60-80%")


def test_gemm_plan_runs_all_cells_of_a_shape_and_m_back_to_back():
    plan = mb.gemm_plan(["o_proj", "down_proj"], (1, 8, 32))
    assert len(plan) == 2 * 3 * len(mb.GEMM_CELLS)
    blocks = [plan[i : i + 8] for i in range(0, len(plan), 8)]
    for j, block in enumerate(blocks):
        assert len({(s, m) for s, m, _, _ in block}) == 1
        cells = [(i, f) for _, _, i, f in block]
        assert cells == (list(mb.GEMM_CELLS) if j % 3 % 2 == 0 else list(mb.GEMM_CELLS)[::-1])
    assert [b[0][:2] for b in blocks] == [
        ("o_proj", 1),
        ("o_proj", 8),
        ("o_proj", 32),
        ("down_proj", 1),
        ("down_proj", 8),
        ("down_proj", 32),
    ]


def test_gemm_plan_filters_impls():
    plan = mb.gemm_plan(["qkv_proj"], (1,), impls=[GemmImpl.CUDNN])
    assert [(i, f) for *_, i, f in plan] == [("cudnn", "mxfp4"), ("cudnn", "nvfp4")]


def test_full_plan_size():
    assert len(mb.gemm_plan()) == 7 * 7 * 8


def test_act_plan():
    jobs = mb.act_plan(impls=None)
    assert len(jobs) == 3 * 2 + 4
    assert ("nv_quant_vllm", "nvfp4", "K=25600", 25600, 25600) in jobs
    assert ("silu_mul_nv_fused", "nvfp4", "down_proj_input", 51200, 25600) in jobs
    assert mb.act_plan(impls=[ActImpl.MX_QUANT_CUTEDSL]) == [
        ("mx_quant_cutedsl", "mxfp4", f"K={k}", k, k) for k in mb.ACT_QUANT_K
    ]


def test_impls_split_by_kind():
    impls = (GemmImpl.CUDNN, ActImpl.NV_QUANT_VLLM, ActImpl.SILU_MUL_NV_FUSED)
    assert [i for i in impls if mb.is_gemm_impl(i)] == [GemmImpl.CUDNN]
    assert [i for i in impls if mb.is_act_impl(i)] == [*impls[1:]]


# --- fp4bench.gpu.microbench, over fakes of the stack only the Modal image installs ---------------


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.fixture
def gpu(monkeypatch):
    """fp4bench.gpu.microbench imported over fake torch, FlashInfer and vLLM modules."""
    fakes = [
        _module("torch", Tensor=object, compile=lambda fn, **_: fn),
        _module("torch.autograd", DeviceType=None),
        _module("torch.profiler", ProfilerActivity=None, profile=None),
        _module("flashinfer"),
        _module("flashinfer.autotuner", AutoTuner=None),
        _module("vllm"),
        _module("vllm._custom_ops", scaled_fp4_quant=None),
    ]
    for fake in fakes:
        monkeypatch.setitem(sys.modules, fake.__name__, fake)
    sys.modules.pop("fp4bench.gpu.microbench", None)
    yield importlib.import_module("fp4bench.gpu.microbench")
    sys.modules.pop("fp4bench.gpu.microbench", None)
    fp4bench.gpu.__dict__.pop("microbench", None)


def test_the_gpu_code_imports_vllms_custom_ops_at_its_top(gpu):
    """Importing vllm._custom_ops registers torch.ops._C, which the activation kernels call (MX-C's
    silu_and_mul too, whose path never touches vLLM's Python): an import the code names and uses,
    so that no tool drops it as unused."""
    assert gpu.vllm_ops is sys.modules["vllm._custom_ops"]


@pytest.mark.parametrize("impl", list(ActImpl))
def test_every_activation_kernel_is_built(gpu, impl):
    assert callable(gpu._act_fn(impl, None))


def test_every_gemm_impl_is_flashinfers_or_torchs(gpu):
    torch_gemms = set(gpu.TORCH_GEMMS)
    assert torch_gemms.isdisjoint(mb.FLASHINFER_IMPLS)
    assert torch_gemms | set(mb.FLASHINFER_IMPLS) == set(GemmImpl)


class _Permutable(np.ndarray):
    def permute(self, *axes):
        return self.transpose(axes)


def _vllm_swizzle(scale: np.ndarray) -> np.ndarray:
    """vllm nvfp4_utils.swizzle_blockscale (2-D case) on an already padded array."""
    rows, cols = scale.shape
    return (
        scale.reshape(rows // 128, 4, 32, cols // 4, 4).transpose(0, 3, 2, 1, 4).reshape(rows, cols)
    )


def test_unswizzle_inverts_the_128x4_layout(gpu):
    rows, cols = 256, 12
    logical = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    swizzled = _vllm_swizzle(logical).view(_Permutable)
    assert np.array_equal(np.asarray(gpu._unswizzle_128x4(swizzled)), logical)


def test_swizzled_offset_matches_the_tcgen05_layout():
    rows, cols = 128, 8
    logical = np.arange(rows * cols).reshape(rows, cols)
    tiles = _vllm_swizzle(logical).reshape(rows // 128, cols // 4, 32, 4, 4)
    for r, c in [(0, 0), (31, 3), (32, 0), (127, 7), (65, 5)]:
        assert tiles[r // 128, c // 4, r % 32, (r % 128) // 32, c % 4] == logical[r, c]


def test_e2m1_table():
    assert len(mb.FP4_E2M1_VALUES) == 16
    assert mb.FP4_E2M1_VALUES[7] == 6.0 and mb.FP4_E2M1_VALUES[15] == -6.0
    assert all(mb.FP4_E2M1_VALUES[i + 8] == -mb.FP4_E2M1_VALUES[i] for i in range(8))


# --- the summary ----------------------------------------------------------------------------------


def _cell_at(impl, fmt, shape, m, frac, extras=None):
    n, k = mb.SHAPE_NK[shape]
    nbytes = mb.gemm_bytes(fmt, m, n, k)["total"]
    us = nbytes / (frac * mb.HBM_BYTES_PER_S) * 1e6
    return mb.make_cell(
        mb.GemmCase(impl, fmt, shape, m, n, k, nbytes, mb.gemm_flops(m, n, k)),
        [us] * 7,
        rel_err=1.7e-3,
        extras=extras,
    )


def _choice(tactic: str, *, fallback: bool) -> mb.AutotunerChoice:
    return {
        "op": "fp4_gemm",
        "runner": "CuteDSLFp4GemmRunner",
        "tactic": tactic,
        "fallback": fallback,
    }


def _results(cells) -> mb.MicrobenchResults:
    smi: dict[str, str | None] = {"name": "NVIDIA B200"}
    return {
        "spec": "EXPERIMENT.md §14",
        "env": {
            "gpu": "NVIDIA B200",
            "capability": "10.0",
            "sm_count": 148,
            "l2_bytes": L2,
            "total_memory": 191_000_000_000,
            "cuda": "13.0",
            "cudnn": 91000,
            "versions": {"torch": "2.13.0"},
            "torch_config_mentions_mslk": False,
            "torch_config": "",
            "env": {},
            "python": "3.12.11",
            "nvidia_smi_start": smi,
            "copy_bandwidth_reference": {"tb_per_s": 6.6, "frac_hbm": 0.83},
            "nvidia_smi_end": smi,
        },
        "config": {
            "shapes": ["gate_up_proj", "down_proj"],
            "m_values": [1, 512],
            "impls": None,
            "n_samples": 7,
            "min_calls": 200,
            "l2_factor": 2.0,
            "max_copies": mb.MAX_COPIES,
            "hbm_bytes_per_s": mb.HBM_BYTES_PER_S,
        },
        "cells": cells,
        "tuning": [
            {"impl": GemmImpl.CUDNN, "fmt": MX, "shape": "down_proj", "M": 512, "tune_s": 10.0}
        ],
        "notes": [],
        "wall_s": 400.0,
    }


def test_bandwidth_bound_pairs_and_ratio():
    cells = [
        _cell_at(GemmImpl.VLLM_CUTEDSL, MX, "gate_up_proj", 1, 0.85),
        _cell_at(GemmImpl.VLLM_CUTEDSL, NV, "gate_up_proj", 1, 0.85),
        _cell_at(GemmImpl.CUDNN, MX, "gate_up_proj", 1, 0.5),
        _cell_at(GemmImpl.CUDNN, NV, "gate_up_proj", 1, 0.9),
    ]
    rows = mb.bandwidth_bound_pairs(_results(cells))
    assert [(r.comparison, r.shape, r.m) for r in rows] == [("a", "gate_up_proj", 1)]
    assert rows[0].ratio == pytest.approx(rows[0].bytes_ratio)


def test_summarize_reports_tables_regimes_and_failures():
    cells = [
        _cell_at(
            GemmImpl.VLLM_CUTEDSL,
            MX,
            "gate_up_proj",
            1,
            0.85,
            {
                "autotuner_choice": [_choice("((128, 8),)", fallback=False)],
                "kernels": ["kernel_cutlass_kernel_x" * 20],
            },
        ),
        _cell_at(
            GemmImpl.VLLM_CUTEDSL,
            NV,
            "gate_up_proj",
            1,
            0.85,
            {"autotuner_choice": [_choice("-1", fallback=True)]},
        ),
        _cell_at(GemmImpl.CUDNN, MX, "down_proj", 1, 0.3),
        _cell_at(GemmImpl.CUDNN, NV, "down_proj", 1, 0.4),
        mb.make_cell(
            mb.GemmCase(GemmImpl.TORCH_MSLK_MX, MX, "down_proj", 1, 5120, 25600, 1, 1),
            error="RuntimeError: M=1 unsupported",
        ),
        mb.make_cell(
            mb.ActCase(ActImpl.MX_QUANT_CUTEDSL, MX, "K=5120", 1, 5120, 20_000),
            [2.0] * 7,
            rel_err=0.115,
        ),
        mb.make_cell(
            mb.ActCase(ActImpl.NV_QUANT_VLLM, NV, "K=5120", 1, 5120, 50_000),
            [1.5] * 7,
            rel_err=0.095,
        ),
        mb.make_cell(
            mb.ActCase(ActImpl.SILU_MUL_THEN_MX_QUANT, MX, "down_proj_input", 1, 25600, 1),
            [6.0] * 7,
        ),
        mb.make_cell(
            mb.ActCase(ActImpl.SILU_MUL_NV_FUSED, NV, "down_proj_input", 1, 25600, 1), [3.0] * 7
        ),
    ]
    text = mb.summarize(_results(cells))
    assert "# FP4 GEMM kernel microbenchmark" in text
    assert "both >= 80%: bytes test applies" in text
    assert "a format < 60%: kernel efficiency, not bytes" in text
    assert "| a | gate_up_proj | 1 | 85% | 85% | 1.059 ≠ | 1.059 |" in text
    assert "torch_mslk_mx / mxfp4 / down_proj / M=1: RuntimeError: M=1 unsupported" in text
    assert "| 5120 | 1 | 2.00 [2.00–2.00]" in text and "0.750 ≠" in text
    assert "| 1 | 6.00 [6.00–6.00]" in text and "0.500 ≠" in text
    assert "6.60 TB/s" in text
    assert "| vllm_cutedsl | nvfp4 | False | 1 | 0 |" in text
    assert "…" in text


def test_summarize_with_no_cells():
    text = mb.summarize(_results([]))
    assert "None: no (comparison, shape, M)" in text
    assert "## Failures" in text and "None." in text


def test_the_kernel_tables_are_keyed_by_the_strings_results_json_stores():
    assert list(mb.GEMM_IMPLS) == [i.value for i in GemmImpl]
    assert [i for i, _ in mb.GEMM_CELLS] == [
        *("vllm_cutedsl", "vllm_cutedsl", "vllm_cutedsl_nv_tuned", "cudnn", "cudnn"),
        *("torch_mslk_mx", "torch_cublaslt_nv1", "torch_cublaslt_nv2"),
    ]
    assert [*mb.ACT_IMPLS, *mb.SILU_IMPLS] == [i.value for i in ActImpl]
    assert [j[0] for j in mb.act_plan(impls=None)][:2] == ["mx_quant_cutedsl", "nv_quant_vllm"]
    cell = mb.make_cell(mb.ActCase(ActImpl.SILU_MUL_NV_FUSED, NV, "x", 8, 1, 2))
    assert json.loads(json.dumps(cell))["impl"] == "silu_mul_nv_fused" and cell["kind"] == "act"


@pytest.mark.parametrize(
    "path",
    [
        "data/kernels/microbench-1/results.json",
        "data/kernels/microbench-1/replicate/results.json",
    ],
)
def test_the_summary_of_a_run_in_memory_is_the_summary_of_its_results_json(path):
    """kernel_microbench summarizes the cells run() returns, whose kinds and impls are members."""
    loaded = json.loads((REPO / path).read_text())
    in_memory = copy.deepcopy(loaded)
    for c in in_memory["cells"]:
        c["kind"] = BenchKind(c["kind"])
        c["impl"] = (GemmImpl if c["kind"] is BenchKind.GEMM else ActImpl)(c["impl"])
        c["fmt"] = Format(c["fmt"])
    assert mb.summarize(in_memory) == mb.summarize(loaded)
    assert json.dumps(in_memory, indent=1) == json.dumps(loaded, indent=1)
