import gzip
import json
import threading
import time
from pathlib import Path

import pytest

from fp4bench import profiling as prof
from fp4bench.core.kernels import CategoryPattern, KernelCategory, KernelGroup
from fp4bench.core.types import Treatment
from fp4bench.studies import model
from fp4bench.studies.expb import EXPB
from fp4bench.studies.main import FULL

PATTERNS = tuple(
    CategoryPattern(KernelCategory(category), regex)
    for category, regex in (
        ("act_quant", r"cvt_fp16_to_fp4|quantize"),
        ("fp4_gemm", r"mm_fp4|BlockScaled"),
        ("silu_mul", r"silu"),
        ("attention", r"fmha"),
        ("bf16_gemm", r"nvjet"),
        ("norm_rope_elementwise", r"^triton_"),
        ("sampling", r"argmax"),
    )
)


def _x(cat, name, ts, dur, corr=None, **extra):
    args = {} if corr is None else {"correlation": corr}
    return {
        "ph": "X",
        "cat": cat,
        "name": name,
        "pid": 0,
        "tid": 7,
        "ts": ts,
        "dur": dur,
        "args": {**args, **extra},
    }


def synthetic_trace() -> dict:
    """Two decode steps of graph replays and eager kernels, with a gap and an overlap."""
    ev = [
        {"ph": "M", "name": "process_name", "pid": 0, "args": {"name": "GPU 0"}},
        _x("cuda_runtime", "cudaGraphLaunch", 1.0, 2.0, corr=100),
        _x("cuda_runtime", "cudaLaunchKernel", 3.0, 1.0, corr=101),
        _x("cuda_runtime", "cudaLaunchKernel", 4.0, 1.0, corr=102),
        _x("cuda_runtime", "cudaGraphLaunch", 101.0, 2.0, corr=200),
        _x("cuda_runtime", "cudaLaunchKernel", 103.0, 1.0, corr=201),
        _x("cuda_runtime", "cudaLaunchKernel", 104.0, 1.0, corr=202),
        _x("user_annotation", "execute_context_0(0)_generation_2(2)", 0.5, 50.0),
        _x("user_annotation", "execute_context_0(0)_generation_2(2)", 100.5, 50.0),
        _x("gpu_user_annotation", "execute_context_0(0)_generation_2(2)", 10.0, 42.0),
        _x("gpu_user_annotation", "execute_context_0(0)_generation_2(2)", 110.0, 50.0),
        {"ph": "s", "cat": "ac2g", "id": 100, "pid": 1, "tid": 1, "ts": 1.0, "name": "ac2g"},
        _x("kernel", "mm_fp4_kernel", 10.0, 20.0, corr=100),
        _x("kernel", "silu_mul_cvt_fp16_to_fp4", 30.0, 4.0, corr=100),
        _x("kernel", "fmhaSm100Kernel", 34.0, 8.0, corr=100),
        _x("kernel", "triton_red_fused_rms_norm_0", 42.0, 2.0, corr=100),
        _x("kernel", "nvjet_hsh_lm_head", 44.0, 6.0, corr=101),
        _x("kernel", "argmax_kernel", 50.0, 2.0, corr=102),
        _x("gpu_memcpy", "Memcpy HtoD (Pinned -> Device)", 5.0, 1.0, corr=103),
        _x("gpu_memset", "Memset (Device)", 6.0, 0.5, corr=104),
        _x("kernel", "mm_fp4_kernel", 110.0, 22.0, corr=200),
        _x("kernel", "silu_mul_cvt_fp16_to_fp4", 132.0, 4.0, corr=200),
        _x("kernel", "fmhaSm100Kernel", 136.0, 8.0, corr=200),
        _x("kernel", "triton_red_fused_rms_norm_0", 143.0, 3.0, corr=200),
        _x("Kernel", "nvjet_hsh_lm_head", 152.0, 6.0, corr=201),
        _x("kernel", "argmax_kernel", 158.0, 2.0, corr=202),
    ]
    return {"schemaVersion": 1, "traceEvents": ev}


def test_categorize_first_match_wins():
    assert prof.categorize("silu_mul_cvt_fp16_to_fp4", PATTERNS) == "act_quant"
    assert prof.categorize("triton_poi_fused_silu_mul_0", PATTERNS) == "silu_mul"
    assert prof.categorize("something_unknown", PATTERNS) == "other"


def test_the_default_patterns_name_every_category_but_other_which_comes_last():
    cats = [p.category for p in prof.CATEGORY_PATTERNS]
    assert sorted(cats) == sorted(set(KernelCategory) - {KernelCategory.OTHER})
    assert list(KernelCategory)[-1] is KernelCategory.OTHER


def test_the_patterns_are_written_as_arrays_and_read_back_as_category_patterns():
    stored = json.loads(json.dumps([list(p) for p in prof.CATEGORY_PATTERNS]))
    assert json.dumps(list(prof.CATEGORY_PATTERNS)) == json.dumps(stored)
    assert stored[0] == ["act_quant", "cvt_fp16_to_fp4|MXFP4Quantize|mxfp4_quantize"]
    read = tuple(CategoryPattern(KernelCategory(c), rx) for c, rx in stored)
    assert read == prof.CATEGORY_PATTERNS and hash(read) == hash(prof.CATEGORY_PATTERNS)


GEMM = (
    "kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaled"
    "PersistentDenseGemmKernel_object_at__TiledMMA_ThrLayoutVMNK11110000_PermutationMNK____"
    "MMAAtom_ThrID1_0"
)
MXQ = (
    "kernel_cutlass_kernel_flashinferquantizationkernelsmxfp4_quantizeMXFP4QuantizeSwizzled"
    "Kernel_object_at__tensorptrbf16gmemalign16o512051201_tensorptri8gmemalign16o256025601_"
    "tensorptri8gmem_0"
)


@pytest.mark.parametrize(
    "name, category",
    [
        (GEMM, "fp4_gemm"),
        (MXQ, "act_quant"),
        (
            "void vllm::silu_mul_cvt_fp16_to_fp4<__nv_bfloat16, false>(int, int, int, "
            "__nv_bfloat16 const*, float const*, unsigned int*, unsigned int*)",
            "act_quant",
        ),
        (
            "void vllm::cvt_fp16_to_fp4<__nv_bfloat16, false, false>(int, int, int, int, "
            "__nv_bfloat16 const*, float const*, unsigned int*, unsigned int*)",
            "act_quant",
        ),
        ("triton_poi_fused_mul_silu_slice_2", "silu_mul"),
        (
            "fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP16MultiCtasKvCgaVarSeqQ8Kv256"
            "StaticGroupedSwapsAbForGen",
            "attention",
        ),
        (
            "void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, "
            "(vllm::Fp8KVCacheDataType)0>(__nv_bfloat16 const*, __nv_bfloat16 const*)",
            "attention",
        ),
        (
            "void at::native::vectorized_elementwise_kernel<8, "
            "at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, "
            "at::native::FillFunctor<unsigned char>, std::array<char*, 1ul>)",
            "attention",
        ),
        ("nvjet_sm100_tst_192x8_64x8_2x1_v_bz_splitK_TNT", "bf16_gemm"),
        (
            "void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float>("
            "cublasLt::cublasSplitKParams<float>)",
            "bf16_gemm",
        ),
        ("triton_red_fused_fused_add_rms_norm_scaled_fp4_quant_0", "norm_rope_elementwise"),
        (
            "triton_red_fused__to_copy_embedding_rms_norm_scaled_fp4_quant_0",
            "norm_rope_elementwise",
        ),
        ("triton_red_fused_fused_add_rms_norm_3", "norm_rope_elementwise"),
        ("triton_poi_fused_0", "norm_rope_elementwise"),
        ("_gumbel_sample_kernel", "sampling"),
        (
            "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, "
            "at::native::ArgMaxOps<float>, unsigned int, long, 4, 4> >("
            "at::native::ReduceOp<float>)",
            "sampling",
        ),
        ("_combine_sampled_and_draft_tokens_kernel", "other"),
        ("_post_update_kernel", "other"),
        ("memcpy32_post", "other"),
        (
            "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, "
            "std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)",
            "other",
        ),
    ],
)
def test_default_patterns_on_known_kernel_names(name, category):
    assert prof.categorize(name) == category


def test_load_trace_reads_gzip_and_plain(tmp_path):
    trace = synthetic_trace()
    gz = tmp_path / "a.pt.trace.json.gz"
    with gzip.open(gz, "wt") as f:
        json.dump(trace, f)
    plain = tmp_path / "b.json"
    plain.write_text(json.dumps(trace))
    assert prof.load_trace(gz) == trace
    assert prof.load_trace(plain) == trace


def test_gpu_events_split_kernels_memcpy_memset():
    ev = prof.gpu_events(synthetic_trace())
    assert len(ev["kernel"]) == 12
    assert [e["name"] for e in ev["memcpy"]] == ["Memcpy HtoD (Pinned -> Device)"]
    assert [e["name"] for e in ev["memset"]] == ["Memset (Device)"]


def test_union_us_merges_overlaps_and_keeps_gaps():
    assert prof.union_us([(0, 10), (5, 12), (20, 25)]) == 17
    assert prof.union_us([]) == 0
    assert prof.union_us([(3, 4), (0, 1)]) == 2


def test_kernel_table_aggregates_and_sorts():
    table = prof.kernel_table(prof.gpu_events(synthetic_trace())["kernel"], PATTERNS)
    first = table[0]
    assert first["name"] == "mm_fp4_kernel" and first["category"] == "fp4_gemm"
    assert first["count"] == 2 and first["total_us"] == 42.0 and first["mean_us"] == 21.0
    assert [row["total_us"] for row in table] == sorted(
        (r["total_us"] for r in table), reverse=True
    )
    nvjet = next(r for r in table if r["name"] == "nvjet_hsh_lm_head")
    assert nvjet["count"] == 2 and nvjet["category"] == "bf16_gemm"


def test_summarize_trace_per_step_categories_span_and_idle():
    s = prof.summarize_trace(synthetic_trace(), n_steps=2, patterns=PATTERNS, top=3)
    assert s["n_steps"] == 2
    assert s["kernel_count"] == 12
    total = 42 + 8 + 16 + 5 + 12 + 4
    assert s["kernel_us"] == pytest.approx(total)
    assert s["kernel_us_per_step"] == pytest.approx(total / 2)
    cats = s["categories"]
    assert list(cats) == list(KernelCategory)
    assert cats["fp4_gemm"]["us_per_step"] == pytest.approx(21.0)
    assert cats["act_quant"]["us_per_step"] == pytest.approx(4.0)
    assert cats["silu_mul"]["us_per_step"] == 0
    assert cats["bf16_gemm"]["count"] == 2
    assert cats["sampling"]["us_per_step"] == pytest.approx(2.0)
    assert cats["other"]["total_us"] == 0
    assert sum(c["share_of_kernel_time"] for c in cats.values()) == pytest.approx(1.0)
    assert s["span_us"] == pytest.approx(150.0)
    assert s["span_us_per_step"] == pytest.approx(75.0)
    assert s["busy_us"] == pytest.approx(86.0)
    assert s["busy_us_per_step"] == pytest.approx(43.0)
    assert s["idle_fraction"] == pytest.approx(1 - 86 / 150)
    assert s["memcpy"] == {"count": 1, "total_us": 1.0}
    assert s["memset"] == {"count": 1, "total_us": 0.5}
    assert [k["name"] for k in s["top_kernels"]] == [
        "mm_fp4_kernel",
        "fmhaSm100Kernel",
        "nvjet_hsh_lm_head",
    ]
    assert len(s["kernels"]) == 6


def test_summarize_trace_graph_replays_give_derived_steps():
    s = prof.summarize_trace(synthetic_trace(), n_steps=2, patterns=PATTERNS)
    g = s["graph_replays"]
    assert g["launches"] == 2
    assert g["launches_with_kernels"] == 2
    assert g["kernels_per_launch"] == {"min": 4, "max": 4}
    assert g["graph_kernel_count"] == 8
    assert g["eager_kernel_count"] == 4
    assert g["eager_kernel_us"] == pytest.approx(16.0)
    assert g["gpu_span_us_per_launch"]["min"] == pytest.approx(34.0)
    assert g["gpu_span_us_per_launch"]["max"] == pytest.approx(36.0)
    assert s["max_kernel_us"]["fp4_gemm"] == pytest.approx(22.0)
    check = s["decode_check"]
    assert check["derived_steps"] == 2
    assert check["steps_match"] is True
    assert check["kernels_per_launch_constant"] is True
    assert check["annotated_steps"] == 2
    assert check["context_requests"] == [0] and check["generation_requests"] == [2]
    assert check["pure_decode"] is True
    assert s["step_annotations"]["gpu_us_per_step"] == {"min": 42.0, "median": 46.0, "max": 50.0}
    assert s["annotation_counts"]["execute_context_0(0)_generation_2(2)"] == 2
    assert s["fusion"] == {"fused_silu_mul_quant_per_step": 1.0, "silu_mul_per_step": 0.0}
    assert s["kernels_per_step"] == 6
    assert s["idle_us_per_step"] == pytest.approx((150 - 86) / 2)


def _replay_trace(gemm_us_per_layer, layers=2, gap=1.0):
    """Two graph replays of `layers` decoder layers, `gap` µs between kernels."""
    ev = []
    for step, corr in enumerate((500, 600)):
        ev.append(_x("cuda_runtime", "cudaGraphLaunch", step * 1000.0, 1.0, corr=corr))
        t = step * 1000.0 + 10
        for _ in range(layers):
            for gemm_us in gemm_us_per_layer:
                for name, dur in (("quantize_x", 1.0), ("mm_fp4_x", gemm_us)):
                    ev.append(_x("kernel", name, t, dur, corr=corr))
                    t += dur + gap
    return {"traceEvents": ev}


def test_by_linear_layer_splits_gemm_and_quant_by_position():
    trace = _replay_trace((2.0, 3.0, 10.0, 8.0))
    s = prof.summarize_trace(trace, n_steps=2, patterns=PATTERNS)
    gemm = s["by_linear_layer"]["fp4_gemm"]
    assert list(gemm) == list(prof.LINEAR_LAYERS)
    assert gemm["gate_up_proj"] == {
        "us_per_step": 20.0,
        "mean_us_per_call": 10.0,
        "calls_per_step": 2,
    }
    assert gemm["qkv_proj"]["us_per_step"] == 4.0
    assert s["by_linear_layer"]["act_quant"]["down_proj"]["us_per_step"] == 2.0
    assert s["graph_replays"]["idle_us_per_launch"]["median"] == pytest.approx(15.0)
    assert s["idle_in_graph_us_per_step"] == pytest.approx(15.0)
    assert s["graph_span_us_per_step"] == pytest.approx(8 + 46 + 15)


def test_by_linear_layer_is_none_when_the_count_does_not_fit():
    trace = _replay_trace((2.0, 3.0, 10.0))
    assert (
        prof.summarize_trace(trace, n_steps=2, patterns=PATTERNS)["by_linear_layer"]["fp4_gemm"]
        is None
    )


def test_summarize_trace_flags_a_prefill_step_by_its_annotation():
    trace = synthetic_trace()
    for e in trace["traceEvents"]:
        if e.get("cat") == "user_annotation":
            e["name"] = "execute_context_1(1024)_generation_1(1)"
            break
    check = prof.summarize_trace(trace, n_steps=2, patterns=PATTERNS)["decode_check"]
    assert check["context_requests"] == [0, 1]
    assert check["pure_decode"] is False


def test_summarize_trace_flags_a_window_that_is_not_pure_decode():
    trace = synthetic_trace()
    trace["traceEvents"] += [
        _x("cuda_runtime", "cudaGraphLaunch", 201.0, 2.0, corr=300),
        _x("kernel", "mm_fp4_kernel", 210.0, 900.0, corr=300),
    ]
    s = prof.summarize_trace(trace, n_steps=2, patterns=PATTERNS)
    assert s["decode_check"]["derived_steps"] == 3
    assert s["decode_check"]["steps_match"] is False
    assert s["decode_check"]["kernels_per_launch_constant"] is False
    assert s["decode_check"]["pure_decode"] is False


def test_summarize_trace_without_kernels_raises():
    with pytest.raises(ValueError, match="no GPU kernel events"):
        prof.summarize_trace({"traceEvents": []}, n_steps=32, patterns=PATTERNS)


def _summary(**per_step):
    """A summary with the given category us/step, span and busy per step (n_steps 1)."""
    cats = {c: {"us_per_step": float(per_step.get(c, 0.0))} for c in KernelCategory}
    kernel = sum(v["us_per_step"] for v in cats.values())
    span = per_step.get("span", kernel)
    return {
        "categories": cats,
        "kernel_us_per_step": kernel,
        "span_us_per_step": span,
        "busy_us_per_step": kernel,
        "idle_us_per_step": span - kernel,
    }


def test_pairwise_deltas_split_into_gemm_act_path_and_rest():
    sessions = {
        "A_MX_c1": _summary(fp4_gemm=100, act_quant=10, silu_mul=5, attention=20, span=200),
        "A_NV_c1": _summary(fp4_gemm=90, act_quant=12, attention=20, span=180),
        "A_NVnf_c1": _summary(fp4_gemm=91, act_quant=14, silu_mul=5, attention=21, span=190),
    }
    rows = prof.pairwise_deltas(sessions, config="A", concurrencies=(1, 32))
    by_pair = {(r["a"], r["b"], r["c"]): r for r in rows}
    assert set(by_pair) == {("NV", "MX", 1), ("NVnf", "MX", 1), ("NV", "NVnf", 1)}
    nv_mx = by_pair[("NV", "MX", 1)]
    assert nv_mx["categories"]["fp4_gemm"] == pytest.approx(-10)
    assert nv_mx["categories"]["silu_mul"] == pytest.approx(-5)
    assert nv_mx["groups"] == pytest.approx(
        {"fp4_gemm": -10, "act_quant_silu_mul": -3, "everything_else": 0}
    )
    assert nv_mx["kernel_us_per_step"] == pytest.approx(-13)
    assert sum(nv_mx["groups"].values()) == pytest.approx(nv_mx["kernel_us_per_step"])
    assert nv_mx["span_us_per_step"] == pytest.approx(-20)
    assert nv_mx["idle_us_per_step"] == pytest.approx(-7)
    nv_nf = by_pair[("NV", "NVnf", 1)]
    assert nv_nf["groups"]["act_quant_silu_mul"] == pytest.approx(-7)
    assert nv_nf["groups"]["everything_else"] == pytest.approx(-1)


def test_pairwise_deltas_per_linear_layer():
    def layers(*us):
        return {
            "fp4_gemm": {
                name: {"us_per_step": u} for name, u in zip(prof.LINEAR_LAYERS, us, strict=True)
            },
            "act_quant": None,
        }

    mx = {**_summary(fp4_gemm=100), "by_linear_layer": layers(10, 20, 40, 30)}
    nv = {**_summary(fp4_gemm=95), "by_linear_layer": layers(10, 19, 38, 28)}
    row = prof.pairwise_deltas({"A_MX_c1": mx, "A_NV_c1": nv}, concurrencies=(1,))[0]
    assert row["by_linear_layer"] == {
        "fp4_gemm": {"qkv_proj": 0, "o_proj": -1, "gate_up_proj": -2, "down_proj": -2}
    }


def test_gemm_share_at_c512():
    sessions = {
        "B_MX_c512": _summary(fp4_gemm=5000, act_quant=500, attention=20000, span=28000),
        "B_NV_c512": _summary(fp4_gemm=4500, act_quant=600, attention=20000, span=27000),
    }
    share = prof.gemm_share(sessions, config="B", c=512)
    assert share["MX"]["s_span"] == pytest.approx(5000 / 28000)
    assert share["MX"]["s_kernel"] == pytest.approx(5000 / 25500)
    assert share["NV"]["act_quant_share_span"] == pytest.approx(600 / 27000)
    assert prof.gemm_share({}, config="B", c=512) == {}


def test_default_sessions_follow_the_preregistration():
    a = [s for s in prof.DEFAULT_SESSIONS if s.config == "A"]
    b = [s for s in prof.DEFAULT_SESSIONS if s.config == "B"]
    assert [s.treatment for s in a] == ["MX", "NV", "NVnf"]
    assert all(
        s.concurrencies == (1, 32) and s.delay_iterations == 16 and s.max_iterations == 32
        for s in a
    )
    assert [s.treatment for s in b] == ["MX", "NV"]
    assert all(
        s.concurrencies == (512,) and s.delay_iterations == 48 and s.max_iterations == 32 for s in b
    )
    assert prof.CONFIG_STUDIES == {"A": FULL, "B": EXPB}


def test_profiler_config_json():
    conf = json.loads(prof.profiler_config("/local/prof/x", delay=16, max_iters=32))
    assert conf == {
        "profiler": "torch",
        "torch_profiler_dir": "/local/prof/x",
        "ignore_frontend": True,
        "torch_profiler_with_stack": False,
        "torch_profiler_record_shapes": False,
        "delay_iterations": 16,
        "max_iterations": 32,
    }
    with pytest.raises(ValueError, match="absolute"):
        prof.profiler_config("relative/dir", delay=16, max_iters=32)


@pytest.mark.parametrize(
    "spec, study",
    [
        (prof.ProfileServer("A", Treatment.NVNF, (1, 32), 16), FULL),
        (prof.ProfileServer("B", Treatment.NV, (512,), 48), EXPB),
    ],
)
def test_server_args_are_the_studys_plus_treatment_plus_profiler(spec, study):
    args = prof.server_args(spec, Path("/local/prof/t"))
    common = study.server.args()
    pin = model.TREATMENTS[spec.treatment].server_args
    assert args[: len(common)] == common
    assert args[len(common) : len(common) + len(pin)] == pin
    assert args[-2] == "--profiler-config"
    conf = json.loads(args[-1])
    assert conf["delay_iterations"] == spec.delay_iterations
    assert conf["max_iterations"] == spec.max_iterations
    assert conf["torch_profiler_dir"] == "/local/prof/t"


def test_expb_server_args_have_fp8_kv_and_095():
    args = prof.server_args(prof.ProfileServer("B", Treatment.MX, (512,), 48), Path("/p"))
    assert args[args.index("--kv-cache-dtype") + 1] == "fp8"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.95"


def test_session_name():
    assert prof.session_name("A", "NVnf", 32) == "A_NVnf_c32"
    assert prof.trace_name("B", "MX", 512) == "B_MX_c512.pt.trace.json.gz"


def _write_gz(path: Path, obj) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(obj, f)


def test_wait_for_new_trace_ignores_known_files_and_waits_for_completion(tmp_path):
    old = tmp_path / "rank0.1.pt.trace.json.gz"
    _write_gz(old, {"traceEvents": []})
    known = prof.trace_files(tmp_path)
    new = tmp_path / "rank0.2.pt.trace.json.gz"

    def writer():
        time.sleep(0.2)
        data = gzip.compress(json.dumps(synthetic_trace()).encode())
        with new.open("wb") as f:
            f.write(data[:10])
            f.flush()
            time.sleep(0.3)
            f.write(data[10:])

    t = threading.Thread(target=writer)
    t.start()
    found = prof.wait_for_new_trace(tmp_path, known, timeout_s=10, poll_s=0.05)
    t.join()
    assert found == new
    assert prof.load_trace(found) == synthetic_trace()


def test_wait_for_new_trace_times_out(tmp_path):
    with pytest.raises(TimeoutError):
        prof.wait_for_new_trace(tmp_path, set(), timeout_s=0.2, poll_s=0.05)


def test_wait_for_new_trace_refuses_two_new_files(tmp_path):
    _write_gz(tmp_path / "a.pt.trace.json.gz", {})
    _write_gz(tmp_path / "b.pt.trace.json.gz", {})
    with pytest.raises(RuntimeError, match="2 new trace files"):
        prof.wait_for_new_trace(tmp_path, set(), timeout_s=1, poll_s=0.05)


def test_profile_concurrency_sequence(tmp_path):
    trace_dir, out_dir = tmp_path / "traces", tmp_path / "out"
    trace_dir.mkdir()
    out_dir.mkdir()
    calls = []
    prompt_sets = [[[0] * 4 for _ in range(8)], [[1] * 4 for _ in range(8)]]

    def wave(prompts, n):
        calls.append(("wave", prompts[0][0], len(prompts), n))
        if calls[-2:-1] == [("post", "/start_profile")]:
            _write_gz(trace_dir / "dp0_rank0.123.pt.trace.json.gz", synthetic_trace())
        return 1.0 if n == 64 else 2.0

    def post(path):
        calls.append(("post", path))
        return {"status": 200, "body": ""}

    spec = prof.ProfileServer("A", Treatment.MX, (2,), 16)
    rec = prof.profile_concurrency(
        spec,
        2,
        prompt_sets,
        trace_dir,
        out_dir,
        wave=wave,
        post=post,
        trace_timeout_s=5,
        poll_s=0.05,
    )
    assert calls == [
        ("wave", 0, 2, 64),
        ("wave", 1, 2, 64),
        ("wave", 1, 2, 256),
        ("post", "/start_profile"),
        ("wave", 1, 2, 256),
        ("post", "/stop_profile"),
    ]
    assert rec["trace_file"] == "A_MX_c2.pt.trace.json.gz"
    assert (out_dir / "A_MX_c2.pt.trace.json.gz").exists()
    assert not list(trace_dir.glob("*.pt.trace.json.gz"))
    assert rec["waves_s"] == {
        "warmup_64": 1.0,
        "reference_64": 1.0,
        "reference_256": 2.0,
        "profiled_256": 2.0,
    }
    assert rec["unprofiled_step_ms"] == pytest.approx(1000 * (2.0 - 1.0) / 192)
    assert rec["start_profile"]["status"] == 200


def test_write_report_from_traces_and_sessions_file(tmp_path):
    _write_gz(tmp_path / "A_MX_c1.pt.trace.json.gz", synthetic_trace())
    _write_gz(tmp_path / "A_NV_c1.pt.trace.json.gz", synthetic_trace())
    meta = {
        "environment": {"gpu": {"name": "NVIDIA B200"}},
        "servers": [
            {
                "config": "A",
                "treatment": t,
                "server_argv": ["vllm", "serve"],
                "facts": {"linear_kernels": ["K"]},
                "concurrencies": [
                    {
                        "c": 1,
                        "trace_file": f"A_{t}_c1.pt.trace.json.gz",
                        "delay_iterations": 16,
                        "max_iterations": 2,
                        "unprofiled_step_ms": 0.1,
                    }
                ],
            }
            for t in ("MX", "NV")
        ],
    }
    (tmp_path / "sessions.json").write_text(json.dumps(meta))
    report = prof.write_report(tmp_path, patterns=PATTERNS)
    assert set(report["sessions"]) == {"A_MX_c1", "A_NV_c1"}
    assert report["sessions"]["A_MX_c1"]["summary"]["n_steps"] == 2
    assert report["category_patterns"] == [list(p) for p in PATTERNS]
    assert report["environment"]["gpu"]["name"] == "NVIDIA B200"
    saved = json.loads((tmp_path / "profiles.json").read_text())
    assert saved["sessions"].keys() == report["sessions"].keys()
    md = (tmp_path / "summary.md").read_text()
    assert "A_MX_c1" in md and "fp4_gemm" in md and "NV − MX" in md


def test_the_profiler_groups_partition_the_categories():
    assert {g: g.members for g in KernelGroup} == {
        "fp4_gemm": ("fp4_gemm",),
        "act_quant_silu_mul": ("act_quant", "silu_mul"),
        "everything_else": ("attention", "bf16_gemm", "norm_rope_elementwise", "sampling", "other"),
    }
    assert [g.value for g in KernelGroup] == ["fp4_gemm", "act_quant_silu_mul", "everything_else"]
    assert sorted(c for g in KernelGroup for c in g.members) == sorted(KernelCategory)
    assert prof.LAYER_CATEGORIES == (KernelCategory.FP4_GEMM, KernelCategory.ACT_QUANT)
    assert prof.PAIRS == (("NV", "MX"), ("NVnf", "MX"), ("NV", "NVnf"))
    assert prof.LINEAR_LAYERS == ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
    assert prof.categorize("some_unknown_kernel") == "other"
