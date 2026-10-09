import json
import posixpath
from types import MappingProxyType

import pytest

from fp4bench import settings
from fp4bench.core.types import (
    ActQuantFusion,
    CheckpointKind,
    LinearBackend,
    LinearKernel,
    Treatment,
)
from fp4bench.studies import model
from fp4bench.studies.kernel_scan import NVA_SCAN, NVX_CROSSCHECK
from fp4bench.studies.main import FULL
from fp4bench.studies.smoke import SMOKE, SMOKE_ONLY_TREATMENTS

T = Treatment
SPECS = model.TREATMENTS
CUTE_DSL_NV = "FlashInferCuteDslNvFp4LinearKernel"
PIN = ("--linear-backend", "flashinfer_cutedsl")


def test_the_source_and_nvidia_checkpoints_are_pinned():
    assert model.SRC_MODEL_ID == "Qwen/Qwen3-32B"
    assert model.SRC_MODEL_REVISION == "9216db5781bf21249d130ec9da846c4624c16137"
    assert model.NVX_MODEL_ID == "nvidia/Qwen3-32B-NVFP4"
    assert model.NVX_MODEL_REVISION == "16426c6eb87be9e27c14cc9fb318f9c7a5f8588c"


def test_volume_dirs():
    assert model.BF16_MODEL_DIR == "/data/qwen3-32b-bf16"
    assert model.MX_MODEL_DIR == "/data/qwen3-32b-mxfp4"
    assert model.NV_MODEL_DIR == "/data/qwen3-32b-nvfp4"
    assert model.NVX_MODEL_DIR == "/data/nvidia-qwen3-32b-nvfp4"
    nv = model.NV_MODEL_DIR
    assert {t: s.volume_dir for t, s in SPECS.items()} == {
        "MX": model.MX_MODEL_DIR,
        "MXp": model.MX_MODEL_DIR,
        "NV": nv,
        "NVa": nv,
        "NVnf": nv,
        "NVx": model.NVX_MODEL_DIR,
        "NVc": nv,
        "NVt": nv,
        "NVd": nv,
        "NVv": nv,
    }


def test_each_checkpoint_kind_has_its_volume_dir():
    assert {kind: model.volume_dir(kind) for kind in CheckpointKind} == {
        CheckpointKind.MX: "/data/qwen3-32b-mxfp4",
        CheckpointKind.NV: "/data/qwen3-32b-nvfp4",
        CheckpointKind.NVX: "/data/nvidia-qwen3-32b-nvfp4",
    }


def test_every_treatment_has_a_spec_and_the_table_is_read_only():
    assert set(SPECS) == set(Treatment) == {*FULL.treatments, *SMOKE_ONLY_TREATMENTS}
    assert isinstance(SPECS, MappingProxyType)


def test_treatments_serve_from_local_disk_not_the_volume():
    nv_local = "/local/qwen3-32b-nvfp4"
    assert {t: s.model_dir for t, s in SPECS.items()} == {
        "MX": "/local/qwen3-32b-mxfp4",
        "MXp": "/local/qwen3-32b-mxfp4",
        "NV": nv_local,
        "NVa": nv_local,
        "NVnf": nv_local,
        "NVx": "/local/nvidia-qwen3-32b-nvfp4",
        "NVc": nv_local,
        "NVt": nv_local,
        "NVd": nv_local,
        "NVv": nv_local,
    }
    for spec in SPECS.values():
        assert posixpath.dirname(spec.model_dir) == settings.LOCAL_DIR
        assert posixpath.basename(spec.model_dir) == posixpath.basename(spec.volume_dir)
    assert SPECS[T.MX].model_dir == SPECS[T.MXP].model_dir != SPECS[T.NV].model_dir
    assert SPECS[T.NV].model_dir == SPECS[T.NVA].model_dir == SPECS[T.NVNF].model_dir


def test_each_treatment_serves_its_checkpoint_kind():
    kind_dir = {
        CheckpointKind.MX: model.MX_MODEL_DIR,
        CheckpointKind.NV: model.NV_MODEL_DIR,
        CheckpointKind.NVX: model.NVX_MODEL_DIR,
    }
    for t, spec in SPECS.items():
        assert spec.volume_dir == kind_dir[spec.checkpoint], t
    assert {t for t, s in SPECS.items() if s.checkpoint is CheckpointKind.MX} == {T.MX, T.MXP}
    assert {t for t, s in SPECS.items() if s.checkpoint is CheckpointKind.NVX} == {T.NVX}


def test_per_treatment_server_args_pin_the_kernel_of_every_treatment():
    scan = {
        "NVc": "flashinfer_cutlass",
        "NVt": "flashinfer_trtllm",
        "NVd": "flashinfer_cudnn",
        "NVv": "cutlass",
    }
    assert {t: s.server_args for t, s in SPECS.items()} == {
        "MX": PIN,
        "NV": PIN,
        "MXp": PIN,
        "NVx": PIN,
        "NVa": ("--linear-backend", "flashinfer_cudnn"),
        "NVnf": (*PIN, "--compilation-config", '{"pass_config": {"fuse_act_quant": false}}'),
        **{t: ("--linear-backend", backend) for t, backend in scan.items()},
    }


def test_nvnf_turns_the_act_quant_fusion_off_with_a_compilation_config_vllm_parses():
    args = SPECS[T.NVNF].server_args
    assert args[:2] == PIN
    assert args.count("--compilation-config") == 1
    config = args[args.index("--compilation-config") + 1]
    assert json.loads(config) == {"pass_config": {"fuse_act_quant": False}}
    assert config == '{"pass_config": {"fuse_act_quant": false}}'
    assert [a for a in args if a not in SPECS[T.NV].server_args] == ["--compilation-config", config]


def test_nva_is_pinned_to_the_kernel_the_smoke_scan_selected():
    assert SPECS[T.NVA].server_args == SPECS[T.NVD].server_args
    assert SPECS[T.NVA].linear_kernel != SPECS[T.NV].linear_kernel


def test_expected_linear_kernel_per_treatment():
    assert {t: s.linear_kernel for t, s in SPECS.items()} == {
        "MX": "FlashInferMxFp4LinearKernel",
        "MXp": "FlashInferMxFp4LinearKernel",
        "NV": CUTE_DSL_NV,
        "NVx": CUTE_DSL_NV,
        "NVnf": CUTE_DSL_NV,
        "NVa": "FlashInferCudnnNvFp4LinearKernel",
        "NVc": "FlashInferCutlassNvFp4LinearKernel",
        "NVt": "FlashInferTrtllmNvFp4LinearKernel",
        "NVd": "FlashInferCudnnNvFp4LinearKernel",
        "NVv": "CutlassNvFp4LinearKernel",
    }
    for t, spec in SPECS.items():
        assert spec.linear_kernel == model.expected_linear_kernel(t, spec.server_args)


@pytest.mark.parametrize(
    "backend, kernel",
    [
        ("flashinfer_cutedsl", "FlashInferCuteDslNvFp4LinearKernel"),
        ("flashinfer_cutlass", "FlashInferCutlassNvFp4LinearKernel"),
        ("flashinfer_trtllm", "FlashInferTrtllmNvFp4LinearKernel"),
        ("flashinfer_cudnn", "FlashInferCudnnNvFp4LinearKernel"),
        ("cutlass", "CutlassNvFp4LinearKernel"),
    ],
)
def test_nva_expectation_follows_its_configured_linear_backend(backend, kernel):
    assert LinearBackend(backend).kernel == kernel
    assert model.expected_linear_kernel(T.NVA, ("--linear-backend", backend)) == kernel


def test_the_expectation_for_an_unpinned_nva_is_the_default_cute_dsl_kernel():
    assert model.expected_linear_kernel(T.NVA, ()) == CUTE_DSL_NV == model.DEFAULT_NVFP4_KERNEL


def test_an_unknown_linear_backend_has_no_expected_kernel():
    with pytest.raises(ValueError, match="flashinfer_unknown"):
        model.expected_linear_kernel(T.NVA, ("--linear-backend", "flashinfer_unknown"))


def test_the_scan_treatments_are_never_cute_dsl_and_nv_stays_pinned_to_it():
    for t in NVA_SCAN.treatments:
        assert SPECS[t].linear_kernel != CUTE_DSL_NV
    assert SPECS[T.NV].linear_kernel == SPECS[T.NVX].linear_kernel == CUTE_DSL_NV
    assert SPECS[T.NVNF].linear_kernel == CUTE_DSL_NV


def test_nvnf_expects_the_cute_dsl_kernel_whatever_its_args():
    assert model.expected_linear_kernel(T.NVNF, ()) == CUTE_DSL_NV


def test_expected_act_quant_fusion_covers_every_treatment():
    assert {t: s.act_quant_fusion for t, s in SPECS.items()} == {
        "MX": False,
        "MXp": False,
        "NVnf": False,
        "NV": True,
        "NVa": True,
        "NVx": True,
        "NVc": True,
        "NVt": True,
        "NVd": True,
        "NVv": True,
    }
    assert all(type(s.act_quant_fusion) is bool for s in SPECS.values())


def test_only_treatments_that_turn_the_fusion_off_on_the_command_line_are_nvfp4_without_it():
    for t, spec in SPECS.items():
        is_mx = spec.checkpoint is CheckpointKind.MX
        turned_off = "--compilation-config" in spec.server_args
        assert spec.act_quant_fusion is (not is_mx and not turned_off), t


def test_the_nvidia_cross_check_and_the_kernel_scan_are_smoke_only():
    assert "NVx" not in FULL.treatments and "NVx" in SMOKE.treatments
    assert (SMOKE.kernel_scan, SMOKE.crosscheck) == (NVA_SCAN, NVX_CROSSCHECK)
    assert NVA_SCAN.treatments == ("NVc", "NVt", "NVd", "NVv")
    for t in NVA_SCAN.treatments:
        assert t not in FULL.treatments and t in SMOKE.treatments
    assert ("NVx", *NVA_SCAN.treatments) == SMOKE_ONLY_TREATMENTS
    assert set(SMOKE.treatments) == {"MX", "NV"} | set(SMOKE_ONLY_TREATMENTS)
    assert NVA_SCAN.selection_c == 32 and NVA_SCAN.selection_c in SMOKE.batches


def test_server_args_stay_plain_strings_as_problems_and_input_diffs_quote_them():
    for spec in SPECS.values():
        assert all(type(arg) is str for arg in spec.server_args)
    assert repr(SPECS[T.NVA].server_args) == "('--linear-backend', 'flashinfer_cudnn')"
    assert model.pinned_to(LinearBackend.CUTLASS) == ("--linear-backend", "cutlass")


def test_each_linear_backend_names_the_kernel_class_vllm_logs():
    assert {b.value: b.kernel.value for b in LinearBackend} == {
        "flashinfer_cutedsl": "FlashInferCuteDslNvFp4LinearKernel",
        "flashinfer_cutlass": "FlashInferCutlassNvFp4LinearKernel",
        "flashinfer_trtllm": "FlashInferTrtllmNvFp4LinearKernel",
        "flashinfer_cudnn": "FlashInferCudnnNvFp4LinearKernel",
        "cutlass": "CutlassNvFp4LinearKernel",
    }
    assert model.MXFP4_KERNEL == LinearKernel.FLASHINFER_MXFP4 == "FlashInferMxFp4LinearKernel"


def test_the_act_quant_fusion_is_stored_as_the_bool_g1_compares():
    assert {t for t, s in SPECS.items() if not s.act_quant_fusion} == {T.MX, T.MXP, T.NVNF}
    assert (
        model.treatment_spec(T.NV, CheckpointKind.NV, PIN, ActQuantFusion.OFF).act_quant_fusion
        is False
    )
