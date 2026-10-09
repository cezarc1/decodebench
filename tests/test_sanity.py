import hashlib
import json
from types import SimpleNamespace

import pytest

from fp4bench import bytes_model as bm
from fp4bench import sanity as sn
from fp4bench.core.types import Format
from tests import QuietHandler, local_server
from tests.ct_checkpoints import (
    QWEN3_BASE_CONFIG,
    as_headers,
    ct_quant_config,
    ct_tensors,
    qwen3_config,
    unquantize,
    write_checkpoint,
    write_header_only,
)


def _shard(path, tensors: dict, metadata: dict | None = None) -> None:
    """A .safetensors file with its zero-filled data: small tensors only (fingerprints hash it)."""
    write_header_only(path, tensors, metadata, zero_body=True)


def _checkpoint(path, tensors: dict | None = None, config=None, hf_quant: dict | None = None):
    """A directory with `tensors` in one shard and the given config.json (a str is written as it
    is) and hf_quant_config.json."""
    path.mkdir(parents=True, exist_ok=True)
    if tensors is not None:
        _shard(path / "model-00001.safetensors", tensors)
    if hf_quant is not None:
        (path / "hf_quant_config.json").write_text(json.dumps(hf_quant))
    if config is not None:
        (path / "config.json").write_text(config if isinstance(config, str) else json.dumps(config))
    return path


CONFIG = {"a": 1}
SMALL = {
    "model.layers.0.mlp.gate_proj.weight_packed": ("U8", [64, 32]),
    "model.layers.0.mlp.gate_proj.weight_scale": ("F8_E4M3", [64, 4]),
    "model.layers.0.mlp.gate_proj.weight_global_scale": ("F32", [1]),
    "model.layers.0.mlp.gate_proj.input_global_scale": ("F32", [1]),
    "model.layers.0.self_attn.q_norm.weight": ("BF16", [128]),
    "lm_head.weight": ("BF16", [16, 64]),
}


# METHODOLOGY.md#checkpoint-checks: the published headers these are excerpts of
REAL_REDHAT_NV_LAYER0 = {
    "model.layers.0.input_layernorm.weight": ("BF16", [5120]),
    "model.layers.0.mlp.down_proj.input_global_scale": ("F32", [1]),
    "model.layers.0.mlp.down_proj.weight_global_scale": ("F32", [1]),
    "model.layers.0.mlp.down_proj.weight_packed": ("U8", [5120, 12800]),
    "model.layers.0.mlp.down_proj.weight_scale": ("F8_E4M3", [5120, 1600]),
    "model.layers.0.mlp.gate_proj.input_global_scale": ("F32", [1]),
    "model.layers.0.mlp.gate_proj.weight_global_scale": ("F32", [1]),
    "model.layers.0.mlp.gate_proj.weight_packed": ("U8", [25600, 2560]),
    "model.layers.0.mlp.gate_proj.weight_scale": ("F8_E4M3", [25600, 320]),
    "model.layers.0.mlp.up_proj.input_global_scale": ("F32", [1]),
    "model.layers.0.mlp.up_proj.weight_global_scale": ("F32", [1]),
    "model.layers.0.mlp.up_proj.weight_packed": ("U8", [25600, 2560]),
    "model.layers.0.mlp.up_proj.weight_scale": ("F8_E4M3", [25600, 320]),
    "model.layers.0.post_attention_layernorm.weight": ("BF16", [5120]),
    "model.layers.0.self_attn.k_norm.weight": ("BF16", [128]),
    "model.layers.0.self_attn.k_proj.input_global_scale": ("F32", [1]),
    "model.layers.0.self_attn.k_proj.weight_global_scale": ("F32", [1]),
    "model.layers.0.self_attn.k_proj.weight_packed": ("U8", [1024, 2560]),
    "model.layers.0.self_attn.k_proj.weight_scale": ("F8_E4M3", [1024, 320]),
    "model.layers.0.self_attn.o_proj.input_global_scale": ("F32", [1]),
    "model.layers.0.self_attn.o_proj.weight_global_scale": ("F32", [1]),
    "model.layers.0.self_attn.o_proj.weight_packed": ("U8", [5120, 4096]),
    "model.layers.0.self_attn.o_proj.weight_scale": ("F8_E4M3", [5120, 512]),
    "model.layers.0.self_attn.q_norm.weight": ("BF16", [128]),
    "model.layers.0.self_attn.q_proj.input_global_scale": ("F32", [1]),
    "model.layers.0.self_attn.q_proj.weight_global_scale": ("F32", [1]),
    "model.layers.0.self_attn.q_proj.weight_packed": ("U8", [8192, 2560]),
    "model.layers.0.self_attn.q_proj.weight_scale": ("F8_E4M3", [8192, 320]),
    "model.layers.0.self_attn.v_proj.input_global_scale": ("F32", [1]),
    "model.layers.0.self_attn.v_proj.weight_global_scale": ("F32", [1]),
    "model.layers.0.self_attn.v_proj.weight_packed": ("U8", [1024, 2560]),
    "model.layers.0.self_attn.v_proj.weight_scale": ("F8_E4M3", [1024, 320]),
    "lm_head.weight": ("BF16", [151936, 5120]),
    "model.embed_tokens.weight": ("BF16", [151936, 5120]),
    "model.norm.weight": ("BF16", [5120]),
}
REAL_INC_MX_LAYER0 = {
    "model.layers.0.mlp.down_proj.weight_packed": ("U8", [5120, 12800]),
    "model.layers.0.mlp.down_proj.weight_scale": ("U8", [5120, 800]),
    "model.layers.0.mlp.gate_proj.weight_packed": ("U8", [25600, 2560]),
    "model.layers.0.mlp.gate_proj.weight_scale": ("U8", [25600, 160]),
    "model.layers.0.mlp.up_proj.weight_packed": ("U8", [25600, 2560]),
    "model.layers.0.mlp.up_proj.weight_scale": ("U8", [25600, 160]),
    "model.layers.0.self_attn.k_proj.weight": ("BF16", [1024, 5120]),
    "model.layers.0.self_attn.o_proj.weight": ("BF16", [5120, 8192]),
    "model.layers.0.self_attn.q_proj.weight": ("BF16", [8192, 5120]),
    "model.layers.0.self_attn.v_proj.weight": ("BF16", [1024, 5120]),
}
REAL_NVIDIA_MODELOPT_Q_PROJ = {
    "model.layers.0.self_attn.q_proj.input_scale": ("F32", []),
    "model.layers.0.self_attn.q_proj.weight": ("U8", [8192, 2560]),
    "model.layers.0.self_attn.q_proj.weight_scale": ("F8_E4M3", [8192, 320]),
    "model.layers.0.self_attn.q_proj.weight_scale_2": ("F32", []),
}
REAL_REDHAT_NV_QUANT_CONFIG = {
    "config_groups": {
        "group_0": {
            "input_activations": {
                "actorder": None,
                "block_structure": None,
                "dynamic": "local",
                "group_size": 16,
                "num_bits": 4,
                "observer": "minmax",
                "observer_kwargs": {},
                "strategy": "tensor_group",
                "symmetric": True,
                "type": "float",
            },
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None,
                "block_structure": None,
                "dynamic": False,
                "group_size": 16,
                "num_bits": 4,
                "observer": "minmax",
                "observer_kwargs": {},
                "strategy": "tensor_group",
                "symmetric": True,
                "type": "float",
            },
        }
    },
    "format": "nvfp4-pack-quantized",
    "global_compression_ratio": None,
    "ignore": ["lm_head"],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
}
REAL_NVIDIA_QUANT_CONFIG = {
    "config_groups": {
        "group_0": {
            "input_activations": {
                "dynamic": False,
                "num_bits": 4,
                "type": "float",
                "group_size": 16,
            },
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16},
            "targets": ["Linear"],
        }
    },
    "ignore": ["lm_head"],
    "quant_algo": "NVFP4",
    "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
    "producer": {"name": "modelopt", "version": "0.35.0"},
    "quant_method": "modelopt",
}

NVFP4, MXFP4 = Format.NVFP4, Format.MXFP4
NV, MX = ct_tensors(Format.NVFP4), ct_tensors(Format.MXFP4)
N_LINEARS = 7 * 64


def _fixture(tensors):
    return {k: (d, list(s)) for k, (d, s) in tensors.items()}


def test_fixtures_reproduce_the_real_published_headers():
    assert all(_fixture(NV)[k] == v for k, v in REAL_REDHAT_NV_LAYER0.items())
    assert all(_fixture(MX)[k] == v for k, v in REAL_INC_MX_LAYER0.items() if k in _fixture(MX))
    assert len(NV) == 2051
    assert sum(k.endswith(".weight_packed") for k in NV) == N_LINEARS == 448


def test_the_compressed_tensors_nvfp4_and_mxfp4_serialisation_is_what_the_fixture_says():
    nv, mx = ct_quant_config(Format.NVFP4), ct_quant_config(Format.MXFP4)
    assert nv["format"] == "nvfp4-pack-quantized" and mx["format"] == "mxfp4-pack-quantized"
    nv_group, mx_group = nv["config_groups"]["group_0"], mx["config_groups"]["group_0"]
    assert (nv_group["weights"]["group_size"], nv_group["weights"]["strategy"]) == (
        16,
        "tensor_group",
    )
    assert (mx_group["weights"]["group_size"], mx_group["weights"]["strategy"]) == (32, "group")
    assert nv_group["input_activations"]["dynamic"] == "local"
    assert mx_group["input_activations"]["dynamic"] is True


def test_headers_and_quantized_bytes_ratio(tmp_path):
    mx = write_checkpoint(tmp_path / "mx", MX)
    nv = write_checkpoint(tmp_path / "nv", NV)
    mx_h, nv_h = sn.read_safetensors_headers(mx), sn.read_safetensors_headers(nv)
    assert set(mx_h) == set(MX) and set(nv_h) == set(NV)
    assert sn.quantized_bytes(nv_h) / sn.quantized_bytes(mx_h) == pytest.approx(
        4.5 / 4.25, abs=1e-4
    )
    assert sn.quantized_bytes(nv_h) / sn.quantized_bytes(mx_h) == pytest.approx(1.0588, abs=1e-4)


def test_quantized_bytes_are_the_packed_weights_plus_their_scales():
    params = bm.QUANT_PARAMS
    packed = sum(v[1][0] * v[1][1] for k, v in NV.items() if k.endswith(".weight_packed"))
    assert packed * 2 == params
    mx = sn.quantized_bytes(as_headers(MX))
    nv = sn.quantized_bytes(as_headers(NV))
    assert mx == params // 2 + params // 32
    assert nv == params // 2 + params // 16 + N_LINEARS * 4
    assert mx / 1e9 == pytest.approx(16.58, abs=0.01) and nv / 1e9 == pytest.approx(17.55, abs=0.01)


def test_quantized_bytes_leave_out_bf16_tensors_biases_and_activation_scales():
    tensors = as_headers(NV)
    base = sn.quantized_bytes(tensors)
    tensors["model.layers.0.self_attn.q_proj.bias"] = {"dtype": "BF16", "shape": [8192]}
    assert sn.quantized_bytes(tensors) == base
    assert sn.quantized_bytes(as_headers(unquantize(NV))) == 0


def test_quantized_bytes_counts_only_the_modules_that_have_weight_packed():
    tensors = as_headers(unquantize(NV, "self_attn"))
    mlp_only = sum(
        sn.tensor_nbytes(v)
        for k, v in tensors.items()
        if ".mlp." in k and not k.endswith("input_global_scale")
    )
    assert sn.quantized_bytes(tensors) == mlp_only


@pytest.mark.parametrize("fmt, tensors", [(NVFP4, NV), (MXFP4, MX)])
def test_check_structure_accepts_a_correct_checkpoint(fmt, tensors):
    assert sn.check_structure(as_headers(tensors), fmt) == []


def test_check_structure_rejects_each_layout_under_the_other_format():
    assert sn.check_structure(as_headers(NV), MXFP4) != []
    assert sn.check_structure(as_headers(MX), NVFP4) != []


@pytest.mark.parametrize("fmt", [NVFP4, MXFP4])
def test_check_structure_rejects_a_bf16_checkpoint_relabelled_as_fp4(fmt):
    problems = sn.check_structure(as_headers(unquantize(ct_tensors(fmt))), fmt)
    assert any(f"{N_LINEARS} of {N_LINEARS} Linear modules not quantized" in p for p in problems)
    assert any("model.layers.0.mlp.down_proj" in p for p in problems)


@pytest.mark.parametrize("fmt", [NVFP4, MXFP4])
def test_check_structure_rejects_weight_packed_that_is_not_u8_or_not_packed(fmt):
    tensors = _fixture(ct_tensors(fmt))
    tensors["model.layers.0.mlp.up_proj.weight_packed"] = ("BF16", [25600, 2560])
    tensors["model.layers.1.mlp.up_proj.weight_packed"] = ("U8", [25600, 5120])
    problems = sn.check_structure(as_headers(tensors), fmt)
    assert len([p for p in problems if "weight_packed" in p]) == 1
    assert "2 weight_packed tensors are not U8 [out, in/2]" in problems[0]
    assert "model.layers.0.mlp.up_proj (BF16 [25600, 2560])" in problems[0]


@pytest.mark.parametrize("fmt", [NVFP4, MXFP4])
def test_check_structure_rejects_attention_left_in_bf16_like_the_inc_recipe(fmt):
    problems = sn.check_structure(as_headers(unquantize(ct_tensors(fmt), "self_attn")), fmt)
    assert len(problems) == 1
    assert f"{4 * 64} of {N_LINEARS} Linear modules not quantized" in problems[0]
    assert "model.layers.0.self_attn.k_proj" in problems[0]


def test_check_structure_rejects_the_inc_mxfp4_layout_from_the_real_header():
    tensors = _fixture(MX)
    for name in [n for n in tensors if "self_attn" in n and "norm" not in n]:
        del tensors[name]
    tensors.update({k: v for k, v in REAL_INC_MX_LAYER0.items() if "self_attn" in k})
    problems = sn.check_structure(as_headers(tensors), MXFP4)
    assert any("Linear modules not quantized" in p and "self_attn.q_proj" in p for p in problems)


def _with_group(tensors: dict, group: int) -> dict:
    """The same checkpoint with every weight_scale shaped for another group size."""
    out = _fixture(tensors)
    for name in [n for n in out if n.endswith(".weight_scale")]:
        rows, packed_cols = out[name.replace("weight_scale", "weight_packed")][1]
        out[name] = (out[name][0], [rows, packed_cols * 2 // group])
    return out


@pytest.mark.parametrize("fmt, wrong, right", [(NVFP4, 32, 16), (MXFP4, 16, 32)])
def test_check_structure_rejects_a_group_size_that_the_scale_shapes_imply(fmt, wrong, right):
    problems = sn.check_structure(as_headers(_with_group(ct_tensors(fmt), wrong)), fmt)
    assert problems == [
        f"{N_LINEARS} weight_scale tensors imply group size {wrong}, expected "
        f"{right}: e.g. model.layers.0.mlp.down_proj, model.layers.0.mlp.gate_proj, "
        "model.layers.0.mlp.up_proj"
    ]


def test_check_structure_rejects_a_scale_shape_that_is_no_group_size_at_all():
    tensors = _fixture(NV)
    tensors["model.layers.3.mlp.gate_proj.weight_scale"] = ("F8_E4M3", [25600, 321])
    tensors["model.layers.3.mlp.up_proj.weight_scale"] = ("F8_E4M3", [25601, 320])
    problems = sn.check_structure(as_headers(tensors), NVFP4)
    assert len(problems) == 1 and "2 weight_scale tensors have the wrong shape" in problems[0]
    assert "expected [out, in/16]" in problems[0]


def test_check_structure_rejects_the_other_formats_scale_dtype():
    nv_with_e8m0 = _fixture(NV)
    for name in [n for n in nv_with_e8m0 if n.endswith(".weight_scale")]:
        nv_with_e8m0[name] = ("U8", nv_with_e8m0[name][1])
    mx_with_e4m3 = _fixture(MX)
    for name in [n for n in mx_with_e4m3 if n.endswith(".weight_scale")]:
        mx_with_e4m3[name] = ("F8_E4M3", mx_with_e4m3[name][1])
    assert any(
        f"{N_LINEARS} weight_scale tensors are not F8_E4M3" in p
        for p in sn.check_structure(as_headers(nv_with_e8m0), NVFP4)
    )
    assert any(
        f"{N_LINEARS} weight_scale tensors are not U8" in p
        for p in sn.check_structure(as_headers(mx_with_e4m3), MXFP4)
    )


def test_check_structure_requires_the_nvfp4_global_scales():
    tensors = _fixture(NV)
    del tensors["model.layers.0.mlp.up_proj.input_global_scale"]
    del tensors["model.layers.1.mlp.up_proj.weight_global_scale"]
    tensors["model.layers.2.mlp.up_proj.weight_global_scale"] = ("F16", [1])
    tensors["model.layers.3.mlp.up_proj.input_global_scale"] = ("F32", [8])
    problems = sn.check_structure(as_headers(tensors), NVFP4)
    assert problems == [
        "1 quantized modules lack weight_global_scale: e.g. model.layers.1.mlp.up_proj",
        "1 weight_global_scale tensors are not F32 [1]: e.g. model.layers.2.mlp.up_proj (F16 [1])",
        "1 quantized modules lack input_global_scale: e.g. model.layers.0.mlp.up_proj",
        "1 input_global_scale tensors are not F32 [1]: e.g. model.layers.3.mlp.up_proj (F32 [8])",
    ]


def test_check_structure_rejects_nvfp4_global_scales_in_an_mxfp4_checkpoint():
    tensors = _fixture(MX)
    tensors["model.layers.0.mlp.up_proj.weight_global_scale"] = ("F32", [1])
    problems = sn.check_structure(as_headers(tensors), MXFP4)
    assert len(problems) == 1 and "1 unexpected tensors" in problems[0]
    assert "model.layers.0.mlp.up_proj.weight_global_scale" in problems[0]


def test_check_structure_rejects_a_leftover_bf16_weight_next_to_the_packed_one():
    tensors = _fixture(NV)
    tensors["model.layers.7.mlp.down_proj.weight"] = ("BF16", [5120, 25600])
    problems = sn.check_structure(as_headers(tensors), NVFP4)
    assert problems == [
        "1 quantized Linear modules still store a .weight tensor: e.g. model.layers.7.mlp.down_proj"
    ]


def test_check_structure_names_missing_modules_and_missing_scales():
    tensors = _fixture(NV)
    for suffix in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"):
        del tensors[f"model.layers.5.self_attn.v_proj.{suffix}"]
    del tensors["model.layers.6.mlp.gate_proj.weight_scale"]
    problems = sn.check_structure(as_headers(tensors), NVFP4)
    assert problems == [
        "1 Linear modules missing (neither weight_packed nor weight): "
        "e.g. model.layers.5.self_attn.v_proj",
        "1 quantized modules have no weight_scale: e.g. model.layers.6.mlp.gate_proj",
    ]


def test_check_structure_rejects_a_quantized_lm_head_and_non_bf16_embeddings_and_norms():
    tensors = _fixture(NV)
    del tensors["lm_head.weight"]
    tensors["lm_head.weight_packed"] = ("U8", [151936, 2560])
    tensors["lm_head.weight_scale"] = ("F8_E4M3", [151936, 320])
    tensors["model.embed_tokens.weight"] = ("F32", [151936, 5120])
    tensors["model.norm.weight"] = ("F16", [5120])
    tensors["model.layers.0.self_attn.q_norm.weight"] = ("BF16", [64])
    problems = sn.check_structure(as_headers(tensors), NVFP4)
    assert any("1 non-quantized tensors missing: e.g. lm_head.weight" in p for p in problems)
    assert any("2 unexpected tensors" in p and "lm_head.weight_packed" in p for p in problems)
    other = [p for p in problems if "not BF16" in p]
    assert len(other) == 1 and "3 non-quantized tensors are not BF16" in other[0]
    assert "model.embed_tokens.weight (F32 [151936, 5120])" in other[0]


def test_check_structure_rejects_stray_quantization_tensors():
    tensors = _fixture(NV)
    tensors["model.layers.0.mlp.up_proj.weight_zero_point"] = ("U8", [25600, 320])
    tensors["model.layers.0.mlp.up_proj.input_scale"] = ("F32", [1])
    problems = sn.check_structure(as_headers(tensors), NVFP4)
    assert len(problems) == 1 and "2 unexpected tensors" in problems[0]


def test_check_structure_rejects_the_nvidia_modelopt_layout():
    tensors = _fixture(NV)
    for suffix in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"):
        del tensors[f"model.layers.0.self_attn.q_proj.{suffix}"]
    tensors.update(REAL_NVIDIA_MODELOPT_Q_PROJ)
    problems = sn.check_structure(as_headers(tensors), NVFP4)
    assert any(
        "1 of 448 Linear modules not quantized" in p and "self_attn.q_proj" in p for p in problems
    )
    assert any("unexpected tensors" in p and "weight_scale_2" in p for p in problems)


def test_check_structure_takes_the_number_of_layers():
    assert sn.check_structure(as_headers(ct_tensors(MXFP4, layers=2)), MXFP4, n_layers=2) == []
    problems = sn.check_structure(as_headers(ct_tensors(MXFP4, layers=2)), MXFP4)
    assert any("Linear modules missing" in p for p in problems)


def test_check_structure_lists_at_most_three_examples_per_problem():
    problems = sn.check_structure(as_headers(unquantize(NV)), NVFP4)
    assert all(p.count("model.layers.") <= 3 for p in problems)


def _qc(tmp_path, fmt=NVFP4, mutate=None):
    qc = ct_quant_config(fmt)
    if mutate:
        mutate(qc)
    return _checkpoint(tmp_path, config=qwen3_config() | {"quantization_config": qc})


@pytest.mark.parametrize("fmt", [NVFP4, MXFP4])
def test_quant_config_accepts_what_compressed_tensors_writes(tmp_path, fmt):
    assert sn.quant_config_problems(_qc(tmp_path, fmt), fmt) == []
    assert sn.quant_format(tmp_path) == f"{fmt}-pack-quantized"


def test_quant_config_accepts_the_real_redhat_config_that_has_no_scale_dtype_or_group_format(
    tmp_path,
):
    d = _checkpoint(
        tmp_path, config=qwen3_config() | {"quantization_config": REAL_REDHAT_NV_QUANT_CONFIG}
    )
    assert sn.quant_config_problems(d, NVFP4) == []


def test_quant_config_accepts_a_group_whose_format_is_null_like_autoround_writes_it(tmp_path):
    def null_format(qc):
        qc["config_groups"]["group_0"]["format"] = None

    assert sn.quant_config_problems(_qc(tmp_path, MXFP4, null_format), MXFP4) == []


def test_quant_config_rejects_the_other_format_in_each_direction(tmp_path):
    nv_as_mx = sn.quant_config_problems(_qc(tmp_path / "a", NVFP4), MXFP4)
    mx_as_nv = sn.quant_config_problems(_qc(tmp_path / "b", MXFP4), NVFP4)
    assert any(
        "format is 'nvfp4-pack-quantized', expected 'mxfp4-pack-quantized'" in p for p in nv_as_mx
    )
    assert any(
        "format is 'mxfp4-pack-quantized', expected 'nvfp4-pack-quantized'" in p for p in mx_as_nv
    )
    assert any("group_size is 16, expected 32" in p for p in nv_as_mx)
    assert any("group_size is 32, expected 16" in p for p in mx_as_nv)


def test_quant_config_rejects_the_nvidia_modelopt_config_with_fp8_kv_cache(tmp_path):
    d = _checkpoint(
        tmp_path, config=qwen3_config() | {"quantization_config": REAL_NVIDIA_QUANT_CONFIG}
    )
    problems = sn.quant_config_problems(d, NVFP4)
    assert any("quant_method is 'modelopt', expected 'compressed-tensors'" in p for p in problems)
    assert any("kv_cache_scheme is set" in p and "'num_bits': 8" in p for p in problems)


def test_quant_config_rejects_fp8_kv_cache_in_an_otherwise_correct_config(tmp_path):
    def fp8_kv(qc):
        qc["kv_cache_scheme"] = {
            "num_bits": 8,
            "type": "float",
            "strategy": "tensor",
            "dynamic": False,
            "symmetric": True,
        }

    problems = sn.quant_config_problems(_qc(tmp_path, NVFP4, fp8_kv), NVFP4)
    assert len(problems) == 1 and "kv_cache_scheme is set" in problems[0]


def test_quant_config_accepts_a_missing_kv_cache_scheme_key(tmp_path):
    assert (
        sn.quant_config_problems(_qc(tmp_path, NVFP4, lambda qc: qc.pop("kv_cache_scheme")), NVFP4)
        == []
    )


@pytest.mark.parametrize(
    "ignore",
    [
        [],
        ["lm_head", "model.layers.0.self_attn.q_proj"],
        ["re:.*self_attn.*", "lm_head"],
        ["model.embed_tokens", "lm_head"],
        None,
    ],
)
def test_quant_config_requires_ignore_to_be_exactly_lm_head(tmp_path, ignore):
    problems = sn.quant_config_problems(
        _qc(tmp_path, MXFP4, lambda qc: qc.update(ignore=ignore)), MXFP4
    )
    assert (
        len(problems) == 1 and "ignore is" in problems[0] and "expected ['lm_head']" in problems[0]
    )


def test_quant_config_rejects_the_real_inc_ignore_list_that_keeps_attention_in_bf16(tmp_path):
    ignore = ["re:.*self_attn.*", "lm_head"] + [
        f"model.layers.{i}.self_attn.{m}_proj" for i in range(64) for m in "qkvo"
    ]
    problems = sn.quant_config_problems(
        _qc(tmp_path, MXFP4, lambda qc: qc.update(ignore=ignore)), MXFP4
    )
    assert len(problems) == 1 and "ignore is" in problems[0]
    assert len(problems[0]) < 300


@pytest.mark.parametrize(
    "fmt, field, value, problem",
    [
        (MXFP4, "weights.num_bits", 8, "weights num_bits is 8, expected 4"),
        (MXFP4, "weights.type", "int", "weights type is 'int', expected 'float'"),
        (NVFP4, "weights.group_size", 32, "weights group_size is 32, expected 16"),
        (NVFP4, "weights.group_size", 8, "weights group_size is 8, expected 16"),
        (MXFP4, "weights.group_size", 16, "weights group_size is 16, expected 32"),
        (MXFP4, "weights.group_size", None, "weights group_size is None, expected 32"),
        (
            NVFP4,
            "weights.strategy",
            "group",
            "weights strategy is 'group', expected 'tensor_group'",
        ),
        (NVFP4, "weights.symmetric", False, "weights symmetric is False, expected True"),
        (
            NVFP4,
            "input_activations.group_size",
            32,
            "input_activations group_size is 32, expected 16",
        ),
        (NVFP4, "targets", ["re:.*mlp.*"], "targets is ['re:.*mlp.*'], expected ['Linear']"),
        (
            NVFP4,
            "format",
            "float-quantized",
            "format is 'float-quantized', expected 'nvfp4-pack-quantized'",
        ),
    ],
)
def test_quant_config_rejects_each_group_field_that_departs_from_the_format(
    tmp_path, fmt, field, value, problem
):
    def change(qc):
        *parents, key = field.split(".")
        node = qc["config_groups"]["group_0"]
        for parent in parents:
            node = node[parent]
        node[key] = value

    assert sn.quant_config_problems(_qc(tmp_path, fmt, change), fmt) == [f"group_0 {problem}"]


@pytest.mark.parametrize("fmt", [NVFP4, MXFP4])
def test_quant_config_requires_w4a4_input_activations(tmp_path, fmt):

    def a16(qc):
        qc["config_groups"]["group_0"]["input_activations"] = None

    def no_key(qc):
        del qc["config_groups"]["group_0"]["input_activations"]

    def a8(qc):
        qc["config_groups"]["group_0"]["input_activations"]["num_bits"] = 8

    for sub, mutate in (("a", a16), ("b", no_key)):
        problems = sn.quant_config_problems(_qc(tmp_path / sub, fmt, mutate), fmt)
        assert len(problems) == 1 and "no input_activations (W4A16, not W4A4)" in problems[0]
    problems = sn.quant_config_problems(_qc(tmp_path / "c", fmt, a8), fmt)
    assert problems == ["group_0 input_activations num_bits is 8, expected 4"]


def test_quant_config_checks_every_config_group(tmp_path):
    def two_groups(qc):
        bad = json.loads(json.dumps(qc["config_groups"]["group_0"]))
        bad["weights"]["group_size"] = 32
        qc["config_groups"]["group_1"] = bad

    assert sn.quant_config_problems(_qc(tmp_path, NVFP4, two_groups), NVFP4) == [
        "group_1 weights group_size is 32, expected 16"
    ]


@pytest.mark.parametrize("config_groups", [{}, None, []])
def test_quant_config_rejects_missing_config_groups(tmp_path, config_groups):
    problems = sn.quant_config_problems(
        _qc(tmp_path, NVFP4, lambda qc: qc.update(config_groups=config_groups)), NVFP4
    )
    assert len(problems) == 1 and "no config_groups" in problems[0]


def test_quant_config_rejects_a_config_without_quantization_config(tmp_path):
    for sub, config in (
        ("a", qwen3_config()),
        ("b", qwen3_config() | {"quantization_config": None}),
        ("c", qwen3_config() | {"quantization_config": "nvfp4"}),
    ):
        d = _checkpoint(tmp_path / sub, config=config)
        problems = sn.quant_config_problems(d, NVFP4)
        assert len(problems) == 1 and "no quantization_config object" in problems[0]
        assert sn.quant_format(d) is None


def test_quant_config_names_a_missing_or_unparseable_config_json(tmp_path):
    problems = sn.quant_config_problems(tmp_path / "a", NVFP4)
    assert len(problems) == 1 and "config.json" in problems[0] and "not found" in problems[0]
    bad = _checkpoint(tmp_path / "b", config="{not json")
    problems = sn.quant_config_problems(bad, NVFP4)
    assert len(problems) == 1 and "config.json is not valid JSON" in problems[0]
    assert sn.quant_format(tmp_path / "a") is None and sn.quant_format(bad) is None


def test_quant_config_ignores_the_model_name_and_hf_quant_config_json(tmp_path):
    d = _checkpoint(
        tmp_path, config=qwen3_config(NVFP4, _name_or_path="someone/Qwen3-32B-NVFP4A16-BF16")
    )
    (d / "hf_quant_config.json").write_text(json.dumps({"quantization": {"quant_algo": "FP8"}}))
    assert sn.quant_config_problems(d, NVFP4) == []


MX_CONFIG = qwen3_config(
    Format.MXFP4,
    torch_dtype="bfloat16",
    _name_or_path="Qwen/Qwen3-32B",
    transformers_version="4.57.0",
)
NV_CONFIG = qwen3_config(Format.NVFP4, transformers_version="4.57.1")


def _pair(tmp_path, nv_config=None, mx_config=None, mx_tensors=None, nv_tensors=None):
    mx = write_checkpoint(
        tmp_path / "mx",
        MX if mx_tensors is None else mx_tensors,
        MX_CONFIG if mx_config is None else mx_config,
    )
    nv = write_checkpoint(
        tmp_path / "nv",
        NV if nv_tensors is None else nv_tensors,
        NV_CONFIG if nv_config is None else nv_config,
    )
    return str(mx), str(nv)


def test_checkpoint_report_for_a_correct_pair_is_clean(tmp_path):
    report = sn.checkpoint_report(*_pair(tmp_path))
    assert report["mx_problems"] == [] and report["nv_problems"] == []
    assert report["config_diff"] == {}
    assert report["mx_quantized_bytes"] == bm.QUANT_PARAMS // 2 + bm.QUANT_PARAMS // 32
    assert report["nv_quantized_bytes"] == (
        bm.QUANT_PARAMS // 2 + bm.QUANT_PARAMS // 16 + N_LINEARS * 4
    )
    assert report["nv_over_mx"] == report["nv_quantized_bytes"] / report["mx_quantized_bytes"]
    assert report["nv_over_mx"] == pytest.approx(1.0588, abs=1e-4)
    assert report["expected_nv_over_mx"] == 4.5 / 4.25
    assert report["mx_quant_format"] == "mxfp4-pack-quantized"
    assert report["nv_quant_format"] == "nvfp4-pack-quantized"
    assert report["nv_has_activation_scales"] is True


def test_checkpoint_report_keeps_the_keys_that_the_runner_publish_and_analysis_read(tmp_path):
    report = sn.checkpoint_report(*_pair(tmp_path))
    assert {
        "nv_problems",
        "nv_over_mx",
        "expected_nv_over_mx",
        "nv_has_activation_scales",
        "config_diff",
    } <= set(report)
    json.dumps(report)


def test_checkpoint_report_records_missing_activation_scales_and_calls_it_a_problem(tmp_path):
    nv = {k: v for k, v in NV.items() if not k.endswith("input_global_scale")}
    report = sn.checkpoint_report(*_pair(tmp_path, nv_tensors=nv))
    assert report["nv_has_activation_scales"] is False
    assert any("lack input_global_scale" in p for p in report["nv_problems"])


def test_activation_scale_presence_needs_every_quantized_module_to_have_one():
    tensors = as_headers(NV)
    assert sn.has_activation_scales(tensors) is True
    del tensors["model.layers.40.mlp.down_proj.input_global_scale"]
    assert sn.has_activation_scales(tensors) is False
    assert sn.has_activation_scales(as_headers(MX)) is False
    assert sn.has_activation_scales({}) is False


def test_checkpoint_report_puts_each_formats_problems_on_its_own_side(tmp_path):
    inc_style_mx = unquantize(MX, "self_attn")
    report = sn.checkpoint_report(*_pair(tmp_path, mx_tensors=inc_style_mx))
    assert report["nv_problems"] == []
    assert len(report["mx_problems"]) == 1
    assert f"{4 * 64} of {N_LINEARS} Linear modules not quantized" in report["mx_problems"][0]
    assert report["mx_quantized_bytes"] < bm.QUANT_PARAMS


def test_checkpoint_report_rejects_an_nvidia_style_nv_checkpoint_with_fp8_kv(tmp_path):
    fp8_kv = {**NV_CONFIG, "quantization_config": REAL_NVIDIA_QUANT_CONFIG}
    report = sn.checkpoint_report(*_pair(tmp_path, nv_config=fp8_kv))
    assert report["mx_problems"] == []
    assert any("kv_cache_scheme is set" in p for p in report["nv_problems"])
    assert report["nv_quant_format"] is None


def test_checkpoint_report_flags_a_checkpoint_that_declares_the_wrong_format(tmp_path):
    report = sn.checkpoint_report(*_pair(tmp_path, nv_config=qwen3_config(Format.MXFP4)))
    assert any(
        "format is 'mxfp4-pack-quantized', expected 'nvfp4-pack-quantized'" in p
        for p in report["nv_problems"]
    )


@pytest.mark.parametrize("side", ["mx", "nv"])
def test_checkpoint_report_names_the_side_without_quantized_bytes(tmp_path, side):
    bf16 = unquantize(NV if side == "nv" else MX)
    kwargs = {"nv_tensors": bf16} if side == "nv" else {"mx_tensors": bf16}
    with pytest.raises(ValueError, match=f"{side.upper()}.*zero quantized bytes"):
        sn.checkpoint_report(*_pair(tmp_path, **kwargs))


def test_checkpoint_report_problem_order_is_structure_then_quant_config_then_parity(tmp_path):
    config = {**qwen3_config(Format.NVFP4, sliding_window=4096)}
    config["quantization_config"]["kv_cache_scheme"] = {"num_bits": 8, "type": "float"}
    report = sn.checkpoint_report(
        *_pair(tmp_path, nv_config=config, nv_tensors=unquantize(NV, "self_attn"))
    )
    problems = report["nv_problems"]
    assert "not quantized" in problems[0]
    assert "kv_cache_scheme" in problems[1]
    assert problems[-1] == "config differs from MX: sliding_window" and len(problems) == 3


def test_config_parity_ignores_the_quantization_section_versions_and_dtype_key(tmp_path):
    report = sn.checkpoint_report(*_pair(tmp_path))
    assert report["config_diff"] == {} and report["nv_problems"] == []


@pytest.mark.parametrize(
    "key, value",
    [
        ("sliding_window", 4096),
        ("layer_types", ["full_attention"] * 63 + ["sliding_attention"]),
        ("rope_theta", 10000),
        ("num_hidden_layers", 32),
    ],
)
def test_config_parity_flags_a_changed_key_and_makes_it_an_nv_problem(tmp_path, key, value):
    report = sn.checkpoint_report(*_pair(tmp_path, nv_config={**NV_CONFIG, key: value}))
    assert report["config_diff"] == {key: {"mx": QWEN3_BASE_CONFIG[key], "nv": value}}
    assert report["nv_problems"] == [f"config differs from MX: {key}"]
    assert report["mx_problems"] == []


def test_config_parity_flags_keys_present_on_one_side_only(tmp_path):
    nv_config = {k: v for k, v in NV_CONFIG.items() if k != "sliding_window"}
    nv_config["rope_scaling"] = {"rope_type": "yarn", "factor": 4.0}
    report = sn.checkpoint_report(*_pair(tmp_path, nv_config=nv_config))
    assert report["config_diff"] == {
        "rope_scaling": {"nv": {"rope_type": "yarn", "factor": 4.0}},
        "sliding_window": {"mx": None},
    }
    assert report["nv_problems"] == [
        "config differs from MX: rope_scaling",
        "config differs from MX: sliding_window",
    ]


def test_config_parity_does_not_equate_true_with_one(tmp_path):
    mx_config = {**MX_CONFIG, "tie_word_embeddings": True}
    report = sn.checkpoint_report(
        *_pair(tmp_path, nv_config={**NV_CONFIG, "tie_word_embeddings": 1}, mx_config=mx_config)
    )
    assert list(report["config_diff"]) == ["tie_word_embeddings"]


@pytest.mark.parametrize("side", ["mx", "nv"])
def test_checkpoint_report_names_the_side_without_a_config_json(tmp_path, side):
    mx, nv = _pair(tmp_path)
    (tmp_path / side / "config.json").unlink()
    with pytest.raises(ValueError, match=f"{side.upper()}.*config.json"):
        sn.checkpoint_report(mx, nv)


def test_checkpoint_report_names_the_side_with_an_unparseable_config_json(tmp_path):
    mx, nv = _pair(tmp_path)
    (tmp_path / "nv" / "config.json").write_text("{not json")
    with pytest.raises(ValueError, match=r"NV.*config\.json"):
        sn.checkpoint_report(mx, nv)


def test_tensor_nbytes_knows_unsigned_wide_dtypes():
    assert sn.tensor_nbytes({"dtype": "U16", "shape": [3]}) == 6
    assert sn.tensor_nbytes({"dtype": "U32", "shape": [3]}) == 12
    assert sn.tensor_nbytes({"dtype": "U64", "shape": [3]}) == 24


def test_tensor_nbytes_names_unknown_dtype():
    with pytest.raises(ValueError, match="F4_E2M1"):
        sn.tensor_nbytes({"dtype": "F4_E2M1", "shape": [2]})


def test_read_headers_ignores_metadata_entry(tmp_path):
    _shard(tmp_path / "model.safetensors", SMALL, {"format": "pt"})
    assert set(sn.read_safetensors_headers(tmp_path)) == set(SMALL)


def test_read_headers_reads_only_shards_listed_in_the_index(tmp_path):
    _shard(tmp_path / "model-00001-of-00002.safetensors", {"a.weight": ("BF16", [2])})
    _shard(tmp_path / "model-00002-of-00002.safetensors", {"b.weight": ("BF16", [2])})
    _shard(tmp_path / "consolidated.safetensors", {"a.weight": ("F32", [2])})
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    assert set(sn.read_safetensors_headers(tmp_path)) == {"a.weight", "b.weight"}


def test_read_headers_rejects_a_tensor_stored_in_two_files(tmp_path):
    _shard(tmp_path / "model-00001.safetensors", {"a.weight": ("BF16", [2])})
    _shard(tmp_path / "model-00002.safetensors", {"a.weight": ("BF16", [2])})
    with pytest.raises(ValueError, match=r"a\.weight"):
        sn.read_safetensors_headers(tmp_path)


def test_read_headers_rejects_a_directory_without_safetensors(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="no safetensors"):
        sn.read_safetensors_headers(tmp_path)


def test_prompt_nll_from_response():
    ids = [5, 7, 9]
    body = {
        "choices": [
            {
                "prompt_logprobs": [
                    None,
                    {"7": {"logprob": -1.0}, "3": {"logprob": -0.1}},
                    {"9": {"logprob": -3.0}},
                ]
            }
        ]
    }
    assert sn.prompt_nll_from_response(body, ids) == pytest.approx(2.0)


def nll_body(ids: list[int], logprob: float = -2.0) -> dict:
    return {
        "choices": [{"prompt_logprobs": [None] + [{str(t): {"logprob": logprob}} for t in ids[1:]]}]
    }


@pytest.fixture
def fake_completions():
    """A local stand-in for vLLM's /v1/completions: answers each payload with `respond`."""
    fake = SimpleNamespace(payloads=[], respond=lambda payload: (200, nll_body(payload["prompt"])))

    class Handler(QuietHandler):
        def do_POST(self):  # noqa: N802 - the name http.server dispatches to
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            fake.payloads.append(payload)
            status, body = fake.respond(payload)
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    with local_server(Handler) as url:
        fake.url = url
        yield fake


def test_prompt_nlls_uses_an_explicit_total_timeout(fake_completions, monkeypatch):
    seen = {}
    real_session = sn.aiohttp.ClientSession

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real_session(*args, **kwargs)

    monkeypatch.setattr(sn.aiohttp, "ClientSession", spy)
    sn.prompt_nlls(fake_completions.url, "fp4bench", [[1, 2, 3]])
    assert seen["timeout"].total == 600


def test_prompt_nlls_returns_one_value_per_prompt_in_prompt_order(fake_completions):
    fake_completions.respond = lambda payload: (
        200,
        nll_body(payload["prompt"], -float(len(payload["prompt"]))),
    )
    prompts = [[1, 2, 3, 4], [5, 6], [7, 8, 9], [1, 2, 3, 4, 5]]
    assert sn.prompt_nlls(fake_completions.url, "fp4bench", prompts) == [4.0, 2.0, 3.0, 5.0]
    assert all(
        p["prompt_logprobs"] == 1 and p["max_tokens"] == 1 and p["model"] == "fp4bench"
        for p in fake_completions.payloads
    )


def test_prompt_nlls_keeps_the_order_when_responses_finish_out_of_order(fake_completions):
    import time

    def respond(payload):
        time.sleep(0.2 if payload["prompt"][0] == 1 else 0.0)
        return 200, nll_body(payload["prompt"], -float(payload["prompt"][0]))

    fake_completions.respond = respond
    prompts = [[1, 9, 9], [2, 9, 9], [3, 9, 9]]
    assert sn.prompt_nlls(fake_completions.url, "fp4bench", prompts) == [1.0, 2.0, 3.0]


def test_prompt_nlls_reports_status_and_truncated_body_on_http_error(fake_completions):
    fake_completions.respond = lambda payload: (500, b"E" * 2000)
    with pytest.raises(RuntimeError) as err:
        sn.prompt_nlls(fake_completions.url, "fp4bench", [[1, 2, 3]])
    assert (
        "500" in str(err.value) and "E" * 500 in str(err.value) and "E" * 501 not in str(err.value)
    )


def test_prompt_nlls_of_no_prompts_is_an_empty_list(fake_completions):
    assert sn.prompt_nlls(fake_completions.url, "fp4bench", []) == []
    assert fake_completions.payloads == []


def test_prompt_nll_rejects_missing_prompt_logprobs():
    with pytest.raises(ValueError, match="prompt_logprobs"):
        sn.prompt_nll_from_response({"choices": [{"prompt_logprobs": None}]}, [1, 2, 3])


def test_prompt_nll_rejects_wrong_length_prompt_logprobs():
    body = {"choices": [{"prompt_logprobs": [None, {"2": {"logprob": -1.0}}]}]}
    with pytest.raises(ValueError, match="3"):
        sn.prompt_nll_from_response(body, [1, 2, 3])


def test_layout_fingerprint_equal_headers_give_equal_fingerprints(tmp_path):
    a = _checkpoint(tmp_path / "a", SMALL)
    b = _checkpoint(tmp_path / "b", dict(SMALL))
    fp = sn.layout_fingerprint(a)
    assert fp == sn.layout_fingerprint(b)
    assert len(fp) == 64 and int(fp, 16) >= 0


def test_layout_fingerprint_is_the_sha256_of_the_sorted_headers(tmp_path):
    path = _checkpoint(tmp_path / "ck", SMALL)
    want = hashlib.sha256(
        json.dumps(sn.read_safetensors_headers(path), sort_keys=True).encode()
    ).hexdigest()
    assert sn.layout_fingerprint(path) == want


def test_layout_fingerprint_changes_with_a_dtype(tmp_path):
    same_size = {**SMALL, "model.layers.0.mlp.gate_proj.weight_packed": ("I8", [64, 32])}
    a = _checkpoint(tmp_path / "a", SMALL)
    b = _checkpoint(tmp_path / "b", same_size)
    assert sn.layout_fingerprint(a) != sn.layout_fingerprint(b)


def test_layout_fingerprint_changes_with_a_shape(tmp_path):
    a = _checkpoint(tmp_path / "a", SMALL)
    b = _checkpoint(tmp_path / "b", {**SMALL, "lm_head.weight": ("BF16", [32, 32])})
    assert sn.layout_fingerprint(a) != sn.layout_fingerprint(b)


def _content_reference(
    shards: dict[str, bytes], config: bytes, extras: dict[str, bytes] | None = None
) -> str:
    pairs = sorted(
        [name, hashlib.sha256(data).hexdigest()]
        for name, data in {**shards, **(extras or {}), "config.json": config}.items()
    )
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()


SERVING_FILES = (
    "hf_quant_config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)
BOOKKEEPING_FILES = ("fp4bench_fetch.json", "fp4bench_quant.json", "README.md")


def test_content_fingerprint_is_the_sha256_of_the_sorted_name_and_file_digest_pairs(tmp_path):
    ck = _checkpoint(tmp_path / "ck", SMALL, CONFIG)
    shard = (ck / "model-00001.safetensors").read_bytes()
    assert sn.content_fingerprint(ck) == _content_reference(
        {"model-00001.safetensors": shard}, json.dumps({"a": 1}).encode()
    )


def test_content_fingerprint_hashes_files_larger_than_one_read_chunk(tmp_path):
    ck = _checkpoint(tmp_path / "ck", SMALL, CONFIG)
    big = ck / "model-00002.safetensors"
    big.write_bytes(bytes(range(256)) * (3 * 4096 + 17))
    assert sn.content_fingerprint(ck) == _content_reference(
        {
            "model-00001.safetensors": (ck / "model-00001.safetensors").read_bytes(),
            "model-00002.safetensors": big.read_bytes(),
        },
        (ck / "config.json").read_bytes(),
    )


def test_content_fingerprint_equal_for_equal_bytes_in_different_directories(tmp_path):
    a = _checkpoint(tmp_path / "a", SMALL, CONFIG)
    b = _checkpoint(tmp_path / "b", dict(SMALL), CONFIG)
    assert sn.content_fingerprint(a) == sn.content_fingerprint(b)


def test_content_fingerprint_sees_a_weight_byte_that_the_layout_fingerprint_cannot(tmp_path):
    a = _checkpoint(tmp_path / "a", SMALL, CONFIG)
    b = _checkpoint(tmp_path / "b", SMALL, CONFIG)
    shard = b / "model-00001.safetensors"
    data = bytearray(shard.read_bytes())
    data[-1] ^= 0x01
    shard.write_bytes(bytes(data))
    assert sn.layout_fingerprint(a) == sn.layout_fingerprint(b)
    assert sn.content_fingerprint(a) != sn.content_fingerprint(b)


def test_content_fingerprint_changes_with_config_json(tmp_path):
    a = _checkpoint(tmp_path / "a", SMALL, {"sliding_window": 128})
    b = _checkpoint(tmp_path / "b", SMALL, {"sliding_window": 4096})
    assert sn.content_fingerprint(a) != sn.content_fingerprint(b)


def test_content_fingerprint_changes_with_a_shard_name(tmp_path):
    a = _checkpoint(tmp_path / "a", SMALL, CONFIG)
    b = _checkpoint(tmp_path / "b", SMALL, CONFIG)
    (b / "model-00001.safetensors").rename(b / "model-00009.safetensors")
    assert sn.content_fingerprint(a) != sn.content_fingerprint(b)


def test_content_fingerprint_covers_only_the_shards_that_the_header_reader_reads(tmp_path):
    ck = _checkpoint(tmp_path, config=CONFIG)
    _shard(ck / "model-00001-of-00002.safetensors", {"a.weight": ("BF16", [2])})
    _shard(ck / "model-00002-of-00002.safetensors", {"b.weight": ("BF16", [2])})
    (ck / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    before = sn.content_fingerprint(ck)
    _shard(ck / "consolidated.safetensors", {"a.weight": ("F32", [2])})
    (ck / "README.md").write_text("not part of the checkpoint")
    assert sn.content_fingerprint(ck) == before


def test_content_fingerprint_includes_every_serving_relevant_file_that_exists(tmp_path):
    ck = _checkpoint(tmp_path / "ck", SMALL, CONFIG)
    for name in SERVING_FILES:
        (ck / name).write_text(f"contents of {name}")
    assert sn.content_fingerprint(ck) == _content_reference(
        {"model-00001.safetensors": (ck / "model-00001.safetensors").read_bytes()},
        (ck / "config.json").read_bytes(),
        {name: f"contents of {name}".encode() for name in SERVING_FILES},
    )


def test_serving_files_are_exactly_the_files_the_reference_hashes():
    assert set(sn.SERVING_FILES) == set(SERVING_FILES)
    assert len(sn.SERVING_FILES) == len(set(sn.SERVING_FILES))


def test_content_fingerprint_covers_every_tokenizer_file_that_quantize_copies():
    from fp4bench import quantize

    assert set(quantize.TOKENIZER_FILES) - set(sn.SERVING_FILES) == {"LICENSE"}


def test_content_fingerprint_changes_with_hf_quant_config_json(tmp_path):
    a = _checkpoint(
        tmp_path / "a", SMALL, CONFIG, hf_quant={"quantization": {"quant_algo": "NVFP4"}}
    )
    b = _checkpoint(
        tmp_path / "b", SMALL, CONFIG, hf_quant={"quantization": {"quant_algo": "MIXED_PRECISION"}}
    )
    assert sn.content_fingerprint(a) != sn.content_fingerprint(b)


def test_content_fingerprint_changes_with_generation_config_json(tmp_path):
    a = _checkpoint(tmp_path / "a", SMALL, CONFIG)
    b = _checkpoint(tmp_path / "b", SMALL, CONFIG)
    (a / "generation_config.json").write_text(json.dumps({"temperature": 1.0, "top_p": 1.0}))
    (b / "generation_config.json").write_text(json.dumps({"temperature": 0.6, "top_p": 0.95}))
    assert sn.content_fingerprint(a) != sn.content_fingerprint(b)


@pytest.mark.parametrize("name", SERVING_FILES)
def test_content_fingerprint_changes_when_a_serving_file_is_edited_or_added(tmp_path, name):
    ck = _checkpoint(tmp_path / "ck", SMALL, CONFIG)
    absent = sn.content_fingerprint(ck)
    (ck / name).write_text("first")
    first = sn.content_fingerprint(ck)
    (ck / name).write_text("second")
    second = sn.content_fingerprint(ck)
    assert len({absent, first, second}) == 3


@pytest.mark.parametrize("name", SERVING_FILES)
def test_content_fingerprint_with_one_optional_file_present_matches_the_reference(tmp_path, name):
    ck = _checkpoint(tmp_path / "ck", SMALL, CONFIG)
    (ck / name).write_text("only this one")
    assert sn.content_fingerprint(ck) == _content_reference(
        {"model-00001.safetensors": (ck / "model-00001.safetensors").read_bytes()},
        (ck / "config.json").read_bytes(),
        {name: b"only this one"},
    )


def test_content_fingerprint_without_any_optional_file_is_the_shards_plus_config_json(tmp_path):
    ck = _checkpoint(tmp_path / "ck", SMALL, CONFIG)
    assert sorted(p.name for p in ck.iterdir()) == ["config.json", "model-00001.safetensors"]
    assert sn.content_fingerprint(ck) == _content_reference(
        {"model-00001.safetensors": (ck / "model-00001.safetensors").read_bytes()},
        json.dumps({"a": 1}).encode(),
    )


@pytest.mark.parametrize("name", BOOKKEEPING_FILES)
def test_content_fingerprint_ignores_fp4bench_bookkeeping_files(tmp_path, name):
    ck = _checkpoint(tmp_path / "ck", SMALL, CONFIG)
    for serving in SERVING_FILES:
        (ck / serving).write_text(f"contents of {serving}")
    before = sn.content_fingerprint(ck)
    (ck / name).write_text("fetched at 2026-10-04T10:00:00Z")
    assert sn.content_fingerprint(ck) == before
    (ck / name).write_text("fetched at 2026-10-05T23:59:59Z")
    assert sn.content_fingerprint(ck) == before


def test_content_fingerprint_needs_config_json_and_shards(tmp_path):
    no_config = _checkpoint(tmp_path / "a", SMALL)
    with pytest.raises(FileNotFoundError, match=r"config\.json"):
        sn.content_fingerprint(no_config)
    _checkpoint(tmp_path / "b", config=CONFIG)
    with pytest.raises(ValueError, match="no safetensors"):
        sn.content_fingerprint(tmp_path / "b")
