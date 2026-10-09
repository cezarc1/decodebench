"""Header-only compressed-tensors W4A4 checkpoints of Qwen3-32B with real names, dtypes and shapes
(METHODOLOGY.md#checkpoint-checks)."""

import copy
import json
import struct

from fp4bench.core.types import Format

HIDDEN, INTERMEDIATE, VOCAB, HEAD_DIM, N_LAYERS = 5120, 25600, 151_936, 128, 64
LINEARS = {
    "self_attn.q_proj": (8192, 5120),
    "self_attn.k_proj": (1024, 5120),
    "self_attn.v_proj": (1024, 5120),
    "self_attn.o_proj": (5120, 8192),
    "mlp.gate_proj": (25600, 5120),
    "mlp.up_proj": (25600, 5120),
    "mlp.down_proj": (5120, 25600),
}
GROUP = {"nvfp4": 16, "mxfp4": 32}
SCALE_DTYPE = {"nvfp4": "F8_E4M3", "mxfp4": "U8"}
CT_FORMAT = {"nvfp4": "nvfp4-pack-quantized", "mxfp4": "mxfp4-pack-quantized"}


def quantized_module_tensors(fmt: Format, out_f: int, in_f: int, group: int | None = None) -> dict:
    """The on-disk tensors of one quantized Linear, as {suffix: (dtype, shape)}."""
    group = group or GROUP[fmt]
    tensors = {
        "weight_packed": ("U8", [out_f, in_f // 2]),
        "weight_scale": (SCALE_DTYPE[fmt], [out_f, in_f // group]),
    }
    if fmt == "nvfp4":
        tensors["weight_global_scale"] = ("F32", [1])
        tensors["input_global_scale"] = ("F32", [1])
    return tensors


def ct_tensors(fmt: Format, layers: int = N_LAYERS) -> dict[str, tuple[str, list[int]]]:
    """A correct checkpoint: every Linear except lm_head quantized, the rest BF16."""
    tensors = {
        "model.embed_tokens.weight": ("BF16", [VOCAB, HIDDEN]),
        "model.norm.weight": ("BF16", [HIDDEN]),
        "lm_head.weight": ("BF16", [VOCAB, HIDDEN]),
    }
    for i in range(layers):
        p = f"model.layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = ("BF16", [HIDDEN])
        tensors[f"{p}.post_attention_layernorm.weight"] = ("BF16", [HIDDEN])
        tensors[f"{p}.self_attn.q_norm.weight"] = ("BF16", [HEAD_DIM])
        tensors[f"{p}.self_attn.k_norm.weight"] = ("BF16", [HEAD_DIM])
        for module, (out_f, in_f) in LINEARS.items():
            for suffix, value in quantized_module_tensors(fmt, out_f, in_f).items():
                tensors[f"{p}.{module}.{suffix}"] = value
    return tensors


def unquantize(tensors: dict, *module_substrings: str) -> dict:
    """Copy of `tensors` with the matching quantized modules (all by default) BF16 again."""
    out = dict(tensors)
    for name in [n for n in tensors if n.endswith(".weight_packed")]:
        module = name[: -len(".weight_packed")]
        if module_substrings and not any(s in module for s in module_substrings):
            continue
        for suffix in (
            "weight_packed",
            "weight_scale",
            "weight_global_scale",
            "input_global_scale",
        ):
            out.pop(f"{module}.{suffix}", None)
        out_f, in_f = LINEARS[module.split(".", 3)[3]]
        out[f"{module}.weight"] = ("BF16", [out_f, in_f])
    return out


def as_headers(tensors: dict) -> dict[str, dict]:
    """The `{name: {"dtype", "shape"}}` form that sanity.check_structure takes."""
    return {k: {"dtype": d, "shape": list(s)} for k, (d, s) in tensors.items()}


def write_header_only(
    path, tensors: dict, metadata: dict | None = None, *, zero_body: bool = False
) -> None:
    """A .safetensors file with a valid header and no data, enough for header readers; with
    `zero_body`, its zero-filled data too (small tensors only: fingerprints hash the bytes)."""
    offset, full = 0, {}
    if metadata is not None:
        full["__metadata__"] = metadata
    sizes = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1, "I8": 1}
    for name, (dtype, shape) in tensors.items():
        n = sizes[dtype]
        for d in shape:
            n *= d
        full[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + n]}
        offset += n
    blob = json.dumps(full).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + (b"\0" * offset if zero_body else b""))


def _args(group_size: int, strategy: str, scale_dtype: str, dynamic, observer) -> dict:
    return {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "group_size": group_size,
        "strategy": strategy,
        "block_structure": None,
        "dynamic": dynamic,
        "actorder": None,
        "scale_dtype": scale_dtype,
        "zp_dtype": None,
        "observer": observer,
        "observer_kwargs": {},
    }


def ct_quant_config(fmt: Format) -> dict:
    if fmt == "nvfp4":
        weights = _args(16, "tensor_group", "torch.float8_e4m3fn", False, None)
        inputs = _args(16, "tensor_group", "torch.float8_e4m3fn", "local", "static_minmax")
    else:
        weights = _args(32, "group", "torch.uint8", False, None)
        inputs = _args(32, "group", "torch.uint8", True, None)
    return {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": weights,
                "input_activations": inputs,
                "output_activations": None,
                "format": CT_FORMAT[fmt],
            }
        },
        "quant_method": "compressed-tensors",
        "kv_cache_scheme": None,
        "format": CT_FORMAT[fmt],
        "quantization_status": "compressed",
        "global_compression_ratio": None,
        "ignore": ["lm_head"],
    }


QWEN3_BASE_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "intermediate_size": 25600,
    "layer_types": ["full_attention"] * 64,
    "max_position_embeddings": 40960,
    "model_type": "qwen3",
    "num_attention_heads": 64,
    "num_hidden_layers": 64,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}


def qwen3_config(fmt: Format | None = None, **overrides) -> dict:
    config = {
        **copy.deepcopy(QWEN3_BASE_CONFIG),
        "dtype": "bfloat16",
        "transformers_version": "4.57.1",
        **overrides,
    }
    if fmt is not None:
        config["quantization_config"] = ct_quant_config(fmt)
    return config


def write_checkpoint(path, tensors: dict, config: dict | None = None):
    """A header-only single-shard checkpoint directory."""
    path.mkdir(parents=True, exist_ok=True)
    write_header_only(path / "model.safetensors", tensors)
    if config is not None:
        (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return path
