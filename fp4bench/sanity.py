# METHODOLOGY.md#checkpoint-checks
import asyncio
import hashlib
import json
import struct
from pathlib import Path

import aiohttp

from fp4bench import bytes_model as bm
from fp4bench.core.files import file_sha256
from fp4bench.core.types import Fingerprint, Format, FormatSpec

DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U16": 2,
    "U32": 4,
    "U64": 8,
    "BOOL": 1,
}
INDEX_FILE = "model.safetensors.index.json"


def _read_header(path: Path) -> dict:
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def _safetensors_files(model_dir: Path) -> list[Path]:
    index = model_dir / INDEX_FILE
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [model_dir / name for name in names]
    return sorted(model_dir.glob("*.safetensors"))


def read_safetensors_headers(model_dir) -> dict[str, dict]:
    model_dir = Path(model_dir)
    files = _safetensors_files(model_dir)
    if not files:
        raise ValueError(f"no safetensors files found in {model_dir}")
    tensors: dict[str, dict] = {}
    for path in files:
        header = _read_header(path)
        duplicated = sorted(tensors.keys() & header.keys())
        if duplicated:
            raise ValueError(f"tensor stored in two files (second: {path.name}): {duplicated[:3]}")
        tensors.update(header)
    return tensors


def layout_fingerprint(model_dir) -> Fingerprint:
    """sha256 over every tensor's dtype, shape and offsets, from the headers only."""
    headers = json.dumps(read_safetensors_headers(model_dir), sort_keys=True)
    return Fingerprint(hashlib.sha256(headers.encode()).hexdigest())


# METHODOLOGY.md#fingerprints
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


def content_fingerprint(model_dir) -> Fingerprint:
    """sha256 over the shards, config.json and the SERVING_FILES present."""
    model_dir = Path(model_dir)
    shards = _safetensors_files(model_dir)
    if not shards:
        raise ValueError(f"no safetensors files found in {model_dir}")
    config = model_dir / "config.json"
    if not config.exists():
        raise FileNotFoundError(f"{config}: config.json is part of the checkpoint's identity")
    serving = [path for path in (model_dir / name for name in SERVING_FILES) if path.is_file()]
    pairs = sorted([path.name, file_sha256(path)] for path in [*shards, config, *serving])
    return Fingerprint(hashlib.sha256(json.dumps(pairs).encode()).hexdigest())


def tensor_nbytes(info: dict) -> int:
    if info["dtype"] not in DTYPE_BYTES:
        raise ValueError(f"unknown safetensors dtype {info['dtype']!r}")
    numel = 1
    for d in info["shape"]:
        numel *= d
    return numel * DTYPE_BYTES[info["dtype"]]


def linear_modules(n_layers: int = bm.N_LAYERS) -> dict[str, tuple[int, int]]:
    """{module path: (out_features, in_features)} of every Linear except lm_head."""
    q_out, kv_out = bm.N_Q_HEADS * bm.HEAD_DIM, bm.N_KV_HEADS * bm.HEAD_DIM
    per_layer = {
        "self_attn.q_proj": (q_out, bm.HIDDEN),
        "self_attn.k_proj": (kv_out, bm.HIDDEN),
        "self_attn.v_proj": (kv_out, bm.HIDDEN),
        "self_attn.o_proj": (bm.HIDDEN, q_out),
        "mlp.gate_proj": (bm.INTERMEDIATE, bm.HIDDEN),
        "mlp.up_proj": (bm.INTERMEDIATE, bm.HIDDEN),
        "mlp.down_proj": (bm.HIDDEN, bm.INTERMEDIATE),
    }
    return {
        f"model.layers.{i}.{name}": shape
        for i in range(n_layers)
        for name, shape in per_layer.items()
    }


def dense_tensors(n_layers: int = bm.N_LAYERS) -> dict[str, list[int]]:
    """{name: shape} of every tensor that stays BF16: embeddings, norms and the untied lm_head."""
    tensors = {
        "model.embed_tokens.weight": [bm.VOCAB, bm.HIDDEN],
        "model.norm.weight": [bm.HIDDEN],
        "lm_head.weight": [bm.VOCAB, bm.HIDDEN],
    }
    for i in range(n_layers):
        p = f"model.layers.{i}"
        tensors[f"{p}.input_layernorm.weight"] = [bm.HIDDEN]
        tensors[f"{p}.post_attention_layernorm.weight"] = [bm.HIDDEN]
        tensors[f"{p}.self_attn.q_norm.weight"] = [bm.HEAD_DIM]
        tensors[f"{p}.self_attn.k_norm.weight"] = [bm.HEAD_DIM]
    return tensors


PACKED_SUFFIX = ".weight_packed"
WEIGHT_TENSOR_SUFFIXES = ("weight_packed", "weight_scale", "weight_global_scale")


def _quantized_modules(tensors: dict[str, dict]) -> list[str]:
    return sorted(k[: -len(PACKED_SUFFIX)] for k in tensors if k.endswith(PACKED_SUFFIX))


def quantized_bytes(tensors: dict[str, dict]) -> int:
    """Stored bytes of the quantized weights, input_global_scale excluded."""
    return sum(
        tensor_nbytes(tensors[f"{module}.{suffix}"])
        for module in _quantized_modules(tensors)
        for suffix in WEIGHT_TENSOR_SUFFIXES
        if f"{module}.{suffix}" in tensors
    )


def has_activation_scales(tensors: dict[str, dict]) -> bool:
    """Whether every quantized module stores an input_global_scale; False without any."""
    modules = _quantized_modules(tensors)
    return bool(modules) and all(f"{m}.input_global_scale" in tensors for m in modules)


def _described(name: str, info: dict) -> str:
    return f"{name} ({info['dtype']} {list(info['shape'])})"


def _short(value, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text)} chars)"


def _examples(items: list[str], limit: int = 3) -> str:
    return ", ".join(sorted(items)[:limit])


def _implied_group_size(shape: list[int], out_f: int, in_f: int) -> int | None:
    if len(shape) == 2 and shape[0] == out_f and shape[1] > 0 and in_f % shape[1] == 0:
        return in_f // shape[1]
    return None


def check_structure(
    tensors: dict[str, dict], fmt: Format, n_layers: int = bm.N_LAYERS
) -> list[str]:
    """G5a: problems with the W4A4 layout of format `fmt`; empty means OK."""
    spec = fmt.spec
    modules = linear_modules(n_layers)
    expected: set[str] = set()
    missing, unquantized, leftover, bad_packed = [], [], [], []
    no_scale, bad_scale_dtype, bad_scale_shape = [], [], []
    implied: dict[int, list[str]] = {}
    lacking = {g: [] for g in spec.global_scales}
    bad_global = {g: [] for g in spec.global_scales}
    for module, (out_f, in_f) in modules.items():
        packed, weight = tensors.get(f"{module}.weight_packed"), tensors.get(f"{module}.weight")
        if packed is None:
            if weight is None:
                missing.append(module)
            else:
                unquantized.append(module)
                expected.add(f"{module}.weight")
            continue
        expected.add(f"{module}.weight_packed")
        if weight is not None:
            leftover.append(module)
            expected.add(f"{module}.weight")
        if packed["dtype"] != "U8" or list(packed["shape"]) != [out_f, in_f // 2]:
            bad_packed.append(_described(module, packed))
        scale = tensors.get(f"{module}.weight_scale")
        if scale is None:
            no_scale.append(module)
        else:
            expected.add(f"{module}.weight_scale")
            if scale["dtype"] != spec.scale_dtype:
                bad_scale_dtype.append(_described(module, scale))
            if list(scale["shape"]) != [out_f, in_f // spec.group_size]:
                group = _implied_group_size(list(scale["shape"]), out_f, in_f)
                if group is None:
                    bad_scale_shape.append(_described(module, scale))
                else:
                    implied.setdefault(group, []).append(module)
        for g in spec.global_scales:
            info = tensors.get(f"{module}.{g}")
            if info is None:
                lacking[g].append(module)
                continue
            expected.add(f"{module}.{g}")
            if info["dtype"] != "F32" or list(info["shape"]) != [1]:
                bad_global[g].append(_described(module, info))
    missing_dense, bad_dense = [], []
    for name, shape in dense_tensors(n_layers).items():
        info = tensors.get(name)
        if info is None:
            missing_dense.append(name)
            continue
        expected.add(name)
        if info["dtype"] != "BF16" or list(info["shape"]) != shape:
            bad_dense.append(_described(name, info))
    unexpected = [name for name in tensors if name not in expected]

    problems = []

    def add(items: list[str], what: str) -> None:
        if items:
            problems.append(f"{len(items)} {what}: e.g. {_examples(items)}")

    add(missing, "Linear modules missing (neither weight_packed nor weight)")
    add(unquantized, f"of {len(modules)} Linear modules not quantized (no weight_packed)")
    add(leftover, "quantized Linear modules still store a .weight tensor")
    add(bad_packed, "weight_packed tensors are not U8 [out, in/2]")
    add(no_scale, "quantized modules have no weight_scale")
    add(bad_scale_dtype, f"weight_scale tensors are not {spec.scale_dtype}")
    for group in sorted(implied):
        add(
            implied[group],
            f"weight_scale tensors imply group size {group}, expected {spec.group_size}",
        )
    add(
        bad_scale_shape,
        f"weight_scale tensors have the wrong shape (expected [out, in/{spec.group_size}])",
    )
    for g in spec.global_scales:
        add(lacking[g], f"quantized modules lack {g}")
        add(bad_global[g], f"{g} tensors are not F32 [1]")
    add(missing_dense, "non-quantized tensors missing")
    add(bad_dense, "non-quantized tensors are not BF16 with the expected shape")
    add(unexpected, "unexpected tensors")
    return problems


def _load_config_json(model_dir) -> tuple[dict | None, str | None]:
    path = Path(model_dir) / "config.json"
    try:
        return json.loads(path.read_text()), None
    except FileNotFoundError:
        return None, f"config.json not found in {model_dir}"
    except ValueError as exc:
        return None, f"config.json is not valid JSON ({exc})"


def _quantization_config(model_dir) -> tuple[dict | None, str | None]:
    config, problem = _load_config_json(model_dir)
    if problem:
        return None, problem
    qc = config.get("quantization_config") if isinstance(config, dict) else None
    if not isinstance(qc, dict):
        return None, (
            "config.json has no quantization_config object (vLLM would serve the "
            "checkpoint unquantized)"
        )
    return qc, None


def quant_format(model_dir) -> str | None:
    """config.json's quantization_config["format"] (e.g. "nvfp4-pack-quantized"), if it has one."""
    qc, _ = _quantization_config(model_dir)
    fmt = (qc or {}).get("format")
    return fmt if isinstance(fmt, str) else None


def _args_problems(group: str, label: str, args, spec: FormatSpec) -> list[str]:
    if not isinstance(args, dict):
        return [f"{group} {label} is {args!r}, expected an object"]
    wanted = {
        "num_bits": 4,
        "type": "float",
        "group_size": spec.group_size,
        "strategy": spec.strategy,
        "symmetric": True,
    }
    got = {**args, "symmetric": args.get("symmetric", True)}  # QuantizationArgs defaults to True

    def same(a, b) -> bool:
        return a == b and type(a) is type(b)

    return [
        f"{group} {label} {field} is {got.get(field)!r}, expected {want!r}"
        for field, want in wanted.items()
        if not same(got.get(field), want)
    ]


def quant_config_problems(model_dir, fmt: Format) -> list[str]:
    """G5a: problems with config.json's quantization_config for format `fmt`."""
    spec = fmt.spec
    qc, problem = _quantization_config(model_dir)
    if problem:
        return [problem]
    assert qc is not None
    problems = []
    if qc.get("quant_method") != "compressed-tensors":
        problems.append(
            f"quant_method is {qc.get('quant_method')!r}, expected 'compressed-tensors'"
        )
    if qc.get("format") != spec.ct_format:
        problems.append(f"format is {qc.get('format')!r}, expected {spec.ct_format!r}")
    groups = qc.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        problems.append("no config_groups")
    else:
        for name, group in groups.items():
            problems += _group_problems(name, group, spec)
    if qc.get("ignore") != ["lm_head"]:
        problems.append(f"ignore is {_short(qc.get('ignore'))}, expected ['lm_head']")
    if qc.get("kv_cache_scheme") is not None:
        problems.append(
            f"kv_cache_scheme is set ({qc['kv_cache_scheme']!r}); the KV cache must stay BF16"
        )
    return problems


def _group_problems(name: str, group, spec: FormatSpec) -> list[str]:
    if not isinstance(group, dict):
        return [f"{name} is {group!r}, expected an object"]
    problems = []
    if group.get("targets") != ["Linear"]:
        problems.append(f"{name} targets is {group.get('targets')!r}, expected ['Linear']")
    if group.get("format") not in (None, spec.ct_format):
        problems.append(f"{name} format is {group['format']!r}, expected {spec.ct_format!r}")
    problems += _args_problems(name, "weights", group.get("weights"), spec)
    if group.get("input_activations") is None:
        problems.append(f"{name} has no input_activations (W4A16, not W4A4)")
    else:
        problems += _args_problems(name, "input_activations", group["input_activations"], spec)
    return problems


def _require_quantized_bytes(label: str, model_dir, n_bytes: int) -> None:
    if n_bytes == 0:
        raise ValueError(
            f"{label} checkpoint {model_dir} has zero quantized bytes: "
            "no '.weight_packed' tensors, so the size ratio is undefined"
        )


# METHODOLOGY.md#config-parity
CONFIG_PARITY_IGNORED = frozenset(
    {"quantization_config", "transformers_version", "_name_or_path", "torch_dtype", "dtype"}
)


def load_config(label: str, model_dir) -> dict:
    path = Path(model_dir) / "config.json"
    try:
        config = json.loads(path.read_text())
    except FileNotFoundError:
        raise ValueError(f"{label} checkpoint {model_dir} has no config.json") from None
    except ValueError as exc:
        raise ValueError(
            f"{label} checkpoint {model_dir}: config.json is not valid JSON ({exc})"
        ) from None
    if not isinstance(config, dict):
        raise ValueError(f"{label} checkpoint {model_dir}: config.json is not a JSON object")
    return config


def config_diff(mx_config: dict, nv_config: dict) -> dict[str, dict]:
    """Top-level keys whose values differ, as {key: {"mx": v, "nv": v}}, compared as JSON."""

    def canonical(value) -> str:
        return json.dumps(value, sort_keys=True)

    diff: dict[str, dict] = {}
    for key in sorted((mx_config.keys() | nv_config.keys()) - CONFIG_PARITY_IGNORED):
        if key in mx_config and key in nv_config:
            if canonical(mx_config[key]) != canonical(nv_config[key]):
                diff[key] = {"mx": mx_config[key], "nv": nv_config[key]}
        else:
            diff[key] = {"mx": mx_config[key]} if key in mx_config else {"nv": nv_config[key]}
    return diff


def checkpoint_report(mx_dir: str, nv_dir: str) -> dict:
    """G2 and G5a of the MX and NV checkpoints; raises without quantized weights or config."""
    mx, nv = read_safetensors_headers(mx_dir), read_safetensors_headers(nv_dir)
    mx_b, nv_b = quantized_bytes(mx), quantized_bytes(nv)
    _require_quantized_bytes("MX", mx_dir, mx_b)
    _require_quantized_bytes("NV", nv_dir, nv_b)
    diff = config_diff(load_config("MX", mx_dir), load_config("NV", nv_dir))
    return {
        "mx_quantized_bytes": mx_b,
        "nv_quantized_bytes": nv_b,
        "nv_over_mx": nv_b / mx_b,
        "expected_nv_over_mx": bm.BITS[Format.NVFP4] / bm.BITS[Format.MXFP4],
        "mx_quant_format": quant_format(mx_dir),
        "nv_quant_format": quant_format(nv_dir),
        "nv_has_activation_scales": has_activation_scales(nv),
        "config_diff": diff,
        "mx_problems": (
            check_structure(mx, Format.MXFP4) + quant_config_problems(mx_dir, Format.MXFP4)
        ),
        "nv_problems": (
            check_structure(nv, Format.NVFP4)
            + quant_config_problems(nv_dir, Format.NVFP4)
            + [f"config differs from MX: {key}" for key in diff]
        ),
    }


def prompt_nll_from_response(body: dict, prompt_ids: list[int]) -> float:
    """Mean negative log-likelihood per predicted prompt token (first token has no logprob)."""
    plp = body["choices"][0].get("prompt_logprobs")
    if plp is None:
        raise ValueError(
            "response has no prompt_logprobs; was the request sent with prompt_logprobs?"
        )
    if len(plp) != len(prompt_ids):
        raise ValueError(
            f"prompt_logprobs has {len(plp)} entries for a prompt of {len(prompt_ids)} tokens"
        )
    total = 0.0
    for pos in range(1, len(prompt_ids)):
        total -= plp[pos][str(prompt_ids[pos])]["logprob"]
    return total / (len(prompt_ids) - 1)


NLL_TIMEOUT_S = 600


async def _nll_one(session, url, model, ids) -> float:
    payload = {
        "model": model,
        "prompt": ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 1,
    }  # METHODOLOGY.md#payloads
    async with session.post(url, json=payload) as resp:
        if not 200 <= resp.status < 300:
            raise RuntimeError(
                f"NLL request failed with HTTP {resp.status}: {(await resp.text())[:500]}"
            )
        return prompt_nll_from_response(await resp.json(), ids)


async def prompt_nlls_async(base_url: str, model: str, prompts: list[list[int]]) -> list[float]:
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0), timeout=aiohttp.ClientTimeout(total=NLL_TIMEOUT_S)
    ) as session:
        url = f"{base_url}/v1/completions"
        return list(await asyncio.gather(*(_nll_one(session, url, model, p) for p in prompts)))


def prompt_nlls(base_url: str, model: str, prompts: list[list[int]]) -> list[float]:
    """Mean NLL per predicted token of each prompt, in prompt order."""
    return asyncio.run(prompt_nlls_async(base_url, model, prompts))
