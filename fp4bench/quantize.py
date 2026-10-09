# METHODOLOGY.md#quantization
import gc
import hashlib
import importlib.metadata
import json
import random
import shutil
from datetime import UTC, datetime
from pathlib import Path

from fp4bench import settings
from fp4bench.core.files import file_sha256, write_text_atomic
from fp4bench.core.types import CheckpointKind, Format
from fp4bench.prompts import load_sharegpt_user_texts
from fp4bench.studies import model as qwen3

IGNORE = ["lm_head"]
CALIB_SAMPLES, CALIB_LEN = 256, 1024
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "LICENSE",
)


def build_calibration_ids(
    tokenizer,
    texts: list[str],
    n: int = CALIB_SAMPLES,
    max_len: int = CALIB_LEN,
    seed: int | None = None,
) -> list[list[int]]:
    """The first `n` texts of a seeded shuffle, tokenized as plain text, cut to `max_len`."""
    seed = settings.SEED + 3 if seed is None else seed
    if len(texts) < n:
        raise ValueError(f"only {len(texts)} ShareGPT prompts, need {n} for calibration")
    pool = list(texts)
    random.Random(seed).shuffle(pool)
    return [
        tokenizer.encode(t, add_special_tokens=False, split_special_tokens=True)[:max_len]
        for t in pool[:n]
    ]


def _ids_sha256(ids: list[list[int]]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def window_nll(model, window: list[int]) -> float:
    """NLL of one token window as the server's prompt_logprobs define it."""
    import torch

    ids = torch.tensor([window], device=model.device)
    logits = model(input_ids=ids, use_cache=False).logits
    return float(torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids[0, 1:]))


def bf16_reference_nll(model, windows: list[list[int]], nll_prompts_sha256: str) -> dict:
    """The BF16 model's NLL per window, their mean, and what they were computed on."""
    import torch

    with torch.no_grad():
        per_prompt = [window_nll(model, w) for w in windows]
    return {
        "per_prompt": per_prompt,
        "mean": sum(per_prompt) / len(per_prompt),
        "source_revision": qwen3.SRC_MODEL_REVISION,
        "nll_prompts_sha256": nll_prompts_sha256,
    }


def reference_problems(ref, n_windows: int, nll_prompts_sha256: str) -> list[str]:
    """What is wrong with a stored reference as a reference for the current NLL prompts."""
    if not isinstance(ref, dict):
        return ["it is not a JSON object"]
    problems = []
    per_prompt = ref.get("per_prompt")
    if not isinstance(per_prompt, list):
        problems.append(
            f"per_prompt is {per_prompt!r}, expected a list with one entry per NLL prompt"
        )
    elif len(per_prompt) != n_windows:
        problems.append(
            f"per_prompt has {len(per_prompt)} entries, expected {n_windows} (one per NLL prompt)"
        )
    if not isinstance(ref.get("mean"), (int, float)):
        problems.append(f"mean is {ref.get('mean')!r}, expected a number")
    if ref.get("source_revision") != qwen3.SRC_MODEL_REVISION:
        problems.append(
            f"source_revision is {ref.get('source_revision')!r}, expected "
            f"{qwen3.SRC_MODEL_REVISION!r}"
        )
    if ref.get("nll_prompts_sha256") != nll_prompts_sha256:
        problems.append(
            f"nll_prompts_sha256 is {ref.get('nll_prompts_sha256')!r}, but the NLL "
            f"prompts file hashes to {nll_prompts_sha256!r}"
        )
    return problems


def linear_quantization(model) -> tuple[list[str], list[str]]:
    """Sorted names of the Linear modules with a W4A4 scheme, and of those without."""
    import torch

    quantized, plain = [], []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        scheme = getattr(module, "quantization_scheme", None)
        w4a4 = (
            scheme is not None
            and scheme.weights is not None
            and scheme.input_activations is not None
        )
        (quantized if w4a4 else plain).append(name)
    return sorted(quantized), sorted(plain)


def assert_all_linear_quantized(quantized: list[str], plain: list[str]) -> None:
    """Every Linear except lm_head is quantized, and lm_head is not (§6)."""
    if not quantized:
        raise RuntimeError("nothing was quantized: no Linear module carries a W4A4 scheme")
    problems = []
    missing = [n for n in plain if n not in IGNORE]
    if missing:
        shown = ", ".join(missing[:20]) + (", ..." if len(missing) > 20 else "")
        problems.append(f"{len(missing)} Linear module(s) not quantized: {shown}")
    quantized_ignored = [n for n in quantized if n in IGNORE]
    if quantized_ignored:
        problems.append(f"{quantized_ignored} must stay BF16 but got quantized")
    if problems:
        raise RuntimeError("coverage is not every Linear except lm_head: " + "; ".join(problems))


def copy_tokenizer_files(src_dir, dst_dir) -> list[str]:
    """Copy the tokenizer, chat template and license files that exist, byte for byte."""
    copied = []
    for name in TOKENIZER_FILES:
        src = Path(src_dir) / name
        if src.exists():
            shutil.copyfile(src, Path(dst_dir) / name)
            copied.append(name)
    return copied


def config_parity(fmt: Format, out_dir) -> dict:
    """Config keys, other than quantization_config, that differ from the other format's."""
    from fp4bench.sanity import config_diff, load_config

    other_fmt = Format.MXFP4 if fmt is Format.NVFP4 else Format.NVFP4
    other = qwen3.volume_dir(CheckpointKind.ours(other_fmt))
    if not (Path(other) / "config.json").is_file():
        return {"compared_with": None, "differing_keys": []}
    mx_dir, nv_dir = (other, out_dir) if fmt is Format.NVFP4 else (out_dir, other)
    diff = config_diff(load_config("MX", mx_dir), load_config("NV", nv_dir))
    return {"compared_with": str(other), "differing_keys": sorted(diff)}


def quantize(fmt: Format) -> dict:
    """Make the `fmt` checkpoint in its volume directory; returns its provenance."""
    scheme, out_dir = fmt.spec.scheme, qwen3.volume_dir(CheckpointKind.ours(fmt))
    installed = importlib.metadata.version("llmcompressor")
    if installed != settings.LLMCOMPRESSOR_VERSION:
        raise RuntimeError(
            f"llmcompressor {installed} is installed, but the experiment pins "
            f"{settings.LLMCOMPRESSOR_VERSION}"
        )

    texts = load_sharegpt_user_texts(settings.SHAREGPT_PATH)
    if len(texts) < CALIB_SAMPLES:
        raise ValueError(
            f"only {len(texts)} ShareGPT prompts, need {CALIB_SAMPLES} for calibration"
        )
    nll_path, ref_path = Path(settings.NLL_PROMPTS_PATH), Path(settings.BF16_REF_NLL_PATH)
    windows, nll_sha = json.loads(nll_path.read_text()), file_sha256(nll_path)
    reference = None
    if ref_path.exists():
        reference = json.loads(ref_path.read_text())
        problems = reference_problems(reference, len(windows), nll_sha)
        if problems:
            raise RuntimeError(
                f"the existing BF16 reference NLL {ref_path} does not match the current NLL "
                f"prompts ({'; '.join(problems)}); delete it to recompute it from the BF16 model"
            )

    import torch
    import transformers
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(qwen3.BF16_MODEL_DIR)
    calib = build_calibration_ids(tok, texts, CALIB_SAMPLES, CALIB_LEN, settings.SEED + 3)
    dataset = Dataset.from_dict(
        {"input_ids": calib, "attention_mask": [[1] * len(ids) for ids in calib]}
    )

    model = AutoModelForCausalLM.from_pretrained(
        qwen3.BF16_MODEL_DIR, dtype=torch.bfloat16, device_map="cuda"
    )
    try:
        computed = reference is None
        if computed:
            reference = bf16_reference_nll(model, windows, nll_sha)
            write_text_atomic(ref_path, json.dumps(reference, indent=2))
        recipe = QuantizationModifier(targets="Linear", scheme=scheme, ignore=list(IGNORE))
        oneshot(
            model=model,
            recipe=recipe,
            dataset=dataset,
            max_seq_length=CALIB_LEN,
            num_calibration_samples=CALIB_SAMPLES,
            shuffle_calibration_samples=False,
        )
        quantized, plain = linear_quantization(model)
        assert_all_linear_quantized(quantized, plain)
        shutil.rmtree(out_dir, ignore_errors=True)
        model.save_pretrained(out_dir, save_compressed=True)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    tokenizer_files = copy_tokenizer_files(qwen3.BF16_MODEL_DIR, out_dir)
    parity = config_parity(fmt, out_dir)

    from fp4bench.sanity import check_structure, quant_config_problems, read_safetensors_headers

    structure = check_structure(read_safetensors_headers(out_dir), fmt)
    if structure:
        raise RuntimeError(f"exported {fmt} checkpoint has structure problems: {structure}")
    config_problems = quant_config_problems(out_dir, fmt)
    if config_problems:
        raise RuntimeError(
            f"exported {fmt} checkpoint has quantization config problems: {config_problems}"
        )

    provenance = {
        "source_model": qwen3.SRC_MODEL_ID,
        "source_revision": qwen3.SRC_MODEL_REVISION,
        "format": fmt,
        "scheme": scheme,
        "llmcompressor_version": installed,
        "compressed_tensors_version": importlib.metadata.version("compressed-tensors"),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "recipe": {
            "modifier": "QuantizationModifier",
            "targets": "Linear",
            "scheme": scheme,
            "ignore": list(IGNORE),
        },
        "ignore": list(IGNORE),
        "calib_dataset": f"{settings.SHAREGPT_REPO}@{settings.SHAREGPT_REVISION}",
        "calib_samples": len(calib),
        "calib_tokens": sum(len(ids) for ids in calib),
        "calib_max_len": CALIB_LEN,
        "calib_seed": settings.SEED + 3,
        "calib_sha256": _ids_sha256(calib),
        "quantized_modules": quantized,
        "tokenizer_files": sorted(tokenizer_files),
        "config_parity": parity,
        "bf16_reference_nll": {
            "path": str(ref_path),
            "mean": reference["mean"],
            "computed_by_this_run": computed,
            "nll_prompts_sha256": reference["nll_prompts_sha256"],
        },
        "utc": datetime.now(UTC).isoformat(),
    }
    write_text_atomic(Path(out_dir) / settings.NV_PROVENANCE_FILE, json.dumps(provenance, indent=2))
    return provenance
