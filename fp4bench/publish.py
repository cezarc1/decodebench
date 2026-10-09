import json
import os
from dataclasses import dataclass
from pathlib import Path

from fp4bench import bytes_model as bm
from fp4bench import settings
from fp4bench.core.types import CheckpointKind, Format, KvDtype, Treatment
from fp4bench.studies import model

REPO_URL = None  # METHODOLOGY.md#publishing

DELETE_PATTERNS = ["*.safetensors", "*.json", "*.md", "*.jinja"]


@dataclass(frozen=True)
class Kind:
    fmt: Format
    label: str
    dir_attr: str
    problems_key: str
    weights: str
    activations: str
    calibration: str


KINDS: dict[str, Kind] = {
    CheckpointKind.MX: Kind(
        Format.MXFP4,
        "MXFP4",
        "MX_MODEL_DIR",
        "mx_problems",
        weights="FP4 E2M1 values in 32-element blocks, E8M0 (power-of-two) block scales, "
        f"{bm.BITS[Format.MXFP4]} bits per weight",
        activations="4-bit, quantized dynamically per 32-element block at runtime; this format "
        "has no static activation scales, so none are stored",
        calibration="MXFP4 has no static activation scales, so the data does not change this "
        "checkpoint's weights. The NVFP4 checkpoint uses the same set.",
    ),
    CheckpointKind.NV: Kind(
        Format.NVFP4,
        "NVFP4",
        "NV_MODEL_DIR",
        "nv_problems",
        weights="FP4 E2M1 values in 16-element blocks, E4M3 block scales plus one FP32 "
        f"per-tensor global scale, {bm.BITS[Format.NVFP4]} bits per weight",
        activations="4-bit, quantized per 16-element block at runtime, with a static per-tensor "
        "activation scale (`input_global_scale`) calibrated on the data below",
        calibration="The data sets NVFP4's static per-tensor activation scale "
        "(`input_global_scale`). The MXFP4 checkpoint uses the same set.",
    ),
}

REQUIRED_PROVENANCE = (
    "source_model",
    "source_revision",
    "format",
    "llmcompressor_version",
    "compressed_tensors_version",
    "recipe",
    "calib_dataset",
    "calib_samples",
    "calib_tokens",
    "calib_max_len",
    "calib_seed",
    "quantized_modules",
    "bf16_reference_nll",
    "utc",
)


def _kind(kind: str) -> Kind:
    try:
        return KINDS[kind]
    except (KeyError, TypeError):
        raise ValueError(
            f"unknown checkpoint kind {kind!r}; expected one of {sorted(map(str, KINDS))}"
        ) from None


def assert_publishable(report: dict, kind: str) -> None:
    """Raise unless the `kind` checkpoint may be published (METHODOLOGY.md#publishing)."""
    key = _kind(kind).problems_key
    problems = report.get(key)
    if not isinstance(problems, list):
        raise ValueError(
            f"checkpoint report has no {key} list ({problems!r}); cannot tell "
            "whether the checkpoint is publishable"
        )
    if problems:
        raise ValueError(f"{kind} checkpoint structure problems: {problems}")
    ratio, expected = report["nv_over_mx"], report["expected_nv_over_mx"]
    if not abs(ratio / expected - 1) < 0.02:
        raise ValueError(
            f"quantized bytes ratio NV/MX {ratio:.4f} is off the format layout "
            f"{expected:.4f} (4.5 / 4.25)"
        )


def _check_provenance(kind: str, prov: dict) -> None:
    spec = _kind(kind)
    if not isinstance(prov, dict):
        raise ValueError(f"provenance is not a JSON object: {prov!r}")
    missing = [k for k in REQUIRED_PROVENANCE if k not in prov]
    if missing:
        raise ValueError(f"provenance lacks {missing}")
    if prov["format"] != spec.fmt:
        raise ValueError(
            f"provenance is for format {prov['format']!r}, but a {kind} checkpoint "
            f"is {spec.fmt.value!r}; the wrong fp4bench_quant.json was copied?"
        )
    if "@" not in str(prov["calib_dataset"]):
        raise ValueError(f"calib_dataset {prov['calib_dataset']!r} is not <repo>@<revision>")
    if not isinstance(prov["bf16_reference_nll"], dict) or "mean" not in prov["bf16_reference_nll"]:
        raise ValueError("provenance bf16_reference_nll has no mean")


def model_card(kind: str, repo_id: str, report: dict, prov: dict) -> str:
    """The README.md of the `kind` checkpoint; refuses a report that fails assert_publishable."""
    spec = _kind(kind)
    assert_publishable(report, kind)
    _check_provenance(kind, prov)
    own_bytes = report[f"{kind}_quantized_bytes"]
    calib_repo, _, calib_revision = str(prov["calib_dataset"]).rpartition("@")
    recipe = prov["recipe"]
    name = repo_id.rsplit("/", maxsplit=1)[-1]
    design = ""
    if REPO_URL:
        design = (
            f"Design and pre-registration: "
            f"[EXPERIMENT.md]({REPO_URL.rstrip('/')}/blob/main/EXPERIMENT.md).\n"
        )
    serve = " ".join(
        [
            "vllm",
            "serve",
            repo_id,
            "--kv-cache-dtype",
            KvDtype.BF16,
            *model.TREATMENTS[Treatment(kind.upper())].server_args,
        ]
    )
    return f"""---
license: apache-2.0
base_model: {prov["source_model"]}
base_model_relation: quantized
pipeline_tag: text-generation
tags: [{spec.fmt}, fp4, compressed-tensors, llm-compressor, qwen3, vllm, blackwell]
---

# {name}

**{spec.label} W4A4** quantization of
[{prov["source_model"]}](https://huggingface.co/{prov["source_model"]})
(revision `{prov["source_revision"]}`, BF16). It was made for a controlled NVFP4-vs-MXFP4 decode
benchmark on NVIDIA B200 (`fp4bench`), not as a general-purpose release: the NVFP4 and MXFP4
checkpoints share the model, the tool, the recipe, the calibration data and the layer coverage,
and differ only in the format.
{design}
| | |
|---|---|
| Weights | {spec.label}: {spec.weights} |
| Activations | {spec.activations} |
| Coverage | all `Linear` layers except `lm_head` ({len(prov["quantized_modules"])} modules); embeddings, norms and `lm_head` stay BF16 |
| KV cache | BF16 at serve time (`--kv-cache-dtype bfloat16`); the checkpoint has no `kv_cache_scheme` |

## How it was made
- Tool: llm-compressor {prov["llmcompressor_version"]}, compressed-tensors {prov["compressed_tensors_version"]}.
  Recipe: `{recipe["modifier"]}(targets="{recipe["targets"]}", scheme="{recipe["scheme"]}", ignore={json.dumps(recipe["ignore"])})`.
  The plain preset recipe, with no weight-rounding optimization (GPTQ, AutoRound and the like).
- Calibration: {prov["calib_samples"]} prompts from the first user turn of ShareGPT conversations,
  `{calib_repo}` at revision `{calib_revision}`, at most {prov["calib_max_len"]} tokens each
  ({prov["calib_tokens"]} tokens in all, seed {prov["calib_seed"]}). {spec.calibration}
- The source weights are the BF16 original, so this is the only quantization step.
- Quantized at {prov["utc"]}.

## Measured size
- Quantized Linear weights stored in this checkpoint (`weight_packed`, `weight_scale` and, for
  NVFP4, `weight_global_scale`): **{own_bytes:,} bytes** ({own_bytes / 2**30:.2f} GiB).
- NVFP4 / MXFP4 stored bytes, measured on the pair: **{report["nv_over_mx"]:.4f}**
  (format-implied 4.5 / 4.25 = {report["expected_nv_over_mx"]:.4f}).

## Accuracy reference
- The BF16 source model's mean NLL is **{prov["bf16_reference_nll"]["mean"]:.4f}** nats/token,
  teacher-forced on the benchmark's {settings.NLL_PROMPTS} NLL prompts of {settings.NLL_LEN} tokens (sha256
  `{prov["bf16_reference_nll"].get("nll_prompts_sha256")}`), defined as vLLM's `prompt_logprobs`
  defines it.
- This checkpoint's own NLL on those prompts is measured by the benchmark run, not by this card.

## Serving
```bash
{serve}
```
Tested with vLLM v{settings.EXPECTED_VERSIONS["vllm"]} on B200 (SM100). The benchmark pins the kernel
with `--linear-backend`.

## License
Apache 2.0, same as the base model.
"""  # noqa: E501 - the template's lines are the card's lines


def publish(kind: str, repo_id: str, private: bool) -> str:
    """Upload the `kind` checkpoint with its model card; returns the commit URL."""
    from huggingface_hub import HfApi

    from fp4bench.sanity import checkpoint_report

    spec = _kind(kind)
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        raise RuntimeError("no HF token in the environment; check the huggingface-secret keys")
    report = checkpoint_report(model.MX_MODEL_DIR, model.NV_MODEL_DIR)
    assert_publishable(report, kind)
    folder = Path(getattr(model, spec.dir_attr))
    prov_path = folder / settings.NV_PROVENANCE_FILE
    if not prov_path.is_file():
        raise FileNotFoundError(
            f"{prov_path} not found: the {kind} checkpoint was not made by `fp4bench quantize` "
            "here (a fetched checkpoint keeps the file only if it was published with it)"
        )
    card = model_card(kind, repo_id, report, json.loads(prov_path.read_text()))
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    if private:
        visibility = api.repo_info(repo_id).private
        if visibility is not True:
            raise RuntimeError(
                f"{repo_id} is public or of unknown visibility "
                f"(private={visibility!r}), but a private upload was requested; "
                "make the repo private or choose another name. Nothing was uploaded."
            )
    (folder / "README.md").write_text(card)
    info = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        commit_message=f"Upload Qwen3-32B {spec.label} W4A4 (llm-compressor)",
        delete_patterns=DELETE_PATTERNS,
    )
    return str(info.commit_url)
