# Methodology: the facts behind fp4bench's constants and checks

[EXPERIMENT.md](EXPERIMENT.md) is the pre-registration: questions, hypotheses, design, decision
rules and the deviation log. [RESULTS.md](RESULTS.md) is what the runs found, and
[docs/run-log.md](docs/run-log.md) is what ran when. This file holds the rest: the vendor source
lines, log formats, platform limits and observations that justify a constant or a behaviour in
`fp4bench/`. Code that depends on one of these facts points here with a one-line
`# METHODOLOGY.md#<anchor>` comment. Each anchor is a stable lowercase slug, and the full list is at
the [end](#anchor-index).

**Version scope.** Every vLLM file:line below is for vLLM v0.31.0 (commit `db9527a`). FlashInfer
lines are for 0.7.0.post1, PyTorch for 2.13.0, and Modal client behaviour for `modal` 1.4.2. Where
a real server log shows a different line number for the same statement, both are given. Re-check
everything here when a pin changes ([EXPERIMENT.md §6][e6sw]).

---

<a id="stack"></a>
## Pinned stack

| Component | Pin | Checked how |
|---|---|---|
| Serving image | `vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b` (v0.31.0, amd64) | Recorded in every manifest line (`image`) |
| vLLM | 0.31.0, build commit `db9527a46873454610df6dbedf79a36d6bf1a7f6` (the image sets `VLLM_BUILD_COMMIT` to it) | `verify_manifest`: package version and `VLLM_BUILD_COMMIT` |
| FlashInfer | `flashinfer-python` and `flashinfer-cubin` 0.7.0.post1 | `verify_manifest` |
| PyTorch | 2.13.0 (`2.13.0+cu130` in the image) | `verify_manifest`; also checked when the quantization image is built |
| CUDA in the image | 13.0.2 (Modal driver 580.95.05 is CUDA 13.0) | Recorded (`cuda`) |
| transformers / nvidia-cutlass-dsl | 5.17.0 / 4.7.1 (as shipped in the image) | Recorded only |
| llm-compressor | 0.14.0, which pins compressed-tensors 0.19.0; makes both checkpoints | `quantize` refuses any other installed version |
| compressed-tensors | 0.19.0. vLLM loads both checkpoints through it | Recorded only |
| llm-inference-bench | v0.7.7, commit `c71ec1f2b34a4f1c8f702f1750ccd70da0e389e2`, cloned to `/opt/llm-inference-bench` | `verify_manifest` runs `git rev-parse HEAD` in that directory |
| httpx, rich, psutil | Unpinned (the bench's runtime dependencies) | Recorded only |
| Modal client | `modal==1.4.2` | `pyproject.toml` |

- **Version match rule.** A package version must equal the pin exactly after its PEP 440 local
  suffix is dropped, so `2.13.0+cu130` matches `2.13.0`.
- **GPU rule.** The name must be exactly `NVIDIA B200`, because a GB200 also reports compute
  capability 10.0. The compute capability must be `10.0`. The manifest records name, UUID,
  compute capability, driver, total memory and power limit.
- **Release re-check (2026-10-05).** vLLM v0.31.0 was still the newest tag; the newer Docker Hub
  tags are nightlies. FlashInfer 0.7.0.post1 was still the newest stable release (`v0.7.1rc2` is
  a pre-release). See [docs/run-log.md](docs/run-log.md).
- **Code version.** The CLI passes `FP4BENCH_CODE_COMMIT` (`git rev-parse HEAD`) and
  `FP4BENCH_CODE_DIRTY` (`1` if `git status --porcelain` lists anything) to the container, and
  `collect_manifest` records them as `code_commit` and `code_dirty`. Manifests from before commit
  `2976d48` lack them, and the analysis ignores them.

**Inputs.**

| Input | Pin | Note |
|---|---|---|
| `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` | BF16 source of both checkpoints. It ships safetensors only, so `*.pt` and `*.bin` are ignored on download; duplicates would add tens of GB |
| `nvidia/Qwen3-32B-NVFP4` | `16426c6eb87be9e27c14cc9fb318f9c7a5f8588c` | NVIDIA's NVFP4 checkpoint (ModelOpt 0.35.0, FP8 KV cache in its config). Treatment NVx: a smoke-run cross-check only, never a full-run input |
| ShareGPT V3 | `anon8231489123/ShareGPT_Vicuna_unfiltered` @ `192ab2185289094fc556ec8ce5ce1e8e587154ca`, file `ShareGPT_V3_unfiltered_cleaned_split.json` | About a 700 MB JSON. Taking the first human turn of each conversation gives 92,779 texts |

<a id="image"></a>
### Images

- **Base image.** The pinned vLLM image with its entrypoint cleared. It adds a `python` symlink to
  `python3`, `git`, the bench cloned at its pinned commit, and `pip install httpx rich psutil`.
  It sets `HF_XET_HIGH_PERFORMANCE=1`, `HF_HOME=/root/.cache/huggingface` and
  `LLM_BENCH_NO_UPDATE_CHECK=1`.
- **Quantization image.** The base image plus `llmcompressor==0.14.0`. Its last build step
  fails unless torch is still 2.13.0 (local suffix ignored), so a dependency swap fails the build
  instead of a B200 run hours later. On 2026-10-04 the image kept torch 2.13.0.

---

<a id="treatments"></a>
## Treatments

| ID | Checkpoint | Arguments after the common ones ([#server-args](#server-args)) | Linear kernel G1 expects | `fuse_act_quant` | Role |
|---|---|---|---|---|---|
| MX | ours, MXFP4 | `--linear-backend flashinfer_cutedsl` | `FlashInferMxFp4LinearKernel` (a fallback, [#kernel-selection](#kernel-selection)) | off | baseline |
| NV | ours, NVFP4 | `--linear-backend flashinfer_cutedsl` | `FlashInferCuteDslNvFp4LinearKernel` | on | R |
| NVa | ours, NVFP4 | `--linear-backend flashinfer_cudnn` ([#nv-alt](#nv-alt)) | `FlashInferCudnnNvFp4LinearKernel` | on | D, K |
| MXp | ours, MXFP4 | `--linear-backend flashinfer_cutedsl` | `FlashInferMxFp4LinearKernel` | off | A/A (G3) |
| NVnf | ours, NVFP4 | `--linear-backend flashinfer_cutedsl --compilation-config '{"pass_config": {"fuse_act_quant": false}}'` | `FlashInferCuteDslNvFp4LinearKernel` | off | F, Φ |
| NVx | NVIDIA's | `--linear-backend flashinfer_cutedsl` | `FlashInferCuteDslNvFp4LinearKernel` | on | smoke cross-check |
| NVc / NVt / NVd / NVv | ours, NVFP4 | `--linear-backend flashinfer_cutlass` / `flashinfer_trtllm` / `flashinfer_cudnn` / `cutlass` | the backend's class ([#kernel-selection](#kernel-selection)) | on | smoke kernel scan |

- **ID history.** The ID "NVa" dates from when the treatment was "NV-auto". EXPERIMENT.md calls
  it NV-alt, and calls MXp MX′ and NVnf NV-nf.
- **Round order.** The order (MX, NV, NVa, MXp, NVnf) is round 1 of the cyclic 5×5 Latin square of
  [§7.7][e77]. Round r is the list rotated by r mod n (`schedule.round_order`), so every treatment
  holds every position once. Experiment B and C rotate (MX, NV, MXp) the same way.
- **Ratios.** They are oriented so that a value above 1 means the denominator is faster: R = NV/MX,
  D = NVa/MX, K = NVa/NV, F = NVnf/MX, Φ = NV/NVnf, AA = MXp/MX. R = F·Φ holds exactly in every
  round.
- **Shared checkpoints.** MX and MXp serve one checkpoint, and so do NV, NVa, NVnf and the scan
  treatments. Each checkpoint is staged once per start ([#staging](#staging)).
- **Formats.** MX and MXp are MXFP4. Every other treatment is NVFP4 (`format_of`).

---

## Checkpoints

<a id="quantization"></a>
### Quantization recipe

This is the llm-compressor 0.14.0 API, read from its sdist:

```python
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
QuantizationModifier(targets="Linear", scheme="NVFP4" | "MXFP4", ignore=["lm_head"])
oneshot(model=..., recipe=..., dataset=<Dataset with "input_ids">, num_calibration_samples=256,
        max_seq_length=1024, shuffle_calibration_samples=False)
model.save_pretrained(out_dir, save_compressed=True)   # patched by oneshot
```

- **Dataset handling.** A Dataset that already has `input_ids` is used as is, with no
  tokenizing. With `shuffle_calibration_samples=False`, oneshot orders the samples by descending
  length (a fixed order) instead of drawing a random one.
- **What calibration does.** NVFP4 needs the data: its `input_global_scale` is a running min/max
  over it. MXFP4 is dynamic, so oneshot runs it data-free; the data is accepted but unused. This
  is the plain preset recipe, with no GPTQ, AutoRound or similar rounding optimization.
- **The calibration set.** The first 256 texts of a seed-3 (`SEED + 3`) shuffle of the ShareGPT
  first human turns. Each is tokenized as plain text (`split_special_tokens=True`) and cut to 1,024
  tokens, with an all-ones attention mask. The set is deterministic and identical for both formats,
  and its sha256 is in the provenance (`calib_sha256`).
- **Config.** `save_pretrained` copies the source `config.json` and patches in
  `quantization_config`, so the two formats' configs agree by construction
  ([#config-parity](#config-parity)).
- **Order of work.**
  1. The cheap checks run before the slow model load: the format, the llmcompressor version, at
     least 256 texts, the NLL prompts, and any existing BF16 reference.
  2. The model is loaded fresh for every call (BF16, `device_map="cuda"`).
  3. The BF16 reference NLL is computed before oneshot if it is missing ([#bf16-reference](#bf16-reference)).
  4. After oneshot, every Linear is checked ([#coverage](#coverage)). Only then is the output
     directory emptied, so a refused export keeps the previous checkpoint.
  5. The checkpoint is saved, the tokenizer files are copied, and parity, structure and
     quantization config are checked.
  6. `fp4bench_quant.json` is written with the provenance.
- **Tokenizer files.** These are copied byte for byte when present: `tokenizer.json`,
  `tokenizer_config.json`, `vocab.json`, `merges.txt`, `special_tokens_map.json`,
  `added_tokens.json`, `chat_template.jinja`, `generation_config.json` and `LICENSE`. M2 goes
  through the server-side chat template, so both treatments need the same files. Qwen3 keeps its
  chat template in `tokenizer_config.json`.

<a id="coverage"></a>
### Coverage

Every `torch.nn.Linear` except `lm_head` must carry a W4A4 scheme, meaning both `weights` and
`input_activations` are set. `lm_head` must not. That is 448 modules (64 layers × 7:
q/k/v/o_proj and gate/up/down_proj). This is NVIDIA's production coverage (`nvidia/Qwen3-32B-NVFP4`
excludes only `lm_head`) applied to both formats. A model with nothing quantized, an unquantized
Linear, or a quantized `lm_head` is refused before saving.

<a id="bf16-reference"></a>
### BF16 reference NLL

It is computed on the 64 NLL windows ([#prompts](#prompts)), defined exactly as the server's
`prompt_logprobs` define it ([#nll](#nll)):
- **Input.** The window's ids go in as stored: no BOS or other token is inserted. One row, no
  padding, no KV cache.
- **Pairing.** `logits[:-1]` are paired with `ids[1:]` (L−1 pairs), and the loss is their plain
  mean.
- **Precision.** Logits are upcast to float32 before the log-softmax, as vLLM does
  (`vllm/v1/sample/sampler.py` `compute_logprobs`: `log_softmax(dim=-1, dtype=float32)`; the
  prompt-logprobs target for index i is prompt token i+1, `gpu_model_runner.py`).
- **Grad mode.** It runs under `torch.no_grad`, not `inference_mode`, because llm-compressor
  warns that tensors created in inference mode break its in-place updates later.
- **Stored file.** `bf16_reference_nll.json` holds `{per_prompt, mean, source_revision,
  nll_prompts_sha256}`. An existing file is reused only if it matches the current windows (count
  and sha256) and the source revision.
- **Use in a run.** A run refuses a reference whose `nll_prompts_sha256` is not that of the run's
  NLL prompts. Only `per_prompt` and `mean` are copied into each manifest line's `inputs`, so the
  run directory is self-contained, and a reference that changes mid-run is refused like any other
  input.
- **Value.** The BF16 mean is 1.7730 nats/token.

<a id="checkpoint-checks"></a>
### G2 and G5a: the checkpoint report

**On-disk layout per quantized Linear.** It was read from the compressed-tensors 0.19.0
compressors (`nvfp4/base.py`, `mxfp4/base.py`; no `weight_zero_point` is stored) and checked on
2026-10-04 against the published headers of `RedHatAI/Qwen3-32B-NVFP4` @ `10a4cab` (llm-compressor
NVFP4, `ignore` `["lm_head"]`, 2,051 tensors) and `INCModel/Qwen3-32B-MXFP4-CT-AutoRound` @
`e32b451` (MXFP4 with BF16 attention). Verbatim excerpts are the `REAL_*` constants of
`tests/test_sanity.py`, and `tests/ct_checkpoints.py` builds header-only fixtures from this layout:

| Format | `weight_packed` | `weight_scale` | Global scales (F32 [1]) |
|---|---|---|---|
| nvfp4 | U8 [out, in/2] | F8_E4M3 [out, in/16] | `weight_global_scale`, `input_global_scale` |
| mxfp4 | U8 [out, in/2] | U8 (E8M0 exponents) [out, in/32] | none |

- **BF16 tensors.** Everything else stays BF16 under its original name: `embed_tokens` and the
  untied `lm_head` [151,936, 5,120], `model.norm` [5,120], and per layer `input_layernorm` and
  `post_attention_layernorm` [5,120] and `q_norm` and `k_norm` [128].
- **Shards.** The shards are the files `model.safetensors.index.json` lists, or every
  `*.safetensors` file when there is no index. A tensor stored in two files is an error.
- **Structure check (G5a).** Every Linear must have `weight_packed` U8 [out, in/2] and a
  `weight_scale` of the format's dtype and shape [out, in/group]. A wrong group size is reported
  as the group the shape implies. NVFP4 also needs both global scales as F32 [1]. No quantized
  module may keep a `.weight`. The BF16 tensors must have their dtype and shape, and any other
  tensor is a problem. Each problem names a count and up to three examples.
- **Format specs.** The `quantization_config.format` and `QuantizationArgs.strategy` values are
  what vLLM's `_is_nvfp4_format` and `_is_mxfp4` test:
  - nvfp4: `nvfp4-pack-quantized`, group 16, strategy `tensor_group`.
  - mxfp4: `mxfp4-pack-quantized`, group 32, strategy `group`.
- **Quantization config check (G5a).**
  - `quant_method` must be `compressed-tensors` and `format` the format's name.
  - Every config group must target `["Linear"]`. Its `format` may be absent or null (older
    writers) or must equal the format's name.
  - Both `weights` and `input_activations` must be `num_bits` 4, `type` `float`, the format's
    group size and strategy, and `symmetric` true. `QuantizationArgs` defaults to true, and the
    comparison is type-exact, so `True` is not `1`. A group without `input_activations` is W4A16,
    which is a problem.
  - `ignore` must be exactly `["lm_head"]`.
  - There must be no `kv_cache_scheme`, because the KV cache stays BF16.
  - Without any `quantization_config`, vLLM would serve the checkpoint unquantized.
- **Bytes check (G2).** The quantized bytes of a checkpoint are `weight_packed` +
  `weight_scale` + `weight_global_scale` of every module with a `weight_packed`.
  `input_global_scale` is an activation scale (4 bytes per module) and is left out. NV/MX must be
  within 2% of 4.5/4.25 = 1.0588235 ([§9][e9]).
- **Measured bytes (published revisions).** MX 16,577,986,560 B, NV 17,553,164,032 B, ratio
  1.0588236 (`data/runs/checkpoint_report.json`). MX is exactly 31,205,621,760 × 4.25/8. NV is
  31,205,621,760 × 4.5/8 plus 448 × 4 B of `weight_global_scale`. The NV bytes match
  `RedHatAI/Qwen3-32B-NVFP4` exactly.
- **Weight memory.** vLLM's load-time weight memory is recorded per session from "Model loading
  took X GiB" (logged at `model_runner.py:407`). NV is 19.53 GiB in `full-1`.
- **When the report is computed.** Every run start computes it from the volume directories. It
  reads headers only, so it takes seconds, and it is recorded in the manifest line. Staging then
  proves the local copies have the same content fingerprints ([#fingerprints](#fingerprints)).
  `fp4bench check` writes the same report to `results/checkpoint_report.json` as a pre-flight
  look. The analysis uses that file only as a labelled fallback for a run whose manifest has no
  report, and with no report at all G2 and G5a fail.
- **NVx.** NVIDIA's checkpoint is not checked: its ModelOpt tensor names (`weight`,
  `weight_scale`, `weight_scale_2`, `input_scale`, `k_scale`/`v_scale`; `nvidia/Qwen3-32B-NVFP4` @
  `16426c6`) and FP8 `kv_cache_scheme` would be rejected by design. It is identified by its content
  fingerprint only.

<a id="config-parity"></a>
### config.json parity

- **Rule.** Any top-level `config.json` key that differs between MX and NV, other than
  `quantization_config`, `transformers_version`, `_name_or_path`, `torch_dtype` and `dtype`, is an
  NV problem (G5a). A difference in `layer_types`, `sliding_window`, rope settings and similar keys
  would change decode speed for reasons unrelated to the weight format. Values are compared as
  JSON, so `True` is not `1`.
- **No repair.** Saving either format records the differing keys in its provenance
  (`config_parity`) and never rewrites the other checkpoint's config; the checkpoint report makes
  any difference an NV problem, so G5a refuses the pair.
- **History.** Revision 1 needed a repair, `align_config_with_mx`, which made NV's config MX's
  config plus NV's `quantization_config`: transformers 5.17 serialized identical RoPE values
  differently (`rope_scaling` + `rope_theta` against `rope_parameters`, and `bos_token_id` absent
  against `null`). In Revision 2 the configs had no differing keys, and the repair was removed
  ([#legacy](#legacy)).

<a id="fingerprints"></a>
### Fingerprints

- **Layout fingerprint.** sha256 over every tensor's dtype, shape and offsets, from the headers
  only. It is cheap but cannot see the weight bytes.
- **Content fingerprint.** sha256 over the sorted `[filename, sha256]` pairs of:
  - every shard;
  - `config.json`;
  - each of these when present: `hf_quant_config.json` (drives quantization detection),
    `generation_config.json` (sets the default sampling), `tokenizer.json`,
    `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja`, `vocab.json`,
    `merges.txt` and `added_tokens.json` (they shape the prompts).
- **What the content fingerprint excludes.** An absent optional file is skipped, so a file's
  presence is itself part of the identity. fp4bench's own bookkeeping files are never covered:
  `fp4bench_fetch.json` varies with fetch time, and `README.md` is written at publish time; the
  same goes for `fp4bench_quant.json`.
- **Cost.** It reads every byte, about a minute for a 13 GB checkpoint, once per run start.

<a id="publishing"></a>
### Publishing

- **Before the Hub is touched.**
  - An HF token must be in the environment (`HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`).
  - The checkpoint report must pass for the published kind. Only that kind's problems list
    counts, and the NV/MX bytes ratio must be within 2% of 4.5/4.25 (NaN fails).
  - The provenance file must be there and build a model card.
- **Both checkpoints needed.** The report compares MX with NV, so both must be in place, quantized
  or fetched, before either is published.
- **Visibility.** Uploads are private unless `--public` is given. `create_repo(exist_ok=True)`
  leaves an existing repo's visibility as it was, so a private upload checks
  `repo_info().private is True` and refuses otherwise. `README.md` is written into the folder only
  after that check.
- **Mirror upload.** Remote `*.safetensors`, `*.json`, `*.md` and `*.jinja` files that are not in
  the local folder are deleted.
- **Model card.** Format, coverage, recipe, calibration, measured bytes and the BF16 reference
  NLL. It has no link back to the code while `publish.REPO_URL` is `None`.
- **Published (private, 2026-10-05).**
  - `ggamecrazy/Qwen3-32B-MXFP4-W4A4` @ `9a719cda64d8748830a2da6c40417374c1e2f8a6`
  - `ggamecrazy/Qwen3-32B-NVFP4-W4A4` @ `6b9b0a43384970072017e0c1d200fcc521bf4561`

<a id="fetch"></a>
### Fetching

- **What it does.** `fetch` makes the volume directory an exact mirror of a published revision:
  the directory is removed first, then `snapshot_download` fills it. It writes
  `fp4bench_fetch.json` (`repo_id`, `revision`, `utc`) next to the weights.
- **Which studies need it.** A study with `require_published` (full, smoke-nf, expb, smoke-c,
  expc) refuses to start unless both MX and NV were fetched: the full run serves exactly the
  published revisions ([§6][e6]). The field defaults to true, so a new study must opt out, as the
  kernel-scan smoke does. That smoke may serve locally quantized checkpoints, but needs
  `fetch --kind nvidia` for NVx.
- **Reproducing from the published checkpoints.** Run `prepare`, then `fetch --kind mx` and
  `fetch --kind nv` with the published revisions, then `run`. This skips `quantize`, `check` and
  `publish`. `prepare` is still needed: it builds the prompts with Qwen3's tokenizer, and it
  downloads the BF16 source the tokenizer is read from.
- **The record.** A fetch record that does not name a `repo_id` and a `revision` is an error. The
  input-identity check ignores its `utc` ([#run-integrity](#run-integrity)).
- **Duration.** Fetching the 17 GB NV checkpoint took 15 minutes.

<a id="staging"></a>
### Local staging

The sessions serve local copies on the container's ephemeral disk (`/local`), never the network
volume ([§6][e6]).
- **Once per checkpoint.** Each distinct volume directory the study's treatments serve is staged
  once.
- **Reuse.** A local copy whose content fingerprint equals the volume's is kept (a restart in the
  same container). Any other existing copy is stale or half written and is replaced.
- **Atomic copy.** The copy goes to `<dir>.copying` next to the final directory, so the rename
  stays on one filesystem, and is then renamed into place. A copy that dies midway never shows up
  under the final name, and what a killed copy left is removed first.
- **Verification.** Every copy's content fingerprint must equal the volume's. Staging stops at the
  first failure, because the run aborts and the remaining copies would only cost time.
- **Last step.** Staging is the last start-up step and runs only if nothing else failed, because
  it copies tens of GB. MX + NV + NVIDIA's checkpoint are about 60 GB.

---

## Serving

<a id="server-args"></a>
### Server arguments

Every treatment of a study gets the same arguments (`ServerSettings.args`), followed by the
treatment's own ([#treatments](#treatments)):

```
--served-model-name fp4bench --host 127.0.0.1 --port 8000 --max-model-len <4096>
--max-num-seqs 512 --max-num-batched-tokens 16384 --gpu-memory-utilization <0.90>
--no-enable-prefix-caching --seed 0 --api-server-count 4 --kv-cache-dtype <bfloat16>
[--hf-overrides <JSON>]
```

| Study | `--kv-cache-dtype` | `--gpu-memory-utilization` | `--max-model-len` | `--hf-overrides` |
|---|---|---|---|---|
| smoke, smoke-nf, full | bfloat16 | 0.90 | 4096 | (not passed) |
| expb | fp8 | 0.95 ([#kv-capacity](#kv-capacity)) | 4096 | (not passed) |
| smoke-c, expc | bfloat16 | 0.90 | 131072 | `{"max_position_embeddings": 131072}` ([#long-context](#long-context)) |

- **Study fields.** The KV dtype and memory fraction are study fields because Experiment B
  changes them. The window and the override are study fields because Experiment C changes them.
  With the defaults, the argv is byte for byte the one every earlier run was served with.
- **Memory fraction format.** It is written with two decimals, as it always was. A value that two
  decimals would round is refused rather than silently changed. `max_model_len` must be a positive
  int.
- **The override.** It is appended last as one argv element holding the JSON. In a shell that is
  `--hf-overrides '{"max_position_embeddings": 131072}'`. `VllmServer` starts vLLM without a
  shell, as for NVnf's `--compilation-config`, and vLLM v0.31.0 parses the value with
  `json.loads`. Anything but a non-empty JSON object is refused before the server starts.
- **API servers.** `--api-server-count 4` keeps the HTTP and detokenization frontend off the
  critical path ([§7.1][e71]). It also puts prometheus_client in multiprocess mode
  ([#preemption-metric](#preemption-metric)) and makes each API server a separate profiler
  frontend ([#profiler](#profiler)).
- **Defaults kept.** No speculative decoding. Async scheduling and CUDA graphs stay at their
  defaults.
- **M2 output length.** `--max-tokens 2048` for the bench, because prompt plus output must fit
  the 4,096-token window.
- **Audit trail.** The argv that actually ran is recorded in every servers row (`server_argv`) for
  G1.

<a id="server-lifecycle"></a>
### Server lifecycle

- **Process group.** `vllm serve` runs in its own process group (`start_new_session`), so that
  `stop()` can also kill an EngineCore that outlived its parent. A group id stays valid while any
  member lives, so the group kill reaches orphans.
- **Port check.** A start is refused if the port already accepts connections, because another
  server's `/health` would be mistaken for ours.
- **Health.** `/health` is polled every 5 s for up to 1,800 s. A process that exits during start
  is an error.
- **Stop.** SIGINT to the group, up to 120 s grace, then SIGKILL to the group. Then wait until
  `nvidia-smi` `memory.used` is below 2,048 MiB, for at most 180 s.
- **Per-server sequence** ([§7.5][e75]):
  1. Wait for health.
  2. Check KV capacity from the log ([#kv-capacity](#kv-capacity)).
  3. Run the preemption preflight read ([#preemption-metric](#preemption-metric)).
  4. Run the NLL check.
  5. Run the M1 blocks.
  6. Run the M2 cells.
  7. Check that the server is alive, parse the log, write the servers row.
- **Why the liveness checks.** `run_lib_decode` records request errors as invalid cells instead
  of raising. A dead server is therefore checked for before every M2 cell and after the last one;
  otherwise it would be reported as a missing bench file, or the session recorded as complete.

<a id="kernel-selection"></a>
### Kernel pins and kernel classes

- **Where vLLM logs the choice.** vLLM v0.31.0 logs the FP4 linear (GEMM) kernel once per engine
  process at INFO, from `vllm/model_executor/kernels/linear/__init__.py`:
  - line 975: `logger.info_once("Using %s for MXFP4 GEMM", kernel_cls.__name__)`
  - lines 1142/1186: `logger.info_once("Using %s for NVFP4 GEMM", <kernel class>.__name__)`
    (`full-1` logs show 1186)

  The captured group is the class name. Both the compressed-tensors W4A4 schemes and ModelOpt
  (NVx) go through this code. G1 requires a session's `linear_kernels` to be exactly the expected
  class.

| `--linear-backend` | NVFP4 class |
|---|---|
| `flashinfer_cutedsl` | `FlashInferCuteDslNvFp4LinearKernel` |
| `flashinfer_cutlass` | `FlashInferCutlassNvFp4LinearKernel` |
| `flashinfer_trtllm` | `FlashInferTrtllmNvFp4LinearKernel` |
| `flashinfer_cudnn` | `FlashInferCudnnNvFp4LinearKernel` |
| `cutlass` | `CutlassNvFp4LinearKernel` |

- **The default is the CuTe-DSL kernel.** With no `--linear-backend` on B200, vLLM picks
  `FlashInferCuteDslNvFp4LinearKernel` for NVFP4: it is first in `_POSSIBLE_NVFP4_KERNELS` on
  SM100 ([§4][e4]). "As deployed" therefore equals NV, and a default-kernel treatment would
  duplicate NV. This is why NV-auto became NV-alt ([§11][e11]) and why `nva_pin_problems` refuses a
  run whose NVa is unpinned or pinned to NV's kernel ([#nv-alt](#nv-alt)). An unknown backend
  raises instead of guessing a class.
- **The MXFP4 fallback.** v0.31.0 has no MXFP4 kernel under `flashinfer_cutedsl`. vLLM warns
  (`warning_once` at lines 365 and 394: "--linear-backend=X has no kernel for this linear layer
  type" / "--linear-backend=X was requested, but no ...") and selects automatically. It lands on
  `FlashInferMxFp4LinearKernel`, the only W4A4 MXFP4 path on SM100, which calls FlashInfer
  `mm_fp4` with `backend="cute-dsl"`. The fallback is expected, not a failure. The warning line
  (logged at :365 in `full-1`) is recorded as `linear_backend_fallback_lines`, the evidence that
  the pin did not choose the MX kernel. Both formats therefore run the same CuTe-DSL GEMM, with
  block size 32 (MXFP4) or 16 (NVFP4). In profiles it is
  `Sm100BlockScaledPersistentDenseGemmKernel` for both ([#profiler](#profiler)).
- **BF16 GEMMs.** The layers that stay BF16 have their own dispatch,
  `vllm/model_executor/layers/utils.py:626` in `dispatch_unquantized_gemm`, which logs
  `logger.info_once("Using FlashInfer %s for eligible unquantized BF16 GEMMs.",
  backend_spec.flashinfer_backend)`.
  - Under `--linear-backend flashinfer_cutedsl` the backend is `cute-dsl`, the only FlashInfer
    BF16 backend there.
  - The scan treatments (cutlass, trtllm, cudnn) do not select it, so this line also tells the
    treatments apart on the BF16 layers.
  - The `warning_once` at line 617 (the backend is unavailable) is a different line.
  - In the profiles, `lm_head` runs eagerly on cuBLAS (`nvjet_sm100_*` plus `splitKreduce_kernel`).

<a id="fusion"></a>
### SiLU·mul + activation-quant fusion

- **Only NVFP4 checkpoints get it.** vLLM v0.31.0 turns `pass_config.fuse_act_quant` on by
  default only for NVFP4 checkpoints. `enable_act_fusion` (`vllm/config/vllm.py:166-175`) is true
  when `ModelConfig.is_nvfp4_quantized()` is (`vllm/config/model.py:2240-2252`): compressed-tensors
  with an `nvfp4` format, or ModelOpt's `modelopt_fp4` (NVx). The reason is that the FP4 quant is
  a custom op that Inductor cannot fuse.
- **No MXFP4 equivalent.** There is no dense MXFP4 SiLU·mul + quant fusion at all; the only fused
  MXFP4 activation op is for MoE experts. MX and MXp therefore run unfused. The comparison can be
  matched only by turning NVFP4's fusion off ([#nv-nf](#nv-nf)), not MXFP4's on.
- **Who runs what.** The scan treatments serve our NVFP4 checkpoint, so they are fused like NV.

| Treatment | `fuse_act_quant` expected |
|---|---|
| MX, MXp, NVnf | off |
| NV, NVa, NVx, NVc, NVt, NVd, NVv | on |

- **The kernels.**
  - NV runs one fused `silu_mul_cvt_fp16_to_fp4` kernel before every `down_proj`. The op is
    `torch.ops._C.silu_and_mul_nvfp4_quant`; its allocation is in
    `vllm/model_executor/layers/fusion/fused_act_quant.py`.
  - MX runs an Inductor `triton_poi_fused_mul_silu_slice_<n>` followed by FlashInfer's CuTe-DSL
    MXFP4 quantize.
  - The servers run with `custom_ops ['none']`, so `SiluAndMul` is its native PyTorch form
    compiled by Inductor. NVFP4's fusion pass replaces it and the quant with one kernel.
- **The engine-config line (G1 evidence).** vLLM logs its whole config once, from the engine core,
  at INFO (`vllm/v1/engine/core.py:129-133`):
  `logger.info("Initializing a V1 LLM engine (v%s) with config: %s", VLLM_VERSION, vllm_config)`.
  - Its `compilation_config` is a dict repr (`vllm/config/compilation.py:818-843`). Its
    `pass_config` holds the values resolved after the optimization-level defaults and leaves out
    every field equal to the class default.
  - The `fuse_*` fields default to `None`, so a resolved `True` or `False` is always printed. For
    example `smoke-r2-2`'s NV session logged `'pass_config': {'fuse_norm_quant': False,
    'fuse_act_quant': True, 'fuse_attn_quant': False, ...}`, and MX the same with
    `'fuse_act_quant': False`.
  - The dict is flat. It is read with regexes and never evaluated, and only the `True`/`False`/
    `None` items (the switches) are taken. A `pass_config` with a nested brace is not matched at
    all: that leaves no evidence, so G1 fails rather than guesses.
  - A missing switch reads as `None`, never as "off".
  - The first engine-config line's `pass_config` is used, and a log with two different ones is
    flagged (`pass_config_conflict`).
- **The fusions line (G1 evidence).** `vllm/config/compilation.py:331-333`:
  `logger.info_once("Enabled custom fusions: %s", ", ".join(enabled_fusions), scope="global")`.
  - It names the switches that are on, without their `fuse_`/`enable_` prefix (`act_quant`).
  - It is logged once per process (the launcher, each API server, the engine core) and skipped
    entirely when none is on. NV logs it; MX and NVnf do not.
- **G1's fusion rule.** A session passes only if `fuse_act_quant` is its expected value,
  `act_quant` is among the logged custom fusions exactly when that value is on, and
  `pass_config_conflict` is false. [§9][e9] has the gate.
- **Size of the effect.** In `smoke-nf-2` (one round, not a result) Φ = 0.991 / 0.985 / 0.997 at
  C = 1 / 32 / 128, and the fusion did not change the NLL measurably (1.8238 for both).

<a id="nv-nf"></a>
### NV-nf: the fusion switched off

- **The arguments.** NV's kernel pin plus `--compilation-config '{"pass_config":
  {"fuse_act_quant": false}}'`.
- **How vLLM reads them.** vLLM v0.31.0 parses the value with `json.loads` and pydantic, so it must
  be JSON with underscore field names.
- **Never overridden.** A value set on the command line is never overridden by the
  optimization-level default: `VllmConfig._set_config_default` fills a field only while it is
  `None` (`vllm/config/vllm.py:1027-1032`).
- **Verified in the fusion smoke (`smoke-nf-2`, published checkpoints).**
  - NVnf ran `FlashInferCuteDslNvFp4LinearKernel` with `'fuse_act_quant': False` and logged no
    "Enabled custom fusions".
  - It was compiled fresh ("Compiling a graph for compile range (1, 16384) takes 28.06 s") under
    its own compile-cache key `torch_aot_compile/482cf30f…`, distinct from NV's `6bbb8ec6…`
    ([#compile-cache](#compile-cache)).
- **Role.** R stays the comparison as deployed; F isolates format + GEMM with the fusion matched
  ([§3][e3]).

<a id="nv-alt"></a>
### NV-alt: the kernel scan and its choice

- **The scan.** The kernel-scan smoke serves our NVFP4 checkpoint on four alternative kernels: NVc
  `flashinfer_cutlass`, NVt `flashinfer_trtllm`, NVd `flashinfer_cudnn` and NVv `cutlass`. The rule
  is pre-registered in [§7.7][e77]. Its constants are the fields of `NVA_SCAN`
  (`studies/kernel_scan.py`), the kernel scan of the `smoke` study:

| Field | Value | Meaning |
|---|---|---|
| `selection_c` | 32 | the batch at which the median M1 step times are compared |
| `max_nll_diff` | 0.01 nats/token | a candidate's mean NLL must be this close to NV's in the same run |
| `tie_margin` | 0.01 | relative difference of two median step times that counts as a tie |
| `tiebreak_c` | 128 | the batch that breaks a tie |

- **Why the NLL bound.** A scanned kernel computes the same thing as the pinned kernel. A kernel
  that moves the NLL is broken or lossy, and its speed is not comparable.
- **Eligibility (`kernel_scan` in `analysis/verdicts.py`).** A kernel is eligible only if:
  - every one of its sessions logged exactly the expected class, and that class is not CuTe-DSL;
  - no session of it failed;
  - it has a median M1 step at C = 32;
  - its mean session NLL is within 0.01 of NV's;
  - it passes G5b on its own (finite degradation, not in G5b's failed list).
- **Ranking.** Candidates are ranked by median step at C = 32, then by scan order.
  - A tie is runner-up/fastest − 1 < 0.01.
  - In a tie, the faster at C = 128 wins, if the two are not also within 1% there (max/min − 1).
  - Otherwise the earlier in (NVc, NVt, NVd, NVv) wins.
- **Failed scan sessions.** A scan session that fails is not fatal. It is a scan result, not a
  reason to lose the other treatments. It is written to `errors.jsonl` with `"nonfatal": true` and
  to `servers.jsonl` as a row with `"failed": true`, so the round completes and the kernel is
  ineligible.
- **The choice (`smoke-r2-2`, 2026-10-05).** All four were eligible. NVd (`flashinfer_cudnn`) led
  at C = 32 with 8.877 ms against NVc's 9.370 ms (+5.55%, not a tie). Median M1 step in ms, with the
  ratio to NV:

| C | NV | NVc | NVt | NVd | NVv |
|---|---|---|---|---|---|
| 1 | 5.824 | 7.260 (1.247) | 8.566 (1.471) | 6.894 (1.184) | 8.272 (1.420) |
| 32 | 8.058 | 9.370 (1.163) | 10.330 (1.282) | 8.877 (1.102) | 11.481 (1.425) |
| 128 | 15.145 | 15.553 (1.027) | 16.127 (1.065) | 15.467 (1.021) | 17.469 (1.153) |

- **The pin.** NVa's pin is `("--linear-backend", "flashinfer_cudnn")`, its `server_args` in the
  `TREATMENTS` table of `studies/model.py`, and its expected G1 class follows the pin. Without a
  pin, vLLM's default would apply, which on B200 is NV's own CuTe-DSL kernel. A full run would
  then spend its rounds comparing NV with itself and call that K and D. So a study that includes
  NVa refuses to start while NVa is unpinned or pinned to NV's kernel
  (`runner.nva_pin_problems`).
- **Restart identity.** Every treatment's extra arguments (`treatment_server_args`) are part of the
  run's input identity, so a restart with a changed NVa pin is refused ([#run-integrity](#run-integrity)).
- **NVx.** In the same smoke NVx/NV was 0.9954 / 1.0044 / 0.9997 at C = 1 / 32 / 128 (`nvx_crosscheck`,
  which gates nothing), and NVx's NLL was 1.8248 against NV's 1.8238.

<a id="compile-cache"></a>
### torch.compile cache

- **Where it lives.** The torch.compile cache is on the `fp4-vllm-cache` volume (`/root/.cache/vllm`)
  and is kept between runs.
- **The directory names.** vLLM names the directory of a compiled graph after a hash and logs it:
  - when it compiles: "Using cache directory: `<cache>/torch_compile_cache/<hash>/rank_0_0/backbone`
    for vLLM's torch.compile" (`backends.py:1090`);
  - when it saves the AOT-compiled model: "saved AOT compiled function to
    `<cache>/torch_compile_cache/torch_aot_compile/<hash>/rank_0_0/model`" (`decorators.py:717`);
  - when it loads one: "Directly load AOT compilation from path ..." (`decorators.py:312`).

  The `smoke-r2-2` lines are in `tests/vllm_logs.py`.
- **G1's NVnf check.** The cache is shared by every session. An NVnf session that reports one of
  NV's hashes loaded the graph compiled for NV, with the fusion in it, whatever its `pass_config`
  says.
  - Every line naming the cache is kept, with the directory names: the one after
    `torch_aot_compile/` for the AOT model, and `torch_aot_compile` itself is never taken for a hash.
  - G1 fails if an NVnf session shares a hash with an NV session.
  - The check reads "unverified" when either side has no hashes. `smoke-nf-2` predates the field,
    so it was checked by hand ([#nv-nf](#nv-nf)).

<a id="server-log"></a>
### Server log facts

`server.parse_server_log` reads the log line by line and keeps whole lines as audit evidence.
Line numbers are vLLM v0.31.0's. "Logged at" is what the `full-1` logs show.

| servers row field | Line | Source | Used by |
|---|---|---|---|
| `linear_kernels`, `linear_kernel_lines` | `Using <class> for MXFP4\|NVFP4 GEMM` | `model_executor/kernels/linear/__init__.py:975`, `:1142/1186` | G1 |
| `linear_backend_fallback_lines` | `--linear-backend=X has no kernel for this linear layer type` / `... was requested, but no ...` | same file, `:365`, `:394` | audit |
| `bf16_gemm_lines` | `Using FlashInfer <backend> for eligible unquantized BF16 GEMMs.` | `model_executor/layers/utils.py:626` | audit |
| `pass_config`, `fuse_act_quant`, `pass_config_conflict` | `Initializing a V1 LLM engine (v…) with config: …` | `v1/engine/core.py:129-133`; repr `config/compilation.py:818-843` | G1 |
| `custom_fusions`, `custom_fusion_lines` | `Enabled custom fusions: act_quant` | `config/compilation.py:331-333` | G1 |
| `autotune_*` | see [#autotune](#autotune) | `model_executor/warmup/kernel_warmup.py`; FlashInfer `autotuner/autotuner.py` | `autotune_fresh` |
| `kv_cache_tokens` | `GPU KV cache size: 558,496 tokens, Maximum concurrency for 4,096 tokens per request: 136.35x` (`info_once`; "GPU" on CUDA; thousands separators; the smallest is kept if several, it is the binding one) | `v1/core/kv_cache_utils.py:2464-2471` | KV capacity (G6a) |
| `kv_cache_memory_gib` | `Available KV cache memory: 136.35 GiB` (`info_once`; smallest kept) | `v1/worker/gpu_worker.py:692-695` | audit |
| `attention_backend_lines` | `Using <X> attention backend out of potential backends: …`; `FlashInfer resolved query dtypes: prefill=…, decode=…, decode_backend=…, kv_cache_dtype=…, arch=…` (both `info_once`; FP8 KV shows here) | `platforms/cuda.py:526-534`; `v1/attention/backends/flashinfer.py:903-911` | audit |
| `compile_cache_lines`, `compile_cache_hashes` | see [#compile-cache](#compile-cache) | `backends.py:1090`, `decorators.py:717`, `decorators.py:312` | G1 (NVnf) |
| `weights_gib` | `Model loading took 19.53 GiB memory and … seconds` | logged at `model_runner.py:407` | G2 record |
| `cudagraph_capture_sizes` | `cudagraph_capture_sizes` in the engine-config line | `v1/engine/core.py:129` | G6a |
| `sampling_defaults_lines` | `Default vLLM sampling parameters have been overridden by the model's generation_config.json: {'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}` | logged at `model.py:1850` | audit ([#m2](#m2)) |
| `moe_backends`, `moe_backend_lines` | `Using '<name>' Mxfp4\|NvFp4 MoE backend` | Revision 1 only; no longer parsed ([#legacy](#legacy)) | none |

The KV example values are from `smoke-r2-2`'s NV session. A BF16 token is 262,144 B, so 558,496
tokens is 136.35 GiB, and 558,496 / 4,096 = 136.35.

---

<a id="prompts"></a>
## Prompts

[§7.2][e72] has the design. These are the construction facts.

- **Texts.** The first human turn of each ShareGPT conversation. If it is empty after stripping,
  the whole conversation is skipped rather than falling through to a later turn.
- **Chat frame.** The chat template is rendered around a sentinel
  (`@@FP4BENCH_CONTENT@@`) with `add_generation_prompt=True`. The ids before and after it are the
  frame, and a template that does not render the content exactly once is refused. Qwen3's frame is
  8 tokens.
- **Content encoding.** `encode(text, add_special_tokens=False, split_special_tokens=True)`, so
  special-token literals in user text stay plain text instead of becoming control tokens.
- **Tokenizer probe.** `prepare` checks the tokenizer before it decides whether to reuse existing
  prompt files, so a run that keeps them is probed too.
  - Every special token that encodes to exactly one id by default must encode to more than one id
    with the flag.
  - A literal that is not a control token of this tokenizer, such as gpt-oss's `<|end|>` in Qwen3,
    spells itself with several ordinary tokens either way and proves nothing.
  - No usable special token is also a failure.
- **M1 prompts.** 6 sets × 512 prompts of exactly 1,024 tokens. Set 0 is warmup.
  - Texts from a seed-0 shuffle are encoded with `"\n\n"` after each and fill the frame's room
    consecutively.
  - The rest of a text carries over into the next prompt.
  - A block at batch C takes the first C prompts of each set.
- **NLL windows.** 64 windows of 512 tokens of plain text (no chat template): the first 512 tokens
  of each text that has at least 512, from a seed-2 shuffle.
- **Experiment C prompts** (`c_prompts.json`, built by `prepare --expc`).
  - It downloads nothing. It needs the ShareGPT file, the Qwen3 tokenizer and the M1 prompts that
    `prepare` put on the volume.
  - 6 sets of C prompts of exactly P tokens per cell, from a deduplicated pool shuffled with seed
    0 + 3 (`CELL_SEED_OFFSET`), so that a cell's prompts do not start with the M1 prompts' texts.
  - Each prompt starts at a text and the rest of its last text is dropped. The pool is read once
    across all cells, so no two prompts share a text. A prompt equal to an earlier one is skipped,
    so every prompt is distinct.
  - The (1, 1,024) cell is not built: set s is the first prompt of M1 set s, exactly what the main
    run's C = 1 block sent (the config-reproduction check).
  - The build is deterministic. Too little text raises before anything is written.
  - The file is 9 cells and 3,141,888 prompt tokens.
- **Seeds.** M1 uses 0, NLL windows 2, calibration 3 ([#quantization](#quantization)) and
  Experiment C cells 3. The Experiment C pool is deduplicated; the calibration pool is not.
- **Byte stability.** Qwen3's chat template embeds no date, so the prompts are byte-stable across
  days. Revision 1's gpt-oss template embedded the date.
- **Reuse rules.**
  - Existing prompt files are reused unless `--force`, and their sha256 values are always
    returned and recorded.
  - M1 prompts without this tokenizer's frame (Revision 1's gpt-oss prompts, from another
    vocabulary) are replaced, never served.
  - A cell prompt file is kept only if it was built from the M1 prompts on the volume now (the
    `m1_prompts_sha256` it records), holds exactly the registered cells, can serve them all, and
    carries the frame. The runner refuses it at start otherwise.
- **Atomic writes.** Prompt files are written via a temp file and `os.replace`. The M1 and NLL
  files are deleted before the pair is rewritten, so a crash between the two writes leaves no
  mismatched pair.
- **Hashes (Revision 2).** `m1_prompts.json` `9930b747…a8f4220d`, `nll_prompts.json`
  `dc20a835…8d8ffd55c`, `c_prompts.json` `6439900e…`.

---

## Measurement

<a id="m1"></a>
### M1: steady-state decode step

[§7.3][e73] has the design.
- **Arithmetic.** With N₁ = 128 and N₂ = 1,152 output tokens, t_step = (T(N₂) − T(N₁)) / 1,024.
  Prefill, HTTP, the admission ramp and the finishing tail are the same in both waves and cancel.
  What remains is 1,024 decode steps at batch exactly C.
- **Context.** During the measured steps the context is P + 128 … P + 1,152. The mean is
  P + 640, which is 1,664 for the 1,024-token prompts (`M1_MEAN_CONTEXT`, used for R_ideal).
- **A block.** One warmup pair (set 0), then `m1_reps` measured pairs (sets 1 … m1_reps): 5 in
  full runs, 3 in smokes. Even sets run N₁ first and odd sets N₂ first, recorded as `first`.
- **Bad windows.** A non-positive window in the warmup pair is recorded as NaN, because analysis
  discards that pair. In a measured pair it raises.
- **Cell value.** The median of the measured pairs. If any rep is not finite, the cell is NaN
  rather than a median of the rest ([#pairing](#pairing)).
- **The client** (`fp4bench/wave.py`).
  - It sends all C requests at once with aiohttp. The connector limit is 0 (unlimited), because
    aiohttp's default caps at 100 connections, which would silently shrink the batch.
  - Total timeout is 1,800 s per wave.
  - A non-2xx response raises with the start of vLLM's error body. `raise_for_status()` would
    drop it.
  - Every response must report `usage.prompt_tokens` equal to the number of ids sent (end-to-end
    proof the server saw exactly our ids) and `usage.completion_tokens` equal to `max_tokens`.
    So an M1 row exists only for an exact block, which G6a relies on.

<a id="payloads"></a>
### Request payloads

| Instrument | Endpoint | Body |
|---|---|---|
| M1 wave | `POST /v1/completions` | `{"model": "fp4bench", "prompt": <ids>, "max_tokens": N, "temperature": 0.0, "ignore_eos": true, "stream": false}` |
| NLL | `POST /v1/completions` | `{"model": "fp4bench", "prompt": <ids>, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 1}`. It is 1, not 0, because the parser picks the prompt's own token by id from the returned dict |
| M2 | the bench's streaming chat requests | the bench's built-in prompt, no `--temperature` ([#m2](#m2)) |
| Profiler | `POST /start_profile`, `POST /stop_profile` | empty body ([#profiler](#profiler)) |

`ignore_eos` forces exact output lengths. Compute is unaffected, because the model is dense
([§10][e10]).

<a id="m2"></a>
### M2: llm-inference-bench Sustained Decode

[§7.4][e74] has the design.
- **One invocation per C.** Each cell gets its own throttle-counter window:

```
LLM_BENCH_NO_UPDATE_CHECK=1 python /opt/llm-inference-bench/llm_decode_bench.py \
  --host 127.0.0.1 --port 8000 --model fp4bench --skip-prefill --contexts 0 --concurrency <C> \
  --max-tokens 2048 --duration <30 | 10 in smokes> --display-mode plain --no-hw-monitor \
  --no-resume --output m2_raw/<session>_c<C>.json
```

- **No self-update.** `LLM_BENCH_NO_UPDATE_CHECK=1` keeps the pinned copy from updating itself.
- **Timeout.** `--duration` + 900 s, which is headroom for warmup and teardown.
- **Stale output.** The output file is deleted before the call, so a stale file is never parsed as
  this cell's result.
- **Which result.** The last result entry with this concurrency and `context_tokens` 0 is taken. A
  missing cell is recorded as invalid (`missing_cell`) with the same keys.
- **Validity, per cell (G6b).** A cell is invalid if any of these holds:
  - one of the bench flags `underfilled`, `capacity_limited`, `loop_detected` or
    `warmup_timed_out` is set;
  - `aggregate_tps` is not finite or is ≤ 0 (NaN ≤ 0 is false, so it is tested separately);
  - `num_errors` > 0;
  - `aggregate_source` is not `openai_continuous_usage`. Only that source is a client-side,
    usage-based measurement of the sustained window; the bench's stream-chunk and prometheus
    fallbacks, `none` and failures are not;
  - for C > 1, `effective_concurrency` < 0.98·C. 0.98 is the bench's own underfill threshold,
    average running requests ≥ 0.98·C. `effective_concurrency` is 0 when vLLM's metrics were
    unavailable, which the bench's `underfilled` flag cannot catch because it needs scheduler
    samples;
  - the preemption counter moved or could not be read ([#preemption-metric](#preemption-metric)).

  Invalid cells are recorded with their reasons, not raised, so a long run keeps going.
- **Units.** `inter_token_latency_p50` and `ttft_p50` are converted to ms.
- **Sampling.** It stays at the model's default. Qwen3-32B's `generation_config.json` sets
  `temperature` 0.6, `top_k` 20 and `top_p` 0.95, which vLLM logs as the warning in
  [#server-log](#server-log). Revision 1's other reason, that with one shared prompt greedy
  decoding would route every stream to the same experts, was for the MoE gpt-oss.
- **Observed validity.** G6b is reported, not blocking. `full-1` had 74 of 250 cells invalid and
  `expb-1` 22 of 45, mostly `underfilled`.

<a id="nll"></a>
### NLL check (G5b)

- **When.** Each server runs the 64 NLL windows before M1, all at once. The aiohttp connector is
  unlimited and the total timeout is 600 s.
- **Per window.** The mean over positions 1 … L−1 of −log p(token_i | tokens_<i) from
  `prompt_logprobs`. Position 0 has no logprob. A response without `prompt_logprobs`, or with a
  length other than the prompt's, raises.
- **Per session.** `nll` is the `fsum` mean over windows, and `nll_per_prompt` keeps each window
  for the accuracy note.
- **G5b.** Each treatment's mean session NLL must be at most 0.5 nats/token above the BF16
  reference ([§9][e9]).
- **Observed.** BF16 1.7730, MX 1.9092, NV 1.8238 (`full-1`). The fusion and the scan kernels do
  not change NV's value.

---

<a id="autotune"></a>
## FlashInfer autotune

- **vLLM reads the cache back.** vLLM v0.31.0 keeps FlashInfer's tuning results in a file per vLLM
  config hash and reads it back at the next start (`vllm/model_executor/warmup/kernel_warmup.py:446-453`).
  One autotune draw would therefore be reused by every round and every treatment of the same
  config: a fixed effect that no confidence interval shows. MX and its A/A replica MXp have the
  same hash.
- **The file.** `vllm/model_executor/warmup/flashinfer_autotune_cache.py:24-41`:
  `<VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR if set, else VLLM_CACHE_ROOT/flashinfer_autotune_cache/<flashinfer version>/<arch>>/<sha256>/autotune_configs.json`.
- **No switch to skip the read.** No flag or variable turns the read off.
  `enable_flashinfer_autotune=False` turns the tuning off instead, which changes what is measured.
- **A fresh directory per start.** `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` (`envs.py:1760`) moves the
  whole root. The runner gives every server start its own new, empty directory,
  `<run>/autotune/<session_id>`. Nothing is read, nothing of the shared cache volume (which also
  holds the torch.compile cache) is deleted, and the file each start writes is kept under the run
  for audit (`autotune_saved_files`, with sha256s). Session ids are unique, so finding the
  directory already there is a bug. Whatever an earlier run left under
  `flashinfer_autotune_cache/` on the volume is inert. [§7.7][e77] pre-registers this.
- **Log lines.** vLLM's `kernel_warmup.py` uses `logger.info`; lines marked `info_once` log once per
  process:

| Line | Source | Meaning |
|---|---|---|
| `Using FlashInfer autotune cache file: %s` | `kernel_warmup.py:437` (`info_once`) | the path in use |
| `Running FlashInfer autotune with %d tokens and token buckets %s.` | `kernel_warmup.py:360` (logged at :359) | the tuning pass; `full-1`: 16,384 tokens, buckets 1, 2, 4, … |
| `Running FlashInfer BF16-only autotune with %d tokens.` | `kernel_warmup.py:391` | BF16 pass |
| `Skipping FlashInfer autotuning for ops %s` | `kernel_warmup.py:430` (logged at :429; `info_once`) | `fp4_gemm`, with the CuTe-DSL NVFP4 kernel |
| `Skipping FlashInfer autotune because it is disabled.` | `kernel_warmup.py:255` (`info_once`) | tuning off |
| `[Autotuner]: Loaded {n} configs from {path}` | FlashInfer `autotuner/autotuner.py:3692` | a cache was read: not fresh |
| `[Autotuner]: Autotuning process starts ...` | `autotuner.py:1046` | |
| `[Autotuner]: Saved {n} configs to {path} ({new} new, {previous} from previous config)` | `autotuner.py:3555` (logged at :3554) | |

- **Fresh means three things** (`autotune_fresh`, `server.autotune_ran_fresh`):
  - vLLM ran its tuning pass;
  - FlashInfer logged no "Loaded … configs";
  - every cache file vLLM named is inside the session's new directory. A file in the volume's
    default location would be a stale one.
- **MXFP4 and NVFP4 are tuned differently.** vLLM autotunes the MXFP4 GEMM at startup but skips
  autotuning for the NVFP4 CuTe-DSL GEMM (`kernel_warmup.py`, `skip_ops={"fp4_gemm"}`). NV and NVnf
  therefore run FlashInfer's heuristic fallback tactic, as on a cache miss. This is the deployed
  behaviour, and the benchmark does not change it. The `full-1` round 0 logs confirm it:
  - MX, MXp and NVa (cuDNN) log "Saved 88 configs";
  - NV and NVnf log "Skipping FlashInfer autotuning for ops ('fp4_gemm',)" and "Saved 0 configs".

  The microbenchmark times both cases ([#microbench](#microbench)).
- **Consequence.** MXFP4's tactic is drawn anew at every start, and that variance enters the
  round-to-round CIs. In the main run two identical MXFP4 servers differed by a per-round SD of
  0.10 ms at batch 1 and 0.07 ms at batch 128. That is why Experiment C's G3 is an A/A test on
  effects ([§16][e16], [#aa](#aa)).

---

<a id="telemetry"></a>
## Telemetry and throttling

- **Clocks can't be locked.** Modal does not permit `nvidia-smi -lgc`, so throttling is detected,
  not prevented ([§6][e6]).
- **Counters (G4).** `nvidia-smi -q -d PERFORMANCE`, section "Clocks Event Reasons Counters", lines
  `<name> : <n> us`. Read are `SW Power Capping`, `Sync Boost`, `SW Thermal Slowdown`,
  `HW Thermal Slowdown` and `HW Power Braking`. A missing section or counter raises.
- **When they are read.** Before and after every M1 block and every M2 cell (`runner.measured`).
  The run also reads them once at start, so a format mismatch fails before the first cell. The
  preemption reads sit outside these windows.
- **The block record.** `window_s`, `counters_delta`, `env_throttle_us` (the sum of the deltas of
  SW Thermal Slowdown, HW Thermal Slowdown and HW Power Braking) and `sw_power_cap_frac` (the SW
  Power Capping delta in seconds over `window_s`).
- **G4.** Environmental throttling (`env_throttle_us` > 0 in any block or M2 cell) fails G4. SW
  power capping is reported, not disqualifying: hitting the 1000 W cap under load is real,
  possibly format-dependent behaviour ([§9][e9]). For Experiment C, G4 also fails a block without
  telemetry numbers, and the power-cap fraction is reported per cell and treatment (blocks, max,
  mean).
- **Sampler.** `nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active --format=csv -lms 200`
  writes `telemetry/<session_id>.csv` for the whole server lifetime. A bad query field makes
  nvidia-smi exit at once, so the sampler checks after 0.5 s and fails loudly. Stopping it is
  terminate, then kill after 10 s, and a cleanup error never replaces an exception in flight.
- **GPU.** B200, 1000 W power limit, driver 580.95.05, 183,359 MiB, MIG disabled.

---

<a id="preemption-metric"></a>
## Preemption metric

- **The counter.** vLLM v0.31.0 (commit `db9527a`) registers `vllm:num_preemptions` in
  `vllm/v1/metrics/loggers.py:676-683`, with labels `engine` and `model_name` and one child per
  engine. It increments it by the PREEMPTED events of the iteration (`loggers.py:1123-1124`,
  counted in `vllm/v1/metrics/stats.py:528-529`).
- **The sample line.** prometheus_client exposes a counter's samples with `_total` appended:
  `vllm:num_preemptions_total{engine="0",model_name="fp4bench"} 0.0`.
- **Only evidence.** The scheduler logs nothing when it preempts, so this counter is the only
  evidence there is.
- **Multiprocess aggregation.** With `--api-server-count 4`, `vllm serve` switches
  prometheus_client to multiprocess mode (`vllm/entrypoints/cli/serve.py:279-280`,
  `vllm/v1/metrics/prometheus.py:17-52`). Each API server counts the preemptions of its own
  requests, and every scrape, whichever API server answers it, sums the counter over all of them.
  One scrape is therefore the total, and adding up scrapes would count it again. Within one scrape
  the samples of every label set are summed (one per engine; one here).
- **Parser rules** (`metrics.parse_preemptions`). The guard must never mistake "not read" for 0.
  - A declared family (`# HELP`/`# TYPE` line: `vllm:num_preemptions_total` in the text format
    vLLM serves by default, `vllm:num_preemptions` in OpenMetrics) with no sample has counted 0.
  - A page without the family raises.
  - A sample that is not a finite non-negative count raises.
  - A line that starts like a sample of the family but does not parse whole raises: an
    OpenMetrics exemplar, a space before the labels, a float timestamp, a cut-off label set.
    Skipping it would leave its count out of the total.
  - A quoted label value may hold braces or spaces.
- **Scrapes.** `GET <base>/metrics` with a 10 s timeout. A scrape can fail for a moment (a busy API
  server, a timeout), so every read is retried, and only `Exception` is caught, so an interrupt
  still stops the run.
  - **Preflight.** Once per server, before anything is measured: 3 attempts, 2 s apart. It raises,
    so a server whose counter cannot be read at all (a missing or renamed metric) fails fast.
  - **In the sweep.** Before and after every M1 block and M2 cell: 5 attempts with pauses of 2, 4,
    8 and 16 s (about 30 s). A lost read there costs more: it stops the session (M1) or
    invalidates the cell (M2) after hours of work.
- **What a delta means.** A preempted request is evicted and recomputed, so a window in which the
  counter moved did not decode at batch exactly C.
  - **M1 (G6a).** Every row of a block carries its `preemptions_delta`, and G6a fails a block whose
    delta is not 0, including `None`. If the counter cannot be read before a block, the session
    stops before it. If it cannot be read after, the rows stay for audit and the session stops,
    because that block could never pass G6a, nor could any later one on a server whose `/metrics`
    stopped answering. A dead server is reported as dead rather than as G6a.
  - **M2 (G6b).** A cell whose delta is not 0 or is `None` is invalid with reason `preemptions`,
    and the session goes on, because M2 is secondary.
- **Observed.** The committed data show 0 preemptions in every M1 block and M2 cell of every run
  that recorded the counter. `smoke-r2-2` predates the guard, so its rows have no delta. At
  C ≤ 128 with BF16 KV the main run uses at most about 50% of the pool ([§11][e11]), so the rule
  was expected never to fire there. It exists for Experiment B ([#kv-capacity](#kv-capacity)).

---

<a id="kv-capacity"></a>
## KV capacity

- **Formula.** An M1 block at (C, P) holds C × (P + N₂) = C × (P + 1,152) KV tokens at the end of
  its N₂ wave (`min_kv_tokens_for`). A study's requirement is that of its largest wave, the cell
  with the largest C × (P + N₂).
- **When it is checked.** At the start of every session, from the log's "GPU KV cache size" line
  ([#server-log](#server-log)). vLLM logs its KV pool before `/health` answers.
  - A pool smaller than the requirement would preempt in that block and fail G6a, so the session
    stops before anything is measured. Experiment B's GPU time is the point of checking early.
  - A log without the line cannot show the pool is large enough, so it fails too.
  - For any treatment except a scan kernel this ends the run, which is intended: every later
    session is served the same way.
  - G6a checks every session's logged pool against the requirement again in analysis ([§13][e13],
    [§16][e16]).
- **Bytes per token.** 64 layers × 8 KV heads × 128 × 2 (K, V) = 131,072 elements per token:
  262,144 B in BF16 and 131,072 B in FP8 (E4M3 on CUDA) ([#bytes-model](#bytes-model)).

| Study | Largest wave | Required tokens | Pool observed |
|---|---|---|---|
| main run (C ≤ 128, P = 1,024, BF16 KV, 0.90) | (128, 1,024) | 278,528 | 555k–560k tokens |
| Experiment B (FP8 KV, 0.95) | (512, 1,024) | 512 × 2,176 = 1,114,112 | 1,190,176–1,193,760 tokens |
| Experiment C (BF16 KV, 0.90, 131,072 window) | (128, 360) | 128 × 1,512 = 193,536 | about 558k–560k tokens |

- **Experiment B's memory fraction.** At 0.90 the FP8 pool would hold about 1,117k tokens, only
  0.26% over the requirement. 0.95 adds about 70k tokens of headroom. C = 512 needs about 94% of
  the FP8 pool.
- **Experiment C's other cells.** (1, 127,360) needs 128,512, (8, 15,360) 132,096 and
  (32, 3,360) 144,384, all below the (128, 360) cell.

---

<a id="long-context"></a>
## Long context (Experiment C)

- **Window.** `--max-model-len 131072`. Qwen3-32B's `max_position_embeddings` is 40,960, so
  `--hf-overrides '{"max_position_embeddings": 131072}'` raises it ([#server-args](#server-args)
  has the argv form).
- **No YaRN.** The override enlarges the stock RoPE table without YaRN, so the position encoding
  and every kernel stay the main run's.
- **Quality past training.** Output quality degrades past the trained ~40k positions. That does
  not affect timing: output lengths are fixed (`ignore_eos`) and the model is dense.
- **Fit.** The largest prompt plus N₂, 127,360 + 1,152 = 128,512, fits the window.
- **Checks.** The design check requires every served argv to have the window, the override, and
  no `rope_scaling`, `rope_parameters` or `yarn` ([#descriptive-checks](#descriptive-checks)).
- **Observed (`smoke-c-1`).** The NLL was unchanged (MX 1.9092, NV 1.8238), and the config
  reproduction R(1, 1,024) was 0.9343 against the main run's 0.934. The within-session step SD was
  0.002 ms at (1, 127,360) and 0.04 ms at (128, 360), which is why `m1_reps` stays 5 ([§16][e16]).

---

<a id="bytes-model"></a>
## Bytes model

Qwen3-32B (`config.json` at `9216db5`):

| Parameter | Value |
|---|---|
| Layers, hidden size, MLP intermediate size | 64, 5,120, 25,600 |
| Query heads, KV heads, head dim | 64, 8, 128 |
| Vocabulary | 151,936 |
| Attention | full in every layer |
| `lm_head` | untied |

- **Weight bytes don't depend on C.** The model is dense, so every token uses every weight.
- **Quantized parameters.** 64 × (2·5,120·8,192 + 2·5,120·1,024 + 3·5,120·25,600) = 31,205,621,760
  (q, o + k, v + gate, up, down).
- **Bits per weight, including block scales.** MXFP4 4 + 8/32 = 4.25; NVFP4 4 + 8/16 = 4.5.
- **`lm_head`.** BF16, read every step: 2 × 151,936 × 5,120 = 1,555,824,640 B.
- **KV per token.** 131,072 elements, BF16 2 B or FP8 1 B each. An unknown KV dtype or weight
  format raises KeyError rather than being guessed.
- **Formulas.** step_bytes(C, ctx) = W + lm_head + C × KV/token × ctx.
  R_ideal = step_bytes(NV)/step_bytes(MX). It is an upper bound on t_NV/t_MX if decode were purely
  bandwidth-bound with no fixed costs. A smaller KV dtype dilutes the weight bytes less, so with
  FP8 R_ideal is 1.0075 at C = 512 and context 1,664 ([§5][e5], [§13][e13]).
- **HBM peak.** 8e12 B/s (B200 HBM3e nominal), used for the effective-bandwidth aid
  ([#descriptive-checks](#descriptive-checks)) and the microbenchmark's %HBM.

---

<a id="profiler"></a>
## Profiling

[§15][e15] has the design. The image has no `nsys`, so vLLM's torch profiler is used.

- **Profiler config** (`--profiler-config`, vLLM v0.31.0): `{"profiler": "torch",
  "torch_profiler_dir": <absolute dir>, "ignore_frontend": true, "torch_profiler_with_stack": false,
  "torch_profiler_record_shapes": false, "delay_iterations": D, "max_iterations": 32}`.
  - `ignore_frontend` is set because with `--api-server-count 4` each API server would otherwise run
    a CPU-only frontend profiler, and a `/stop_profile` could land on another server than the
    `/start_profile`.
  - The directory must be absolute.
  - The rest of the argv is exactly what the runner serves (the study's arguments, then the
    treatment's), plus `--profiler-config`.
- **Sessions.**
  - Config A (§7.1): MX, NV and NVnf at C = 1 and 32, delay 16. Delay 16 skips the prefill of
    C = 32: 32 × 1,024 tokens at 16,384 per step is 2–3 steps.
  - Config B (§13: FP8 KV, 0.95): MX and NV at C = 512, delay 48. At C = 512 the prefill takes
    about 32–34 steps, because the admitted requests' decode tokens share the budget.
  - The profiler window is a server setting, so it is the same for every C on a server.
- **Per C on a server.**
  1. One unprofiled warmup wave (set 0, 64 tokens).
  2. Two unprofiled reference waves (set 1, 64 and 256 tokens). (t₂₅₆ − t₆₄)/192 is an M1-style
     step time on the same server without the profiler.
  3. `POST /start_profile`.
  4. One wave of C prompts (M1 set 1, 1,024 tokens) with 256 output tokens.
  5. `POST /stop_profile`, which vLLM ignores if `max_iterations` already stopped the profiler.
- **What the profiler records.** It skips the first `delay_iterations` engine steps (the prefill
  chunks), records exactly 32 steps, then stops and writes the trace by itself.
- **Trace handling.**
  - The trace is moved to `<out>/<config>_<treatment>_c<C>.pt.trace.json.gz`.
  - A trace counts as complete when its size is unchanged between two polls and its gzip stream
    is whole. With TP = 1, exactly one new trace file must appear.
  - vLLM also writes a `key_averages` table (`profiler_out_<rank>.txt`) after the trace. It is
    overwritten by every profile, so a fresh one is copied as `<session>.profiler_out.txt`, best
    effort.
- **Trace semantics.**
  - GPU activity is the complete events (`ph` "X") of category `kernel`, `gpu_memcpy`/`memcpy` and
    `gpu_memset`/`memset`. The category is compared lowercased, because older PyTorch wrote
    `Kernel`.
  - Decode steps run as one full CUDA-graph replay each. CUPTI reports the kernels inside a replay
    and gives each one the correlation id of its `cudaGraphLaunch`/`cuGraphLaunch`, so kernels
    group by launch. That gives an independent step count and per-step kernel counts; a mixed
    prefill step would show up as piecewise replays with different counts. This was checked on the
    first traces, so nsys was not needed.
  - vLLM wraps every engine step in a `record_function` named
    `execute_context_<reqs>(<tokens>)_generation_<reqs>(<tokens>)`: a CPU `user_annotation` and a
    GPU-side `gpu_user_annotation`. A pure decode step has 0 context (prefill) requests.
  - A window is pure decode when there are 32 graph launches with kernels, the same kernel count in
    each, and every annotated step held decode requests only. All 8 windows of `profile-1` were,
    with annotations `execute_context_0(0)_generation_C(C)`.
  - The span is first kernel start to last kernel end. Busy time is the union of kernel intervals,
    and idle is span − busy. Idle inside replays is GPU-side gaps (launch latency, tails); idle
    outside them is eager kernels and the gaps between steps (CPU scheduling, inflated by the
    profiler's own overhead).
- **Categories.** The patterns are ordered, and the first `re.search` match wins; `other` is the
  fallback. They were set from the kernel lists of `profile-debug-1` (MX at C = 1 and 32, NV at
  C = 1), which §15 allows, and are recorded with every result.

| Category | Pattern | Kernels (vLLM v0.31.0 / FlashInfer 0.7.0.post1, B200) |
|---|---|---|
| `act_quant` | `cvt_fp16_to_fp4\|MXFP4Quantize\|mxfp4_quantize` | vLLM's NVFP4 `cvt_fp16_to_fp4` and the fused `silu_mul_cvt_fp16_to_fp4`; FlashInfer CuTe-DSL `…mxfp4_quantizeMXFP4QuantizeSwizzledKernel…` |
| `fp4_gemm` | `BlockScaled\|blockscaled_gemm` | FlashInfer CuTe-DSL `…dense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDenseGemmKernel…` (`mm_fp4`, both formats) |
| `silu_mul` | `silu` | MX's and NVnf's Inductor `triton_poi_fused_mul_silu_slice_<n>` |
| `attention` | `^fmha\|reshape_and_cache\|FillFunctor<unsigned char>` | trtllm-gen decode `fmhaSm100fKernel_…`, the KV write `reshape_and_cache_flash_kernel`, and the one-byte zero fill `vectorized_elementwise_kernel<8, FillFunctor<unsigned char>>` that runs right before every fmha in the graph (64 per step at every C in both formats; its name alone does not say so, its place does) |
| `bf16_gemm` | `^nvjet\|splitKreduce` | `lm_head`, eager, cuBLAS `nvjet_sm100_…` plus `splitKreduce_kernel` |
| `norm_rope_elementwise` | `^triton_` | every other Inductor kernel: RMSNorms, q/k norm + RoPE, embedding, MX's small `triton_poi_fused_0` before each `o_proj` GEMM |
| `sampling` | `gumbel_sample\|ArgMaxOps` | `_gumbel_sample_kernel`, the argmax reduction |
| `other` | (fallback) | vLLM's eager step bookkeeping (`_post_update_kernel`, `_gather_block_tables_kernel`, `_compute_slot_mappings_kernel`, `_combine_sampled_and_draft_tokens_kernel`, …), aten index/fill/scatter/arange, `memcpy32_post` |

- **NV's fused norm kernels.** NV's Inductor kernels
  `triton_red_fused_fused_add_rms_norm_scaled_fp4_quant_<n>` only list the quant op in their name.
  They are the residual add + RMSNorm in front of it, taking the same µs as MX's
  `triton_red_fused_fused_add_rms_norm_<n>`. The quantization is the `cvt_fp16_to_fp4` that
  follows, so they fall through to `norm_rope_elementwise`.
- **Per linear layer.** A Qwen3 decoder layer has four FP4 linears in graph order: `qkv_proj`,
  `o_proj`, `gate_up_proj`, `down_proj`. Each has one activation quantization in front and one FP4
  GEMM. The i-th such kernel of a replay is attributed to layer i mod 4, only when every replay has
  the same multiple-of-4 count.
- **Use.**
  - The three-way split of a difference: FP4 GEMM; act-quant + SiLU·mul; everything else.
  - The pairs NV−MX, NVnf−MX and NV−NVnf.
  - §13's GEMM share s at C = 512, against the GPU span and against summed kernel time.
  - The fusion check: count of `silu_mul_cvt_fp16_to_fp4` per step.
  - Each server's G1-style facts (linear kernel, `fuse_act_quant`) are recorded with its profile.
- **Cost.** About 47 B200-minutes for `profile-1` (8 sessions).

---

<a id="microbench"></a>
## Kernel microbenchmark

[§14][e14] has the design and predictions. These are the method facts (`fp4bench/microbench.py`).

- **Shapes and M.** Shapes are N × K. The merged layers vLLM runs: `qkv_proj` 10,240 × 5,120,
  `o_proj` 5,120 × 8,192, `gate_up_proj` 51,200 × 5,120, `down_proj` 5,120 × 25,600. The
  individual ones: `q_proj` 8,192 × 5,120, `k_proj`=`v_proj` 1,024 × 5,120,
  `gate_proj`=`up_proj` 25,600 × 5,120. Shapes shared by two layers are timed once.
  M ∈ {1, 8, 32, 64, 128, 256, 512}.
- **Implementations.**

| Key | Formats | Library | Tuned |
|---|---|---|---|
| `vllm_cutedsl` | MX, NV | FlashInfer `mm_fp4(backend="cute-dsl")`, vLLM's path as deployed | MX yes, NV no |
| `vllm_cutedsl_nv_tuned` | NV | same, NVFP4 autotuned | yes |
| `cudnn` | MX, NV | FlashInfer `mm_fp4(backend="cudnn")`, same library for both | yes |
| `torch_cublaslt_nv1` | NV | `torch._scaled_mm`, single-level NVFP4 (cuBLASLt) | no |
| `torch_cublaslt_nv2` | NV | `F.scaled_mm`, two-level (BlockWise1x16 + TensorWise) | no |
| `torch_mslk_mx` | MX | `F.scaled_mm`, MXFP4 BlockWise1x32 | no |

- **Untuned NV cells.** They are the same calls inside `flashinfer.autotune(False,
  skip_ops={"fp4_gemm"})`. That returns FlashInfer's fallback tactic exactly as a cache miss does
  in the deployed NVFP4 server, whose autotune cache has no `fp4_gemm` entries
  ([#autotune](#autotune)).
- **Tuned cells.** Each is tuned once per (implementation, format, shape) at the largest M.
  FlashInfer profiles every bucket up to it (1, 2, 4, …, 512), so each timed M is a cache hit.
  Tactic −1 from the autotuner is the heuristic fallback (no tuned or bundled entry for the
  bucket), and the chosen (runner, tactic) is recorded per cell.
- **Library differences.**
  - `torch_mslk_mx`: in PyTorch 2.13 on CUDA, BlockWise1x32 runs MSLK's CUTLASS `f4f4bf16` kernel,
    not cuBLASLt. (b) therefore compares two libraries; it is the MAMF reference, not a format
    comparison.
  - cuDNN chooses its own engines per format, and the kernels each cell ran are recorded.
- **Operands, as vLLM holds them.**
  - Weights are (N, K/2) uint8 plus 128×4-swizzled scales of shape (pad128(N), pad4(K/bs)).
    The swizzle is (Mt, 4, 32, Kt, 4) ↔ (Mt, Kt, 32, 4, 4), a self-inverse permutation.
  - Activations are quantized exactly as vLLM's runtime does: MXFP4
    `flashinfer.mxfp4_quantize(backend="cute-dsl")`, NVFP4 `scaled_fp4_quant(...,
    is_sf_swizzled_layout=True, backend="flashinfer-cutedsl")`.
  - The NVFP4 per-tensor scale is 448·6/amax (FP32), as vLLM and FlashInfer use it, and static,
    like vLLM's calibrated one.
  - `alpha` is 1/(g_x·g_w) for NVFP4 (vLLM: `input_global_scale` × the weight's). For MXFP4 vLLM
    passes `alpha=None`, which its wrapper turns into `ones(1)`.
  - The cuDNN backend gets uint8 views of the scales, as vLLM's `flashinfer_scaled_fp4_mm` does.
  - The single-level `torch._scaled_mm` has no global scale in the GEMM, so its output is
    multiplied by `alpha` afterwards.
  - E2M1 decoding uses the 16-value LUT, low nibble = even element.
- **SiLU·mul + quantize on the `down_proj` input** (M, 2 × 25,600).
  - The servers run `custom_ops ['none']`, so MX as deployed is Inductor-compiled native SiLU·mul
    (dynamic batch dimension, like vLLM's compiled graph) followed by `mxfp4_quantize`.
  - NV as deployed is the fused `torch.ops._C.silu_and_mul_nvfp4_quant`.
  - Extras: MX with vLLM's CUDA `silu_and_mul`, and NV unfused.
  - %HBM uses the fused minimum (input read once, FP4 + scales written once) for all, so MXFP4's
    extra BF16 round trip shows as lower %HBM.
- **Timing.**
  - One CUDA graph per cell holds ≥ 200 calls, cycling through R input copies (weights,
    activations and scales) with (R − 1) × bytes per copy ≥ 2 × L2. At least 2 copies and at most
    32,768; the cap only binds for the tiniest activation-quant inputs and is recorded per cell.
  - Warmup (JIT, plan caches, workspaces) runs on a side stream, off the capture.
  - 7 replays per cell; median and min–max µs per call.
  - All GEMM cells of one (shape, M) run back to back, with the order reversed on every other M,
    so paired MX/NV cells are seconds apart and a slow drift favours neither format.
  - Seeds: the weight 0, tuning input 999, activation 1000 + M, activation-quant input 1.
- **Accounting.** GEMM bytes = packed weights + weight scales (1 B each, swizzled and padded) +
  packed activations + activation scales + BF16 output, each read or written once. NVFP4's FP32
  global scales and alpha are a few bytes and left out. FLOPs = 2·M·N·K.
- **Reading.**
  - Both formats at ≥ 80% of 8 TB/s: the bytes test applies.
  - Either below 60%: the ratio measures kernel efficiency, not bytes.
  - A ratio outside [0.98, 1.02] is "practically different". This is descriptive only, with no
    TOST.
  - A device-to-device copy of 4 GiB (read + write counted) is a reference bandwidth that was not
    pre-registered.
  - Relative error against dequantized operands in FP32 is reported, not gated.
- **Run.** One B200, 8 CPU cores, 64 GiB. It reads no model or data, so only the results volume
  is mounted. `microbench-1` ran the full grid in a primary and a replicate container with 0
  failed cells. The two sources on its duration disagree: the deleted `modal_app.py` comment says
  about 7 minutes each, and the run log says about 26 GPU-minutes for both together.

---

<a id="modal"></a>
## Modal

<a id="modal-detach"></a>
### Detached apps and spawn

- **Long steps live in a detached app.** `fp4bench run`, `microbench` and `profile` spawn their
  function inside `app.run(detach=True)` and return the call id. `prepare` and `quantize` spawn
  the same way and then wait for the result; if the CLI stops waiting, the call goes on. Short
  steps (`env`, `check`, `publish`, `fetch`) use `.remote()` inside `app.run()`. Two incidents
  shaped this:
  - **`smoke-nf-1` (2026-10-04).** It was launched without `--detach`. The ephemeral app stopped
    when the local entrypoint returned, which cancelled the spawned call before it did anything.
  - **`smoke-r2-1` (2026-10-05).** The client held a blocking `experiment.remote()`. The laptop
    entered clamshell sleep at 21:40:59 EDT on battery, and the container logged "Received a
    cancellation signal while processing input" at 01:44:19 UTC. The input was cancelled despite
    `--detach`, mid-scan. The fix was `spawn()`.
- **History (`modal_app.py`, removed).** With `modal` 1.4.2, `modal run` stored `--detach` in
  `ctx.obj["detach"]` of its click command (`modal/cli/run.py`). The old entrypoint read that to
  refuse a non-detached long step and warned when it could not tell. It was not a public Modal API.
- **Interruptions still happen.** During `full-1`'s extension (2026-10-05 15:33 UTC) the
  container received a KeyboardInterrupt in round 9's NVa session. Modal restarted the function on
  another GPU, and the restart passed the identity checks. Round 9 spanned two starts, so it was
  re-run whole ([#run-integrity](#run-integrity)).

<a id="modal-volumes"></a>
### Volumes and paths

| Volume | Mount | Holds |
|---|---|---|
| `fp4-hf-cache` | `/root/.cache/huggingface` (`HF_HOME`) | Hub downloads |
| `fp4-vllm-cache` | `/root/.cache/vllm` | the torch.compile cache, kept between runs ([#compile-cache](#compile-cache)); FlashInfer autotune files there are inert ([#autotune](#autotune)) |
| `fp4-data` | `/data` | where checkpoints are made, fetched, checked and published (`qwen3-32b-bf16`, `qwen3-32b-mxfp4`, `qwen3-32b-nvfp4`, `nvidia-qwen3-32b-nvfp4`); ShareGPT; the prompt files; `bf16_reference_nll.json` |
| `fp4-results` | `/results` | one directory per run id |

- **Local scratch.** `/local` is the container's ephemeral disk. Sessions serve `/local/<checkpoint
  dir>` ([#staging](#staging)), and the profiler's traces go under `/local/profile`.
- **Local executor.** `--executor local` sets `FP4BENCH_DATA_DIR` and `FP4BENCH_LOCAL_DIR` for a
  child process instead of using volumes. The scratch directory must not be the data directory.
- **Commits.** A failed volume commit is logged and the run continues: Modal commits again on
  container exit, and raising would abort a run of hours. The runner commits after the manifest
  line, after every session and at the end. The handlers catch `Exception`, not `BaseException`,
  so a cancelled container still stops.
- **Download.** `fp4bench download <run-id>` runs `modal volume get fp4-results <run-id> <dest>`
  (with `--force` to overwrite).
- **Fresh run ids.** `microbench` and `profile` do not resume, so each needs a fresh `--run-id`. A
  run directory that already has files is refused. `microbench-1` and `profile-1` are the
  2026-10-05 runs.

<a id="modal-disk"></a>
### Ephemeral disk

- **Bounds.** Modal accepts `ephemeral_disk` only between 512 GiB and 3 TiB. The check is
  server-side, at app creation. The first Revision 2 `env` was refused for asking 200 GiB (fixed in
  `eac29fd`). The functions therefore ask for the minimum, 512 GiB.
- **Units.** Memory and disk are given in MiB (`GIB = 1024`).
- **What needs it.** The experiment's local copies of the checkpoints (MX + NV + NVIDIA's are about
  60 GB) plus vLLM and torch caches. `quantize` asks for the same size as headroom for `/tmp` and
  the torch/triton caches; `HF_HOME` and the checkpoints are on volumes.

<a id="modal-resources"></a>
### Resources and timeouts

| Function | GPU | CPU | Memory | Ephemeral disk | Timeout | Volumes | Secret |
|---|---|---|---|---|---|---|---|
| `env_check` | B200 | 4 | 16 GiB | | 10 min | none | |
| `prepare_data` | | 8 | 32 GiB | | 3 h (the ~64 GB BF16 download) | all | |
| `prepare_expc_data` | | 8 | 32 GiB | | 1 h | all | |
| `quantize_checkpoint` (quantization image) | B200 | 16 | 256 GiB | 512 GiB | 3 h | all | |
| `fetch_checkpoint` | | 8 | 32 GiB | | 1 h | all | `huggingface-secret` |
| `check_checkpoints` | | 4 | 16 GiB | | 10 min | all | |
| `publish_checkpoint` | | 4 | 16 GiB | | 1 h | all | `huggingface-secret` |
| `experiment` | B200 | 16 | 64 GiB | 512 GiB | 10 h | all | |
| `kernel_microbench` | B200 | 8 | 64 GiB | | 1 h | results only | |
| `profile_sessions` | B200 | 16 | 64 GiB | 512 GiB | 3 h | all | |

- **Why these sizes.** `experiment` asks for 16 cores and 64 GiB explicitly, the same for every
  treatment, because the vLLM server and the load generators need host CPU ([§6][e6]). Experiment
  C takes about 6.5 h. `profile_sessions` is sized like `experiment` because it stages and serves
  the checkpoints too. `prepare_expc_data` reads ShareGPT, the tokenizer and the M1 prompts from
  the volume, tokenizes about 3.1M tokens and writes one file in minutes, not hours.

<a id="modal-secrets"></a>
### Secrets

- **The secret must exist.** Reproducers need a Modal secret named `huggingface-secret`. The app
  attaches it to the fetch and publish functions, and Modal resolves the secrets of every function
  in the app on each start, so even `env` fails without it.
- **What it may hold.** A read-only HF token, or effectively nothing when every repo involved is
  public. Publishing needs a token with write access.
  - With a token: `modal secret create huggingface-secret HF_TOKEN=<read-only token>`.
  - Empty: `modal secret create huggingface-secret UNUSED=1`, because the CLI wants one KEY=VALUE.
  - Don't put a made-up `HF_TOKEN` in an "empty" secret: public repos reject an invalid token too.

<a id="modal-platform"></a>
### Platform constraints

- **Same GPU.** One function call is one container and one GPU. Every round of a run, with all its
  treatments, runs inside one call. G7 checks that each round's sessions share a GPU UUID and a
  start.
- **Exact GPU.** The function asks for exactly `"B200"`, and the manifest must show
  `NVIDIA B200` / 10.0 ([#stack](#stack)).
- **gVisor sandbox.** It may add host-side overhead. That overhead is the same for every
  treatment, so it affects absolute times (mostly at C = 1), not the paired ratios ([§6][e6]).
- **Clocks.** They can't be locked ([#telemetry](#telemetry)).

<a id="modal-costs"></a>
### Run times and costs

| Step | Pre-registered budget | Observed |
|---|---|---|
| Two quantizations | ~$10 | |
| Kernel-scan smoke (7 servers + staging) | ~$10 | `smoke-r2-2` 3,201 s |
| Fusion smoke | ~$5 | `smoke-nf-2` ~31 min |
| Main run, 5 rounds × 5 treatments | ~$34 (~4.6 h; ~11 min per server) | `full-1` rounds 0–4: 00:16–06:18 EDT; extension to 10 rounds: 06:20–12:54 EDT |
| Experiment B, 15 servers | ~$33 (~4.5 h, ~18 min per server) | `expb-1` 00:17–04:45 EDT |
| Experiment C, 15 servers | ~$40 (~6.5 h, ~26 min per server) | `expc-1` 19:21–00:02 EDT |
| Experiment C smoke | ~$5 | |
| Microbenchmark | | ~26 GPU-min (primary + replicate) |
| Profiling | | ~47 B200-min |

The budgets are from [§7.7][e77], [§13][e13] and [§16][e16]. The times are from
[docs/run-log.md](docs/run-log.md).

---

<a id="run-integrity"></a>
## Run integrity and restarts

A run resumes in its run directory. These rules keep one run from mixing inputs, GPUs or
protocols ([§11][e11], 2026-10-04).

- **Start-up order.**
  1. Check the protocol identity, then the code identity.
  2. Collect the environment manifest and verify it ([#stack](#stack)).
  3. Collect the input identities and the NVa pin check.
  4. Check the inputs against the first start (only if no input probe failed, since an
     environment error is the better diagnosis then).
  5. Check the published-checkpoint requirement.
  6. Compute the checkpoint report.
  7. Stage the checkpoints, only if nothing failed so far.
  8. Append the manifest line and commit it, then abort if anything failed.
  9. Read the throttle counters once.
  10. Load the prompt files once per start.
- **`start_id`.** A uuid per container start, written into its manifest line and each of its
  servers rows.
  - A round is complete only if every treatment's latest session exists and all of them share one
    `start_id`.
  - A partial round, including an interrupted re-run that leaves new sessions next to old ones,
    therefore re-runs whole, and every analysed round has all its treatments on one GPU.
  - A row without a `start_id` proves nothing.
- **Protocol identity.** On a restart, every recorded protocol field except `rounds` must equal the
  first start's: changing one mid-run would mix two experiments in one result set. `rounds` may
  only rise, and never below any earlier start's (even an aborted one), because analysis reads the
  registered count from the last manifest line and a lower one is a restart that forgot `--rounds`.
  A first start from before Experiment C's fields existed counts as having their defaults (no
  cells, a 4,096-token window, no override), because its argv was byte for byte the same.
- **Code identity.** On a restart, the code must be the first start's `code_commit`, and neither
  start may have had uncommitted or unknown changes, because those cannot be compared: a run is
  one code version. A first start from before `code_commit` existed cannot be checked, so the
  restart goes on with a warning.
- **Input identity.** These must equal those of the first start that passed its environment
  check, because a start that aborted ran no session:
  - the prompt hashes (M1, NLL and, for cell studies, `c_prompts.json`);
  - the BF16 reference;
  - both checkpoints' layout and content fingerprints;
  - NVx's content fingerprint, when NVx is in the study;
  - the fetch records, without their `utc`;
  - `treatment_server_args`.

  How the start staged the checkpoints is not part of identity. Without this check, a resume
  after `prepare --force`, a re-quantize or a fetch would silently mix prompts or checkpoints.
- **Rounds.**
  - `--rounds` 0 keeps the study's count.
  - Any other value must be the study's count or its registered extension (full, expb and expc,
    to 10); the same `--rounds` must be passed on every restart.
  - `--rerun-rounds` re-runs complete rounds whole (for example after a G4 flag), without editing
    `servers.jsonl`; analysis keeps the latest session per (round, treatment).
  - The values must lie in 0 … rounds−1.
- **Probes never raise.** `collect_manifest` and `collect_inputs` record a failed probe as an
  error value, which the start then treats as an environment problem. The GPU UUID of a finished
  session is read the same way, so a session is never lost to a failed `nvidia-smi` query; an
  unreadable UUID then fails G7.
- **Failures.** Every session failure is fatal except a scan treatment's
  ([#nv-alt](#nv-alt)). It is recorded in `errors.jsonl` (with `utc`, `session_hint`, `error` and
  `traceback`), then raised. A cancelled or timed-out container is recorded too. Writing the
  diagnostics never hides the error being diagnosed.
- **Validation before a container starts.**
  - Run ids are plain directory names (`[A-Za-z0-9][A-Za-z0-9._-]*`).
  - Repo ids are `<user-or-org>/<name>`.
  - Revisions are the full 40-character lowercase hex sha.
  - `smoke-1` is Revision 1's smoke directory. A start there is refused, not resumed, because its
    protocol and inputs differ.

---

## Analysis rules

<a id="study-matching"></a>
### Studies and study matching

- **The studies.** `smoke` (the kernel scan), `smoke-nf`, `full`, `expb`, `smoke-c` and `expc`
  (`studies/registry.py`).
  - `smoke`: 1 round, MX, NV, NVx and the scan treatments, C = 1, 32, 128, 3 reps, M2 10 s, local
    checkpoints allowed. NVa isn't in it, because the scan picks its kernel, and MXp isn't needed.
  - `smoke-nf`: the fusion smoke. NVnf must show the CuTe-DSL kernel with the fusion off on the
    published checkpoints. Not a result.
  - `smoke-c`: 1 round, MX and NV, cells (1, 1,024), (1, 32,768), (1, 127,360) and (128, 360), 3
    reps. It checks the override, exact token counts, timeouts and KV capacity.
- **Recorded protocol.** A manifest line records `protocol` with the fields, in order: `rounds`,
  `treatments`, `concurrencies`, `m1_reps`, `m2_duration_s`, `require_published`,
  `kv_cache_dtype`, `gpu_memory_utilization`, `primary_concurrencies`, `cells`, `max_model_len`,
  `hf_overrides`. `cells` is empty unless the cells have their own prompts. Two registered studies
  may not record the same protocol, and this is checked at import.
- **Which study a run is analysed as** (`study_for`):
  1. The study the manifest line names (`study`, recorded since commit `4e14a41`), with its
     recorded rounds.
  2. Otherwise, the study whose protocol the line recorded exactly, except for `rounds`. Keys that
     older runs lack read as the values they had: `kv_cache_dtype` bfloat16,
     `gpu_memory_utilization` 0.9, `primary_concurrencies` [8, 32, 128], `cells` [],
     `max_model_len` 4096, `hf_overrides` "".
  3. Otherwise, a fallback (`fallback_study`): Experiment C's rules if the protocol has `cells`,
     Experiment B's if its KV dtype is FP8 (`fp8`, `fp8_e4m3`, `fp8_e5m2`), the kernel-scan
     smoke's if one of its treatments is a scan treatment (NVc, NVt, NVd, NVv), and the main
     run's otherwise.
- **Primary endpoints.** `primary_concurrencies` are the batches whose verdicts decide: (8, 32,
  128) for the main run and the smokes ([§8][e8]), (256, 512) for Experiment B ([§13][e13]), and
  (1,) for Experiment C, whose decision is on its contrasts ([§16][e16]).
- **Unknown KV dtype.** A KV dtype that is neither BF16 nor FP8 is analysed as a main run, with a
  note in the summary.
- **Where the rules come from.** Gates, verdict and G3 follow the study's rules (`Study.verdict`,
  `Study.g3`). Rounds, reps, cells and primary batches come from the last manifest line.
  `primary_concurrencies` falls back to (8, 32, 128), since every run recorded before the field
  was a main run.

<a id="pairing"></a>
### Values and pairing

- **Which sessions count.** The latest servers row per (round, treatment), so later lines win.
  Failed rows are left out of every computation and listed; a failed re-run replaces an earlier
  good session.
- **Rows of uncounted sessions.** M1 and M2 rows of sessions that don't count are never read, and
  one that does not load is skipped.
- **Cell values.** An M1 cell value per (round, treatment, cell) is the median of the measured reps
  (not the warmup). It is NaN if any rep is not finite. M2 values come from valid cells only.
- **Usable values.** A value must be finite and > 0. Unusable values are skipped with a warning
  and listed. A pair needs both treatments usable in the same round.
- **Ratios.** A ratio is the exponentiated mean of paired log-ratios with a two-sided t-CI at 90%
  (equivalent to TOST at α = 0.05). It needs at least 2 paired rounds for a CI; with one round it
  is a point estimate with no verdict.
- **Differences.** Δ (ms) is a paired mean with a t-CI. A contrast of Δs is paired per round over
  the rounds both cells have.
- **M2 ratios.** Throughput ratios are inverted, so that > 1 still means the second-named treatment
  is faster. The A/A replica gets no ITL ratio.
- **R = F·Φ.** The decomposition is reported per batch, not gated.

<a id="decision-rules"></a>
### Decision tables

The tables are pre-registered in [§8][e8] (δ = 0.02, 90% CIs, primary C ∈ {8, 32, 128}).

- **Labels.** K's verdicts are relabelled "pinned (CuTe-DSL) faster" and "alternative faster".
  Φ's are relabelled "fusion slower" and "fusion faster".
- **Complete rounds required.** The overall H_eq verdict is computed only if every primary C has
  exactly the registered number of paired NV/MX rounds, at least 2. Otherwise it reads "incomplete
  (k/R rounds)" as a typed `Incomplete`, so code tells it from a decision without parsing text.
- **Extension.** When G3 fails, the pre-registered remedy is extending to 10 rounds
  (`--rounds 10`). For `full-1`, G3 still failed after the extension, and RESULTS.md reports the
  verdict as formally not interpretable ([§11][e11]).

<a id="gates"></a>
### Gates

The definitions are in [§9][e9], as §13 and §16 amend them. Implementation facts:

- **No sessions.** With no sessions at all, every gate fails.
- **Blocking.** G6b is the only non-blocking gate. A run is interpretable when no blocking gate
  fails.
- **G1.** Every session's `linear_kernels` equals `[expected]`, its fusion facts match
  ([#fusion](#fusion)), no NVnf session loaded NV's graph ([#compile-cache](#compile-cache)), and
  the last manifest line's `problems` is exactly `[]`. The expected class and fusion are those of
  the server args that line records for the treatment (`inputs.treatment_server_args`), or the
  `TREATMENTS` table's where it records none; the kernel scan expects the same classes.
- **G2.** The checkpoint report's NV/MX bytes are within 2% of the format ratio. A missing or
  malformed report fails.
- **G3.** See [#aa](#aa).
- **G4.** No environmental throttling ([#telemetry](#telemetry)).
- **G5a.** The report's `mx_problems` and `nv_problems` are both empty lists. No report fails.
- **G5b.** Every treatment's mean NLL is at most 0.5 nats/token above the manifest's BF16
  reference. No usable reference fails.
- **G6a (main run and Experiment B), per batch.**
  - Every session's `cudagraph_capture_sizes` includes every batch.
  - Every (session, batch) has exactly `m1_reps` measured rows.
  - No M1 row has a `preemptions_delta` other than 0.
  - Every session's `kv_cache_tokens` ≥ the largest wave's requirement ([#kv-capacity](#kv-capacity)).
- **G6a (Experiment C), per (C, P) cell.** The same checks, and also:
  - `protocol.cells` must parse: non-empty, [C, P] pairs of positive ints, no duplicates;
  - no row is malformed;
  - no measured cell is outside `protocol.cells`.

  Exact prompt-token counts are enforced by the M1 client, so a row exists only for an exact block
  ([#m1](#m1)).
- **G6b.** Every M2 cell is valid ([#m2](#m2)), every valid cell has the usage-based source, and no
  valid cell has a non-zero preemption delta.
- **G7.** Each round has one GPU UUID and one `start_id`. An unreadable GPU UUID fails it.

<a id="aa"></a>
### A/A variants (G3)

| Rule | Studies | Passes when |
|---|---|---|
| `aa_ratio` | main run, Experiment B, smokes | at every measured batch, the A/A ratio t_MXp/t_MX has a 90% CI that is classified Equivalent and contains 1 |
| `aa_effects` | Experiment C (§16 amendment) | with Δ_AA = t_MXp − t_MX per cell, both E_tok,AA = Δ_AA(1, 127,360) − Δ_AA(1, 1,024) and E_batch,AA = Δ_AA(128, 360) − Δ_AA(1, 127,360) have 90% CIs that contain 0 and lie inside ±0.25 ms; it fails closed without an MXp session, without data, or with fewer than 2 rounds |

- **Smokes.** A smoke has no MXp, so G3 fails there by construction and is not read.
- **Why Experiment C's rule differs.** Per-start MXFP4 autotune gives a per-round A/A SD of
  0.07–0.10 ms ([#autotune](#autotune)). A cross-cell A/A difference then has a 5-round 90% CI
  half-width of about 0.12 ms, and the per-cell ratio rule failed in the main run on sub-0.5%
  offsets. ±0.25 ms is well below the effects that separate the hypotheses (about 0.35 ms and
  0.6 ms). Per-cell A/A ratios are still reported for Experiment C, not gated.
- **Extension for Experiment C.** A run whose G3 fails or whose answer is inconclusive gets the
  extension to 10 rounds. A study without an A/A replica or a registered extension (`smoke-c`) is
  not that run.

<a id="expb"></a>
### Experiment B reading

[§13][e13] has the design.
- **Verdict.** H_eq's rule is applied to R_B at the primary C ∈ {256, 512}.
- **The reading.**
  - VALIDATED: H_B,eq holds.
  - MXFP4 practically faster at any primary C: contradicts H_B,nv.
  - NVFP4 practically faster at every primary C: H_B,nv holds; at only some: it does not.
  - Anything else: neither.
- **Prediction.** R_B(512) ∈ (0.973, 0.987).
- **FP8 KV scales.** Neither checkpoint carries KV scales, so vLLM uses 1.0 for K, V and the FP8
  query (`CompressedTensorsKVCacheMethod`). Decode attention is FlashInfer trtllm-gen with an FP8
  query.
- **NLL against the main run.** `--reference-run` compares the FP8-KV NLL with the main run's
  BF16-KV NLL ([#descriptive-checks](#descriptive-checks)). It was +0.0039 / +0.0038 nats/token.

<a id="expc"></a>
### Experiment C rules

[§16][e16] has the design, predictions and amendment.
- **Effects.** E_tok = Δ(1, 127,360) − Δ(1, 1,024) and E_batch = Δ(128, 360) − Δ(1, 127,360), with
  Δ = t_MX − t_NV in ms.
- **Classification** (margin 0.1 ms).
  - CI inside ±0.1: no effect.
  - Upper bound < −0.1: shrinks the saving.
  - Lower bound > 0.1: grows the saving.
  - Otherwise: inconclusive.
- **Answer.**
  - (no effect, shrinks): batch (GEMM size) drives the gap.
  - (shrinks, no effect): tokens drive it.
  - (shrinks, shrinks): both.
  - (no effect, no effect): neither, which is inconsistent with the main run.
  - Anything else, including "grows": inconclusive.

  The answer needs both effects to have exactly the registered rounds, at least 2.
- **Predictions.** The registered predictions come from the main run's Δ = 0.42 / 0.49 / 0.32 /
  0.14 / −0.19 ms at batch 1 / 8 / 32 / 64 / 128. H_tokens reads that curve against C × 1,664 mean
  KV tokens (`np.interp`). Cell KV tokens are C × (P + 640), and every batch-arm cell holds 128,000.
- **Config reproduction.** R(1, 1,024) must be within ±0.01 of the main run's batch-1 R (0.934),
  with a 1e-12 float tolerance. It is reported, not gated.

<a id="descriptive-checks"></a>
### Descriptive checks

None of these decides anything.
- **Accuracy note** ([§8][e8]).
  - It uses MX and NV only. MXp, NVa and NVnf serve the same checkpoints, and their NLLs are in
    G5b.
  - Per-window NLLs are averaged over a treatment's sessions, and every session must have a usable
    `nll_per_prompt` with the same window count.
  - NV − MX gets a paired percentile bootstrap: 10,000 resamples, seed 0, 95%, numpy's legacy
    `RandomState`. That is a frozen stream, so the interval is the same every time.
  - A BF16 reference with a different window count is flagged.
- **Reference run** (`--reference-run`). This run's mean NLL per treatment next to another run's.
  Problems are flagged: a reference without sessions, a missing treatment, the same KV dtype (the
  difference would then not be the KV dtype's), or different NLL windows (`nll_prompts_sha256`).
- **Effective bandwidth** ([§8][e8]). Bytes-model step bytes at context P + 640 over the median M1
  step, as a fraction of 8 TB/s. It covers MX, MXp, NV, NVa and NVnf; a KV dtype the bytes model
  does not know leaves the bytes as None.
- **Order effect.** The median step split by which wave of the pair ran first.
- **NVx cross-check.** NVx/NV median step per C.
- **Experiment C design check.**
  - The manifest's cells are against the registered ones: an unregistered cell is a deviation, a
    subset is a smoke.
  - The treatments must be a subset of (MX, NV, MXp).
  - `max_model_len` 131072, the override, BF16 KV, 0.90 and `m2_duration_s` 0 must hold.
  - Every served argv must have the window, the override and no RoPE-scaling marker.
- **Experiment C NLL cross-check.** Each treatment's mean session NLL against `full-1`'s (MX
  1.9092, MXp 1.9092, NV 1.8238), and whether the NLL windows are the main run's
  (`nll_prompts_sha256` `dc20a83531303e1559a5eafe5d1f907cc77680363a3b6d05cd0438d8d8ffd55c`).

---

## Data formats

<a id="run-dir"></a>
### Run directory

| Path | Content |
|---|---|
| `manifests.jsonl` | one `ManifestLine` per container start: environment, `code_commit`/`code_dirty`, `study`, `start_id`, `inputs`, `checkpoint_report`, `protocol`, `problems` |
| `servers.jsonl` | one `ServerRow` per finished session (the log facts of [#server-log](#server-log), NLLs, argv, env, GPU UUID, autotune record), or `failed: true` with `error` for a scan session that could not be served |
| `m1.jsonl` | one `M1Row` per wave pair (`set`, `warmup`, `first`, `t1_s`, `t2_s`, `step_s`, `prompt_len`), with its block's `block_telemetry` and `preemptions_delta` |
| `m2.jsonl` | one `M2Row` per cell: validity, reasons, metrics, `telemetry`, `preemptions_delta` |
| `errors.jsonl` | `utc`, `session_hint`, `error`, `traceback`, and `nonfatal` for a scan failure |
| `servers/<session_id>.log` | the `vllm serve` log |
| `telemetry/<session_id>.csv` | the 200 ms sampler |
| `autotune/<session_id>/` | the session's FlashInfer autotune cache |
| `m2_raw/<session_id>_c<C>.json`, `lib_logs/<session_id>.log` | the bench's output and log |
| `summary.md`; `ratio_vs_c.png`, `pareto_m2.png`; or `results_c.json`, `expc_delta.png` | written by `fp4bench analyze` |

- **Ids.** `session_id` is `r<round>_<treatment>-<8 hex>`, and `start_id` is a uuid4 hex.
- **Profile runs.** `sessions.json`, `profiles.json`, `summary.md`,
  `<config>_<treatment>_c<C>.pt.trace.json.gz`, `*.profiler_out.txt`, `servers/` and `autotune/`.
  The report can be rebuilt from the traces with `python -m fp4bench.profiling report <dir>`.
- **Microbenchmark runs.** `results.json` and `summary.md`.
- **Committed data.** `data/` holds the committed compact data (manifests, servers rows, M1 and M2
  rows) of the 2026-10 runs, produced by the pre-release code.
- **Shareable figures.** `fp4bench plot` writes `docs/figures/r_vs_concurrency.png`,
  `nvfp4_vs_mxfp4_decode_b200.png`, `nvfp4_vs_mxfp4_decode_b200_simple.png` and
  `nvfp4_vs_mxfp4_batch_vs_tokens_b200.png`.

<a id="jsonl"></a>
### JSONL rows and schema versions

- **Versions.** Rows written now carry `schema_version` 2 (`core/schema.py`). v1 rows have no
  version and load as they are. Any other version is refused.
- **Missing and extra keys.** A key a stored row lacks reads as its default and is left out again
  when written back, so a v1 row round-trips unchanged. Keys the schema doesn't know are kept in
  `extra` and written back after the schema's own, so such a key comes back at the end of its row.
- **M1 rows without `prompt_len`.** They are the main run's 1,024-token M1. In a cell study a row
  without one has no cell.
- **Strict JSON.** NaN and infinities are written as `null`.
- **Truncated lines.** A container killed mid-write leaves a last line without its newline.
  - Reading skips such a last line with a warning.
  - The next append cuts it off, or adds the newline if the line is complete, so the next row is
    not glued onto it.
  - A bad line anywhere else is corruption and raises, naming the file and the line.

<a id="data-files"></a>
### Files on the data volume

| File | Content |
|---|---|
| `m1_prompts.json` | 6 × 512 lists of 1,024 token ids |
| `nll_prompts.json` | 64 lists of 512 token ids |
| `c_prompts.json` | `{"cells": {"<C>x<P>": sets}, "seed", "n_sets", "m1_prompts_sha256", "reused_m1_cells", "sharegpt_revision", "tokenizer_revision"}`; a cell key is `<C>x<P>` with no leading zeros |
| `bf16_reference_nll.json` | `{"per_prompt", "mean", "source_revision", "nll_prompts_sha256"}` |
| `<checkpoint>/fp4bench_quant.json` | provenance: source and revision, format, scheme, tool versions, recipe, calibration (dataset@revision, samples, tokens, max length, seed, sha256), quantized modules, tokenizer files, config parity, BF16 reference, `utc` |
| `<checkpoint>/fp4bench_fetch.json` | `{"repo_id", "revision", "utc"}` |
| `results/checkpoint_report.json` (local) | `mx_quantized_bytes`, `nv_quantized_bytes`, `nv_over_mx`, `expected_nv_over_mx`, the quant formats, `nv_has_activation_scales`, `config_diff`, `mx_problems`, `nv_problems` |

---

<a id="legacy"></a>
## Revision 1 leftovers

Revision 1 was gpt-oss-20b, a MoE ([§11][e11]). Task 5.2 of the cleanup plan removed its code:
the MoE backend log parsing (`Using '<name>' (Mxfp4|NvFp4) MoE backend`), the config alignment
`align_config_with_mx` with its gpt-oss tests ([#config-parity](#config-parity)) and the MoE
sampling rationale in `lib_bench.py`'s docstring ([#m2](#m2)). What remains:
- **MoE servers-row fields.** Every committed servers row carries `moe_backends` and
  `moe_backend_lines` (both `[]` for the dense Qwen3-32B). `ServerRow` keeps the two fields so
  those rows round-trip byte for byte: as unknown keys they would move to the end of the row
  ([#jsonl](#jsonl)). Rows written now lack them.
- **Stale-prompt detection.** The frame check that replaces gpt-oss prompts ([#prompts](#prompts)).
- **Revision 1 run.** Its prepare measured a 67-token gpt-oss chat frame and date-dependent prompts,
  and its NVFP4 quantization was ModelOpt 0.47.0 on the experts only. Neither is used by
  Revision 2.

---

<a id="anchor-index"></a>
## Anchor index

| Anchor | Topic |
|---|---|
| [#stack](#stack) | pinned versions, inputs, version and GPU checks, code commit |
| [#image](#image) | Modal images |
| [#treatments](#treatments) | treatment table, order, ratios |
| [#quantization](#quantization) | llm-compressor recipe and calibration |
| [#coverage](#coverage) | every Linear except `lm_head` |
| [#bf16-reference](#bf16-reference) | BF16 reference NLL |
| [#checkpoint-checks](#checkpoint-checks) | G2 and G5a layout, quantization config, bytes |
| [#config-parity](#config-parity) | `config.json` parity |
| [#fingerprints](#fingerprints) | layout and content fingerprints |
| [#publishing](#publishing) | Hub publishing |
| [#fetch](#fetch) | fetching published revisions |
| [#staging](#staging) | local copies on ephemeral disk |
| [#server-args](#server-args) | common `vllm serve` arguments |
| [#server-lifecycle](#server-lifecycle) | start, stop, per-server sequence |
| [#kernel-selection](#kernel-selection) | kernel pins, classes, fallback, BF16 GEMMs |
| [#fusion](#fusion) | SiLU·mul + activation-quant fusion |
| [#nv-nf](#nv-nf) | NV-nf's compilation config |
| [#nv-alt](#nv-alt) | kernel scan rule and the `smoke-r2-2` choice |
| [#compile-cache](#compile-cache) | torch.compile cache and G1's NVnf check |
| [#server-log](#server-log) | every parsed log line and its source |
| [#prompts](#prompts) | prompt construction, seeds, reuse |
| [#m1](#m1) | M1 wave design and arithmetic |
| [#payloads](#payloads) | request bodies |
| [#m2](#m2) | llm-inference-bench usage and validity |
| [#nll](#nll) | NLL check (G5b) |
| [#autotune](#autotune) | fresh autotune per start, MXFP4 vs NVFP4 tuning |
| [#telemetry](#telemetry) | clock-event counters, sampler, G4 |
| [#preemption-metric](#preemption-metric) | `vllm:num_preemptions_total`, aggregation, retries |
| [#kv-capacity](#kv-capacity) | KV capacity formula and sizing |
| [#long-context](#long-context) | `max_position_embeddings` override, no YaRN |
| [#bytes-model](#bytes-model) | Qwen3-32B bytes model |
| [#profiler](#profiler) | torch profiler config, semantics, categories |
| [#microbench](#microbench) | kernel microbenchmark method |
| [#modal](#modal) | Modal |
| [#modal-detach](#modal-detach) | detached apps, spawn, incidents |
| [#modal-volumes](#modal-volumes) | volumes, paths, commits |
| [#modal-disk](#modal-disk) | ephemeral disk bounds |
| [#modal-resources](#modal-resources) | resources and timeouts |
| [#modal-secrets](#modal-secrets) | `huggingface-secret` |
| [#modal-platform](#modal-platform) | same GPU, gVisor, clocks |
| [#modal-costs](#modal-costs) | budgets and run times |
| [#run-integrity](#run-integrity) | restarts, identity checks, failures |
| [#study-matching](#study-matching) | studies and how a run's study is found |
| [#pairing](#pairing) | sessions, cell values, paired CIs |
| [#decision-rules](#decision-rules) | decision tables and completeness |
| [#gates](#gates) | gate implementation |
| [#aa](#aa) | A/A rules (G3) |
| [#expb](#expb) | Experiment B reading |
| [#expc](#expc) | Experiment C effects and answer |
| [#descriptive-checks](#descriptive-checks) | accuracy note, bandwidth, design checks |
| [#run-dir](#run-dir) | run directory layout |
| [#jsonl](#jsonl) | JSONL rows and schema versions |
| [#data-files](#data-files) | files on the data volume |
| [#legacy](#legacy) | Revision 1 leftovers |
| [#anchor-index](#anchor-index) | this list |

[e3]: EXPERIMENT.md#3-hypotheses
[e4]: EXPERIMENT.md#4-definitions
[e5]: EXPERIMENT.md#5-model-analysis-and-quantitative-predictions
[e6]: EXPERIMENT.md#6-materials
[e6sw]: EXPERIMENT.md#software-pinned
[e71]: EXPERIMENT.md#71-server-identical-for-every-treatment-except-the-checkpoint-path-and-kernel-selection
[e72]: EXPERIMENT.md#72-prompt-data-built-once-and-stored-on-the-volume
[e73]: EXPERIMENT.md#73-m1-steady-state-decode-step-primary-instrument
[e74]: EXPERIMENT.md#74-m2-llm-inference-bench-sustained-decode-secondary-instrument
[e75]: EXPERIMENT.md#75-per-server-sequence
[e77]: EXPERIMENT.md#77-schedule
[e8]: EXPERIMENT.md#8-analysis-plan
[e9]: EXPERIMENT.md#9-validity-gates-all-blocking-gates-must-pass-before-results-are-interpreted
[e10]: EXPERIMENT.md#10-threats-to-validity-scope-and-follow-ups
[e11]: EXPERIMENT.md#11-deviation-log
[e13]: EXPERIMENT.md#13-experiment-b-high-batch-with-fp8-kv-secondary-pre-registered-2026-10-05
[e14]: EXPERIMENT.md#14-kernel-microbenchmark-secondary-pre-registered-2026-10-05
[e15]: EXPERIMENT.md#15-profiling-sessions-diagnostic-pre-registered-2026-10-05
[e16]: EXPERIMENT.md#16-experiment-c-batch-size-or-tokens-secondary-pre-registered-2026-10-05
