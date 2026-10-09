# Experiment: NVFP4 vs MXFP4 on decode-bound LLM inference

- **Date:** 2026-10-04, **Revision 2.1** (see [Revision history](#revision-history))
- **Author:** Cezar Cocu
- **Status:** Data collected 2026-10-05 (`full-1`, 10 rounds; `expb-1`; microbenchmark;
  profiling). Results in [RESULTS.md](RESULTS.md). Revision 2.1 added secondary studies (§6 NV-nf,
  §13–§15) before the full run.
- **Model / hardware:** `Qwen/Qwen3-32B` (dense) on 1× NVIDIA B200 (SM100), served by vLLM v0.31.0

> Pre-registration rule: the hypotheses, primary endpoints, equivalence margin and decision rules
> below are fixed **before** any data is collected. Any later change goes in the
> [Deviation log](#11-deviation-log) with a reason. Results must not be reinterpreted against
> criteria chosen after the fact.

### Revision history

| Revision | Date | Change |
|---|---|---|
| 1 | 2026-10-04 | gpt-oss-20b (MoE). MX = OpenAI's native MXFP4 release; NV = ModelOpt re-quantization of it. |
| **2.1** | 2026-10-05 | **Secondary additions before the full run, motivated by the smoke runs** (§11): the NV-nf treatment (fusion-matched F and the fusion effect Φ), the preemption guard, Experiment B (high batch, FP8 KV, §13), a kernel-level microbenchmark (§14) and profiling sessions (§15). H_eq, R's primary endpoints, δ and the decision rules are unchanged. |
| **2** | 2026-10-04 | **Switched to dense Qwen3-32B with four treatments.** The Revision 1 smoke run showed that vLLM (v0.31.0 and `main`) and SGLang (v0.5.21 and `main`) cannot serve gpt-oss with NVFP4 experts, because their NVFP4 MoE paths have no expert-bias support (§11). Revision 1 collected no experimental data: its smoke run measured only MX and is not a result. Revision 2 also removes Revision 1's known confounds: mismatched activation precision, NV quantized twice, and NV-only publishing. |

---

## 1. Background

On 2026-10-02 Stas Bekman (author of [ml-engineering](https://github.com/stas00/ml-engineering))
reported that on B200, NVFP4 beats MXFP4 by ~9% in maximum achievable matmul FLOPS (MAMF):
6624 vs 6087 TFLOPS with torch 2.14. He recommended NVFP4 over MXFP4 on Blackwell and newer.
His [FP4 formats write-up](https://github.com/stas00/ml-engineering/blob/master/training/dtype.md#fp4-formats)
says both formats run through the same tensor-core instruction at the same theoretical peak
(9000 TFLOPS), so the measured gap is "a software difference, not a hardware one".

MAMF measures large, **compute-bound** GEMMs. LLM **decode** at realistic batch sizes is
**memory-bandwidth-bound**: each step streams the weights from HBM and does little arithmetic per
byte. In that regime speed should follow bytes moved, not FLOPS. Stas agreed it is cheap to test
both formats on a real workload. If the results are close, MXFP4 has an advantage of its own:
it is an OCP standard that also runs on AMD MI355X and other accelerators.

### The two formats

| | MXFP4 (OCP MX) | NVFP4 (NVIDIA) |
|---|---|---|
| Element | FP4 E2M1 | FP4 E2M1 |
| Block size | 32 | 16 |
| Block scale | E8M0 (power of two) | E4M3, plus one FP32 scale per tensor |
| Storage | **4.25 bits/element** | **4.5 bits/element** |

NVFP4 weights are 4.5 / 4.25 = **1.059×** the size of MXFP4 weights. If streaming the weights
is the bottleneck, MXFP4 can be at most 1 − 4.25/4.5 ≈ **5.6%** faster on the quantized part.

## 2. Research questions

> **Q1 (primary).** In the decode-bound regime, does the choice between NVFP4 and MXFP4 weights,
> **on the same kernel family**, produce a practically significant difference in decode speed?
>
> **Q2 (secondary).** How does NVFP4 on its best *alternative* vLLM kernel compare with MXFP4?
>
> **Q3 (secondary).** How sensitive is NVFP4 decode speed to **kernel choice**, compared with the
> format effect? This is the decode-time counterpart of the MAMF "software difference".
>
> **Q4 (secondary, added 2026-10-05).** How much of R is the format plus its GEMM, and how much is
> vLLM's NVFP4-only SiLU·mul + activation-quant fusion (§6, NV-nf)?
>
> **Q5 (secondary, added 2026-10-05).** Above the compute ridge, does NVFP4's MAMF advantage show
> up end to end (Experiment B, §13), and at the kernel level, which format's GEMM is faster at
> each M, and do both reach HBM bandwidth at small M (kernel microbenchmark, §14)?

## 3. Hypotheses

The treatments are defined in §6: MX, NV (CuTe-DSL kernel), NV-alt (alternative kernel), MX′ (A/A),
and NV-nf (NV with the activation-quant fusion off; secondary, added 2026-10-05).
All ratios are oriented so that **> 1 means the second-named treatment is faster**. They are
measured with instrument M1 (§7.3).

- **R = t_NV / t_MX**, the format effect on the same kernel family. R > 1 means MXFP4 is faster.
- **D = t_NValt / t_MX**, NVFP4 on its alternative kernel vs MXFP4. D > 1 means MXFP4 is faster.
- **K = t_NValt / t_NV**, NVFP4's kernel sensitivity. K > 1 means the CuTe-DSL kernel is faster
  than the alternative.

| ID | Hypothesis | Prediction |
|---|---|---|
| **H_eq** (the hypothesis under test, on R) | In decode-bound operation neither format has a practical advantage | 1 − δ ≤ R ≤ 1 + δ |
| **H_mx** (bytes model) | Speed follows bytes moved, so MXFP4's smaller weights make it faster | R > 1, bounded above by the bytes model in §5 |
| **H_nv** (kernel advantage carries over) | NVFP4's ~9% MAMF advantage carries over into decode | R < 1 |

D and K are estimated and classified with the same decision table (§8). They answer Q2 and Q3 and
do not decide H_eq.

Two secondary ratios answer Q4 (added 2026-10-05, §11). They do not decide H_eq either:
- **F = t_NVnf / t_MX**, format + GEMM with the activation-quant fusion matched (off in both).
  F > 1 means MXFP4 is faster.
- **Φ = t_NV / t_NVnf**, the fusion effect within NVFP4. Φ < 1 means the fusion makes NVFP4
  faster. R = F · Φ exactly (per round), so F and Φ split R into its two parts.

**Equivalence margin: δ = 2%.** Rationale:
1. The bytes model's best case for MXFP4 is about 3–5% (§5). δ = 2% is roughly half the largest
   effect physics allows.
2. Serving benchmarks typically vary 0.5–1% between runs, so a 2% margin can be resolved with
   paired repetitions (§8).
3. In practice, a throughput difference under 2% is unlikely to outweigh accuracy (favours NVFP4)
   or portability (favours MXFP4) when choosing a format.

## 4. Definitions

- **Decode-bound:** most of the per-step time goes to streaming weights and the KV cache from
  HBM, and prefill is excluded from the metric. In this experiment it means (a) M1 measures pure
  decode steps, with prefill removed by construction, and (b) every tested concurrency is below
  the compute-bound ridge (§5).
- **Treatment:** the checkpoint plus the kernel selection it is served with. Everything else is
  held constant.
- **Same kernel family:** MX, NV and MX′ run with `--linear-backend flashinfer_cutedsl`. NV gets
  `FlashInferCuteDslNvFp4LinearKernel`. vLLM v0.31.0 has no MXFP4 kernel under that backend, so
  MX logs a fallback warning and gets `FlashInferMxFp4LinearKernel`, its only W4A4 path on SM100.
  That class also calls FlashInfer's `mm_fp4` with `backend="cute-dsl"`. Both formats therefore run
  the same CuTe-DSL GEMM, with block size 32 (MXFP4) or 16 (NVFP4).
- **vLLM's default NVFP4 kernel is the same CuTe-DSL kernel.** It is first in
  `_POSSIBLE_NVFP4_KERNELS` on SM100, so "as deployed" equals NV, and a default-kernel treatment
  would duplicate it.
- **Alternative kernel (NV-alt):** the NVFP4 checkpoint pinned to a *different* W4A4 NVFP4 kernel.
  It is chosen from the smoke kernel scan by the rule in §7.7.

## 5. Model analysis and quantitative predictions

Qwen3-32B (`config.json`, revision `9216db5`): 64 layers; hidden size 5120; MLP intermediate size
25600; 64 query heads and 8 KV heads, head dim 128; vocab 151,936; untied `lm_head`; every layer
uses full attention.

**What is quantized:** every Linear layer except `lm_head` (§6), i.e. `q/k/v/o_proj` and
`gate/up/down_proj` in all 64 layers.

| Component | Size |
|---|---|
| Quantized Linear weights (31,205,621,760 params) | MXFP4 **16.58 GB**, NVFP4 **17.55 GB** (Δ 0.97 GB) |
| BF16 `lm_head`, read every step | 1.56 GB |
| KV cache (BF16) per token | 262,144 B (64 layers × 8 KV heads × 128 × K,V × 2 B) |
| KV cache read per sequence per step at M1's mean context (1664 tokens) | 436 MB |

**Ideal bytes-model ratio:** R_ideal(C) = (W_nv + S + C·K) / (W_mx + S + C·K). Here W is the
quantized weight bytes, S the BF16 `lm_head` bytes and K the KV bytes per sequence. In a dense
model every token uses every weight, so W does not depend on C.

| Concurrency C | Ideal step at 8 TB/s | R_ideal (upper bound) |
|---|---|---|
| 1 | ~2.4 ms | 1.052 |
| 8 | ~2.8 ms | 1.045 |
| 32 | ~4.1 ms | 1.030 |
| 64 | ~5.9 ms | 1.021 |
| 128 | ~9.4 ms | 1.013 |

Real step times include fixed costs: kernel launches, per-layer activation quantization (W4A4),
attention compute, norms, sampling and scheduling. These pull R toward 1. **Realistic expectation
under H_mx: R ≈ 1.01–1.04, shrinking with C as the KV cache dilutes the weight bytes.**

**Compute-bound ridge.** In a dense model all C tokens share every weight. With W4A4 at roughly
6.1–6.6 PFLOPS (Stas's MAMF) and 8 TB/s, the ridge is at about C ≈ 200–235. **Every tested C
(≤ 128) is memory-bound.**

**Memory ceiling:** with BF16 KV, 128 sequences at the maximum M1 context of 2176 tokens need
~73 GB of KV plus ~19 GB of weights, which fits. C = 512 (Revision 1) would need ~290 GB of KV,
so the sweep stops at 128.

## 6. Materials

### Hardware
- **1× NVIDIA B200 SXM (SM100).** It is the GPU Stas measured, and it has native FP4 tensor-core
  support for both formats. One GPU is enough (§5).
- **Not used:**
  - B300/GB300 (SM103): different FP4 throughput and kernels. A follow-up.
  - RTX PRO 6000 / RTX 5090 (SM120): different FP4 instructions, GDDR7, different vLLM backends.
  - H100/H200: no FP4 tensor cores, so both formats would fall back to weight-only Marlin.
  - GB200: same GPU, but an ARM host and rack-scale rental, which this experiment doesn't need.
- **Platform: [Modal](https://modal.com)**, `gpu="B200"`. Environment check on 2026-10-04: NVIDIA
  B200, compute capability 10.0, driver 580.95.05, CUDA 13.0, 183,359 MiB, MIG disabled, 1000 W
  power limit (`docs/run-log.md`).
- **Platform constraints and how we handle them:**
  - **Clocks can't be locked** (`nvidia-smi -lgc` is not permitted). Throttling is detected
    instead: before and after each measurement window, read `nvidia-smi -q -d PERFORMANCE` and
    take the difference in its clock-event-reason counters (gate G4).
  - **Same physical GPU:** the whole protocol (all rounds, all treatments) runs inside **one
    Modal function call**, i.e. one container, launched with `modal run --detach`.
  - **Weights on local disk:** at container start, both checkpoints are copied from the Modal
    volume to the container's local ephemeral disk. Their content fingerprints are computed on
    the local copies, and vLLM serves only from the local copies, never from the network volume.
  - **CPU and memory:** request 16 CPU cores and 64 GiB explicitly, the same for every
    treatment. The vLLM server and the load generators need host CPU.
  - **Sandboxed runtime (gVisor):** it may add host-side overhead. That overhead is identical
    across treatments, so it affects absolute times, mostly at C = 1, not the paired ratios.
  - **Exact GPU:** request exactly `"B200"`. Abort if the manifest doesn't show `NVIDIA B200` /
    compute capability 10.0.
- **Record:** GPU name, UUID, compute capability, power limit, driver, CUDA version, and the
  clock-event counters per window.

### Software (pinned)
Kernel availability changes quickly and is a major confounder, so we use the newest release and
pin it exactly. Versions as of 2026-10-04:

| Component | Pinned version | Why |
|---|---|---|
| Serving image | `vllm/vllm-openai:v0.31.0` @ `sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b` (amd64), vLLM commit `db9527a` | Newest release image (built 2026-10-03). Verified in-container: vLLM 0.31.0, the commit above, `problems: []` |
| CUDA (in image) | 13.0.2 | Matches Modal's driver 580.95.05 (CUDA 13.0) |
| FlashInfer | 0.7.0.post1, plus `flashinfer-cubin` 0.7.0.post1 | Latest on PyPI (2026-09-29) |
| PyTorch / transformers | 2.13.0+cu130 / 5.17.0 | As shipped in the image |
| Quantization | `llmcompressor==0.14.0` (with `compressed-tensors==0.19.0`) | Latest (2026-09-22). Supports transformers ≤ 5.17.0 and torch ≤ 2.14. One tool for both formats |
| [llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench) | v0.7.7, commit `c71ec1f2b34a4f1c8f702f1750ccd70da0e389e2`, run with `LLM_BENCH_NO_UPDATE_CHECK=1` | M2's community-standard Sustained Decode measurement (MIT license; documented AIPerf parity) |

Alternatives we rejected:
- **`nightly`** (commit `18f8f96`): same FlashInfer, PyTorch and CUDA, and a release is easier
  for others to reproduce.
- **`cu134-*`** images: CUDA 13.4 is newer than the driver's CUDA 13.0, and the image disables
  CUDA forward-compatibility.

Re-check this table right before the full run. If a newer release has appeared, diff the FP4
kernel-selection code and log the change in §11.

### Treatments

| ID | Checkpoint | Served with | Role |
|---|---|---|---|
| **MX** | Qwen3-32B, MXFP4 W4A4 (ours) | `--linear-backend flashinfer_cutedsl` | baseline |
| **NV** | Qwen3-32B, NVFP4 W4A4 (ours) | `--linear-backend flashinfer_cutedsl` | **Q1: format effect (R)** |
| **NV-alt** (`NVa`) | the same NVFP4 checkpoint | `--linear-backend flashinfer_cudnn` (selected by the §7.7 rule in `smoke-r2-2`) | Q2 (D) and Q3 (K) |
| **MX′** | the same MXFP4 checkpoint, separate server launch | `--linear-backend flashinfer_cutedsl` | A/A noise floor (G3) |
| **NV-nf** (`NVnf`) | the same NVFP4 checkpoint | `--linear-backend flashinfer_cutedsl --compilation-config '{"pass_config": {"fuse_act_quant": false}}'` | Q4 (secondary): F and Φ, fusion matched to MX (added 2026-10-05, §11) |

**Why NV-nf exists.** vLLM v0.31.0 turns on its SiLU·mul + activation-quant fusion
(`fuse_act_quant`) automatically for NVFP4 checkpoints only (`vllm/config/vllm.py:166-175`,
`enable_act_fusion` → `ModelConfig.is_nvfp4_quantized()`). NV therefore runs one fused
`silu_and_mul_nvfp4_quant` kernel before every `down_proj`, while MX runs an Inductor SiLU·mul
kernel followed by FlashInfer's MXFP4 quantize. v0.31.0 has no dense SiLU·mul + MXFP4 fusion at all
(the only fused MXFP4 activation op is for MoE experts), so the comparison can only be matched by
turning the fusion **off** for NVFP4, not on for MXFP4. A value set on the command line is never
overridden by vLLM's default (`vllm/config/vllm.py:1027-1032`). R stays the format comparison
as deployed (with the fusion difference); F isolates format + GEMM with the fusion matched.

**How both checkpoints are made** (identically, except for the preset):
- **Source:** `Qwen/Qwen3-32B`, revision `9216db5781bf21249d130ec9da846c4624c16137`, BF16.
- **Tool:** llm-compressor `oneshot`, using the `NVFP4` preset for NV and the `MXFP4` preset for
  MX. Both are W4A4: weights fp4 and activations fp4, dynamic per block. NVFP4 also has a static
  per-tensor activation global scale, set from calibration.
- **Coverage:** every `Linear` layer except `lm_head`. This is NVIDIA's own production recipe
  (`nvidia/Qwen3-32B-NVFP4` excludes only `lm_head`), applied to both formats.
- **Calibration:** the same 256 ShareGPT prompts (≤ 1024 tokens, fixed seed) for both. MXFP4 has
  no static activation scales, so the data only affects NVFP4's global scale.
- **KV cache:** BF16 for every treatment (`--kv-cache-dtype bfloat16`). No KV quantization is
  written into either checkpoint.
- **`config.json` parity:** both share the source config apart from `quantization_config`. This
  is checked at every run start (G5a).
- **Publishing:** both checkpoints go to the Hugging Face Hub (private first, then public on
  confirmation). Each has a provenance model card: source revision, tool versions, preset,
  coverage, calibration data, and measured bytes. **The full run serves the exact published
  revisions**, fetched with `--step fetch` and verified by content fingerprint.

**Third-party checkpoints, inspected and not used as treatments:**
- `nvidia/Qwen3-32B-NVFP4` (revision `16426c6`): ModelOpt 0.35.0, all Linear except `lm_head`,
  but **FP8 KV cache**.
- `INCModel/Qwen3-32B-MXFP4-CT-AutoRound` (revision `e32b451`): MXFP4 W4A4, but **attention left
  in BF16**.

Pairing them would compare recipes (layer coverage, KV dtype), not formats. NVIDIA's checkpoint
is used once, in the smoke run only, as a cross-check: served with BF16 KV and the pinned kernel,
its step time should match our NV.

**Prompt data:** ShareGPT V3 (`anon8231489123/ShareGPT_Vicuna_unfiltered`, file
`ShareGPT_V3_unfiltered_cleaned_split.json`, revision `192ab2185289094fc556ec8ce5ce1e8e587154ca`),
first user turn of each conversation. It is wrapped in Qwen3's chat template. Qwen3's template
embeds no date, so the prompts are byte-stable across days.

## 7. Procedure

Two instruments, both run against the same `vllm serve` instance:

- **M1, steady-state decode** (the hypothesis test): decode step time at an exact batch size,
  with prefill and HTTP overhead removed by construction.
- **M2, community-standard sustained decode**: llm-inference-bench's Sustained Decode, i.e.
  aggregate decode throughput with C streams kept busy for a fixed time, through the normal
  streaming chat API. It is the number the local-inference community publishes.

### 7.1 Server, identical for every treatment except the checkpoint path and kernel selection
```
vllm serve <local checkpoint dir> --served-model-name fp4bench --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 --max-num-seqs 512 --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.90 --no-enable-prefix-caching --seed 0 --api-server-count 4 \
  --kv-cache-dtype bfloat16 [--linear-backend flashinfer_cutedsl]   # bracketed: MX, NV, MX′
# NV-alt: --linear-backend flashinfer_cudnn
# NV-nf:  --linear-backend flashinfer_cutedsl --compilation-config '{"pass_config": {"fuse_act_quant": false}}'
```
- No speculative decoding. Async scheduling and CUDA graphs stay at their defaults.
- The FP4 linear kernel each treatment actually ran is recorded from the server log (gate G1),
  and so is the resolved `pass_config` (whether `fuse_act_quant` is on) from the engine-config line.
- `--api-server-count 4` keeps the HTTP/detokenization frontend off the critical path.

### 7.2 Prompt data, built once and stored on the volume
- **M1 prompt sets:** 6 sets × 512 prompts (set 0 is warmup; sets 1–5 are measured). Each prompt
  is ShareGPT user text in Qwen3's chat template, cut so that the templated prompt is **exactly
  1024 tokens**.
- **NLL prompts:** 64 fixed 512-token windows of ShareGPT text.
- **BF16 reference NLL:** while the BF16 model is loaded for quantization, its mean NLL on the
  same 64 windows is computed once and stored. Gate G5b uses it.
- M2 uses llm-inference-bench's own built-in prompt.
- Every treatment gets byte-identical prompts.

### 7.3 M1: steady-state decode step (primary instrument)
For each C ∈ {1, 8, 32, 64, 128}:
1. Send a **synchronized wave of exactly C** non-streaming `/v1/completions` requests (token-id
   prompts, `ignore_eos`, temperature 0). Time from the first send to the last response = T(N).
2. Do this with N₁ = 128 and N₂ = 1152 output tokens: **t_step = (T(N₂) − T(N₁)) / 1024**.
   Prefill, HTTP, the admission ramp and the finishing tail are identical in both waves and
   cancel. What remains is 1024 decode steps at batch exactly C, with context 1152–2176.
3. Decode throughput = C / t_step. Interactivity = 1 / t_step.
4. One warmup pair (set 0), then 5 measured pairs (sets 1–5). Alternate which of N₁ or N₂ runs
   first. The cell value is the median of the 5.
5. Integrity checks:
   - Every response must report exactly N completion tokens and exactly 1024 prompt tokens.
   - The client allows ≥ C concurrent connections.

### 7.4 M2: llm-inference-bench Sustained Decode (secondary instrument)
Run once per C, so each cell has its own throttle-counter window:
```
LLM_BENCH_NO_UPDATE_CHECK=1 python /opt/llm-inference-bench/llm_decode_bench.py \
  --host 127.0.0.1 --port 8000 --model fp4bench \
  --skip-prefill --contexts 0 --concurrency <C> --max-tokens 2048 --duration 30 \
  --display-mode plain --no-hw-monitor --no-resume --output <cell>.json
```
- `--duration 30` is the bench's default, kept for comparability.
- Sampling stays at the model's default; no `--temperature` is passed.
- **Validity, per cell.** A cell counts only if all of these hold:
  - it is not `underfilled` or `capacity_limited`;
  - no loop was detected and warmup did not time out;
  - `aggregate_tps` is positive and finite, with no request errors;
  - `aggregate_source` is `openai_continuous_usage`;
  - for C > 1, `effective_concurrency ≥ 0.98·C`.

  Invalid cells are recorded with reasons and excluded (G6b).

### 7.5 Per-server sequence
`vllm serve` → wait for `/health` → quality check (G5b: mean NLL on the 64 windows) → M1 sweep →
M2 sweep → shut down → wait until GPU memory is released.

### 7.6 Telemetry
- An `nvidia-smi` sampler (200 ms) runs for the whole server lifetime.
- Clock-event **counters** are snapshotted before and after every M1 C-block and every M2 cell
  (gate G4).

### 7.7 Schedule
- **5 rounds** × {MX, NV, NV-alt, MX′, NV-nf}, one server each, in cyclically rotated order (a
  5×5 Latin square: every treatment holds every position once):
  R1: MX, NV, NV-alt, MX′, NV-nf · R2: NV, NV-alt, MX′, NV-nf, MX · R3: NV-alt, MX′, NV-nf, MX, NV ·
  R4: MX′, NV-nf, MX, NV, NV-alt · R5: NV-nf, MX, NV, NV-alt, MX′.
  (NV-nf was added on 2026-10-05, before the full run; §11.)
- **Fresh kernel autotune per server start.** Each start gets an empty
  `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`, so FlashInfer's tactic choice is drawn independently per
  session and its variance enters the round-to-round CIs. Otherwise one cached draw would be
  reused in every round, a fixed effect invisible to the CIs.
- **Estimated time:** ~3 min startup + ~4 min M1 + ~4 min M2 ≈ 11 min per server, so ~4.6 h
  for 25 servers.
- **Smoke run first:** 1 round at C ∈ {1, 32, 128}, with 3 measured M1 pairs. Treatments:
  - MX and NV;
  - NVx, the NVIDIA-checkpoint cross-check (§6);
  - the **NVFP4 kernel scan**: our NVFP4 checkpoint on `flashinfer_cutlass` (NVc),
    `flashinfer_trtllm` (NVt), `flashinfer_cudnn` (NVd) and vLLM `cutlass` (NVv).

  A scan kernel that fails to start or crashes is logged and treated as ineligible; the smoke
  continues.
- **NV-alt selection rule** (pre-registered 2026-10-05, before any Revision 2 data):
  1. **Eligible** scan kernels must meet all of these:
     - every session logged exactly the expected kernel class, and it is not CuTe-DSL;
     - no session failed;
     - it has an M1 step at C=32;
     - its mean NLL is within 0.01 nats/token of NV's in the same smoke run;
     - it passes G5b.
  2. **Pick** the eligible kernel with the lowest median M1 step time at C=32.
  3. **Tie:** if the runner-up is within 1% of it at C=32, pick whichever of the two is faster at
     C=128. If they are also within 1% there, take the earlier in the order NVc, NVt, NVd, NVv.
  4. The chosen `--linear-backend` is written into the config before the full run. The full run
     refuses to start if NV-alt is unpinned or pinned to NV's kernel.
- **Fusion smoke (`smoke-nf`, added 2026-10-05):** before the full run, 1 round of MX, NV and NV-nf
  at C ∈ {1, 32, 128} with 3 measured M1 pairs, serving the published checkpoints. It checks that
  NV-nf runs the CuTe-DSL kernel with the fusion off (G1) before 5 rounds depend on it. Not a result.
- **Budget:** ~$10 for two quantizations, ~$10 for smoke (7 servers, plus staging), ~$5 for the
  fusion smoke, ~$34 for the full run.

## 8. Analysis plan

- **Primary metric (decides H_eq):** M1 step time t_step per (treatment, C, round), i.e. the
  median of the 5 measured pairs.
- **Paired log-ratio** per C and round, e.g. d_r = ln t_NV,r − ln t_MX,r. Pairing within a round
  cancels drift between rounds.
- **Estimate:** ratio = exp(mean d). 90% CI: exp(mean d ± t₀.₉₅,₄ · s_d/√5). This is a two
  one-sided tests (TOST) equivalence test at α = 0.05.
- **Computed for:** R (NV vs MX, primary), D (NV-alt vs MX), K (NV-alt vs NV), the A/A
  ratio MX′ vs MX (gate G3), and, secondary (Q4), F (NV-nf vs MX) and Φ (NV vs NV-nf). F and Φ
  use the same paired CIs and decision table; for Φ, "above 1.02" reads "fusion practically
  slower" and "below 0.98" reads "fusion practically faster".
- **M2:** the same paired CIs for the sustained-decode tok/s ratio of each pair, plus the p50 ITL
  ratios. Also Pareto curves (tok/s/user vs tok/s/GPU) for all three formats/kernels. Reported;
  does not decide H_eq.
- **Interpretation aids:**
  - Effective bandwidth = bytes-model step bytes (§5, at context 1664) / t_step, as a fraction of
    8 TB/s.
  - R̂ next to R_ideal per C.
  - An order-effect table (step time split by which wave ran first).
- **Accuracy (secondary, descriptive):** mean NLL on the 64 windows for BF16 (reference), MX and
  NV. Report NV − MX with a paired bootstrap 95% CI over the 64 windows. This measures Stas's
  "NVFP4 is more accurate" point for this model. It is not a hypothesis test and does not affect
  any verdict.

### Decision rules, per concurrency level (for R; D and K use the same table)

| 90% CI of the ratio | Verdict at that C |
|---|---|
| Entirely inside [0.98, 1.02] | **Equivalent** |
| Entirely above 1.02 | **MXFP4 practically faster** (for K: CuTe-DSL kernel practically faster) |
| Entirely below 0.98 | **NVFP4 practically faster** (for K: alternative kernel practically faster) |
| Excludes 1 but overlaps the margin | Statistically different, practically uncertain |
| Includes 1 and extends past the margin | Inconclusive (underpowered) |

### Overall verdict on H_eq
- **Primary endpoints:** C ∈ {8, 32, 128}.
- **Secondary endpoints:** C = 1 (dominated by launch and scheduling overhead) and C = 64.
  They are reported but do not decide the verdict.
- **H_eq validated:** R is Equivalent at all three primary endpoints.
- **H_eq nullified:** R shows one format practically faster at ≥ 1 primary endpoint. The
  direction says whether the evidence supports H_mx or H_nv.
- **Otherwise:** partial or inconclusive. Extend to 10 rounds, then consider the follow-ups in §10.

## 9. Validity gates (all blocking gates must pass before results are interpreted)

| Gate | Check | If it fails |
|---|---|---|
| **G1** | Kernel audit. For every treatment, save the FP4 linear kernel line from vLLM's log. Each session must show exactly its expected class: `FlashInferMxFp4LinearKernel` for MX and MX′, `FlashInferCuteDslNvFp4LinearKernel` for NV, NV-nf (and NVx), and the selected alternative's class for NV-alt. **Fusion audit (added 2026-10-05):** each session's resolved `pass_config.fuse_act_quant` must be its expected value (on for every NVFP4 treatment except NV-nf; off for MX, MX′ and NV-nf), and the INFO line "Enabled custom fusions: act_quant" must appear exactly when it is on. Manifest problems must be empty and in-container versions must match §6 | Stop and investigate: we would not know what was compared |
| **G2** | Bytes check. From the safetensors headers, NV/MX quantized-weight bytes ≈ 4.5/4.25 = 1.0588 (format plus scales); record vLLM's load-time weight memory for both | Bytes model invalid. Use the measured bytes in the predictions |
| **G3** | A/A (MX′ vs MX) 90% CI of the M1 ratio contains 1 and lies inside [0.98, 1.02] at every C. Also reported for M2 | Too noisy: reduce noise or add rounds |
| **G4** | No *environmental* throttling in a measurement window: the deltas of "HW Thermal Slowdown", "SW Thermal Slowdown" and "HW Power Braking" are 0. "SW Power Capping" is reported, not discarded: hitting the 1000 W cap under load is real, possibly format-dependent behaviour | Re-run the affected rounds |
| **G5a** | Checkpoint structure. Every quantized Linear stores packed FP4 (U8) with the format's block scales: E4M3 per 16 for NV, E8M0 per 32 for MX. `lm_head`, embeddings and norms are BF16. The quantization config names the right format. `config.json` matches the source apart from `quantization_config` | Re-quantize |
| **G5b** | Not broken: each treatment's mean NLL on the 64 windows is within 0.5 nats/token of the BF16 reference | Investigate; a broken checkpoint invalidates its treatment |
| **G6a** | Run integrity (M1, blocking): CUDA-graph capture sizes include every tested C; every M1 response has exactly N completion and 1024 prompt tokens; no (session, C) pair is missing; **no preemption during any M1 wave** (vLLM's preemption counter does not move across an M1 block; added 2026-10-05). An M2 cell with a preemption is invalid (G6b) | Fix the configuration and re-run |
| **G6b** | M2 cell validity (reported; does **not** block the M1 verdict): invalid cells per §7.4 are excluded and listed | Investigate; re-run affected rounds if M2 coverage matters |
| **G7** | Each round's sessions ran on one GPU (same UUID) within one container start (same `start_id`) | Re-run the affected round |

## 10. Threats to validity, scope, and follow-ups

**Known limitations:**
1. **One model, one shape family.** Dense Qwen3-32B, so GEMM shapes are 5120×{8192, 1024,
   25600}. The MoE regime (many small expert GEMMs) is not covered; see §11 for why.
2. **CuTe-DSL may not be NVFP4's fastest kernel.** That is why NV-alt exists. If K > 1 + δ, the
   alternative is slower than CuTe-DSL; if K < 1 − δ, the format comparison R was run on a
   kernel that handicaps NVFP4, and D is the more relevant number for practitioners.
3. **W4A4 activation quantization** is part of every FP4 GEMM in both formats. Its per-layer
   cost is included in the measurement by design.
4. **One GPU and one software snapshot.** Kernels change quickly, and the results are tied to
   the pinned versions and this physical GPU.
5. **`ignore_eos`** forces exact output lengths, so outputs past the natural end are less natural
   text. Compute is unaffected (dense). M2's loop detection invalidates degenerate cells.
6. **M2 contains small restart prefills** when streams finish together. M1 has none.

**Out of scope:** accuracy beyond the descriptive NLL comparison, multi-GPU or tensor
parallelism, long context (> 4k), FP8 KV cache (except Experiment B, §13), other engines, and prefill (excluded from M1 by
construction; skipped in M2).

**Follow-ups, if the results call for them:**
- an nsys kernel breakdown (`--cuda-graph-trace=node`; §15 uses the torch profiler instead)
- an MoE model once vLLM's NVFP4 MoE path supports expert biases (or a bias-free MoE such as
  Qwen3-30B-A3B)
- an SGLang cross-check
- other hardware (B300 / SM103, MI355X for MXFP4)

## 11. Deviation log

| Date | Deviation | Reason |
|---|---|---|
| 2026-10-04 (pre-data, Rev 1) | M2 validity rules tightened; G6 split into G6a (blocking) and G6b (reported) | §7.4 and §9 contradicted each other. M1 is primary, so an M2 cell must not veto the M1 verdict |
| 2026-10-04 (pre-data, Rev 1) | Run-integrity guards: `start_id` per container start; input-identity and rounds-floor checks on restart; published-checkpoint requirement for the full run; checkpoint report at every start | They prevent mixing inputs or GPUs within a round. They change no measurement |
| 2026-10-04 (pre-data, Rev 1) | gpt-oss NV `config.json` aligned to MX apart from `quantization_config` | transformers 5.17 re-serialized identical RoPE values; the treatments must differ only in quantization |
| 2026-10-04 (pre-data) | **Revision 2: model switched from gpt-oss-20b to dense Qwen3-32B; NV-auto treatment added** | Revision 1's smoke run: vLLM v0.31.0 failed to load the NVFP4 gpt-oss export with `KeyError: 'layers.0.mlp.experts.w2_bias'`. Source inspection: vLLM's NVFP4 MoE methods (ModelOpt and compressed-tensors, v0.31.0 and `main`) register no expert biases, while gpt-oss experts have them. SGLang v0.5.21 and `main` have the same gap. Community "NVFP4 gpt-oss" checkpoints were inspected and rejected (BF16 relabelled as NVFP4; GGUF with Q8_0 non-experts). A dense model avoids MoE routing entirely, and with both formats made by us from BF16 the activation precision matches (W4A4). NV-auto was added to separate the format effect from kernel selection |
| 2026-10-05 (pre-data) | **NV-auto replaced by NV-alt**, chosen by a pre-registered smoke kernel scan (§7.7); fresh FlashInfer autotune per server start; smoke uses 3 measured pairs | The Revision 2 code audit found that vLLM v0.31.0's default NVFP4 kernel on SM100 is the same CuTe-DSL kernel NV is pinned to, so NV-auto would have duplicated NV and made K a second A/A comparison. NV-alt keeps the intent: separate the format effect (R) from NVFP4's kernel sensitivity (K). The cached autotune would have frozen one tactic draw across all rounds. Also added: the full run refuses an unpinned NV-alt, `treatment_server_args` is part of the restart identity check, and the BF16 reference must match the NLL prompts hash. Decided before any Revision 2 data |
| 2026-10-05 (pre-full-run) | **NV-alt set to `flashinfer_cudnn` (NVd)** | Applied the §7.7 rule to `smoke-r2-2`, unchanged: all four scanned kernels eligible; NVd fastest at C=32 (8.877 ms vs NVc 9.370 ms, +5.55%, not a tie). Not a deviation from the rule, recorded here because it fixes a pre-registered free parameter. The first smoke (`smoke-r2-1`) was cancelled by a sleeping client and discarded in full; the experiment step now spawns the call |
| 2026-10-05 (pre-full-run, Rev 2.1) | **NV-nf treatment added to the full run** (NVFP4, CuTe-DSL pin, `fuse_act_quant` off), with secondary ratios F = t_NVnf/t_MX and Φ = t_NV/t_NVnf (§3, §8), a fusion audit in G1, and a `smoke-nf` check before the full run (§7.7) | `smoke-r2-1` and `smoke-r2-2` server logs: every NVFP4 server resolved `'fuse_act_quant': True` and logged "Enabled custom fusions: act_quant"; MX resolved `False`. Source: vLLM enables the fusion for NVFP4 checkpoints only, and has no dense MXFP4 equivalent (§6). So R mixes the format with a compile-fusion difference. NV-nf matches the fusion (off for both) without changing R. Secondary only: H_eq, R's endpoints, δ and the decision rules are unchanged. The schedule becomes a 5×5 Latin square (~4.6 h) |
| 2026-10-05 (pre-full-run, Rev 2.1) | **Preemption guard:** vLLM's `vllm:num_preemptions_total` is read before and after every M1 block and M2 cell; a moved counter fails G6a (M1) or invalidates the M2 cell (G6b) | Needed for Experiment B, whose C = 512 needs ~94% of the FP8 KV pool; recorded for every run so all runs share one validity rule. At C ≤ 128 with BF16 KV the main run uses ≤ 50% of the pool, so the rule should never fire there |
| 2026-10-05 (pre-full-run, Rev 2.1) | **Experiment B (§13), kernel microbenchmark (§14), profiling sessions (§15) pre-registered** | Smoke observation: achieved HBM bandwidth was only ~37–61% of 8 TB/s (C=1: NV 3.35, MX 2.97 TB/s; C=128: ~4.9 TB/s), so decode at low C is kernel/overhead-bound and the bytes hypothesis' premise is not met. §14 tests the bytes question at the kernel level, §15 attributes the step time, and §13 tests the high-batch regime where Stas's MAMF advantage should appear. Separate runs; they do not feed the H_eq verdict |
| 2026-10-05 (procedural) | Experiment B (§13) ran concurrently with the full run on a separate B200 instead of after it; the microbenchmark (§14) ran before the full run | Wall-clock only (overnight unattended run). Separate containers, separate GPUs, separate A/A controls; the checkpoint volume is only read. No analysis rule changes |
| 2026-10-05 (post-data) | **G3 failed after the pre-registered 10-round extension** (A/A CI excludes 1 at C = 32 and 64, both within ±0.6%); RESULTS.md reports the H_eq verdict as formally not interpretable and adds a clearly-labelled **post-hoc** robustness check (R against MX′ and against the pooled MX/MX′ baseline) | No further remedy is pre-registered. The post-hoc check does not replace G3; it only shows that R's verdicts do not depend on which MXFP4 server start is the baseline |
| 2026-10-05 (post-main-run) | **Experiment C pre-registered (§16)**: batch-vs-tokens follow-up, M1 only, token arm at batch 1 (1k–127k) and a constant-token batch arm; context limit raised via `max_position_embeddings` override (no YaRN) | Stas Bekman's question on the results. A new, separate experiment with its own pre-registered predictions; it changes nothing in the main run, Experiment B or their analysis |
| 2026-10-05 (pre-data, Experiment C) | §16 amended after the smoke: G3 becomes an A/A test on E_tok and E_batch (CIs contain 0, inside ±0.25 ms), per-cell ratio A/A reported only; extension to 10 rounds if G3 fails or the answer is inconclusive; m1_reps stays 5 | Main-run A/A noise (per-round SD 0.07–0.10 ms from per-start MXFP4 autotune) would make the per-cell ratio rule fail on irrelevant offsets; the smoke's within-session precision (0.002–0.04 ms) needs no more pairs. Decided before any Experiment C data |

## 12. Deliverables

1. Results table: R̂, D̂ and K̂ (M1) and the M2 ratios, each with a 90% CI per C, verdicts per C,
   the overall verdict on H_eq, and gate status.
2. Plots:
   - R, D, K, F and Φ vs C with CI bands, the ±δ band and R_ideal (§5).
   - M2 Pareto curves (tok/s/user vs tok/s/GPU) for MX, NV and NV-alt.
3. Accuracy note: BF16 / MX / NV mean NLL and the NV − MX difference with its CI.
4. A short write-up for Stas: setup, verdict, the kernel decomposition, and the accuracy note.
5. Raw data: M1/M2 rows, bench JSONs, server logs, telemetry CSVs, and per-start manifests.
6. **Both checkpoints, published on the Hugging Face Hub**, with provenance model cards. The
   repo ids and visibility are confirmed before upload.

## 13. Experiment B: high batch with FP8 KV (secondary; pre-registered 2026-10-05)

A separate run (`--mode expb`) after the main full run. It does not feed the H_eq verdict.

**Question (Q5).** Above the compute ridge (§5, C ≈ 200–235), where every FP4 GEMM is
compute-bound, does NVFP4's MAMF advantage (~9%, §1) appear in end-to-end decode?

**Design.**
- **Treatments:** MX, NV and MX′ (A/A), served exactly as in §7.1 (same kernel pins; NV keeps
  vLLM's default fusion, i.e. as deployed), except for two flags, identical for all three:
  - `--kv-cache-dtype fp8`. Neither checkpoint carries KV scales, so vLLM uses 1.0 for K, V and
    the FP8 query (vLLM v0.31.0 `CompressedTensorsKVCacheMethod`; decode attention is FlashInfer
    trtllm-gen with an FP8 query). Any accuracy cost of the uncalibrated scales shows in the NLL
    check (G5b) and is reported against the main run's BF16-KV NLL; it is the same for both formats.
  - `--gpu-memory-utilization 0.95`. M1 at C = 512 holds 512 × 2176 = 1,114,112 tokens. At 0.90
    the FP8 pool would be ~1,117k tokens (0.26% headroom); 0.95 adds ~70k tokens.
- **C ∈ {128, 256, 512}**, M1 and M2 exactly as §7.3–§7.4 (1024-token prompts, N₁ = 128,
  N₂ = 1152, 5 measured pairs; M2 30 s per C). CUDA-graph capture sizes include 256 and 512.
- **5 rounds**, cyclic rotation: R1 MX, NV, MX′ · R2 NV, MX′, MX · R3 MX′, MX, NV · R4 = R1 · R5 = R2.
- **Gates:** as §9, with G3 (A/A) at every C. Plus a **capacity check** (blocking, part of G6a):
  every session's logged "GPU KV cache size" must be ≥ 1,114,112 tokens, and the preemption counter
  must not move during any M1 block.
- **Endpoints:** R_B = t_NV / t_MX at **C = 256 and C = 512 (primary for Experiment B)**;
  C = 128 is secondary (the bridge to the main run's C = 128, which has BF16 KV). Same paired CIs,
  δ = 2% and decision table as §8.

**Hypotheses.** H_B,nv: R_B < 1 − δ at C ∈ {256, 512} (the MAMF advantage carries over).
H_B,eq: R_B within [0.98, 1.02].

**Prediction (pre-registered).** At high C the step is dominated by attention reading the KV
cache, not by the GEMMs:

| C | FP8 KV read per step | weights + `lm_head` | FP4 GEMM time at 6.1–6.6 PFLOPS | R_ideal (bytes) |
|---|---|---|---|---|
| 128 | 27.9 GB | 18.1 / 19.1 GB (MX / NV) | 1.2–1.3 ms | 1.021 |
| 256 | 55.8 GB | same | 2.4–2.6 ms | 1.013 |
| 512 | 111.7 GB | same | 4.8–5.2 ms | 1.0075 |

At the ~4.9 TB/s the smoke run achieved at C = 128, the C = 512 step should take ~25–30 ms, of
which the FP4 GEMMs are a share s ≈ 0.15–0.30. A full 9% NVFP4 GEMM advantage would then move
the step by only 9% × s ≈ 1.4–2.7%: **predicted R_B(512) ≈ 0.973–0.987**, so most likely
"Equivalent" or "practically uncertain". Such an outcome would **not** refute a kernel-level
NVFP4 advantage (that is §14's question); it would mean the advantage is diluted below δ in
end-to-end decode at this context length. R_B(512) < 0.973 would mean NVFP4's end-to-end
advantage is larger than its GEMM share allows, pointing at non-GEMM effects (e.g. the fusion).
R_B > 1.02 would contradict H_B,nv outright.

**Budget:** 15 servers × ~18 min ≈ 4.5 h, ~$33.

## 14. Kernel microbenchmark (secondary; pre-registered 2026-10-05)

A separate Modal step (`--step microbench`, one B200, vLLM image of §6), not part of any run. It
asks the bytes question directly: at small M, does each format's GEMM reach HBM bandwidth, and is
MXFP4 then ~5.6% faster?

**Shapes** (Qwen3-32B, N × K): the merged layers vLLM actually runs, qkv_proj 10240 × 5120,
o_proj 5120 × 8192, gate_up_proj 51200 × 5120, down_proj 5120 × 25600, and the individual
q 8192 × 5120, k = v 1024 × 5120, gate = up 25600 × 5120.
**M ∈ {1, 8, 32, 64, 128, 256, 512}.**

**Implementations**, each for MXFP4 and NVFP4 on the same packed bytes:
- **(a) vLLM's path:** FlashInfer `mm_fp4(..., backend="cute-dsl")` with `block_size` 32 / 16,
  scales in the 128×4 swizzled layout, exactly as `FlashInferMxFp4LinearKernel` and
  `FlashInferCuteDslNvFp4LinearKernel` call it. vLLM autotunes the MXFP4 GEMM at startup but skips
  autotuning for the NVFP4 CuTe-DSL GEMM (`kernel_warmup.py`, `skip_ops={"fp4_gemm"}`), so (a)
  runs MXFP4 autotuned and NVFP4 on its heuristic tactic, **as deployed**; an autotuned-NVFP4
  variant is also timed.
- **(b) Stas's MAMF path:** NVFP4 through `torch._scaled_mm` (single-level, cuBLASLt) and MXFP4
  through `torch._scaled_mm_v2` with `BlockWise1x32` scales, which in PyTorch 2.13 on CUDA runs
  MSLK's CUTLASS `f4f4bf16` kernel, **not** cuBLASLt. (b) therefore compares two different
  libraries; it is reported as the MAMF reference, not as a format comparison. Also timed:
  NVFP4 two-level (`_scaled_mm_v2` with a tensor-wise global scale, cuBLASLt).
- **(c) Same library, both formats:** FlashInfer `mm_fp4(..., backend="cudnn")`, which supports
  both formats on SM100: the cleanest format-only comparison outside vLLM's own path.
- **Activation quantization**, timed separately per format at each M and K ∈ {5120, 8192, 25600}:
  MXFP4 `flashinfer.mxfp4_quantize(backend="cute-dsl")` (vLLM's MX path), NVFP4 vLLM
  `scaled_fp4_quant` (vLLM's NV path), and the fused NVFP4 `silu_and_mul_nvfp4_quant` on the
  down_proj input (2 × 25600 wide) against MXFP4's unfused SiLU·mul + quantize.

**Timing.** CUDA graphs; L2 is defeated by rotating through enough input copies to exceed 2× the
L2 size per graph; ≥ 200 timed calls per sample, 7 samples, median reported with the min–max.
Outputs: µs per call, achieved bandwidth (bytes = packed weights + weight scales + packed
activations + activation scales + BF16 output, read or written once) as a fraction of 8 TB/s, and
TFLOPS. A one-off numerical check confirms each implementation's output against a BF16 reference
on dequantized operands (relative error reported, not gated).

**Prediction (pre-registered).**
- **Small M (≤ 32) on the large shapes** (gate_up, down, qkv): if both formats' kernels reach
  ≥ 80% of 8 TB/s, t_NV / t_MX ≈ the bytes ratio ≈ 1.05–1.06 (**MXFP4 ~5.6% faster**). If either
  stays below 60%, the GEMM is not bandwidth-bound and the ratio measures kernel efficiency, not
  bytes; the bytes hypothesis then cannot be tested at that point.
- **Large M (256, 512):** compute-bound. Both formats run the same block-scaled MMA at the same
  peak, so no direction is predicted for (a) and (c); for (b), MAMF predicts NVFP4 ~9% faster.
- **Activation quantization:** NVFP4 needs twice as many scales per element, but MXFP4's scale is
  a power of two; no direction is predicted. The fused NVFP4 SiLU·mul + quantize is predicted to
  beat MXFP4's two unfused kernels on the down_proj input.

Descriptive: no verdict and no TOST; ratios outside [0.98, 1.02] are called practically different.

## 15. Profiling sessions (diagnostic; pre-registered 2026-10-05)

A separate Modal step (`--step profile`), not timed with any measurement. The image has no
`nsys`, so vLLM's torch profiler is used (`--profiler-config` with `ignore_frontend: true`,
`with_stack: false`).

- **Sessions:** MX, NV and NV-nf with the §7.1 configuration at C = 1 and C = 32; MX and NV with
  the §13 configuration (FP8 KV, 0.95) at C = 512.
- **Window:** a wave of C requests (1024-token prompts, 256 output tokens); the profiler skips the
  first steps (`delay_iterations` 16, or 48 at C = 512) so the prefill chunks are excluded, and
  records exactly 32 decode steps (`max_iterations`).
- **Attribution:** GPU kernel time per decode step, split into FP4 GEMM, activation quant
  (including fused SiLU·mul + quant), SiLU·mul, attention, BF16 GEMM (`lm_head`), norms/RoPE and
  other, from the kernel names in the trace. The category patterns are set from the first trace's
  kernel list and recorded with the result; the full per-kernel table is published with it.
- **Use:** attribute the NV-vs-MX gap at C = 1 and C = 32 to GEMM vs activation quant/fusion vs
  everything else, and measure the GEMM share s at C = 512 that §13's prediction depends on. The
  sum of kernel time per step is compared with M1's step time to show how much of the step is GPU
  idle (launch and scheduling overhead).

## 16. Experiment C: batch size or tokens? (secondary; pre-registered 2026-10-05)

A separate run (`--mode expc`), motivated by Stas Bekman's question on the results: *is it
batch size or the number of tokens that changes the gap?* In the main run both grew together
(context fixed at 1.15–2.18k, batch 1 → 128). In decode the two act on different work: **batch**
sets the GEMM size (M = batch; one new token per sequence per step) and the per-step activation
quantization; **tokens** (sequences × context) set the KV cache that attention reads, which is
identical for both formats.

**Design.** Two arms in one run, M1 only (§7.3 waves with N₁ = 128, N₂ = 1152, 5 measured pairs; no
M2), 5 rounds, treatments MX, NV, MX′ (A/A), cyclic rotation as §13:
- **Token arm (batch fixed at 1):** prompt length P ∈ {1,024, 4,096, 16,384, 32,768, 65,536,
  127,360}. Batch 1 keeps every GEMM at M = 1; only the KV cache grows.
- **Batch arm (tokens fixed):** cells with the same mean KV tokens during the measured window,
  C × (P + 640) = 128,000: (C, P) = (1, 127,360), (8, 15,360), (32, 3,360), (128, 360). Only the
  batch changes. (1, 127,360) is shared with the token arm.
- **Server:** §7.1 exactly (same kernel pins, BF16 KV, fusion defaults, `--gpu-memory-utilization
  0.90`), except `--max-model-len 131072` and `--hf-overrides '{"max_position_embeddings": 131072}'`.
  The override enlarges the stock RoPE table; it does **not** use YaRN, so the position encoding
  and every kernel stay those of the main run. Past the trained ~40k positions output quality
  degrades; that does not affect timing (fixed output lengths, `ignore_eos`, dense compute).
- **Prompts:** exact-length ShareGPT text in Qwen3's chat template, 6 sets (1 warmup + 5 measured)
  of C distinct prompts per cell, built once, byte-identical for every treatment, hashed into the
  manifest. The (1, 1,024) cell reuses the main run's M1 prompt sets.

**Metrics.** Per cell: R = t_NV / t_MX (90% CI, §8 table) and the **absolute per-step saving
Δ = t_MX − t_NV** (ms, paired per round, 90% CI). Δ is primary: attention cost is the same for both
formats, so anything that only adds format-independent work leaves Δ unchanged while shrinking R.
Equivalence margin for Δ differences: **±0.1 ms** (≈ 25% of the main run's batch-1 saving of
0.42 ms).

**Decision rules.**
- **Token effect** E_tok = Δ(1, 127,360) − Δ(1, 1,024): CI inside ±0.1 ms → *no token effect*; CI
  entirely below −0.1 ms → *tokens shrink the saving*; otherwise inconclusive.
- **Batch effect at fixed tokens** E_batch = Δ(128, 360) − Δ(1, 127,360): CI entirely below −0.1 ms →
  *batch shrinks the saving*; CI inside ±0.1 ms → *no batch effect*; otherwise inconclusive.
- **Answer:** batch effect and no token effect → *batch (GEMM size) drives the gap*; token effect
  and no batch effect → *tokens drive it*; both → both; neither → inconsistent with the main run.

**Predictions (pre-registered).** From the main run: Δ = 0.42 / 0.49 / 0.32 / 0.14 / −0.19 ms at
batch 1 / 8 / 32 / 64 / 128 (context ~1.7k).
- **H_batch** (Δ depends on batch only): token arm Δ ≈ 0.42 ms at every P, so E_tok ≈ 0; R shrinks
  only by dilution: ≈ 0.93 (1k), 0.94 (16k), 0.95 (32k), 0.96 (64k), 0.97 (127k), assuming
  batch-1 attention reads the KV cache at ~4.5 TB/s (0.964–0.982 at 127k for 2–6 TB/s). Batch arm
  Δ ≈ 0.42 / 0.49 / 0.32 / −0.19 ms at C = 1 / 8 / 32 / 128, so E_batch ≈ −0.6 ms and R(128, 360)
  > 1.
- **H_tokens** (Δ depends on KV tokens only, i.e. the main-run curve read against C × 1,664):
  token arm Δ falls to ≈ 0.07 ms at 127k (E_tok ≈ −0.35 ms), R ≈ 0.99 there; batch arm Δ the same
  in all four cells (E_batch ≈ 0).
- **Our prediction: H_batch.** The main run's per-step saving shrinks and reverses with batch,
  which KV dilution alone cannot do (§ RESULTS.md), and nothing format-dependent grows with
  context at batch 1.

**Checks.**
- **Config reproduction:** R(1, 1,024) within ±0.01 of the main run's batch-1 R (0.934); otherwise
  the context-limit change moved the timing, and that is reported.
- **Gates:** as §9 with: G6a requires exactly P prompt tokens per response in each cell, CUDA-graph
  capture sizes including 1, 8, 32 and 128, no preemption, and a KV pool ≥ max over cells of
  C × (P + 1,152) tokens; G3 is evaluated per cell (and an A/A Δ = t_MX′ − t_MX is reported); SW
  power-cap fractions are reported per cell (the batch arm's high-batch cells may be power-capped,
  the batch-1 cells likely not).

**Amendment before the full run (2026-10-05, after the smoke `smoke-c-1`, before any Experiment C
data):**
- **G3 for Experiment C is an A/A test on the effects, not on per-cell ratios.** With
  Δ_AA = t_MX′ − t_MX per cell: E_tok,AA = Δ_AA(1, 127,360) − Δ_AA(1, 1,024) and
  E_batch,AA = Δ_AA(128, 360) − Δ_AA(1, 127,360). G3 passes if both 90% CIs contain 0 and lie inside
  **±0.25 ms**, i.e. well below the effects that separate the hypotheses (≈ 0.35 ms for E_tok under
  H_tokens, ≈ 0.6 ms for E_batch under H_batch). Reason: in the main run two identical MXFP4
  servers differed by a per-round SD of 0.10 ms at batch 1 and 0.07 ms at batch 128 (fresh MXFP4
  autotune per start, §7.7), so a cross-cell A/A difference has a 5-round 90% CI half-width of
  ≈ 0.12 ms; the per-cell ratio rule (CI contains 1) failed in the main run on sub-0.5% offsets. The
  per-cell ratio A/A is still reported, not gated. If G3 fails, or the answer is inconclusive, the
  run is extended to 10 rounds.
- **m1_reps stays 5:** the smoke's within-session SD of the step was 0.002 ms at (1, 127,360) and
  0.04 ms at (128, 360), far below the ±0.1 ms margin.
- **"Grows the saving"** (an effect CI entirely above +0.1 ms) is reported as such; the answer
  matrix treats it as "inconclusive" (§16 names only a shrinking saving as an effect).

**Smoke first:** 1 round, MX and NV, cells (1, 1,024), (1, 32,768), (1, 127,360), (128, 360), to
confirm the override, exact token counts, timeouts and KV capacity. **Budget:** ~26 min per
server × 15 ≈ 6.5 h (~$40), smoke ~$5.
