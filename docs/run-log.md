# Run log

Chronological record of cloud steps (Modal). Inputs and pins are listed in
EXPERIMENT.md §6. Deviations from the protocol go in EXPERIMENT.md §11.

## 2026-10-04: environment and data

### `modal run modal_app.py --step env` (Task 16.1)
App `ap-3fkJbRANQzVNU1lQ1PWm6C`. `problems: []`.

| Field | Value |
|---|---|
| GPU | NVIDIA B200, compute capability 10.0, driver 580.95.05, 183,359 MiB, power limit 1000 W |
| GPU UUID | GPU-9ddee40d-b8e3-77d8-782c-256e74b62783 (env-check container only; the experiment records its own) |
| Image | `vllm/vllm-openai@sha256:a4a4c0437bf7240089da5f08aa370c4aee17ae5290f7a3b468825ee26c4c3a6b` |
| vLLM | 0.31.0, build commit `db9527a46873454610df6dbedf79a36d6bf1a7f6` |
| FlashInfer | flashinfer-python 0.7.0.post1, flashinfer-cubin 0.7.0.post1 |
| PyTorch / CUDA | 2.13.0+cu130 / 13.0 |
| transformers | 5.17.0 |
| nvidia-cutlass-dsl | 4.7.1 |
| llm-inference-bench | `c71ec1f2b34a4f1c8f702f1750ccd70da0e389e2` (deps: httpx 0.28.1, rich 15.0.0, psutil 7.2.2) |

### `modal run modal_app.py --step prepare` (Task 16.2)
App `ap-Z0ghRfR1nS4ptJu8fFAUej`.

| Output | Value |
|---|---|
| ShareGPT user texts | 92,779 |
| M1 prompt sets | 6 × 512, each exactly 1024 tokens (gpt-oss chat frame = 67 tokens) |
| NLL prompts | 64 × 512 tokens |
| `split_special_tokens` honoured by the real tokenizer | yes (prep would have aborted otherwise) |
| `m1_prompts.json` sha256 | `b3e402d75c85f8a036486cd224260964bf0fbad839a6ea7745c4ae77a84186ec` |
| `nll_prompts.json` sha256 | `d1baa193a88c98a654bd44ed36e2c0f9d6db6d006b1f9b320a9519912631a797` |

The prompts embed the template's current date (2026-10-04). Re-running `prepare` without
`--force` keeps these exact files, and the runner refuses to resume a run if their hashes
change.

## 2026-10-04: NVFP4 checkpoint

### `modal run modal_app.py --step quantize` (Task 17.1)
App `ap-TN6wftytRv7tyPUdXog2bU`. ModelOpt 0.47.0, `NVFP4_DEFAULT_CFG` with non-expert quantizers
disabled.
- Enabled quantizers: exactly `model.layers.{0..23}.mlp.experts.{gate_up,down}_proj_{weight,input}_quantizer`
  (W4A4 on the experts only). The in-run experts-only assertion passed.
- Calibration: 256 ShareGPT prompts, 32,522 tokens (max 1024 each, seed 3).
- The full provenance is in `fp4bench_quant.json` inside the checkpoint directory.

### First `--step check` (Task 17.2)
App `ap-NCmR5Yvk4sJQ3Kk98PHjVu`. Bytes and quantization were correct, but config parity flagged
4 keys: `rope_scaling` + `rope_theta` (MX) vs `rope_parameters` (NV), and `bos_token_id` absent vs
`null`. The values were identical; transformers 5.17 serializes them differently.

### `modal run modal_app.py --step align-config` (pre-data fix, EXPERIMENT.md §11)
App `ap-JUjluKD1zEC3W5zpuXSCIy`. NV `config.json` = MX `config.json` + NV `quantization_config`.
Replaced keys: `bos_token_id`, `dtype`, `rope_parameters`, `rope_scaling`, `rope_theta`,
`transformers_version`. No weights changed.

### Second `--step check`
App `ap-rq4cuhBdAvvIeX1pQeo6NH`. **G2 and G5a pass.** Report: `docs/artifacts/checkpoint_report.json`.

| | Value |
|---|---|
| MX expert bytes | 10,152,345,600 |
| NV expert bytes | 10,749,542,784 |
| NV / MX | 1.0588236 (format-implied 4.5/4.25 = 1.0588235) |
| quant algo | MX `MXFP4`, NV `NVFP4` |
| NV activation scales | present (W4A4) |
| config diff | none |
| `nv_problems` | none |

## 2026-10-04/05: Revision 2 (dense Qwen3-32B)

EXPERIMENT.md was revised before any Revision 2 data existed (see §11). Steps:

| Step | App | Result |
|---|---|---|
| `env` (first attempt) | n/a | App creation refused: Modal requires `ephemeral_disk` of 512 GiB to 3 TiB, and we had asked for 200 GiB. Fixed (`eac29fd`), with a test pinning the bounds |
| `env` | `ap-xoYBBcJQdRX8CmSlBW7sZh` | `problems: []`. Same pins as before; the quantization image (llmcompressor 0.14.0) kept torch 2.13.0 |
| `prepare --force` | `ap-Jg69f0lB7RZrlR2UauBIAx` | Qwen3-32B @ `9216db5` downloaded. M1 prompts: 6 × 512 × 1024 tokens (Qwen3 chat frame = 8 tokens). `m1_prompts` sha256 `9930b747…a8f4220d`, `nll_prompts` sha256 `dc20a835…8d8ffd55c` |
| `quantize --fmt mxfp4` | `ap-7oJcCMgtZo2AARuSX1Hp7s` | All 448 Linear layers W4A4 MXFP4; `lm_head` BF16. Computed the BF16 reference NLL: **1.7730 nats/token** (64 windows) |
| `quantize --fmt nvfp4` | `ap-IGjy3HvSdDPHB6ErCk7uIi` | All 448 Linear layers W4A4 NVFP4 (calibrated input global scale); `config.json` parity with MX: no differing keys |
| `check` | `ap-qBJrJsUmRJGqWjHQKcJ2I4` | **G2 and G5a pass.** MX 16,577,986,560 B, NV 17,553,164,032 B, NV/MX = 1.0588236 (format-implied 1.0588235). The NV bytes match `RedHatAI/Qwen3-32B-NVFP4` exactly. `config_diff: {}`, no problems |
| `fetch --kind nvidia` | `ap-vq0UuGIAmdKRqhHHXhgkro` | `nvidia/Qwen3-32B-NVFP4` @ `16426c6` mirrored for the smoke cross-check |
| `prepare` (no `--force`) | `ap-ZQ4ZCEeGH5uOPi1UGwzxei` | Re-ran the fixed `split_special_tokens` probe against the real Qwen3 tokenizer: passed, prompts unchanged (same hashes) |

### Smoke `smoke-r2-1`: interrupted, incomplete (not a result)
App `ap-8Sc9Jq2W7ysdNUtwCJbLJd`.
- **Completed sessions:** MX, NV, NVx, NVc and NVt, all on one GPU.
  - G1: each logged exactly its expected kernel class.
  - Autotune was fresh in every session.
  - Local-staging fingerprints matched.
- **Where it stopped:** NVd finished C=1, then the run was cancelled mid-C=32. NVv never ran.
- **Cause: not preemption.**
  - The container log says `Received a cancellation signal while processing input` at 01:44:19 UTC.
  - `pmset` shows the controlling Mac entered **clamshell sleep at 21:40:59 EDT**, on battery.
  - The local `modal run` client held the in-flight `experiment.remote()` call. When the sleeping client dropped, the input was cancelled, even with `--detach`.
- **Fix (`a2f2c09`):** the experiment step now uses `experiment.spawn()`, so the run lives on Modal independently of the client.
- **Disposition:** the smoke is re-run in full as `smoke-r2-2` (same rule, same treatments). The pre-registered NV-alt rule needs the complete scan from one container. The partial numbers below are kept for the record only.
- **Partial M1 data (not a result: one round, no A/A):**
  - NV/MX ratio: 0.934 / 0.953 / 1.001 at C = 1 / 32 / 128.
  - Scan, ratio to NV at C=32: NVc 1.130, NVt 1.329. NVd at C=1 was 1.181.
  - NVx/NV: 0.996 / 1.003 / 1.002.
  - NLL: BF16 1.773, MX 1.909, NV 1.824, NVc and NVt 1.824, NVx 1.825.

### Smoke `smoke-r2-2`: complete, all gates pass (G3 not applicable)
App `ap-wdDuRQyWcxHEkCNHXfYrqe`, call `fc-01M44WPNKW14K6ABSH8MH73Q59` (spawned), 3201 s, one
B200 (`GPU-0f642246…`, driver 580.95.05), `start_id` `f68761d2…`, manifest `problems: []`.
- **Gates:** G1, G2, G4, G5a, G5b, G6a, G6b and G7 pass. G3 fails by construction: the smoke
  has no MX′.
- **G1:** every session logged exactly its expected kernel class, scan treatments included.
- **NLL (nats/token):** MX 1.9092, NV 1.8238, NVx 1.8248, every scan kernel 1.8238. BF16
  reference 1.7730.
- **Kernel scan, median M1 step in ms (ratio to NV):**

  | C | NV (CuTe-DSL) | NVc (FI CUTLASS) | NVt (FI TRT-LLM) | NVd (FI cuDNN) | NVv (vLLM CUTLASS) |
  |---|---|---|---|---|---|
  | 1 | 5.824 | 7.260 (1.247) | 8.566 (1.471) | 6.894 (1.184) | 8.272 (1.420) |
  | 32 | 8.058 | 9.370 (1.163) | 10.330 (1.282) | 8.877 (1.102) | 11.481 (1.425) |
  | 128 | 15.145 | 15.553 (1.027) | 16.127 (1.065) | 15.467 (1.021) | 17.469 (1.153) |

- **NV-alt selection (§7.7 rule):** all four candidates eligible. **NVd** (`flashinfer_cudnn`,
  `FlashInferCudnnNvFp4LinearKernel`) leads NVc by 5.55% at C=32, so it is not a tie.
  `TREATMENT_SERVER_ARGS["NVa"]` is now `("--linear-backend", "flashinfer_cudnn")`.
- **NVx cross-check:** NVx/NV = 0.9954 / 1.0044 / 0.9997 at C = 1 / 32 / 128. NVIDIA's
  checkpoint and ours decode at the same speed under the same kernel and BF16 KV.
- **MX vs NV (one round, not a result):**
  - M1 step: MX 6.226 / 8.441 / 15.072 ms, NV 5.824 / 8.058 / 15.145 ms. NV/MX = 0.935 / 0.955 /
    1.005 at C = 1 / 32 / 128. Within-cell spread ≤ 0.53%.
  - M2: MX 157 / 3822 / 9565 tok/s, NV 168 / 3966 / 10163 tok/s; all cells valid.
  - The bytes model predicts R_ideal 1.052 / 1.030 / 1.013 if decode were bandwidth-bound. The
    direction is reversed at low C, so kernel efficiency, not bytes, dominates there.

## 2026-10-05: published checkpoints, fetched back, checked

| Step | App | Result |
|---|---|---|
| `publish --kind mx` (private) | `ap-wecHSHlNeUWBaRsdE6wFdY` | `ggamecrazy/Qwen3-32B-MXFP4-W4A4` @ `9a719cda64d8748830a2da6c40417374c1e2f8a6` |
| `publish --kind nv` (private) | `ap-rhx34xUYo7hXVtOVYmW6He` | `ggamecrazy/Qwen3-32B-NVFP4-W4A4` @ `6b9b0a43384970072017e0c1d200fcc521bf4561` |
| `fetch --kind mx` | `ap-Ex1k36OVdS2m5jurf9L0Tw` | Volume dir is now an exact mirror of the published MX revision |
| `fetch --kind nv` | `ap-SU3OLmHymoZ4HZevltwIx0` | Same for NV (the 17 GB download took 15 min) |
| `check` | `ap-98kcertAyK4mf5yfuubjLA` | **G2 and G5a pass on the published revisions:** MX 16,577,986,560 B, NV 17,553,164,032 B, NV/MX = 1.0588236; formats `mxfp4-pack-quantized` / `nvfp4-pack-quantized`; NV activation scales present; `config_diff: {}`; no problems |

Release re-check (§6), 2026-10-05: vLLM `v0.31.0` is still the newest tag (GitHub tags; the
Docker Hub tags newer than it are nightlies), FlashInfer `v0.7.0.post1` is still the newest
stable release (`v0.7.1rc2` is a pre-release). The pinned stack is unchanged.

### Smoke `smoke-nf-1`: cancelled at launch (not a result)
App `ap-lXHY2xJ5YFA7YPUubXBtED`. Launched without `--detach`: the ephemeral app stopped when the
local entrypoint returned and cancelled the spawned call before it did anything. `modal_app.py`
now refuses `--step experiment` without `--detach` (`229eefe`).

### Smoke `smoke-nf-2`: fusion check passes (not a result: one round, no A/A)
App `ap-AxdKPOh4fQNemSrJJygKOD`, code `63fd680`, serving the published revisions, ~31 min.
- **Gates:** G1, G2, G4, G5a, G5b, G6a, G6b and G7 pass; G3 fails by construction (no MX′).
- **NV-nf works as intended:** `FlashInferCuteDslNvFp4LinearKernel`, resolved `'fuse_act_quant': False`,
  no "Enabled custom fusions" line; NV resolved `True` with `act_quant`, MX `False`. NV-nf was
  compiled fresh ("Compiling a graph for compile range (1, 16384) takes 28.06 s") under its own
  compile-cache key `torch_aot_compile/482cf30f…`, distinct from NV's `6bbb8ec6…`, so it did not
  load NV's fused graph. (The automated compile-cache check in G1 reads "unverified" for this run
  because the field was added after it was collected; checked by hand from the logs.)
- **NLL:** MX 1.9092, NV 1.8238, NV-nf 1.8238 (the fusion does not change the numerics measurably).
- **Preemptions:** 0 in every M1 block and M2 cell. KV pools 555k–560k tokens (BF16).
- **M1 step (ms), one round:**

  | C | MX | NV | NV-nf | R = NV/MX | F = NVnf/MX | Φ = NV/NVnf |
  |---|---|---|---|---|---|---|
  | 1 | 6.333 | 5.807 | 5.859 | 0.917 | 0.925 | 0.991 |
  | 32 | 8.423 | 7.999 | 8.120 | 0.950 | 0.964 | 0.985 |
  | 128 | 15.050 | 15.213 | 15.261 | 1.011 | 1.014 | 0.997 |

  The fusion accounts for ~1–1.5% of NV's advantage at C ≤ 32; most of it remains with the fusion
  matched (F).

## 2026-10-05: full run and Experiment B launched (in parallel, separate B200s)

| Run | App | Code | Start | Notes |
|---|---|---|---|---|
| `full-1` (`--mode full`) | `ap-so9uAjkmide4CyWjr3a9Jq` | `8bbbbb7` (+ untracked WIP `fp4bench/profiling.py`, not imported by the runner) | 00:16 EDT | 5 treatments × 5 rounds; manifest `problems: []`; both checkpoints staged locally and verified |
| `expb-1` (`--mode expb`) | `ap-wzQG32qR53Jzvquby2qvex` | same | 00:17 EDT | MX, NV, MX′ × 5 rounds, FP8 KV, 0.95; manifest `problems: []` |

Experiment B was pre-registered to run after the full run (§13). It runs at the same time on a
different B200 instead, to finish overnight. They are independent experiments with their own A/A
controls and their own GPUs; nothing is shared but the read-only checkpoint volume (§11).

The microbenchmark (§14) ran earlier the same night on two separate B200 containers
(`ap-eH2fLLIikgWnNBZbcxKQDZ` primary, `ap-7cBuXvk9eprPASl4bAsG3q` replicate; ~26 GPU-min; code
`8bbbbb7`). Results in `results/microbench-1/`.

## 2026-10-05: profiling sessions (§15) and microbenchmark (§14)

- **Microbenchmark** (`8bbbbb7`): primary `ap-eH2fLLIikgWnNBZbcxKQDZ`, replicate
  `ap-7cBuXvk9eprPASl4bAsG3q`; full pre-registered grid, 0 failed cells; results
  `results/microbench-1/` (+ `replicate/`). Ran from a scratch Modal app calling
  `fp4bench.microbench.run()`; `--step microbench` was added to `modal_app.py` afterwards.
- **Profiling** (`7ce7ba7`): debug `ap-Jc1OEHPTfbxHtMJW3ezPzU` (MX C=1/32, NV C=1), full
  `ap-OxVE0hKep5HrqRxM8fjrWV` (8 sessions); ~47 B200-min; results `results/profile-1/`. The torch
  profiler sees the kernels inside replayed CUDA graphs (32 graph launches per session, constant
  kernel count), so nsys was not needed. All 8 windows are pure decode (vLLM step annotations
  `execute_context_0(0)_generation_C(C)`). Kernel categories were set from the debug traces and are
  recorded in `profiles.json` (§15 allows this). Addition to the procedure: two unprofiled
  reference waves before each profiled wave (diagnostic only; M1 remains the measurement).

## 2026-10-05: results

### `expb-1` (Experiment B): complete
App `ap-wzQG32qR53Jzvquby2qvex`, 15 sessions, 00:17–04:45 EDT, one B200, no failed session.
All blocking gates pass; G6b fails (22/45 M2 cells `underfilled`). R_B = 1.0006 / 1.0065 / 0.9987
at C = 128 / 256 / 512 (all Equivalent; A/A 1.0002 / 0.9986 / 0.9986): **H_B,eq validated**.
FP8 KV pools 1,190,176–1,193,760 tokens, 0 preemptions. Analysis: `python -m analysis.analyze
results/expb-1`. Details in RESULTS.md.

### `full-1` (main run), rounds 0–4: complete; G3 failed → pre-registered extension started
App `ap-so9uAjkmide4CyWjr3a9Jq`, 25 sessions, 00:16–06:18 EDT, one B200
(`GPU-6a97cfb2…`), no failed session. Gates: G1, G2, G4, G5a, G5b, G6a, G7 pass; **G3 fails**
(A/A at C = 1: 0.9861 [0.9681, 1.0046], CI past −2%; at C = 32: 1.0051 [1.0017, 1.0084], excludes 1);
G6b fails (42/125 M2 cells `underfilled`). R = 0.9285 / 0.9304 / 0.9542 / 0.9819 / 1.0088 at
C = 1 / 8 / 32 / 64 / 128. Local snapshot: `results/full-1-5rounds-snapshot/`.

**Extension to 10 rounds** (pre-registered remedy for a failed G3, §8/§9): app
`ap-b7MXxaqd9RpbLxPK23EVfS`, started 06:20 EDT with `--rounds 10`, code `d633c51` (the runner's
measurement path is unchanged since `8bbbbb7`; the differences are preemption-read retries, the
detach guard and analysis). The restart passed the protocol and input identity checks
(manifest line 2: `rounds 10`, `problems: []`).

### `full-1` extension to 10 rounds: complete; G3 still fails
App `ap-b7MXxaqd9RpbLxPK23EVfS`, 06:20–12:54 EDT. Rounds 5–8 on `GPU-a54e4eb9…` (start
`d3ad2425`). At 11:33 EDT (15:33 UTC) the container received a KeyboardInterrupt during round 9's
NV-alt session (a Modal interruption; `errors.jsonl`); Modal restarted the function, the restart
passed the protocol/input identity checks (start `9bb05f3b`, `GPU-85218b45…`, `problems: []`),
and round 9 was re-run whole (analysis keeps the latest complete round; G7 passes). 53 servers
rows, 50 counted. Gates: G1, G2, G4, G5a, G5b, G6a, G7 pass; **G3 fails** at C = 32 (1.0032
[1.0009, 1.0055]) and C = 64 (0.9956 [0.9916, 0.9997]), both on "CI contains 1" only; C = 1 now
passes (0.9918 [0.9820, 1.0016]). G6b: 74/250 M2 cells invalid. R = 0.9339 / 0.9297 / 0.9596 /
0.9869 / 1.0118 at C = 1 / 8 / 32 / 64 / 128. `python -m analysis.analyze results/full-1`;
Experiment B's FP8-vs-BF16 NLL: `python -m analysis.analyze results/expb-1 --reference-run
results/full-1` (+0.0039 / +0.0038 nats/token).

## 2026-10-05: Experiment C (§16)

- `prepare-c` (`ap-IF6cOIshcrCalse6ufvNOE`): `/data/c_prompts.json`, 9 cells, 3,141,888 prompt tokens,
  sha256 `6439900e…`; the (1, 1,024) cell reuses the main run's M1 prompts (`9930b747…`).
- **Smoke `smoke-c-1`** (`ap-yIZJWXAywCYfue2v5NTzmG`, code `bcb4386`, 1 round, MX and NV, 4 cells):
  the `max_position_embeddings` override and `--max-model-len 131072` work (NLL unchanged: MX 1.9092,
  NV 1.8238; KV pools ~558–560k tokens); all gates pass except G3 (no MX′, by construction);
  config reproduction R(1, 1,024) = 0.9343 (main run 0.934). Within-session step SD 0.002 ms at
  (1, 127,360), 0.04 ms at (128, 360). One round, not a result: Δ = 0.414 / 0.405 / 0.401 ms at
  P = 1k / 32k / 127k (batch 1) and −0.071 ms at (128, 360).
- **Full run `expc-1`** (`ap-HKYrMP98LZjierxdxbQbjg`, code `ac9d71f`, 19:21–00:02 EDT, 15 sessions,
  no errors): all gates pass (G3 = A/A on effects, per the pre-data amendment). **Answer: batch
  (GEMM size) drives the gap.** E_tok = −0.031 ms [−0.040, −0.023] (no effect); E_batch = −0.556 ms
  [−0.640, −0.472] (shrinks the saving); A/A effects −0.003 and +0.020 ms. Config reproduction
  R(1, 1,024) = 0.9354. `python -m analysis.analyze_c results/expc-1`.

## 2026-10-08: end-to-end smoke of the refactored code (`smoke-nf-v2`)

App `ap-ugW8YZ3HSmrdoejPRpJsXs`, launched with `fp4bench run smoke-nf --run-id smoke-nf-v2`
(code `d884bbe`, clean tree), 12:13–12:41 EDT. The first run of the new CLI, Modal executor and
package layout. The manifest records `study: smoke-nf`, `code_commit: d884bbe…`, `code_dirty:
false`, schema 2, `problems: []`. Gates as `smoke-nf-2`: all pass except G3 (no MX′, by
construction).

Compared with `smoke-nf-2` (old code): server argv identical for MX, NV and NV-nf; same kernels and
fusion state; NLL identical (MX 1.9092, NV/NV-nf 1.8238); same KV pools. M1 rows add the intended
`schema_version` and `prompt_len`; M2 rows are otherwise unchanged. R = 0.936 / 0.947 / 1.008 and
F = 0.943 / 0.960 / 1.011 at C = 1 / 32 / 128 (smoke-nf-2: R 0.917 / 0.950 / 1.011), within the
known MXFP4 per-start autotune spread at C = 1 (main run 10-round R(1) = 0.934).
