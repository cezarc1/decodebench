# decodebench - LLM decode speed on real serving stacks


## Goal

Benchmark realistic production-like workloads on non GEMM-bound tasks.

#### NVFP4 vs MXFP4 TL;DR — _Last updated October 2026. See accompanying [blog post](https://cezarcocu.com/blog/nvfp4-vs-mxfp4-decode-bench/)._

If you're using a B200 GPU all of the evidence points to always preferring NVFP4 over MXFP4, by a non-significant margin, on
both GEMM-bound workloads as well as low-batch decode workloads. Additionally, quantization from bf16 to the respective formats indicate that nvfp4 should additionally yield non-trivial superior eval perf. 
However, you should always strive to [benchmark your actual production configuration](https://x.com/StasBekman/status/2107221020197429422?s=20).

- **Kernel implementations dominate perf differences.** On a B200 (SM100), with vLLM v0.31, the
  NVFP4/MXFP4 decode perf difference mostly comes down to kernel implementation and not from the
  4.5 vs 4.25 bits difference. _Note that we tested this on only one dense model (Qwen3-32B, w/ bf16 KV)._
- **@ small-batch decode: NVFP4 is meaningfully faster.** +7.1%, +7.6% and +4.2% decode throughput at
  batch 1, 8 and 32. Worth highlighting here that decode for this workload is neither bandwidth-bound (~40–50% of theoretical
  HBM bandwidth) nor compute-bound (the GEMMs are tiny), so kernel implementation differences dominate.
  NVFP4's superior perf here seems to come mostly from the kernels around the GEMMs (activation
  quantization, fewer kernels per step, etc).
- **@ Batch ≥ 64: there is no meaningful difference**, so you might as well pick the format based on eval
  perf, portability or other concerns if you're in this regime.
- **Batch size dominates the perf difference, not HBM memory bandwidth.** NVFP4's non-GEMM kernels are
  faster, but it's GEMMs fall behind MXFP4's as they grow in size, and from batch 64 up the two differences cancel each other
  out.
- **Large GEMMs (prefill/training) depend heavily on the kernel implementation.** In PyTorch, NVFP4 gets
  ~9% more TFLOPS ([ml-engineering book](https://github.com/stas00/ml-engineering/blob/master/training/dtype.md#fp4-formats)),
  but there NVFP4 runs on cuBLASLt and MXFP4 on a different library (MSLK). In vLLM's kernels (CuTe-DSL),
  NVFP4's largest GEMMs were from 4% to 20% slower @ batch 512.
- **Serve NVFP4 on vLLM's default kernel**, CuTe-DSL (`--linear-backend flashinfer_cutedsl`). On
  cuDNN (`flashinfer_cudnn`) the same weights are between 10 to 18% slower and lose to MXFP4 @ every batch
  size. MXFP4 only has a single W4A4 kernel in vLLM 0.31 (FlashInfer CuTe-DSL, aka `mm_fp4`).
- **NVFP4 is more accurate:** its NLL penalty post-quantization from BF16 is 2.7× smaller. While no evals were run, the lower NLL (negative log-likelihood) suggests
  that NVFP4 should yield better eval results.

![NVFP4 vs MXFP4 decode throughput on B200, by batch size and by kernel](docs/figures/nvfp4_vs_mxfp4_decode_b200.png)


![NVFP4 vs MXFP4: context length against batch size](docs/figures/nvfp4_vs_mxfp4_batch_vs_tokens_b200.png)

## Results

vLLM v0.31.0 · 1× NVIDIA B200 SXM (SM100) on Modal · Qwen3-32B (dense, W4A4) ·

| Date | vLLM | FlashInfer | GPU | Model | NVFP4 decode throughput vs MXFP4 | Details |
|---|---|---|---|---|---|---|
| 2026-10 | 0.31.0 | 0.7.0.post1 | 1× B200 | Qwen3-32B | +7.1 / +7.5 / +4.2% at batch 1 / 8 / 32; within ±2% at 64–512 | [RESULTS.md](RESULTS.md) |

## What we found on vLLM v0.31.0 

- **The kernels dominate this perf difference.** Most of NVFP4's lead comes from
  MXFP4's slower activation quantization, extra small kernels and idle gaps between kernels.
  vLLM's NVFP4-only fused SiLU + quantization adds about 1%.
- **The lead depends on batch size.** At batch 1, NVFP4 saves ~0.4 ms per step at any context
  from 1k to 127k tokens. By batch 128 the lead is gone as theoretically things get memory bandwidth bound.
- **At large batch the formats tie.** MAMF runs showed NVFP4 ~9% faster on large GEMMs;
  in decode at batch 256–512 the two run at the same speed. Again, we can infer that this is due to memory bandwidth saturation.

## When GEMMs are the bottleneck

Large matrix multiplies (prefill, training, decode at large batch) are generally considered compute-bound:

| Large GEMMs on B200 | NVFP4 vs MXFP4 | Source |
|---|---|---|
| PyTorch 2.14, MAMF (best shape found) | ~9% more TFLOPS (6,624 vs 6,087; 73.6% vs 67.6% of peak) | [ml-engineering](https://github.com/stas00/ml-engineering/blob/master/training/dtype.md#fp4-formats) |
| PyTorch 2.13 `_scaled_mm`, M = 512 | 17% / 26% faster on gate_up / qkv; mixed on down | this |
| vLLM v0.31.0 (FlashInfer CuTe-DSL `mm_fp4`), M = 512 | 4–20% slower on gate_up and down; 2–3% faster on qkv | this |
| FlashInfer cuDNN, M = 512 | 10–30% faster on every shape | this |
| End-to-end decode, batch 256 / 512 | within 0.7% | this |

In PyTorch 2.13 the two formats run in different libraries (NVFP4 in cuBLASLt, MXFP4 in CUTLASS),
so that row compares libraries as much as formats. Our rows are medians that held on two B200s
(they differ by up to ~22% between containers); details in [RESULTS.md](RESULTS.md).

## How it measures

- **Checkpoints.** [MXFP4](https://huggingface.co/ggamecrazy/Qwen3-32B-MXFP4-W4A4) and
  [NVFP4](https://huggingface.co/ggamecrazy/Qwen3-32B-NVFP4-W4A4) are W4A4 quantizations of the
  same BF16 [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B) (llm-compressor 0.14.0,
  every Linear except `lm_head`), served with identical vLLM settings except the checkpoint and
  the kernel. The runs used revisions
  [`9a719cd`](https://huggingface.co/ggamecrazy/Qwen3-32B-MXFP4-W4A4/tree/9a719cda64d8748830a2da6c40417374c1e2f8a6)
  and [`6b9b0a4`](https://huggingface.co/ggamecrazy/Qwen3-32B-NVFP4-W4A4/tree/6b9b0a43384970072017e0c1d200fcc521bf4561).
- **Instrument M1, the decode step.** Synchronized waves of exactly C requests with 128 and 1,152
  output tokens: (T(1152) − T(128)) / 1024 is one decode step at batch C, with prefill and HTTP
  cancelled out. M2 is llm-inference-bench's Sustained Decode, the number the community
  publishes.
- **Design.** Treatments rotate in a Latin square over 5–10 rounds, on one GPU per run. Gates
  check the kernel each server ran, checkpoint bytes, A/A noise, throttling, accuracy,
  completeness and that every session used the same GPU.
- **Pre-registered.** Hypotheses, the ±2% equivalence margin and the decision rules were fixed in
  [EXPERIMENT.md](EXPERIMENT.md) before any data was collected.

## Run it

Requirements: [uv](https://docs.astral.sh/uv/), a Modal account with B200 access, and a Modal
secret `huggingface-secret`.

```bash
uv sync                                    # .venv on Python 3.12 with the dev tools
uv run modal setup
uv run modal secret create huggingface-secret HF_TOKEN=<read-only token>   # or UNUSED=1
```

Then, with `uv run fp4bench …` (`fp4bench <command> --help` documents every option):

```bash
fp4bench env                    # GPU, driver, pinned versions in the image; expect "problems": []
fp4bench prepare                # BF16 model, ShareGPT and prompt files on the fp4-data volume
fp4bench prepare --expc         # only for expc / smoke-c: the per-cell prompts
fp4bench quantize --fmt mxfp4   # also computes the BF16 reference NLL every run needs
fp4bench fetch --kind mx --repo-id ggamecrazy/Qwen3-32B-MXFP4-W4A4 \
    --revision 9a719cda64d8748830a2da6c40417374c1e2f8a6
fp4bench fetch --kind nv --repo-id ggamecrazy/Qwen3-32B-NVFP4-W4A4 \
    --revision 6b9b0a43384970072017e0c1d200fcc521bf4561
fp4bench check                  # checkpoint report (G2, G5a) -> results/checkpoint_report.json
fp4bench run full --run-id full-2   # spawns a detached Modal app and returns at once
fp4bench download full-2        # copies the run directory to results/full-2
fp4bench analyze results/full-2 # verdict, gates, summary.md and figures, written into the run dir
```

- **`quantize` before `fetch`:** the first `quantize` writes the BF16 reference NLL that every run
  needs; the fetches then replace the checkpoint with the published revision, which `full`,
  `expb` and `expc` require.
- **The `smoke` study** also serves NVIDIA's checkpoint: `fp4bench fetch --kind nvidia --repo-id
  nvidia/Qwen3-32B-NVFP4 --revision 16426c6eb87be9e27c14cc9fb318f9c7a5f8588c`.
- **Resume** by re-running the same `run` command; only incomplete rounds run again. `--rounds 10`
  applies a study's registered extension, `--rerun-rounds 2,4` re-runs complete rounds, and a
  restart refuses changed inputs or protocol ([METHODOLOGY.md#run-integrity](METHODOLOGY.md#run-integrity)).
- **Kernel studies:** `fp4bench microbench --run-id <id>` and `fp4bench profile --run-id <id>`
  (EXPERIMENT.md §14–§15). **`--executor local`** runs the same runner on a local B200.

## Re-run it on a new vLLM version or config

1. Bump the image digest and version pins in `fp4bench/settings.py`.
2. Re-check the version-specific facts in [METHODOLOGY.md](METHODOLOGY.md) (kernel classes, log
   lines, fusion) and the kernel classes gate G1 expects in `fp4bench/studies/model.py`.
3. Run `smoke` and `smoke-nf`, then `full` under a new run id, analyse it, and add a row to
   [Results by stack](#results-by-stack).

New kernels, checkpoints or studies: [docs/extending.md](docs/extending.md).

## Re-analyse the data (no GPU required)

`data/runs/` holds the raw rows of the published runs ([data/README.md](data/README.md)).
`analyze` writes into the run directory, so work on a copy:

```bash
cp -R data/runs /tmp/fp4bench-runs
fp4bench analyze /tmp/fp4bench-runs/full-1
fp4bench analyze /tmp/fp4bench-runs/expb-1 --reference-run /tmp/fp4bench-runs/full-1
fp4bench analyze /tmp/fp4bench-runs/expc-1
fp4bench plot share data/runs/full-1 data/runs/expb-1 --out-dir /tmp/fp4bench-figures
```

`full-1` reports its G3 and G6b failures, as RESULTS.md describes; the smokes report "incomplete"
because one round has no CI.

## Reproducibility

These results should be reproducible. If not, please file a bug or PR request.

```bash
make check            # what CI runs: ruff format --check, ruff, pyright, ty's ratchet, pytest
```
