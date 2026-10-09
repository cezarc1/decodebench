# Raw data

Compact raw data of the 2026-10 runs (per-start manifests, per-session server facts, M1 and M2
rows). Server logs, telemetry CSVs, profiler traces and figures are not included. Produced by the
pre-release code, whose analysis the current code reproduces exactly (`tests/test_golden.py`);
details in `docs/run-log.md`, results in `RESULTS.md`.

| Run | What | EXPERIMENT.md |
|---|---|---|
| `runs/full-1` | Main experiment, 10 rounds (5 + the pre-registered extension), MX / NV / NV-alt / MX′ / NV-nf, batch 1–128, BF16 KV | §3–§9 |
| `runs/expb-1` | Experiment B: MX / NV / MX′, batch 128–512, FP8 KV, 5 rounds | §13 |
| `runs/expc-1` | Experiment C: batch size vs tokens, 9 (batch, prompt) cells, 5 rounds | §16 |
| `runs/smoke-r2-2` | Revision 2 smoke and NVFP4 kernel scan (selected NV-alt = cuDNN) | §7.7 |
| `runs/smoke-nf-2` | Fusion smoke (MX / NV / NV-nf) | §7.7 |
| `runs/smoke-c-1` | Experiment C smoke | §16 |
| `runs/checkpoint_report.json` | G2/G5a checkpoint report of the published checkpoints | §9 |
| `kernels/microbench-1` | Kernel microbenchmark (primary and replicate) | §14 |
| `kernels/profile-1` | Profiling sessions (kernel time per decode step) | §15 |

Analyse a run with the CLI (`fp4bench analyze data/runs/<run>`); the analysis writes its outputs
into the run directory, so work on a copy.
