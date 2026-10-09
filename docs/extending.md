# Extending fp4bench

## Adding a study

A study is a frozen `Study` dataclass (`fp4bench/studies/base.py`):

- **Design.** `name`; `treatments` (keys of `studies/model.TREATMENTS`); `cells`
  (`Cell(batch, prompt_len)`); `server` (`ServerSettings`: KV dtype, GPU memory utilization,
  `max_model_len`, `hf_overrides`); `rounds`, `extension_rounds`, `m1_reps`, `m2_duration_s`
  (0 = no M2); `require_published`; `prompts` (`Prompts.M1`, the 1,024-token prompts, or
  `Prompts.CELLS`, per-cell prompts from `prepare --expc`).
- **Analysis rules.** `primary_batches`; `ratios` (`Ratio(name, numer, denom)`); `verdict`
  (`H_EQ`, `H_B` or `BATCH_VS_TOKENS`); `g3` (`AA_RATIO` or `AA_EFFECTS`); `contrasts` with
  `effect_margin_ms` and `aa_effect_margin_ms` (batch-vs-tokens rule only); `kernel_scan`;
  `crosscheck`; `smoke`.

`Study.__post_init__` refuses inconsistent designs, such as unknown treatments, primary batches
the study doesn't measure, or contrasts without margins. To add one:

1. Write down its design, hypotheses and decision rule in EXPERIMENT.md before running it.
2. Define it in a module under `fp4bench/studies/`. `expb.py`, a variant of the main run in a few
   lines, is a good model.
3. Add it to `STUDIES` in `fp4bench/studies/registry.py`. At import, the registry refuses two
   studies that record the same protocol, because a manifest could not tell them apart.
4. For a new kernel, backend or checkpoint, add a `Treatment` in `fp4bench/core/types.py` and a
   `TreatmentSpec` in `fp4bench/studies/model.py`: its checkpoint, server args, the linear-kernel
   class gate G1 expects, and whether the fusion is on.
5. The identity test pins every registered study's argv, schedule and prompts in
   `tests/golden/identity.json`, which a refactor must not change. A new study, treatment or
   constant may only add to it: `python -m tests.golden.make_identity --check-additive` lists
   what the regenerated view adds and fails if it removes or changes anything. Then regenerate it
   with `python -m tests.golden.make_identity`, which refuses a view that is not additive too.

**What the analysis needs: facts from the manifest, rules from the study.** Each run start writes
a line to `manifests.jsonl`: the environment, the input hashes, the study's name and its protocol.
`fp4bench analyze` takes rounds, reps, cells and primary batches from the last manifest line, and
gates, verdict, ratios and contrasts from the registered study that line names. Once a study has
data, treat it as frozen and add a new study instead of editing it. A verdict outside the three
rules needs a new `VerdictRule` and code in `fp4bench/analysis/verdicts.py`.

## Hardware and engine

They are not study fields yet. The GPU is requested in `fp4bench/executors/modal.py` and checked
(`NVIDIA B200`, compute capability 10.0) in `fp4bench/manifest.py`. The image digest and package
pins live in `fp4bench/settings.py`, vLLM's log lines are parsed in `fp4bench/server.py`, and the
kernel classes G1 expects are in `fp4bench/studies/model.py`. Every version-specific fact in
[METHODOLOGY.md](../METHODOLOGY.md) is tied to vLLM v0.31.0.

## Golden PNGs across platforms

The figures pinned by sha256 in `tests/golden/outputs.json` and `tests/golden/mutations.json`, and
`docs/figures`, were drawn on the platform `tests/golden/pngs.json` records: macOS arm64 with
matplotlib 3.11.2. There they must be byte-identical. On any other platform, such as CI's
ubuntu-latest, they are compared by pixels within a tolerance (`tests/golden/pngs.py`). Every
other written file, `.md` and `.json`, is byte-exact everywhere.
