"""Experiment C (EXPERIMENT.md §16): does batch size or the number of KV tokens drive the gap?"""

import json

from fp4bench import settings
from fp4bench.core.types import Arm, Cell, ContrastName, Hypothesis, RatioName, Treatment
from fp4bench.studies.base import (
    Contrast,
    G3Rule,
    Prompts,
    Ratio,
    ServerSettings,
    Study,
    VerdictRule,
)

T = Treatment
TOKEN_ARM = tuple(Cell(1, p) for p in (1024, 4096, 16384, 32768, 65536, 127360))
BATCH_ARM = (Cell(1, 127360), Cell(8, 15360), Cell(32, 3360), Cell(128, 360))
REGISTERED_CELLS = TOKEN_ARM + tuple(c for c in BATCH_ARM if c not in TOKEN_ARM)
E_TOK = (Cell(1, 127360), Cell(1, 1024))
E_BATCH = (Cell(128, 360), Cell(1, 127360))
DELTA_MARGIN_MS = 0.1
AA_EFFECT_MARGIN_MS = 0.25
HF_OVERRIDES = {"max_position_embeddings": 131072}
SERVER = ServerSettings(max_model_len=131072, hf_overrides=json.dumps(HF_OVERRIDES))
ROPE_SCALING_MARKERS = ("rope_scaling", "rope_parameters", "yarn")


def _study(
    name: str,
    treatments: tuple[Treatment, ...],
    cells: tuple[Cell, ...],
    rounds: int,
    m1_reps: int,
    extension_rounds: int | None,
    smoke: bool,
) -> Study:
    return Study(
        name=name,
        treatments=treatments,
        cells=cells,
        server=SERVER,
        rounds=rounds,
        extension_rounds=extension_rounds,
        m1_reps=m1_reps,
        m2_duration_s=0,
        primary_batches=(1,),
        prompts=Prompts.CELLS,
        ratios=(Ratio(RatioName.R, T.NV, T.MX), Ratio(RatioName.AA, T.MXP, T.MX)),
        contrasts=(
            Contrast(ContrastName.E_TOK, E_TOK, Arm.TOKEN),
            Contrast(ContrastName.E_BATCH, E_BATCH, Arm.BATCH),
        ),
        effect_margin_ms=DELTA_MARGIN_MS,
        aa_effect_margin_ms=AA_EFFECT_MARGIN_MS,
        g3=G3Rule.AA_EFFECTS,
        verdict=VerdictRule.BATCH_VS_TOKENS,
        smoke=smoke,
    )


EXPC = _study(
    "expc",
    (T.MX, T.NV, T.MXP),
    REGISTERED_CELLS,
    rounds=5,
    m1_reps=5,
    extension_rounds=10,
    smoke=False,
)
SMOKE_C = _study(
    "smoke-c",
    (T.MX, T.NV),
    (Cell(1, 1024), Cell(1, 32768), Cell(1, 127360), Cell(128, 360)),
    rounds=1,
    m1_reps=3,
    extension_rounds=None,
    smoke=True,
)
EXPC_TREATMENTS = EXPC.treatments
C_CELLS = EXPC.cells

REPRO_CELL = Cell(1, 1024)
MAIN_RUN_R_BATCH1 = 0.934
REPRO_TOL = 0.01
# METHODOLOGY.md#descriptive-checks
MAIN_RUN_NLL = {T.MX: 1.9092, T.MXP: 1.9092, T.NV: 1.8238}
MAIN_RUN_NLL_PROMPTS_SHA256 = "dc20a83531303e1559a5eafe5d1f907cc77680363a3b6d05cd0438d8d8ffd55c"
MEAN_CONTEXT_EXTRA = (settings.M1_N1 + settings.M1_N2) // 2

# METHODOLOGY.md#expc
MAIN_RUN_DELTA_MS = {1: 0.42, 8: 0.49, 32: 0.32, 64: 0.14, 128: -0.19}
PREDICTED_DELTA_MS: dict[Hypothesis, dict[Cell, float]] = {
    Hypothesis.H_BATCH: {
        **dict.fromkeys(TOKEN_ARM, 0.42),
        Cell(8, 15360): 0.49,
        Cell(32, 3360): 0.32,
        Cell(128, 360): -0.19,
    },
    Hypothesis.H_TOKENS: {
        Cell(1, 1024): 0.42,
        Cell(1, 127360): 0.07,
        Cell(8, 15360): 0.07,
        Cell(32, 3360): 0.07,
        Cell(128, 360): 0.07,
    },
}
PREDICTED_R: dict[Hypothesis, dict[Cell, float | str]] = {
    Hypothesis.H_BATCH: {
        Cell(1, 1024): 0.93,
        Cell(1, 4096): 0.934,
        Cell(1, 16384): 0.94,
        Cell(1, 32768): 0.95,
        Cell(1, 65536): 0.96,
        Cell(1, 127360): 0.97,
        Cell(128, 360): "> 1",
    },
    Hypothesis.H_TOKENS: {Cell(1, 127360): 0.99},
}
PREDICTED_EFFECT_MS: dict[Hypothesis, dict[ContrastName, float]] = {
    Hypothesis.H_BATCH: {ContrastName.E_TOK: 0.0, ContrastName.E_BATCH: -0.6},
    Hypothesis.H_TOKENS: {ContrastName.E_TOK: -0.35, ContrastName.E_BATCH: 0.0},
}
