"""Experiment C's cells against §16's design: arms, mean KV tokens and H_tokens' Δ."""

import numpy as np

from fp4bench import settings
from fp4bench.core.types import Arm, Cell
from fp4bench.studies.expc import BATCH_ARM, MAIN_RUN_DELTA_MS, TOKEN_ARM


def arms_of(cell: Cell) -> list[Arm]:
    return [name for name, arm in ((Arm.TOKEN, TOKEN_ARM), (Arm.BATCH, BATCH_ARM)) if cell in arm]


def kv_tokens(cell: Cell, extra: int) -> int:
    """Mean KV tokens of a cell during the measured window: C × (P + extra), §16's design with
    expc.MEAN_CONTEXT_EXTRA (640), a run's with its rows' N1 and N2."""
    return cell[0] * (cell[1] + extra)


def h_tokens_delta_ms(tokens: float) -> float:
    """H_tokens' Δ at `tokens` mean KV tokens: the main run's Δ curve against C × 1,664."""
    cs = sorted(MAIN_RUN_DELTA_MS)
    return float(
        np.interp(
            tokens, [c * settings.M1_MEAN_CONTEXT for c in cs], [MAIN_RUN_DELTA_MS[c] for c in cs]
        )
    )
