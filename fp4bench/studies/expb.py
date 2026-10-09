"""Experiment B (EXPERIMENT.md §13): high batch with an FP8 KV cache."""

from fp4bench import settings
from fp4bench.core.types import KvDtype, Treatment
from fp4bench.studies.base import ServerSettings, Study, VerdictRule, cells_at
from fp4bench.studies.main import RATIOS

EXPB = Study(
    name="expb",
    treatments=(Treatment.MX, Treatment.NV, Treatment.MXP),
    cells=cells_at(settings.M1_INPUT_LEN, (128, 256, 512)),
    server=ServerSettings(kv_dtype=KvDtype.FP8, gpu_memory_utilization=0.95),
    extension_rounds=10,
    primary_batches=(256, 512),
    ratios=RATIOS,
    verdict=VerdictRule.H_B,
)
EXPB_PREDICTION_C = 512
EXPB_PREDICTED_R = (0.973, 0.987)
