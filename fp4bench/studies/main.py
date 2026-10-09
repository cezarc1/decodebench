"""The main run (EXPERIMENT.md §7–§8): MXFP4 against NVFP4 at C = 1 … 128, BF16 KV."""

from fp4bench import settings
from fp4bench.core.types import RatioName, Treatment, VerdictLabels
from fp4bench.studies.base import Ratio, Study, cells_at

T = Treatment
RATIOS = (
    Ratio(RatioName.R, T.NV, T.MX),
    Ratio(RatioName.D, T.NVA, T.MX),
    Ratio(RatioName.K, T.NVA, T.NV, VerdictLabels.KERNEL),
    Ratio(RatioName.F, T.NVNF, T.MX),
    Ratio(RatioName.PHI, T.NV, T.NVNF, VerdictLabels.FUSION),
    Ratio(RatioName.AA, T.MXP, T.MX),
)

FULL = Study(
    name="full",
    treatments=(T.MX, T.NV, T.NVA, T.MXP, T.NVNF),
    cells=cells_at(settings.M1_INPUT_LEN, (1, 8, 32, 64, 128)),
    extension_rounds=10,
    ratios=RATIOS,
)
