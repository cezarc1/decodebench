"""The smokes before the main run: the kernel scan with NVIDIA's checkpoint, the fusion smoke."""

from fp4bench import settings
from fp4bench.core.types import Treatment
from fp4bench.studies.base import Study, cells_at
from fp4bench.studies.kernel_scan import NVA_SCAN, NVX_CROSSCHECK
from fp4bench.studies.main import RATIOS

T = Treatment
SMOKE_ONLY_TREATMENTS = (NVX_CROSSCHECK.treatment, *NVA_SCAN.treatments)
SMOKE = Study(
    name="smoke",
    treatments=(T.MX, T.NV, *SMOKE_ONLY_TREATMENTS),
    cells=cells_at(settings.M1_INPUT_LEN, (1, 32, 128)),
    rounds=1,
    m1_reps=3,
    m2_duration_s=10,
    require_published=False,
    ratios=RATIOS,
    kernel_scan=NVA_SCAN,
    crosscheck=NVX_CROSSCHECK,
    smoke=True,
)
SMOKE_NF = Study(
    name="smoke-nf",
    treatments=(T.MX, T.NV, T.NVNF),
    cells=cells_at(settings.M1_INPUT_LEN, (1, 32, 128)),
    rounds=1,
    m1_reps=3,
    m2_duration_s=10,
    ratios=RATIOS,
    smoke=True,
)
