"""The smoke's NVFP4 kernel scan that chooses NVa's kernel, and the NVx cross-check."""

from dataclasses import dataclass

from fp4bench.core.types import CheckpointKind, Treatment
from fp4bench.studies import model


@dataclass(frozen=True)
class KernelScanSpec:
    """The NVa selection rule (METHODOLOGY.md#nv-alt)."""

    treatments: tuple[Treatment, ...]
    reference: Treatment
    selection_c: int
    tie_margin: float
    tiebreak_c: int
    max_nll_diff: float

    def __post_init__(self) -> None:
        if not self.treatments or len(set(self.treatments)) != len(self.treatments):
            raise ValueError(f"a kernel scan needs distinct treatments, got {self.treatments}")
        if self.reference in self.treatments:
            raise ValueError(f"the kernel scan's reference {self.reference} is not scanned")
        kernel = {t: model.TREATMENTS[t].linear_kernel for t in (self.reference, *self.treatments)}
        if len(set(kernel.values())) != len(kernel):
            raise ValueError(
                "the kernel scan's treatments do not each serve a kernel of their "
                "own: " + ", ".join(f"{t} {k}" for t, k in kernel.items())
            )
        if not (self.tie_margin > 0 and self.max_nll_diff > 0):
            raise ValueError("the kernel scan's tie margin and NLL bound must be positive")


@dataclass(frozen=True)
class Crosscheck:
    """NVIDIA's checkpoint against ours on the same kernel; gates nothing."""

    treatment: Treatment
    reference: Treatment

    def __post_init__(self) -> None:
        theirs, ours = model.TREATMENTS[self.treatment], model.TREATMENTS[self.reference]
        if theirs.checkpoint is not CheckpointKind.NVX or ours.checkpoint is CheckpointKind.NVX:
            raise ValueError(
                f"the cross-check puts NVIDIA's checkpoint against ours, not "
                f"{self.treatment} against {self.reference}"
            )
        if (theirs.linear_kernel, theirs.act_quant_fusion) != (
            ours.linear_kernel,
            ours.act_quant_fusion,
        ):
            raise ValueError(
                f"the cross-check needs {self.treatment} and {self.reference} on "
                f"the same kernel and fusion"
            )


NVA_SCAN = KernelScanSpec(
    treatments=(Treatment.NVC, Treatment.NVT, Treatment.NVD, Treatment.NVV),
    reference=Treatment.NV,
    selection_c=32,
    tie_margin=0.01,
    tiebreak_c=128,
    max_nll_diff=0.01,
)
NVX_CROSSCHECK = Crosscheck(Treatment.NVX, Treatment.NV)
