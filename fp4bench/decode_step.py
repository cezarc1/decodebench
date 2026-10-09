# METHODOLOGY.md#m1
from collections.abc import Callable

from fp4bench.core.types import PromptSet, WaveOrder

NAN = float("nan")

Wave = Callable[[PromptSet, int], float]


def step_time(
    t_short: float, t_long: float, n_short: int, n_long: int, strict: bool = True
) -> float:
    """Seconds per decode step; a non-positive window raises, or is NaN when not `strict`."""
    if n_long <= n_short:
        raise ValueError("n_long must exceed n_short")
    window = t_long - t_short
    if window <= 0:
        if strict:
            raise ValueError(f"non-positive decode window: {window:.6f}s")
        return NAN
    return window / (n_long - n_short)


def decode_block(
    wave: Wave, prompt_sets: list[PromptSet], c: int, n1: int, n2: int, reps: int
) -> list[dict]:
    """One warmup pair (set 0), then `reps` measured pairs, alternating which length runs first."""
    rows = []
    for s in range(reps + 1):
        prompts = prompt_sets[s][:c]
        if len(prompts) != c:
            raise ValueError(f"prompt set {s} has fewer than {c} prompts")
        warmup, n1_first = s == 0, s % 2 == 0
        if n1_first:
            t1 = wave(prompts, n1)
            t2 = wave(prompts, n2)
        else:
            t2 = wave(prompts, n2)
            t1 = wave(prompts, n1)
        step = step_time(t1, t2, n1, n2, strict=not warmup)
        rows.append(
            {
                "c": c,
                "set": s,
                "warmup": warmup,
                "first": WaveOrder.N1 if n1_first else WaveOrder.N2,
                "n1": n1,
                "n2": n2,
                "t1_s": t1,
                "t2_s": t2,
                "step_s": step,
                "decode_tok_s": c / step,
            }
        )
    return rows
