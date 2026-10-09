from fp4bench.core.types import Treatment


def round_order(r: int, treatments: tuple[Treatment, ...]) -> list[Treatment]:
    k = r % len(treatments)
    return list(treatments[k:] + treatments[:k])
