"""Command-line arguments read as vLLM's argument parser reads them."""

from collections.abc import Sequence


def flag_value(argv: Sequence[str], flag: str) -> str | None:
    """The value of the last `flag` ("--flag value" or "--flag=value"), None if absent; a `flag`
    with no value after it raises ValueError."""
    value = None
    for i, arg in enumerate(argv):
        if arg == flag:
            if i + 1 == len(argv):
                raise ValueError(f"{flag} has no value: it is the last of {list(argv)!r}")
            value = argv[i + 1]
        elif arg.startswith(flag + "="):
            value = arg[len(flag) + 1 :]
    return value
