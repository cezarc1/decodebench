"""A flag's value in a server command line."""

from collections.abc import Sequence


def flag_value(argv: Sequence[str], flag: str) -> str | None:
    """The value of `flag` written "--flag value" or "--flag=value", the last occurrence winning
    as in vLLM; None if absent, and ValueError for a `flag` with nothing after it. vLLM's other
    spellings (dotted "--compilation-config.x", "-O", underscores) are not read."""
    value = None
    for i, arg in enumerate(argv):
        if arg == flag:
            if i + 1 == len(argv):
                raise ValueError(f"{flag} has no value: it is the last of {list(argv)!r}")
            value = argv[i + 1]
        elif arg.startswith(flag + "="):
            value = arg[len(flag) + 1 :]
    return value
