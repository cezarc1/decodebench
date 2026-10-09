# `make check` is what CI runs (.github/workflows/ci.yml): its five steps, in CI's order, stopping
# at the first that fails.
.NOTPARALLEL:
.PHONY: check lint test fmt

check: lint test

lint:
	uv run ruff format --check
	uv run ruff check
	uv run pyright
	uv run python -m tests.ty_baseline

test:
	uv run pytest -q -n auto # xdist's default --dist load: one file's cases (the mutations) spread out

fmt:
	uv run ruff format
	uv run ruff check --fix
