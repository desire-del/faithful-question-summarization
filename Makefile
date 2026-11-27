.PHONY: install install-train lint format test check

install:          ## Core + dev tools (what CI uses)
	uv sync

install-train:    ## Add GPU training stack (torch, transformers, trl, peft)
	uv sync --extra train

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

check: lint test
