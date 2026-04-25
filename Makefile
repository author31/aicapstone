.PHONY: install install-dev test

install:
	uv sync

install-dev:
	uv sync --extra dev

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --extra dev pytest tests/test_repo_layout.py
