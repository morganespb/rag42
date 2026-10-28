PYTHON = uv run python

install:
	uv sync

run:
	$(PYTHON) -m src index --max_chunk_size 2000

debug:
	$(PYTHON) -m pdb -m src index --max_chunk_size 2000

clean:
	rm -rf __pycache__ src/__pycache__ .mypy_cache .pytest_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

.PHONY: install run debug clean lint lint-strict
