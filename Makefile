.PHONY: install dev test lint format check clean docker run

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

run:
	python -m docmind

test:
	pytest

test-verbose:
	pytest -v --tb=long

test-fast:
	pytest -x -q

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

check: lint test

clean:
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

docker-build:
	docker build -t docmind .

docker-run:
	docker run --name docmind-app -p 7860:7860 docmind
