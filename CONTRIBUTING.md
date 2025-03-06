# Contributing to DocMind

Thank you for your interest in improving DocMind.

## Getting Started

1. Fork the repository and clone your fork.
2. Install in development mode:

```bash
pip install -e ".[dev]"
```

3. Verify everything works:

```bash
make check
```

## Development Workflow

1. Create a branch from `main`:
   ```bash
   git checkout -b feature/your-feature
   ```
2. Write code with tests.
3. Ensure lint and tests pass: `make check`
4. Commit with a clear, descriptive message.
5. Push and open a pull request.

## Commit Conventions

- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation changes
- `test:` test additions or fixes
- `refactor:` code restructuring
- `chore:` maintenance tasks

## Running Tests

```bash
make test           # full suite with coverage report
make test-fast      # stop on first failure
make lint           # ruff linter
make format         # auto-fix style issues
```

## Code Style

- **Linter:** ruff (configured in `pyproject.toml`)
- **Line length:** 79 characters
- **Type hints:** Required on all public functions
- **Docstrings:** Required on all public classes and functions

## Pull Request Process

1. Fork the repository.
2. Create a feature branch from `main`.
3. Write tests for your changes.
4. Ensure `make check` passes.
5. Submit a PR with a clear description.
6. Address review feedback promptly.

## Reporting Issues

Open an issue with: a clear title, steps to reproduce, expected vs
actual behaviour, and your Python version and OS.
