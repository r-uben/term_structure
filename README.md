# Term Structure Analysis

This project implements term structure models for analyzing yield curves and macroeconomic factors.

## Project Structure

```
term_structure/
├── data/               # Data directory
│   ├── raw/           # Raw data files
│   └── processed/     # Processed data files
├── src/               # Source code
│   ├── core/         # Core functionality
│   │   ├── models/   # Model implementations
│   │   └── utils/    # Utility functions
│   ├── data/         # Data handling
│   ├── analysis/     # Analysis tools
│   └── visualization/ # Visualization tools
├── tests/            # Unit tests
├── notebooks/        # Jupyter notebooks for analysis
├── scripts/          # Scripts for data processing
├── config/          # Configuration files
└── docs/            # Documentation
```

## Setup

1. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```

3. Activate the virtual environment:
```bash
poetry shell
```

4. Prepare data:
- Place yield curve data in `data/raw/Q/yield_curve.csv`
- Place macro data in `data/raw/Q/macro_data.csv`

## Usage

1. Data Processing:
```bash
poetry run python scripts/process_data.py
```

2. Run Analysis:
```bash
poetry run python scripts/run_analysis.py
```

## Development

The project uses several development tools, all configured in `pyproject.toml`:

- `black` for code formatting:
```bash
poetry run black .
```

- `flake8` for linting:
```bash
poetry run flake8 src tests
```

- `mypy` for type checking:
```bash
poetry run mypy src
```

- `pytest` for testing:
```bash
poetry run pytest
```

## License

MIT License 