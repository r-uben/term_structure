# Term Structure Analysis

This project implements term structure models for analyzing yield curves and macroeconomic factors, focusing on the comparison between ACM (Adrian, Crump, Moench) and FF (Favero-Fuertes) models.

## Project Structure

```
term_structure/
├── data/                    # Data directory
│   ├── raw/                # Raw data files
│   │   └── Q/             # Quarterly data
│   │       ├── yield_curve.csv
│   │       └── macro_data.csv
│   ├── processed/         # Processed data files
│   │   ├── term_premia.csv
│   │   ├── term_premia_acm.csv
│   │   └── term_premia_ff.csv
│   └── figures/           # Generated figures
│       ├── acm/          # ACM model figures
│       │   ├── model_parameters/
│       │   ├── returns/
│       │   └── yield_curves/
│       ├── ff/           # FF model figures
│       │   ├── model_parameters/
│       │   ├── returns/
│       │   └── yield_curves/
│       └── model_results_{model}.{png,svg}
├── src/                   # Source code
│   └── model/            # Model implementations
│       ├── data/         # Data handling
│       ├── figures/      # Visualization components
│       │   ├── base.py
│       │   ├── model_parameters.py
│       │   ├── returns.py
│       │   └── yield_curves.py
│       ├── time/         # Time-related utilities
│       ├── common_trend.py
│       ├── estimation.py
│       ├── forecast.py
│       ├── params.py
│       ├── pricing_factors.py
│       └── trend.py
└── mains/                # Main execution scripts
    ├── create_figures.py
    ├── get_term_premia.py
    ├── main.py
    └── out_of_sample_comparisons.py
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

The project includes several main scripts that can be run using Poetry:

1. Create Model Figures:
```bash
poetry run create_figures
```
This generates both individual and combined plots for ACM and FF models in the `data/figures` directory.

2. Generate Term Premia:
```bash
poetry run get_term_premia
```

3. Run Main Analysis:
```bash
poetry run main
```

4. Run Out-of-Sample Comparisons:
```bash
poetry run out_of_sample_comparisons
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

## Models

The project implements and compares two term structure models:

1. **ACM Model** (Adrian, Crump, Moench):
   - Standard affine term structure model
   - Uses principal components as pricing factors

2. **FF Model** (Favero-Fuertes):
   - Data-congruent term structure model
   - Incorporates macroeconomic factors
   - Features a common trend component

## License

**PRIVATE SOFTWARE**

This is private software. All rights reserved. No part of this software may be reproduced, distributed, or modified without explicit permission from the authors:

- Carlo A. Favero (carlo.favero@unibocconi.it)
- Rubén Fernández-Fuertes (ruben.fernandez@phd.unibocconi.it)

Any modifications, distributions, or use of this software require prior written consent from the authors. 