# MLBUDDY-LEARN

[![PyPI version](https://badge.fury.io/py/mlbuddy-learn.svg)](https://pypi.org/project/mlbuddy-learn/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Machine Learning automation and guidance system.

## Project Structure

- **mlpilot/auto/**: Automated machine learning tasks
  - `data.py`: Data handling and preprocessing
  - `trainer.py`: Model training utilities
  
- **mlpilot/guide/**: ML guidance and suggestions
  - `suggest.py`: Suggestions engine for ML workflows
  
- **mlpilot/explain/**: Model interpretation and visualization
  - `visualizer.py`: Model visualization tools
  
- **tests/**: Test suite

## Installation

```bash
pip install -e .
```

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```
