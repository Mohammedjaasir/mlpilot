# 🤖 MLBuddy

[![PyPI version](https://img.shields.io/pypi/v/mlbuddy-learn.svg?color=blue)](https://pypi.org/project/mlbuddy-learn/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Support](https://img.shields.io/badge/python-3.8%2B-blue)](https://pypi.org/project/mlbuddy-learn/)

> **Your friendly machine learning companion.** Build models, understand what they learned, and receive step-by-step guidance to improve them — all with just a few lines of code.

`MLBuddy` is designed to bridge the gap between complex machine learning workflows and beginners. Instead of treating models as a "black box," `MLBuddy` actively guides you through the process, explains your model's decisions visually, and suggests actionable next steps to boost performance.

---

## ✨ Key Features

*   **⚡ Automated Model Comparison (`mlbuddy.compare`)**: Compare multiple ML architectures (Logistic Regression, Decision Trees, Random Forests, SVMs) in one command to find the absolute best match for your dataset.
*   **📊 Transparent Explanations (`model.explain()`)**: Instantly generate feature importance bar charts or coefficients to see exactly which features drive your model's predictions.
*   **💬 Actionable Feedback suggestions (`model.suggest()`)**: Receive direct, intelligent hints on how to improve your model based on its performance metrics (e.g., handling overfitting, engineering features, or tuning hyperparameters).
*   **📦 Zero-Config Data Prep (`mlbuddy.load_csv`)**: Automatically handles data scaling and splitting with intelligent defaults.

---

## 🚀 Quickstart

Get up and running in less than 60 seconds.

### 1. Installation

```bash
pip install mlbuddy-learn
```

### 2. The Golden Workflow

Here is how simple it is to load, compare, train, explain, and improve a machine learning model:

```python
import mlbuddy as ml

# 1. Load and prepare your dataset
X_train, X_test, y_train, y_test = ml.load_csv("your_data.csv", target_column="hired")

# 2. Compare all models to find the winner
results = ml.compare(X_train, X_test, y_train, y_test)

# 3. Train the best performing model
model = ml.train(X_train, X_test, y_train, y_test, model="random_forest")

# 4. Peer into the black box (generates a feature importance plot!)
model.explain()

# 5. Get tailored recommendations to improve accuracy
model.suggest()
```

---

## 🛠️ Project Architecture

```
mlbuddy/
├── auto/
│   ├── data.py         # Scaling, train/test splitting, and CSV preprocessing
│   └── trainer.py      # Guided Model wrapper, training routines, and model comparison
├── explain/
│   └── visualizer.py   # Automatic feature importance & coefficient plotting
└── guide/
    └── suggest.py      # Rule-based suggestions engine tailored to accuracy thresholds
```

---

## 💡 How it Guides You

When you call `model.suggest()`, `MLBuddy` analyzes your model's accuracy and provides contextual advice:

*   **Low Accuracy (< 60%)**: Recommends looking into data quality, checking labels, or gathering more samples.
*   **Moderate Accuracy (60% - 80%)**: Suggests trying more robust architectures (like Random Forests) and hyperparameter tuning.
*   **High Accuracy (> 90%)**: Flags potential overfitting risks, duplicates in the dataset, and guides you on preparing for production.

---

## 💻 Development Setup

Clone the repository and set up a local development environment:

```bash
# Clone the repository
git clone https://github.com/Mohammedjaasir/mlbuddy-learn.git
cd mlbuddy-learn

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
