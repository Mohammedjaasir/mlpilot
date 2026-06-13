import numpy as np
import matplotlib.pyplot as plt


def explain_model(model, X_test, y_test, model_name, feature_names=None):
    print(f"\n📊 mlbuddy: Explaining your '{model_name}' model...\n")

    if hasattr(model, "feature_importances_"):
        _plot_feature_importance(model.feature_importances_, model_name, feature_names)

    elif hasattr(model, "coef_"):
        _plot_coefficients(model.coef_[0], model_name, feature_names)

    else:
        print("  ℹ This model doesn't expose internals directly.")
        print("  → Try model='logistic' or model='random_forest'.\n")


def _plot_feature_importance(importances, model_name, feature_names=None):
    n = len(importances)
    if feature_names is None:
        features = [f"Feature {i+1}" for i in range(n)]
    else:
        features = feature_names
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(8, 4))
    plt.bar(range(n), importances[indices], color="steelblue", alpha=0.8)
    plt.xticks(range(n), [features[i] for i in indices], rotation=45, ha="right")
    plt.title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.show()
    print("  ✓ Saved as 'feature_importance.png'")
    print("  → Features on the left matter most to your model.\n")


def _plot_coefficients(coefs, model_name, feature_names=None):
    n = len(coefs)
    if feature_names is None:
        features = [f"Feature {i+1}" for i in range(n)]
    else:
        features = feature_names
    colors = ["crimson" if c < 0 else "steelblue" for c in coefs]

    plt.figure(figsize=(8, 4))
    plt.bar(range(n), coefs, color=colors, alpha=0.8)
    plt.xticks(range(n), features, rotation=45, ha="right")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title(f"Model Coefficients — {model_name}", fontsize=13, fontweight="bold")
    plt.ylabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig("coefficients.png", dpi=120)
    plt.show()
    print("  ✓ Saved as 'coefficients.png'")
    print("  → Blue = pushes toward class 1. Red = pushes toward class 0.\n")