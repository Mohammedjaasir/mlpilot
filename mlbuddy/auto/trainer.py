from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from ..guide.suggest import suggest


MODELS = {
    "logistic":      LogisticRegression(max_iter=1000),
    "tree":          DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(),
    "svm":           SVC(),
}


class GuidedModel:
    def __init__(self, model, X_test, y_test, model_name, feature_names=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.feature_names = feature_names

        y_pred = model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.y_pred = y_pred

    def predict(self, X):
        return self.model.predict(X)

    def explain(self):
        from ..explain.visualizer import explain_model
        explain_model(self.model, self.X_test, self.y_test, self.model_name, feature_names=self.feature_names)

    def suggest(self):
        suggest(self)


def train(X_train, X_test, y_train, y_test, model="logistic"):
    if model not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(
            f"\n✗ Unknown model '{model}'.\n"
            f"  → Available models: {available}"
        )

    print(f"\n🤖 mlbuddy: Training '{model}' model...\n")

    clf = MODELS[model]
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"  ✓ Training complete!")
    print(f"  📊 Accuracy: {acc * 100:.1f}%")
    print()
    print(classification_report(y_test, y_pred))
    print("💡 Tip: Call model.explain() to see what the model learned.")
    print("💡 Tip: Call model.suggest() to get advice on improving it.\n")

    from .data import _FEATURE_NAMES
    feature_names = None
    if _FEATURE_NAMES and len(_FEATURE_NAMES) == X_train.shape[1]:
        feature_names = _FEATURE_NAMES

    return GuidedModel(clf, X_test, y_test, model, feature_names=feature_names)


def compare(X_train, X_test, y_train, y_test):
    """Try all available models and show which one performs best.
    
    This is the quickest way to find the best model for your data.
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        
    Returns:
        Dictionary with model names and their accuracies
    """
    print("\n🔬 mlbuddy: Comparing all models...\n")

    results = {}
    for name, clf in MODELS.items():
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        results[name] = acc
        print(f"  {name:15s} → {acc * 100:6.1f}% accuracy")

    print()
    best_model = max(results, key=results.get)
    best_acc = results[best_model]
    
    print(f"  🏆 Best: {best_model} with {best_acc * 100:.1f}% accuracy\n")
    print(f"💡 Tip: Use ml.train(..., model='{best_model}') to train the winner.\n")
    
    return results