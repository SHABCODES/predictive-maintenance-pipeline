import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def evaluate_against_baseline(model, X_train, y_train, X_test, y_test, out_dir="results"):
    """
    Compare the trained model against a majority-class baseline.

    The failure class is a minority class (roughly 1 in 7 rows in this
    dataset). A classifier that always predicts "no failure" would already
    score a high raw accuracy without learning anything useful, so accuracy
    alone is a misleading headline metric here. This function reports the
    baseline's accuracy alongside the model's, and uses F1 / ROC-AUC on the
    failure class as the metrics that actually reflect model skill.
    """
    os.makedirs(out_dir, exist_ok=True)

    baseline = DummyClassifier(strategy="most_frequent", random_state=42)
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_acc = baseline.score(X_test, y_test)
    baseline_f1 = f1_score(y_test, baseline_pred, zero_division=0)

    model_pred = model.predict(X_test)
    model_proba = model.predict_proba(X_test)[:, 1]
    model_acc = model.score(X_test, y_test)
    model_f1 = f1_score(y_test, model_pred)
    model_auc = roc_auc_score(y_test, model_proba)

    summary = {
        "baseline_majority_class_accuracy": round(baseline_acc, 4),
        "baseline_majority_class_f1": round(baseline_f1, 4),
        "model_accuracy": round(model_acc, 4),
        "model_f1_failure_class": round(model_f1, 4),
        "model_roc_auc": round(model_auc, 4),
        "failure_class_prevalence": round(float(y_test.mean()), 4),
    }

    logging.info(f"Baseline vs. model: {summary}")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_test, model_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Failure", "Failure"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Random Forest")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, model_proba, name="Random Forest")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.title("ROC Curve - Failure Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close()

    return summary


def plot_feature_importance(model, feature_names, out_dir="results", top_n=15):
    os.makedirs(out_dir, exist_ok=True)

    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(idx)), importances[idx], color="#2E5266")
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances - Random Forest")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close()

    ranked = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:top_n]
    return ranked
