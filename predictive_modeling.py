"""
Predictive Modeling Using Machine Learning
------------------------------------------
A supervised-learning classification project using the scikit-learn
Breast Cancer Wisconsin dataset.

Models:
1. Logistic Regression
2. Decision Tree
3. Random Forest

Evaluation:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curve and AUC
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# 1. Load the dataset
# -------------------------------------------------------------------
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Save a copy so the project has a visible dataset file.
dataset = X.copy()
dataset["target"] = y
dataset.to_csv(BASE_DIR / "breast_cancer_data.csv", index=False)

print(f"Dataset shape: {X.shape}")
print(f"Classes: {dict(zip(data.target_names, range(len(data.target_names))))}")

# -------------------------------------------------------------------
# 2. Train/test split
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# -------------------------------------------------------------------
# 3. Define models
# -------------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=42)),
        ]
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    ),
}

metrics_rows = []
roc_data = {}

# -------------------------------------------------------------------
# 4. Train, test and evaluate
# -------------------------------------------------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)

    metrics_rows.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "ROC AUC": auc,
        }
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)
    print(classification_report(
        y_test,
        predictions,
        target_names=data.target_names,
    ))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=data.target_names,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    fig.savefig(
        RESULTS_DIR / f"{safe_name}_confusion_matrix.png",
        dpi=150,
    )
    plt.close(fig)

    # ROC information
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_data[name] = (fpr, tpr, auc)

# -------------------------------------------------------------------
# 5. Compare model performance
# -------------------------------------------------------------------
metrics_df = pd.DataFrame(metrics_rows)
metrics_df = metrics_df.sort_values("ROC AUC", ascending=False)
metrics_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

print("\nModel comparison:")
print(metrics_df.to_string(index=False))

# Bar chart
fig, ax = plt.subplots(figsize=(9, 5))
metrics_plot = metrics_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]]
metrics_plot.plot(kind="bar", ax=ax)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
plt.close(fig)

# -------------------------------------------------------------------
# 6. ROC curves
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
plt.close(fig)

print(f"\nResults saved in: {RESULTS_DIR}")
