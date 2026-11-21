
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

print("\n========== CHARGEMENT DES DONNEES ==========\n")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv")).squeeze()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

print("Training :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing :", X_test.shape)


# ==========================================
# 3. CONSTRUCTION DU MODELE
# ==========================================

print("\n========== ENTRAINEMENT BASELINE ==========\n")

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    solver="lbfgs"
)

model.fit(X_train, y_train)

print("Modèle entraîné avec succès !")


# ==========================================
# 4. OPTIMISATION DU SEUIL SUR VALIDATION
# ==========================================

print("\n========== OPTIMISATION DU SEUIL ==========\n")

# Prédire les probabilités sur Validation
y_val_proba = model.predict_proba(X_val)[:, 1]

# Tester différents seuils
thresholds = np.arange(0.05, 0.96, 0.01)

best_threshold = 0.5
best_f1 = 0

threshold_results = []

for threshold in thresholds:

    y_val_pred = (y_val_proba >= threshold).astype(int)

    precision_val = precision_score(
        y_val, y_val_pred, zero_division=0
    )

    recall_val = recall_score(
        y_val, y_val_pred, zero_division=0
    )

    f1_val = f1_score(
        y_val, y_val_pred, zero_division=0
    )

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_val,
        "Recall": recall_val,
        "F1-score": f1_val
    })

    if f1_val > best_f1:
        best_f1 = f1_val
        best_threshold = threshold

print(f"Meilleur seuil : {best_threshold:.2f}")
print(f"Meilleur F1 Validation : {best_f1:.4f}")

# Sauvegarder les résultats des seuils
threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    os.path.join(RESULTS_DIR, "baseline_threshold_analysis.csv"),
    index=False
)

# Visualiser l'évolution du F1-score
plt.figure(figsize=(9, 5))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1-score"],
    label="F1-score",
    linewidth=2
)

plt.axvline(
    best_threshold,
    color="red",
    linestyle="--",
    label=f"Best threshold = {best_threshold:.2f}"
)

plt.xlabel("Classification Threshold")
plt.ylabel("F1-score")
plt.title("Threshold Optimization - Validation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "baseline_threshold_optimization.png"),
    dpi=300
)

plt.close()


# ==========================================
# 5. PREDICTIONS FINALES SUR TEST
# ==========================================

print("\n========== PREDICTIONS SUR TEST ==========\n")

# Probabilités de fraude sur Test
y_proba = model.predict_proba(X_test)[:, 1]

# Utilisation du seuil optimal trouvé sur Validation
y_pred = (y_proba >= best_threshold).astype(int)

print("Seuil utilisé :", round(best_threshold, 2))


# ==========================================
# 5. EVALUATION
# ==========================================

print("\n========== EVALUATION DU MODELE ==========\n")

precision = precision_score(y_test, y_pred, zero_division=0)

recall = recall_score(y_test, y_pred, zero_division=0)

f1 = f1_score(y_test, y_pred, zero_division=0)

pr_auc = average_precision_score(y_test, y_proba)

roc_auc = roc_auc_score(y_test, y_proba)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ==========================================
# 6. RAPPORT DE CLASSIFICATION
# ==========================================

print("\n========== CLASSIFICATION REPORT ==========\n")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Normal", "Fraud"],
    zero_division=0
))


# ==========================================
# 7. MATRICE DE CONFUSION
# ==========================================

print("\n========== CONFUSION MATRIX ==========\n")

cm = confusion_matrix(y_test, y_pred)

print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Fraud"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Logistic Regression - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "baseline_confusion_matrix.png"),
    dpi=300
)

plt.close()


# ==========================================
# 8. COURBE PRECISION-RECALL
# ==========================================

fig, ax = plt.subplots(figsize=(8, 6))

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_proba,
    ax=ax
)

plt.title("Baseline - Precision Recall Curve")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "baseline_precision_recall.png"),
    dpi=300
)

plt.close()


# ==========================================
# 9. SAUVEGARDE DU MODELE
# ==========================================

joblib.dump(
    model,
    os.path.join(BASE_DIR, "models", "baseline_model.pkl")
)


# ==========================================
# 10. SAUVEGARDE DES RESULTATS
# ==========================================

results = pd.DataFrame([{
    "Model": "Logistic Regression",
    "Threshold": best_threshold,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "PR-AUC": pr_auc,
    "ROC-AUC": roc_auc
}])

results.to_csv(
    os.path.join(RESULTS_DIR, "baseline_metrics.csv"),
    index=False
)

print("\nModèle sauvegardé !")
print("Graphiques sauvegardés dans :", FIGURES_DIR)

print("\n========== BASELINE TERMINEE ==========\n")
