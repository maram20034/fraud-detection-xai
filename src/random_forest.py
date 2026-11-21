
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# =====================================================
# 1. CONFIGURATION
# =====================================================

RANDOM_STATE = 42

DATA_PATH = "data/processed"
MODEL_PATH = "models"
FIGURES_PATH = "results/figures"
RESULTS_PATH = "results/random_forest"

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

print("\n========== RANDOM FOREST : DETECTION DE FRAUDE ==========\n")


# =====================================================
# 2. CHARGEMENT DES DONNEES
# =====================================================

X_train = pd.read_csv(f"{DATA_PATH}/X_train.csv")
X_test = pd.read_csv(f"{DATA_PATH}/X_test.csv")

y_train = pd.read_csv(f"{DATA_PATH}/y_train.csv")
y_test = pd.read_csv(f"{DATA_PATH}/y_test.csv")


def prepare_labels(data):
    """Récupère la colonne des labels."""
    if "Class" in data.columns:
        return data["Class"].values

    return data.iloc[:, 0].values


y_train = prepare_labels(y_train)
y_test = prepare_labels(y_test)

print("Données chargées avec succès !")
print("X_train :", X_train.shape)
print("X_test :", X_test.shape)

print("\nDistribution des classes dans le train :")
print(pd.Series(y_train).value_counts())

print("\nDistribution des classes dans le test :")
print(pd.Series(y_test).value_counts())


# =====================================================
# 3. DIVISION TRAIN / VALIDATION
# =====================================================

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train
)

print("\n========== DIVISION DES DONNEES ==========")
print("Training :", X_fit.shape)
print("Validation :", X_val.shape)


# =====================================================
# 4. CREATION DU MODELE RANDOM FOREST
# =====================================================

print("\n========== ENTRAINEMENT RANDOM FOREST ==========")

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_model.fit(X_fit, y_fit)

print("Entraînement terminé avec succès !")


# =====================================================
# 5. PREDICTIONS SUR LA VALIDATION
# =====================================================

y_val_proba = rf_model.predict_proba(X_val)[:, 1]


# =====================================================
# 6. RECHERCHE DU MEILLEUR SEUIL
# =====================================================

precision, recall, thresholds = precision_recall_curve(
    y_val,
    y_val_proba
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

best_index = np.argmax(f1_scores)

best_threshold = float(thresholds[best_index])

print("\n========== SEUIL DE CLASSIFICATION ==========")
print("Seuil sélectionné :", best_threshold)
print("F1-score validation :", f1_scores[best_index])


# =====================================================
# 7. PREDICTIONS SUR LE TEST
# =====================================================

print("\n========== EVALUATION SUR LE TEST ==========")

y_test_proba = rf_model.predict_proba(X_test)[:, 1]

# Seuil classique
y_pred_05 = (y_test_proba >= 0.5).astype(int)

# Seuil sélectionné sur validation
y_pred_optimal = (
    y_test_proba >= best_threshold
).astype(int)


# =====================================================
# 8. FONCTION D'EVALUATION
# =====================================================

def evaluate_model(y_true, y_pred, y_proba, threshold_name):

    print(f"\n========== {threshold_name} ==========")

    print("\nMatrice de confusion :")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    print("\nRapport de classification :")
    print(
        classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0
        )
    )

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "PR-AUC": average_precision_score(
            y_true, y_proba
        )
    }

    print("\nMétriques :")
    for name, value in metrics.items():
        print(f"{name} : {value:.4f}")

    return cm, metrics


cm_05, metrics_05 = evaluate_model(
    y_test,
    y_pred_05,
    y_test_proba,
    "SEUIL 0.5"
)

cm_optimal, metrics_optimal = evaluate_model(
    y_test,
    y_pred_optimal,
    y_test_proba,
    f"SEUIL OPTIMAL {best_threshold:.4f}"
)


# =====================================================
# 9. SAUVEGARDE DES METRIQUES
# =====================================================

results = {
    "threshold_0.5": metrics_05,
    "threshold_optimal": metrics_optimal,
    "selected_threshold": best_threshold
}

with open(
    f"{RESULTS_PATH}/rf_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4)

with open(
    f"{MODEL_PATH}/random_forest_threshold.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        {"threshold": best_threshold},
        file,
        indent=4
    )


# =====================================================
# 10. MATRICES DE CONFUSION
# =====================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay(
    confusion_matrix=cm_05,
    display_labels=["Normal", "Fraude"]
).plot(ax=axes[0], cmap="Blues", colorbar=False)

axes[0].set_title("Random Forest - Seuil 0.5")

ConfusionMatrixDisplay(
    confusion_matrix=cm_optimal,
    display_labels=["Normal", "Fraude"]
).plot(ax=axes[1], cmap="Blues", colorbar=False)

axes[1].set_title("Random Forest - Seuil optimal")

plt.tight_layout()

plt.savefig(
    f"{FIGURES_PATH}/rf_confusion_matrices.png",
    dpi=300
)

plt.close()


# =====================================================
# 11. COURBE PRECISION-RECALL
# =====================================================

precision_curve, recall_curve, _ = precision_recall_curve(
    y_test,
    y_test_proba
)

pr_auc = average_precision_score(
    y_test,
    y_test_proba
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall_curve,
    precision_curve,
    label=f"Random Forest (AP = {pr_auc:.4f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Random Forest - Precision Recall Curve")
plt.legend()
plt.grid()

plt.savefig(
    f"{FIGURES_PATH}/rf_precision_recall.png",
    dpi=300
)

plt.close()


# =====================================================
# 12. IMPORTANCE DES VARIABLES
# =====================================================

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    f"{RESULTS_PATH}/rf_feature_importance.csv",
    index=False
)

print("\n========== TOP 10 VARIABLES IMPORTANTES ==========")
print(feature_importance.head(10))

plt.figure(figsize=(10, 7))

top_features = feature_importance.head(10).sort_values(
    by="Importance"
)

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.title("Random Forest - Top 10 Feature Importance")
plt.tight_layout()

plt.savefig(
    f"{FIGURES_PATH}/rf_feature_importance.png",
    dpi=300
)

plt.close()


# =====================================================
# 13. SAUVEGARDE DU MODELE
# =====================================================

joblib.dump(
    rf_model,
    f"{MODEL_PATH}/random_forest_model.joblib"
)

print("\n========== SAUVEGARDE ==========")
print("Modèle sauvegardé.")
print("Seuil sauvegardé.")
print("Métriques sauvegardées.")
print("Graphiques sauvegardés.")

print("\n========== RANDOM FOREST TERMINE ==========\n")
