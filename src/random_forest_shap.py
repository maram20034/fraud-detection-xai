
# ============================================
# SHAP - RANDOM FOREST
# Explication du modèle Random Forest déjà entraîné
# Aucun réentraînement
# ============================================

import os
import json
import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================
# 1. CHEMINS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "random_forest_model.joblib"

X_TRAIN_PATH = BASE_DIR / "data" / "processed" / "X_train.csv"
X_TEST_PATH = BASE_DIR / "data" / "processed" / "X_test.csv"
Y_TEST_PATH = BASE_DIR / "data" / "processed" / "y_test.csv"

RESULTS_DIR = BASE_DIR / "results" / "shap"
FIGURES_DIR = BASE_DIR / "results" / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# 2. CHARGEMENT DU MODELE ET DES DONNEES
# ============================================

print("\n========== CHARGEMENT DU MODELE ==========")

model = joblib.load(MODEL_PATH)

X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]

y_test = y_test.astype(int)

print("Modèle Random Forest chargé.")
print("X_test :", X_test.shape)
print("Nombre de fraudes dans le test :", int((y_test == 1).sum()))


# ============================================
# 3. SELECTION DES TRANSACTIONS A EXPLIQUER
# ============================================

print("\n========== SELECTION DES TRANSACTIONS ==========")

# Même principe que pour l'analyse SHAP du MLP :
# 20 fraudes et 180 transactions normales.

fraud_indices = np.where(y_test.to_numpy() == 1)[0]
normal_indices = np.where(y_test.to_numpy() == 0)[0]

rng = np.random.RandomState(42)

n_fraud = min(20, len(fraud_indices))
n_normal = min(180, len(normal_indices))

selected_fraud = rng.choice(
    fraud_indices,
    size=n_fraud,
    replace=False
)

selected_normal = rng.choice(
    normal_indices,
    size=n_normal,
    replace=False
)

selected_indices = np.concatenate([
    selected_fraud,
    selected_normal
])

X_explain = X_test.iloc[selected_indices].copy()
y_explain = y_test.iloc[selected_indices].to_numpy()

print("Transactions sélectionnées :", len(X_explain))
print("Fraudes :", int((y_explain == 1).sum()))
print("Transactions normales :", int((y_explain == 0).sum()))


# ============================================
# 4. CREATION DE L'EXPLAINER SHAP
# ============================================

print("\n========== CREATION DE L'EXPLAINER SHAP ==========")

# Un échantillon des données d'entraînement sert
# de référence pour expliquer les prédictions.

background = X_train.sample(
    n=min(100, len(X_train)),
    random_state=42
)

explainer = shap.TreeExplainer(
    model,
    data=background,
    model_output="probability",
    feature_perturbation="interventional"
)

print("Explainer créé avec succès.")


# ============================================
# 5. CALCUL DES VALEURS SHAP
# ============================================

print("\n========== CALCUL DES VALEURS SHAP ==========")

raw_shap_values = explainer.shap_values(X_explain)

# Selon la version de SHAP, les résultats peuvent
# être retournés sous différentes formes.

if isinstance(raw_shap_values, list):
    # Cas fréquent : une matrice par classe
    shap_values = np.asarray(raw_shap_values[1])

else:
    shap_values = np.asarray(raw_shap_values)

    if shap_values.ndim == 3:

        # Forme : (transactions, variables, classes)
        if shap_values.shape[-1] == 2:
            shap_values = shap_values[:, :, 1]

        # Forme : (classes, transactions, variables)
        elif shap_values.shape[0] == 2:
            shap_values = shap_values[1]

    elif shap_values.ndim != 2:
        raise ValueError(
            f"Format SHAP non reconnu : {shap_values.shape}"
        )

# Valeur de base correspondant à la classe fraude (1)
base_value = np.asarray(explainer.expected_value)

if base_value.ndim > 0 and base_value.size > 1:
    base_value = float(base_value.reshape(-1)[1])
else:
    base_value = float(base_value.reshape(-1)[0])

print("Forme des valeurs SHAP :", shap_values.shape)
print("Valeur de base :", base_value)


# ============================================
# 6. PREDICTIONS DU MODELE
# ============================================

print("\n========== PREDICTIONS ==========")

fraud_probabilities = model.predict_proba(X_explain)[:, 1]

predictions = (fraud_probabilities >= 0.5).astype(int)


# ============================================
# 7. SAUVEGARDE DES VALEURS SHAP
# ============================================

print("\n========== SAUVEGARDE DES RESULTATS ==========")

shap_df = pd.DataFrame(
    shap_values,
    columns=X_explain.columns
)

shap_df["true_label"] = y_explain
shap_df["predicted_label"] = predictions
shap_df["fraud_probability"] = fraud_probabilities

shap_df.to_csv(
    RESULTS_DIR / "random_forest_shap_values.csv",
    index=False
)

# Importance globale : moyenne des valeurs absolues SHAP
feature_importance = pd.DataFrame({
    "feature": X_explain.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
})

feature_importance = feature_importance.sort_values(
    by="mean_abs_shap",
    ascending=False
)

feature_importance.to_csv(
    RESULTS_DIR / "random_forest_shap_importance.csv",
    index=False
)

print("\nVariables les plus importantes selon SHAP :")
print(feature_importance.head(10).to_string(index=False))


# ============================================
# 8. GRAPHIQUE SHAP BAR
# ============================================

print("\n========== GRAPHIQUE SHAP BAR ==========")

plt.figure()

shap.summary_plot(
    shap_values,
    X_explain,
    plot_type="bar",
    max_display=15,
    show=False
)

plt.title("Random Forest - SHAP Feature Importance")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "random_forest_shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================
# 9. GRAPHIQUE SHAP SUMMARY
# ============================================

print("\n========== GRAPHIQUE SHAP SUMMARY ==========")

plt.figure()

shap.summary_plot(
    shap_values,
    X_explain,
    max_display=15,
    show=False
)

plt.title("Random Forest - SHAP Summary Plot")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "random_forest_shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================
# 10. EXPLICATION D'UNE FRAUDE ET D'UNE TRANSACTION NORMALE
# ============================================

print("\n========== EXPLICATIONS INDIVIDUELLES ==========")

explanation = shap.Explanation(
    values=shap_values,
    base_values=np.full(len(X_explain), base_value),
    data=X_explain.to_numpy(),
    feature_names=X_explain.columns.tolist()
)

# Première transaction frauduleuse de notre échantillon
fraud_position = np.where(y_explain == 1)[0][0]

plt.figure()

shap.plots.waterfall(
    explanation[fraud_position],
    max_display=15,
    show=False
)

plt.title("Random Forest - Explication d'une fraude")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "random_forest_shap_waterfall_fraud.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Première transaction normale de notre échantillon
normal_position = np.where(y_explain == 0)[0][0]

plt.figure()

shap.plots.waterfall(
    explanation[normal_position],
    max_display=15,
    show=False
)

plt.title("Random Forest - Explication d'une transaction normale")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "random_forest_shap_waterfall_normal.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================
# 11. RESUME FINAL
# ============================================

print("\n========== TERMINE ==========")

print("Analyse SHAP Random Forest terminée.")
print("\nFichiers générés :")

print("- results/shap/random_forest_shap_values.csv")
print("- results/shap/random_forest_shap_importance.csv")

print("- results/figures/random_forest_shap_bar.png")
print("- results/figures/random_forest_shap_summary.png")
print("- results/figures/random_forest_shap_waterfall_fraud.png")
print("- results/figures/random_forest_shap_waterfall_normal.png")
