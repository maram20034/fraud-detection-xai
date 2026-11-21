
# ============================================
# SHAP - AUTOENCODER
# Explication de l'erreur de reconstruction
# Aucun réentraînement
# ============================================

import json
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.models import load_model


# ============================================
# 1. CHEMINS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "autoencoder.keras"
SCALER_PATH = BASE_DIR / "models" / "autoencoder_scaler.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "autoencoder_threshold.json"

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

autoencoder = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0].astype(int)

print("Autoencoder chargé avec succès.")
print("Scaler chargé avec succès.")
print("X_train :", X_train.shape)
print("X_test :", X_test.shape)


# ============================================
# 3. CHARGEMENT DU SEUIL D'ANOMALIE
# ============================================

print("\n========== CHARGEMENT DU SEUIL ==========")

with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
    threshold_data = json.load(file)

# Recherche du seuil dans le fichier JSON
if isinstance(threshold_data, dict):

    threshold = None

    for key, value in threshold_data.items():
        if "threshold" in key.lower() and isinstance(value, (int, float)):
            threshold = float(value)
            break

    if threshold is None:
        raise ValueError(
            "Impossible de trouver le seuil dans autoencoder_threshold.json. "
            "Vérifie son contenu."
        )

else:
    threshold = float(threshold_data)

print("Seuil d'anomalie :", threshold)


# ============================================
# 4. SELECTION DES TRANSACTIONS
# ============================================

print("\n========== SELECTION DES TRANSACTIONS ==========")

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
# 5. PREPARATION DES DONNEES
# ============================================

print("\n========== PREPARATION DES DONNEES ==========")

# Le scaler a été entraîné uniquement sur les
# transactions normales pendant l'entraînement.

X_train_normal = X_train[
    y_train_normal_mask
] if False else None

# Pour construire le background, on utilise ici
# un échantillon des données d'entraînement.
# Le scaler et l'Autoencoder restent inchangés.

background = X_train.sample(
    n=min(50, len(X_train)),
    random_state=42
)

# Transformation avec le scaler déjà entraîné
background_scaled = scaler.transform(background)
X_explain_scaled = scaler.transform(X_explain)


# ============================================
# 6. FONCTION DE CALCUL DE L'ERREUR
# ============================================

print("\n========== FONCTION DE RECONSTRUCTION ==========")

def reconstruction_error(data):

    # SHAP transmet les données sous forme de tableau
    data = np.asarray(data)

    # Normalisation avec le scaler existant
    data_scaled = scaler.transform(data)

    # Reconstruction par l'Autoencoder
    reconstructed = autoencoder.predict(
        data_scaled,
        verbose=0
    )

    # Calcul de l'erreur MSE pour chaque transaction
    errors = np.mean(
        np.square(data_scaled - reconstructed),
        axis=1
    )

    return errors


# ============================================
# 7. CALCUL DES SCORES D'ANOMALIE
# ============================================

print("\n========== CALCUL DES SCORES ==========")

anomaly_scores = reconstruction_error(X_explain)

anomaly_predictions = (
    anomaly_scores >= threshold
).astype(int)

print("Score moyen :", anomaly_scores.mean())
print("Anomalies détectées :", int(anomaly_predictions.sum()))


# ============================================
# 8. CREATION DE L'EXPLAINER SHAP
# ============================================

print("\n========== CREATION DE L'EXPLAINER SHAP ==========")

# KernelExplainer permet d'expliquer la sortie
# de notre fonction personnalisée : reconstruction_error.

explainer = shap.KernelExplainer(
    reconstruction_error,
    background,
    link="identity"
)

print("Explainer créé avec succès.")


# ============================================
# 9. CALCUL DES VALEURS SHAP
# ============================================

print("\n========== CALCUL DES VALEURS SHAP ==========")

raw_shap_values = explainer.shap_values(
    X_explain,
    nsamples=100
)

# Adaptation aux différentes versions de SHAP
if isinstance(raw_shap_values, list):
    shap_values = np.asarray(raw_shap_values[0])
else:
    shap_values = np.asarray(raw_shap_values)

# Certaines versions retournent une dimension supplémentaire
if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
    shap_values = shap_values[:, :, 0]

if shap_values.ndim != 2:
    raise ValueError(
        f"Format SHAP non reconnu : {shap_values.shape}"
    )

print("Forme des valeurs SHAP :", shap_values.shape)


# ============================================
# 10. SAUVEGARDE DES VALEURS SHAP
# ============================================

print("\n========== SAUVEGARDE DES RESULTATS ==========")

shap_df = pd.DataFrame(
    shap_values,
    columns=X_explain.columns
)

shap_df["true_label"] = y_explain
shap_df["anomaly_prediction"] = anomaly_predictions
shap_df["reconstruction_error"] = anomaly_scores

shap_df.to_csv(
    RESULTS_DIR / "autoencoder_shap_values.csv",
    index=False
)

# Importance globale des variables
feature_importance = pd.DataFrame({
    "feature": X_explain.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
})

feature_importance = feature_importance.sort_values(
    by="mean_abs_shap",
    ascending=False
)

feature_importance.to_csv(
    RESULTS_DIR / "autoencoder_shap_importance.csv",
    index=False
)

print("\nVariables les plus importantes selon SHAP :")
print(feature_importance.head(10).to_string(index=False))


# ============================================
# 11. GRAPHIQUE SHAP BAR
# ============================================

print("\n========== GRAPHIQUE SHAP BAR ==========")

shap.summary_plot(
    shap_values,
    X_explain,
    plot_type="bar",
    max_display=15,
    show=False
)

plt.title("Autoencoder - SHAP Feature Importance")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "autoencoder_shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================
# 12. GRAPHIQUE SHAP SUMMARY
# ============================================

print("\n========== GRAPHIQUE SHAP SUMMARY ==========")

shap.summary_plot(
    shap_values,
    X_explain,
    max_display=15,
    show=False
)

plt.title("Autoencoder - SHAP Summary Plot")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "autoencoder_shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================
# 13. EXPLICATION D'UNE FRAUDE ET D'UNE TRANSACTION NORMALE
# ============================================

print("\n========== EXPLICATIONS INDIVIDUELLES ==========")

base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

explanation = shap.Explanation(
    values=shap_values,
    base_values=np.full(len(X_explain), base_value),
    data=X_explain.to_numpy(),
    feature_names=X_explain.columns.tolist()
)

# Première fraude de l'échantillon
fraud_position = np.where(y_explain == 1)[0][0]

shap.plots.waterfall(
    explanation[fraud_position],
    max_display=15,
    show=False
)

plt.title("Autoencoder - Explication d'une fraude")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "autoencoder_shap_waterfall_fraud.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Première transaction normale de l'échantillon
normal_position = np.where(y_explain == 0)[0][0]

shap.plots.waterfall(
    explanation[normal_position],
    max_display=15,
    show=False
)

plt.title("Autoencoder - Explication d'une transaction normale")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "autoencoder_shap_waterfall_normal.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================
# 14. RESUME FINAL
# ============================================

print("\n========== ANALYSE TERMINEE ==========")

print("SHAP Autoencoder terminé.")

print("\nFichiers générés :")

print("- results/shap/autoencoder_shap_values.csv")
print("- results/shap/autoencoder_shap_importance.csv")

print("- results/figures/autoencoder_shap_bar.png")
print("- results/figures/autoencoder_shap_summary.png")
print("- results/figures/autoencoder_shap_waterfall_fraud.png")
print("- results/figures/autoencoder_shap_waterfall_normal.png")
