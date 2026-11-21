
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from tensorflow.keras.models import load_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(".")

RESULTS_DIR = BASE_DIR / "results"
LIME_DIR = RESULTS_DIR / "lime"
FIGURES_DIR = RESULTS_DIR / "figures"

LIME_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
NUM_FEATURES = 10
NUM_SAMPLES = 3000


# ============================================================
# 2. CHARGEMENT DES DONNEES
# ============================================================

print("\n========== CHARGEMENT DES DONNEES ==========")

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_test_df = pd.read_csv("data/processed/y_test.csv")
y_test = y_test_df.iloc[:, 0].to_numpy().astype(int)

feature_names = X_train.columns.tolist()

X_train_np = X_train.to_numpy(dtype=float)
X_test_np = X_test.to_numpy(dtype=float)

print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)

# Selection deterministe d'une fraude et d'une transaction normale
rng = np.random.RandomState(RANDOM_STATE)

fraud_indices = np.where(y_test == 1)[0]
normal_indices = np.where(y_test == 0)[0]

if len(fraud_indices) == 0 or len(normal_indices) == 0:
    raise ValueError("Le jeu de test doit contenir des exemples normaux et frauduleux.")

fraud_index = int(rng.choice(fraud_indices))
normal_index = int(rng.choice(normal_indices))

selected_transactions = {
    "fraud": fraud_index,
    "normal": normal_index
}

print("\nTransactions selectionnees :")
print("Fraud index  :", fraud_index)
print("Normal index :", normal_index)


# ============================================================
# 3. FONCTION POUR CHARGER LES SEUILS
# ============================================================

def load_threshold(file_path, default_value):
    """
    Charge le seuil depuis un fichier JSON.
    Utilise la valeur par defaut si le fichier est absent
    ou si son contenu ne contient pas de seuil exploitable.
    """
    path = Path(file_path)

    if not path.exists():
        return default_value

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, (int, float)):
        return float(data)

    if isinstance(data, dict):
        for key, value in data.items():
            if "threshold" in key.lower() and isinstance(value, (int, float)):
                return float(value)

        for value in data.values():
            if isinstance(value, (int, float)):
                return float(value)

    return default_value


mlp_threshold = load_threshold(
    "models/mlp_threshold.json", 0.99
)

rf_threshold = load_threshold(
    "models/random_forest_threshold.json", 0.3533333333
)

ae_threshold = load_threshold(
    "models/autoencoder_threshold.json", 4.6457848
)

print("\nSeuil MLP :", mlp_threshold)
print("Seuil RF  :", rf_threshold)
print("Seuil AE  :", ae_threshold)


# ============================================================
# 4. CHARGEMENT DES MODELES
# ============================================================

print("\n========== CHARGEMENT DES MODELES ==========")

mlp_model = load_model("models/mlp_model.keras")

rf_model = joblib.load("models/random_forest_model.joblib")

autoencoder = load_model("models/autoencoder.keras")

scaler = joblib.load("models/autoencoder_scaler.pkl")

print("Les trois modeles sont charges avec succes.")


# ============================================================
# 5. FONCTIONS DE PREDICTION
# ============================================================

# MLP : probabilite de fraude
def mlp_predict_proba(data):
    data = np.asarray(data, dtype=np.float32)

    probabilities = mlp_model.predict(
        data,
        verbose=0
    ).reshape(-1)

    probabilities = np.clip(probabilities, 0, 1)

    return np.column_stack((
        1 - probabilities,
        probabilities
    ))


# Random Forest : probabilite de fraude
def rf_predict_proba(data):
    data = np.asarray(data, dtype=float)
    return rf_model.predict_proba(data)


# Autoencoder : erreur de reconstruction
def ae_reconstruction_error(data):
    data = np.asarray(data, dtype=float)

    data_scaled = scaler.transform(data)

    reconstructed = autoencoder.predict(
        data_scaled,
        verbose=0
    )

    errors = np.mean(
        np.square(data_scaled - reconstructed),
        axis=1
    )

    return errors


# ============================================================
# 6. CREATION DES EXPLAINERS LIME
# ============================================================

print("\n========== INITIALISATION DE LIME ==========")

# LIME pour les modeles de classification
classification_explainer = LimeTabularExplainer(
    training_data=X_train_np,
    feature_names=feature_names,
    class_names=["Normal", "Fraud"],
    mode="classification",
    discretize_continuous=True,
    random_state=RANDOM_STATE
)

# LIME pour l'Autoencoder (regression)
regression_explainer = LimeTabularExplainer(
    training_data=X_train_np,
    feature_names=feature_names,
    mode="regression",
    discretize_continuous=True,
    random_state=RANDOM_STATE
)

print("Les explainers LIME sont prets.")


# ============================================================
# 7. FONCTION POUR SAUVEGARDER UNE EXPLICATION
# ============================================================

all_explanations = []


def save_classification_explanation(
    model_name,
    transaction_type,
    index,
    predict_function,
    threshold
):
    print(f"\n--- {model_name} : {transaction_type} ---")

    transaction = X_test_np[index]

    probabilities = predict_function(
        transaction.reshape(1, -1)
    )[0]

    fraud_probability = float(probabilities[1])

    predicted_class = int(fraud_probability >= threshold)
    actual_class = int(y_test[index])

    print("Index :", index)
    print("Vraie classe :", actual_class)
    print("Probabilite de fraude :", round(fraud_probability, 6))
    print("Prediction avec le seuil du projet :", predicted_class)

    explanation = classification_explainer.explain_instance(
        data_row=transaction,
        predict_fn=predict_function,
        labels=(1,),
        num_features=NUM_FEATURES,
        num_samples=NUM_SAMPLES
    )

    # Sauvegarde HTML
    html_path = LIME_DIR / f"{model_name}_lime_{transaction_type}.html"
    explanation.save_to_file(str(html_path))

    # Sauvegarde PNG
    figure = explanation.as_pyplot_figure(label=1)

    png_path = FIGURES_DIR / f"{model_name}_lime_{transaction_type}.png"

    figure.savefig(
        png_path,
        bbox_inches="tight",
        dpi=300
    )

    plt.close(figure)

    # Sauvegarde des poids LIME
    for feature, weight in explanation.as_list(label=1):
        all_explanations.append({
            "model": model_name,
            "transaction_type": transaction_type,
            "actual_class": actual_class,
            "fraud_probability": fraud_probability,
            "project_prediction": predicted_class,
            "feature": feature,
            "lime_weight": weight
        })

    print("Explication HTML :", html_path)
    print("Graphique PNG :", png_path)


def save_autoencoder_explanation(transaction_type, index):
    print(f"\n--- AUTOENCODER : {transaction_type} ---")

    transaction = X_test_np[index]

    reconstruction_error = float(
        ae_reconstruction_error(
            transaction.reshape(1, -1)
        )[0]
    )

    predicted_class = int(reconstruction_error > ae_threshold)
    actual_class = int(y_test[index])

    print("Index :", index)
    print("Vraie classe :", actual_class)
    print("Erreur de reconstruction :", round(reconstruction_error, 6))
    print("Prediction avec le seuil du projet :", predicted_class)

    explanation = regression_explainer.explain_instance(
        data_row=transaction,
        predict_fn=ae_reconstruction_error,
        num_features=NUM_FEATURES,
        num_samples=NUM_SAMPLES
    )

    # Sauvegarde HTML
    html_path = LIME_DIR / f"autoencoder_lime_{transaction_type}.html"
    explanation.save_to_file(str(html_path))

    # Sauvegarde PNG
    figure = explanation.as_pyplot_figure()

    png_path = FIGURES_DIR / f"autoencoder_lime_{transaction_type}.png"

    figure.savefig(
        png_path,
        bbox_inches="tight",
        dpi=300
    )

    plt.close(figure)

    # Sauvegarde des poids LIME
    for feature, weight in explanation.as_list():
        all_explanations.append({
            "model": "Autoencoder",
            "transaction_type": transaction_type,
            "actual_class": actual_class,
            "fraud_probability": np.nan,
            "project_prediction": predicted_class,
            "feature": feature,
            "lime_weight": weight
        })

    print("Explication HTML :", html_path)
    print("Graphique PNG :", png_path)


# ============================================================
# 8. EXPLICATIONS MLP
# ============================================================

print("\n========== EXPLICATIONS MLP ==========")

for transaction_type, index in selected_transactions.items():
    save_classification_explanation(
        model_name="mlp",
        transaction_type=transaction_type,
        index=index,
        predict_function=mlp_predict_proba,
        threshold=mlp_threshold
    )


# ============================================================
# 9. EXPLICATIONS RANDOM FOREST
# ============================================================

print("\n========== EXPLICATIONS RANDOM FOREST ==========")

for transaction_type, index in selected_transactions.items():
    save_classification_explanation(
        model_name="random_forest",
        transaction_type=transaction_type,
        index=index,
        predict_function=rf_predict_proba,
        threshold=rf_threshold
    )


# ============================================================
# 10. EXPLICATIONS AUTOENCODER
# ============================================================

print("\n========== EXPLICATIONS AUTOENCODER ==========")

for transaction_type, index in selected_transactions.items():
    save_autoencoder_explanation(
        transaction_type=transaction_type,
        index=index
    )


# ============================================================
# 11. SAUVEGARDE DU FICHIER CSV
# ============================================================

explanations_df = pd.DataFrame(all_explanations)

csv_path = LIME_DIR / "lime_all_explanations.csv"

explanations_df.to_csv(
    csv_path,
    index=False
)

print("\n========== TERMINÉ ==========")

print("Nombre total d'explications :", len(all_explanations))
print("Fichier CSV :", csv_path)
print("Dossier HTML :", LIME_DIR)
print("Dossier PNG :", FIGURES_DIR)

print("\nTous les fichiers LIME ont ete generes avec succes !")
