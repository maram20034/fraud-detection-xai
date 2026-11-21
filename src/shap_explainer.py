
import os
import json
import shap
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
SHAP_DIR = os.path.join(RESULTS_DIR, "shap")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(SHAP_DIR, exist_ok=True)

RANDOM_STATE = 42

BACKGROUND_SIZE = 100
EXPLAIN_SIZE = 200
MAX_FRAUD_EXAMPLES = 20
NSAMPLES = 100

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("\n======================================")
print("       SHAP EXPLAINABILITY - MLP")
print("======================================")


# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

print("\n1. CHARGEMENT DES DONNEES")

X_train_path = os.path.join(DATA_DIR, "X_train.csv")
X_test_path = os.path.join(DATA_DIR, "X_test.csv")
y_test_path = os.path.join(DATA_DIR, "y_test.csv")

X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)

y_test = pd.read_csv(y_test_path).squeeze("columns")

# Verifier que y_test contient une seule colonne
if isinstance(y_test, pd.DataFrame):
    raise ValueError(
        "y_test doit contenir une seule colonne."
    )

# Verifier les dimensions
if len(X_test) != len(y_test):
    raise ValueError(
        "Erreur : X_test et y_test n'ont pas le meme nombre de lignes."
    )

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Les donnees d'entrainement ou de test sont vides.")

# Verifier la coherence des variables
if X_train.columns.tolist() != X_test.columns.tolist():
    raise ValueError(
        "Les colonnes de X_train et X_test ne sont pas identiques."
    )

feature_names = X_train.columns.tolist()

# Conversion numerique
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

y_test = y_test.astype(int)

print("Training :", X_train.shape)
print("Testing :", X_test.shape)

print("\nDistribution des classes dans le test :")
print(y_test.value_counts())

print("\nNombre de fraudes :", int((y_test == 1).sum()))
print("Nombre de transactions normales :", int((y_test == 0).sum()))


# ==========================================
# 3. CHARGEMENT DU MODELE
# ==========================================

print("\n2. CHARGEMENT DU MODELE")

model_path = os.path.join(
    MODELS_DIR,
    "mlp_model.keras"
)

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Modele introuvable : {model_path}"
    )

model = tf.keras.models.load_model(model_path)

print("Modele MLP charge avec succes !")

# Chargement du seuil optimal
threshold_path = os.path.join(
    MODELS_DIR,
    "mlp_threshold.json"
)

if not os.path.exists(threshold_path):
    raise FileNotFoundError(
        f"Seuil introuvable : {threshold_path}"
    )

with open(threshold_path, "r") as f:
    threshold = float(json.load(f)["threshold"])

if not 0 <= threshold <= 1:
    raise ValueError("Le seuil doit etre compris entre 0 et 1.")

print("Seuil optimal :", threshold)


# ==========================================
# 4. FONCTION DE PREDICTION
# ==========================================

def predict_fraud(data):

    data = np.asarray(data, dtype=np.float32)

    predictions = model.predict(
        data,
        verbose=0
    )

    predictions = np.asarray(predictions)

    if predictions.ndim == 2 and predictions.shape[1] == 1:
        return predictions.reshape(-1)

    raise ValueError(
        "Le modele doit retourner une probabilite par transaction."
    )


# ==========================================
# 5. SELECTION DES ECHANTILLONS
# ==========================================

print("\n3. SELECTION DES ECHANTILLONS")

# Background : donnees representatives du train
background = X_train.sample(
    n=min(BACKGROUND_SIZE, len(X_train)),
    random_state=RANDOM_STATE
)

# Identifier les transactions frauduleuses et normales
fraud_positions = np.where(
    y_test.to_numpy() == 1
)[0]

normal_positions = np.where(
    y_test.to_numpy() == 0
)[0]

print("Fraudes disponibles :", len(fraud_positions))
print("Transactions normales :", len(normal_positions))

# Selectionner les fraudes en priorite
n_frauds = min(
    MAX_FRAUD_EXAMPLES,
    len(fraud_positions)
)

selected_frauds = np.random.choice(
    fraud_positions,
    size=n_frauds,
    replace=False
) if n_frauds > 0 else np.array([], dtype=int)

# Completer avec des transactions normales
remaining_size = EXPLAIN_SIZE - len(selected_frauds)

n_normal = min(
    remaining_size,
    len(normal_positions)
)

selected_normals = np.random.choice(
    normal_positions,
    size=n_normal,
    replace=False
) if n_normal > 0 else np.array([], dtype=int)

# Fusionner les positions selectionnees
selected_positions = np.concatenate([
    selected_frauds,
    selected_normals
])

# Melanger les positions
np.random.shuffle(selected_positions)

# Construire l'echantillon final
X_explain = X_test.iloc[selected_positions].copy()

y_explain = y_test.iloc[selected_positions].copy()

background = background.astype(np.float32)
X_explain = X_explain.astype(np.float32)

print("\nBackground :", background.shape)
print("Echantillon a expliquer :", X_explain.shape)

print(
    "Fraudes dans l'echantillon :",
    int((y_explain == 1).sum())
)

print(
    "Transactions normales dans l'echantillon :",
    int((y_explain == 0).sum())
)


# ==========================================
# 6. CONSTRUCTION DE L'EXPLAINER
# ==========================================

print("\n4. INITIALISATION DE SHAP")

# KernelExplainer explique la probabilite de fraude
explainer = shap.KernelExplainer(
    predict_fraud,
    background
)

print("SHAP Explainer initialise avec succes !")


# ==========================================
# 7. CALCUL DES VALEURS SHAP
# ==========================================

print("\n5. CALCUL DES VALEURS SHAP")

shap_values = explainer.shap_values(
    X_explain,
    nsamples=NSAMPLES,
    silent=False,
    rseed=RANDOM_STATE
)

# Compatibilite avec les anciennes versions de SHAP
if isinstance(shap_values, list):
    shap_values = shap_values[0]

shap_values = np.asarray(shap_values)

# Adapter les dimensions si necessaire
if shap_values.ndim == 3:

    if shap_values.shape[-1] == 1:
        shap_values = shap_values[:, :, 0]

    elif shap_values.shape[0] == 1:
        shap_values = shap_values[0]

if shap_values.shape != X_explain.shape:
    raise ValueError(
        f"Dimensions SHAP incorrectes : {shap_values.shape}. "
        f"Dimensions attendues : {X_explain.shape}"
    )

print("Calcul SHAP termine !")
print("Dimensions SHAP :", shap_values.shape)


# ==========================================
# 8. SAUVEGARDE DES VALEURS SHAP
# ==========================================

print("\n6. SAUVEGARDE DES VALEURS SHAP")

shap_df = pd.DataFrame(
    shap_values,
    columns=feature_names,
    index=X_explain.index
)

shap_df.to_csv(
    os.path.join(SHAP_DIR, "mlp_shap_values.csv"),
    index=True
)

print("Valeurs SHAP sauvegardees !")


# ==========================================
# 9. GLOBAL FEATURE IMPORTANCE
# ==========================================

print("\n7. IMPORTANCE GLOBALE DES VARIABLES")

mean_importance = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_SHAP": mean_importance
})

importance_df = importance_df.sort_values(
    by="Mean_SHAP",
    ascending=False
)

importance_df.to_csv(
    os.path.join(SHAP_DIR, "mlp_feature_importance.csv"),
    index=False
)

print("\nTop 10 variables importantes :")
print(importance_df.head(10).to_string(index=False))


# ==========================================
# 10. SUMMARY BAR PLOT
# ==========================================

print("\n8. GENERATION DU SUMMARY BAR PLOT")

plt.figure(figsize=(10, 7))

shap.summary_plot(
    shap_values,
    X_explain,
    feature_names=feature_names,
    plot_type="bar",
    max_display=15,
    show=False
)

plt.title(
    "SHAP - Global Feature Importance (MLP)"
)

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "mlp_shap_bar.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Summary Bar Plot sauvegarde !")


# ==========================================
# 11. SUMMARY BEESWARM
# ==========================================

print("\n9. GENERATION DU SUMMARY BEESWARM")

shap.summary_plot(
    shap_values,
    X_explain,
    feature_names=feature_names,
    max_display=15,
    show=False
)

plt.title(
    "SHAP - Summary Plot (MLP)"
)

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "mlp_shap_summary.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Summary Beeswarm sauvegarde !")


# ==========================================
# 12. EXPLICATION INDIVIDUELLE
# ==========================================

print("\n10. EXPLICATION INDIVIDUELLE")

# Rechercher une vraie transaction frauduleuse
fraud_positions_in_sample = np.where(
    y_explain.to_numpy() == 1
)[0]

if len(fraud_positions_in_sample) > 0:

    # Choisir la premiere fraude de l'echantillon
    fraud_position = fraud_positions_in_sample[0]

    fraud_shap = shap_values[fraud_position]

    fraud_data = X_explain.iloc[fraud_position]

    real_class = int(y_explain.iloc[fraud_position])

    fraud_probability = float(
        predict_fraud(
            fraud_data.values.reshape(1, -1)
        )[0]
    )

    predicted_class = int(
        fraud_probability >= threshold
    )

    print("\nTransaction frauduleuse selectionnee")

    print("Index :", X_explain.index[fraud_position])

    print("Classe reelle :", real_class)

    print("Probabilite de fraude :", fraud_probability)

    print("Seuil optimal :", threshold)

    print("Classe predite :", predicted_class)

    if predicted_class == 1:
        print("Resultat : FRAUDE DETECTEE")
    else:
        print("Resultat : FRAUDE NON DETECTEE (FAUX NEGATIF)")

    # Valeur de reference SHAP
    base_value = float(
        np.asarray(explainer.expected_value).reshape(-1)[0]
    )

    # Construction de l'explication
    explanation = shap.Explanation(
        values=fraud_shap,
        base_values=base_value,
        data=fraud_data.values,
        feature_names=feature_names
    )

    # Waterfall Plot
    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False
    )

    plt.gcf().set_size_inches(10, 7)

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "mlp_shap_waterfall_fraud.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nWaterfall Plot sauvegarde avec succes !")

else:

    print(
        "Aucune fraude disponible dans le dataset de test."
    )


# ==========================================
# 13. EXPLICATION D'UNE TRANSACTION NORMALE
# ==========================================

print("\n11. EXPLICATION D'UNE TRANSACTION NORMALE")

normal_positions_in_sample = np.where(
    y_explain.to_numpy() == 0
)[0]

if len(normal_positions_in_sample) > 0:

    normal_position = normal_positions_in_sample[0]

    normal_explanation = shap.Explanation(
        values=shap_values[normal_position],
        base_values=float(
            np.asarray(explainer.expected_value).reshape(-1)[0]
        ),
        data=X_explain.iloc[normal_position].values,
        feature_names=feature_names
    )

    shap.plots.waterfall(
        normal_explanation,
        max_display=15,
        show=False
    )

    plt.gcf().set_size_inches(10, 7)

    plt.savefig(
        os.path.join(
            FIGURES_DIR,
            "mlp_shap_waterfall_normal.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Waterfall normal sauvegarde !")

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score
)

print("\n========== EVALUATION DU MLP ==========")

# Prediction des probabilites sur tout le test
y_prob = predict_fraud(X_test)

# Comparaison de plusieurs seuils
for current_threshold in [0.5, threshold]:

    y_pred = (y_prob >= current_threshold).astype(int)

    print("\nSeuil :", current_threshold)

    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report :")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

print(
    "\nPR-AUC :",
    average_precision_score(y_test, y_prob)
)
# ==========================================
# 14. RESUME FINAL
# ==========================================

print("\n======================================")
print("          SHAP TERMINE")
print("======================================")

print("\nFichiers generes :")

print("\nDossier SHAP :")
print("- mlp_shap_values.csv")
print("- mlp_feature_importance.csv")

print("\nDossier FIGURES :")
print("- mlp_shap_bar.png")
print("- mlp_shap_summary.png")

if len(fraud_positions_in_sample) > 0:
    print("- mlp_shap_waterfall_fraud.png")

if len(normal_positions_in_sample) > 0:
    print("- mlp_shap_waterfall_normal.png")

print("\nDossier :", RESULTS_DIR)

print("\n======================================")
print("       EXECUTION TERMINEE")
print("======================================")
