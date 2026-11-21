
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    average_precision_score
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

RANDOM_STATE = 42
THRESHOLD_PERCENTILE = 99.5

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
AUTOENCODER_DIR = os.path.join(RESULTS_DIR, "autoencoder")

for folder in [MODEL_DIR, FIGURES_DIR, AUTOENCODER_DIR]:
    os.makedirs(folder, exist_ok=True)


# ==========================================================
# 2. CHARGEMENT DES DONNEES
# ==========================================================

print("\n========== AUTOENCODER : DETECTION DE FRAUDE ==========")

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))

# Conversion des labels en tableaux 1D
y_train = y_train.iloc[:, 0].values
y_test = y_test.iloc[:, 0].values

# Vérification
if len(X_train) != len(y_train):
    raise ValueError("X_train et y_train ont des tailles différentes.")

if len(X_test) != len(y_test):
    raise ValueError("X_test et y_test ont des tailles différentes.")

if not np.isfinite(X_train.values).all():
    raise ValueError("X_train contient des valeurs manquantes ou infinies.")

if not np.isfinite(X_test.values).all():
    raise ValueError("X_test contient des valeurs manquantes ou infinies.")

print("\nDonnées chargées avec succès !")
print("X_train :", X_train.shape)
print("X_test :", X_test.shape)

print("\nDistribution des classes :")
print(pd.Series(y_train).value_counts())


# ==========================================================
# 3. DIVISION TRAIN / VALIDATION
# ==========================================================

print("\n========== DIVISION DES DONNEES ==========")

X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train
)

# Garder uniquement les transactions normales
X_train_normal = X_train_part[y_train_part == 0]
X_val_normal = X_val[y_val == 0]

print("Training normal :", X_train_normal.shape)
print("Validation normale :", X_val_normal.shape)


# ==========================================================
# 4. NORMALISATION
# ==========================================================

print("\n========== NORMALISATION ==========")

scaler = StandardScaler()

# Le scaler apprend uniquement sur les transactions normales
X_train_normal_scaled = scaler.fit_transform(X_train_normal)

X_val_scaled = scaler.transform(X_val)
X_val_normal_scaled = scaler.transform(X_val_normal)
X_test_scaled = scaler.transform(X_test)

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "autoencoder_scaler.pkl")
)

print("Normalisation terminée !")


# ==========================================================
# 5. CONSTRUCTION DE L'AUTOENCODER
# ==========================================================

print("\n========== CONSTRUCTION AUTOENCODER ==========")

input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))

# Encoder
encoded = Dense(16, activation="relu")(input_layer)
encoded = Dense(8, activation="relu")(encoded)
encoded = Dense(4, activation="relu")(encoded)

# Decoder
decoded = Dense(8, activation="relu")(encoded)
decoded = Dense(16, activation="relu")(decoded)
output_layer = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(input_layer, output_layer)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()


# ==========================================================
# 6. ENTRAINEMENT
# ==========================================================

print("\n========== ENTRAINEMENT AUTOENCODER ==========")

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = autoencoder.fit(
    X_train_normal_scaled,
    X_train_normal_scaled,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(
        X_val_normal_scaled,
        X_val_normal_scaled
    ),
    callbacks=[early_stopping],
    verbose=1
)

print("\nEntraînement terminé !")


# ==========================================================
# 7. CALCUL DE L'ERREUR DE RECONSTRUCTION
# ==========================================================

print("\n========== ERREURS DE RECONSTRUCTION ==========")

# Erreur moyenne quadratique par transaction
def reconstruction_error(model, data):
    reconstructed = model.predict(data, verbose=0)

    errors = np.mean(
        np.square(data - reconstructed),
        axis=1
    )

    return errors


train_errors = reconstruction_error(
    autoencoder,
    X_train_normal_scaled
)

val_errors = reconstruction_error(
    autoencoder,
    X_val_normal_scaled
)

test_errors = reconstruction_error(
    autoencoder,
    X_test_scaled
)

print("Erreur moyenne train :", np.mean(train_errors))
print("Erreur moyenne validation :", np.mean(val_errors))
print("Erreur moyenne test :", np.mean(test_errors))


# ==========================================================
# 8. SELECTION DU SEUIL
# ==========================================================

print("\n========== SEUIL DE DETECTION ==========")

threshold = np.percentile(
    val_errors,
    THRESHOLD_PERCENTILE
)

print("Percentile utilisé :", THRESHOLD_PERCENTILE)
print("Seuil sélectionné :", threshold)

# Sauvegarde du seuil
with open(
    os.path.join(MODEL_DIR, "autoencoder_threshold.json"),
    "w"
) as file:
    json.dump(
        {
            "threshold": float(threshold),
            "percentile": THRESHOLD_PERCENTILE
        },
        file,
        indent=4
    )


# ==========================================================
# 9. PREDICTION SUR LE TEST
# ==========================================================

print("\n========== EVALUATION SUR LE TEST ==========")

y_pred = (test_errors > threshold).astype(int)

cm = confusion_matrix(y_test, y_pred)

print("\nMatrice de confusion :")
print(cm)

print("\nRapport de classification :")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0
    )
)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
pr_auc = average_precision_score(y_test, test_errors)

print("\n========== METRIQUES ==========")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")


# ==========================================================
# 10. SAUVEGARDE DES RESULTATS
# ==========================================================

autoencoder.save(
    os.path.join(MODEL_DIR, "autoencoder.keras")
)

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "pr_auc": float(pr_auc),
    "threshold": float(threshold),
    "threshold_percentile": THRESHOLD_PERCENTILE
}

with open(
    os.path.join(AUTOENCODER_DIR, "metrics.json"),
    "w"
) as file:
    json.dump(metrics, file, indent=4)

pd.DataFrame({
    "Real_Class": y_test,
    "Reconstruction_Error": test_errors,
    "Predicted_Class": y_pred
}).to_csv(
    os.path.join(AUTOENCODER_DIR, "predictions.csv"),
    index=False
)


# ==========================================================
# 11. VISUALISATIONS
# ==========================================================

print("\n========== VISUALISATIONS ==========")

# Courbe de perte
plt.figure(figsize=(9, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Autoencoder - Evolution de la perte")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "autoencoder_loss.png")
)
plt.close()


# Distribution des erreurs de reconstruction
plt.figure(figsize=(10, 6))

sns.histplot(
    test_errors[y_test == 0],
    bins=50,
    color="green",
    label="Transactions normales",
    stat="density",
    element="step",
    fill=False
)

sns.histplot(
    test_errors[y_test == 1],
    bins=50,
    color="red",
    label="Transactions frauduleuses",
    stat="density",
    element="step",
    fill=False
)

plt.axvline(
    threshold,
    color="black",
    linestyle="--",
    label="Seuil de détection"
)

plt.title("Distribution des erreurs de reconstruction")
plt.xlabel("Erreur de reconstruction")
plt.ylabel("Densité")
plt.legend()

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "autoencoder_errors.png")
)
plt.close()


# Matrice de confusion
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Matrice de confusion - Autoencoder")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_DIR, "autoencoder_confusion_matrix.png")
)
plt.close()


print("\n========== SAUVEGARDE ==========")
print("Modèle sauvegardé.")
print("Scaler sauvegardé.")
print("Seuil sauvegardé.")
print("Métriques sauvegardées.")
print("Graphiques sauvegardés.")

print("\n========== AUTOENCODER TERMINE ==========")
