
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

RANDOM_STATE = 42
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 0.001

# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

print("\n" + "=" * 55)
print("1. CHARGEMENT DES DONNEES")
print("=" * 55)

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.ravel()
y_val = pd.read_csv(os.path.join(DATA_DIR, "y_val.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.ravel()

# Garantir le même ordre des features
X_val = X_val[X_train.columns]
X_test = X_test[X_train.columns]

# Conversion en tableaux NumPy
X_train = X_train.to_numpy(dtype=np.float32)
X_val = X_val.to_numpy(dtype=np.float32)
X_test = X_test.to_numpy(dtype=np.float32)

y_train = y_train.astype(np.int32)
y_val = y_val.astype(np.int32)
y_test = y_test.astype(np.int32)

print("X_train :", X_train.shape)
print("X_val   :", X_val.shape)
print("X_test  :", X_test.shape)

print("\nRépartition initiale des classes dans Train :")
print(pd.Series(y_train).value_counts().sort_index())

# ==========================================
# 3. APPLICATION DE SMOTE
# ==========================================

print("\n" + "=" * 55)
print("2. APPLICATION DE SMOTE")
print("=" * 55)

smote = SMOTE(
    sampling_strategy="auto",
    random_state=RANDOM_STATE,
    k_neighbors=5
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\nDimensions après SMOTE :")
print("X_train_smote :", X_train_smote.shape)
print("y_train_smote :", y_train_smote.shape)

print("\nRépartition après SMOTE :")
print(pd.Series(y_train_smote).value_counts().sort_index())

# Sauvegarder la répartition des classes
class_distribution = pd.DataFrame({
    "Class": [0, 1],
    "Before_SMOTE": [
        np.sum(y_train == 0),
        np.sum(y_train == 1)
    ],
    "After_SMOTE": [
        np.sum(y_train_smote == 0),
        np.sum(y_train_smote == 1)
    ]
})

class_distribution.to_csv(
    os.path.join(RESULTS_DIR, "smote_class_distribution.csv"),
    index=False
)

# ==========================================
# 4. CONSTRUCTION DU MODELE MLP
# ==========================================

print("\n" + "=" * 55)
print("3. CONSTRUCTION DU MLP")
print("=" * 55)

tf.keras.utils.set_random_seed(RANDOM_STATE)

model = Sequential([
    Input(shape=(X_train_smote.shape[1],)),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(
            curve="PR",
            name="pr_auc"
        )
    ]
)

model.summary()

# ==========================================
# 5. ENTRAINEMENT
# ==========================================

print("\n" + "=" * 55)
print("4. ENTRAINEMENT DU MLP + SMOTE")
print("=" * 55)

early_stopping = EarlyStopping(
    monitor="val_pr_auc",
    mode="max",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train_smote,
    y_train_smote,

    validation_data=(X_val, y_val),

    epochs=EPOCHS,
    batch_size=BATCH_SIZE,

    callbacks=[early_stopping],

    verbose=1
)

# ==========================================
# 6. RECHERCHE DU MEILLEUR SEUIL
# ==========================================

print("\n" + "=" * 55)
print("5. OPTIMISATION DU SEUIL")
print("=" * 55)

y_val_proba = model.predict(
    X_val,
    verbose=0
).ravel()

precisions, recalls, thresholds = precision_recall_curve(
    y_val,
    y_val_proba
)

# Calcul du F1 pour chaque seuil
f1_scores = (
    2 * precisions[:-1] * recalls[:-1]
    / (precisions[:-1] + recalls[:-1] + 1e-10)
)

best_index = np.argmax(f1_scores)

best_threshold = float(thresholds[best_index])
best_val_f1 = float(f1_scores[best_index])

print(f"Meilleur seuil : {best_threshold:.4f}")
print(f"Meilleur F1 validation : {best_val_f1:.4f}")

# ==========================================
# 7. EVALUATION SUR TEST
# ==========================================

print("\n" + "=" * 55)
print("6. EVALUATION SUR TEST")
print("=" * 55)

y_test_proba = model.predict(
    X_test,
    verbose=0
).ravel()

y_test_pred = (
    y_test_proba >= best_threshold
).astype(int)

precision = precision_score(
    y_test, y_test_pred, zero_division=0
)

recall = recall_score(
    y_test, y_test_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_test_pred, zero_division=0
)

accuracy = accuracy_score(
    y_test, y_test_pred
)

roc_auc = roc_auc_score(
    y_test, y_test_proba
)

pr_auc = average_precision_score(
    y_test, y_test_proba
)

cm = confusion_matrix(y_test, y_test_pred)

print("\nRESULTATS MLP + SMOTE")
print("-" * 40)
print(f"Threshold : {best_threshold:.4f}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nMatrice de confusion :")
print(cm)

print("\nRapport de classification :")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=["Normal", "Fraud"],
    zero_division=0
))

# ==========================================
# 8. SAUVEGARDE DU MODELE ET DES RESULTATS
# ==========================================

print("\n" + "=" * 55)
print("7. SAUVEGARDE")
print("=" * 55)

model_path = os.path.join(
    MODELS_DIR,
    "mlp_smote_model.keras"
)

model.save(model_path)

# Sauvegarder le meilleur seuil
threshold_path = os.path.join(
    MODELS_DIR,
    "mlp_smote_threshold.json"
)

with open(threshold_path, "w", encoding="utf-8") as f:
    json.dump({
        "threshold": best_threshold,
        "validation_f1": best_val_f1
    }, f, indent=4)

# Sauvegarder les métriques
metrics = {
    "model": "MLP + SMOTE",
    "threshold": best_threshold,
    "validation_f1": best_val_f1,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "pr_auc": float(pr_auc),
    "roc_auc": float(roc_auc),
    "confusion_matrix": cm.tolist()
}

metrics_path = os.path.join(
    RESULTS_DIR,
    "mlp_smote_metrics.json"
)

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

# ==========================================
# 9. VISUALISATION DE L'ENTRAINEMENT
# ==========================================

print("\n" + "=" * 55)
print("8. VISUALISATIONS")
print("=" * 55)

# Courbe de loss
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("MLP + SMOTE - Loss")
plt.xlabel("Epoch")
plt.ylabel("Binary Crossentropy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "mlp_smote_loss.png"),
    dpi=300
)

plt.close()

# Courbe PR-AUC
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["pr_auc"],
    label="Training PR-AUC"
)

plt.plot(
    history.history["val_pr_auc"],
    label="Validation PR-AUC"
)

plt.title("MLP + SMOTE - PR-AUC")
plt.xlabel("Epoch")
plt.ylabel("PR-AUC")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "mlp_smote_pr_auc.png"),
    dpi=300
)

plt.close()

# Matrice de confusion
plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - MLP + SMOTE")
plt.colorbar()

plt.xticks([0, 1], ["Normal", "Fraud"])
plt.yticks([0, 1], ["Normal", "Fraud"])

plt.xlabel("Predicted label")
plt.ylabel("True label")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=12
        )

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "mlp_smote_confusion_matrix.png"),
    dpi=300
)

plt.close()

print("\nToutes les sauvegardes sont terminées.")
print("Modèle :", model_path)
print("Métriques :", metrics_path)
print("Seuil :", threshold_path)
print("Figures :", FIGURES_DIR)
