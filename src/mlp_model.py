
import os
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
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
os.makedirs(FIGURES_DIR, exist_ok=True)

tf.keras.utils.set_random_seed(42)

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

X_train = X_train.to_numpy(dtype=np.float32)
X_val = X_val.to_numpy(dtype=np.float32)
X_test = X_test.to_numpy(dtype=np.float32)

y_train = y_train.to_numpy(dtype=np.int32)
y_val = y_val.to_numpy(dtype=np.int32)
y_test = y_test.to_numpy(dtype=np.int32)

print("Training :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing :", X_test.shape)

# ==========================================
# 3. GESTION DU DESEQUILIBRE
# ==========================================

print("\n========== CLASS WEIGHTS ==========\n")

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("Poids des classes :", class_weights)

# ==========================================
# 4. CONSTRUCTION DU MLP
# ==========================================

print("\n========== CONSTRUCTION DU MLP ==========\n")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(curve="PR", name="pr_auc")
    ]
)

model.summary()

# ==========================================
# 5. ENTRAINEMENT
# ==========================================

print("\n========== ENTRAINEMENT DU MLP ==========\n")

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_pr_auc",
    mode="max",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=512,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

print("\nEntrainement terminé !")

# ==========================================
# 6. RECHERCHE DU MEILLEUR SEUIL
# ==========================================

print("\n========== OPTIMISATION DU SEUIL ==========\n")

val_proba = model.predict(X_val, verbose=0).ravel()

thresholds = np.arange(0.05, 1.00, 0.005)

best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:

    val_pred = (val_proba >= threshold).astype(int)

    score = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold
print("Dernier seuil testé :", thresholds[-1])
print("Meilleur seuil :", best_threshold)
print("Meilleur F1 validation :", best_f1)
# ==========================================
# 7. EVALUATION FINALE SUR TEST
# ==========================================

print("\n========== EVALUATION SUR TEST ==========\n")

test_proba = model.predict(X_test, verbose=0).ravel()

test_pred = (test_proba >= best_threshold).astype(int)

precision = precision_score(
    y_test, test_pred, zero_division=0
)

recall = recall_score(
    y_test, test_pred, zero_division=0
)

f1 = f1_score(
    y_test, test_pred, zero_division=0
)

pr_auc = average_precision_score(y_test, test_proba)

roc_auc = roc_auc_score(y_test, test_proba)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nClassification Report :")

print(classification_report(
    y_test,
    test_pred,
    target_names=["Normal", "Fraud"],
    zero_division=0
))

# ==========================================
# 8. MATRICE DE CONFUSION
# ==========================================

cm = confusion_matrix(y_test, test_pred)
tn, fp, fn, tp = cm.ravel()

print("\n========== CONFUSION MATRIX DETAILS ==========")

print(f"True Negatives  (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives  (TP): {tp}")

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"]
)

plt.title("MLP - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "mlp_confusion_matrix.png"),
    dpi=300
)

plt.close()

# ==========================================
# 9. COURBES D'ENTRAINEMENT
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("MLP - Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "mlp_training_loss.png"),
    dpi=300
)

plt.close()

# ==========================================
# 10. SAUVEGARDE DU MODELE
# ==========================================

model.save(
    os.path.join(MODELS_DIR, "mlp_model.keras")
)

# Sauvegarde du seuil optimal
with open(
    os.path.join(MODELS_DIR, "mlp_threshold.json"),
    "w"
) as f:
    json.dump({"threshold": float(best_threshold)}, f)

# ==========================================
# 11. SAUVEGARDE DES RESULTATS
# ==========================================

results = pd.DataFrame([{
    "Model": "MLP_Class_Weights",
    "Threshold": best_threshold,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
    "PR-AUC": pr_auc,
    "ROC-AUC": roc_auc
}])

results.to_csv(
    os.path.join(RESULTS_DIR, "mlp_metrics.csv"),
    index=False
)

print("\nModèle sauvegardé !")
print("Résultats sauvegardés !")

print("\n========== MLP TERMINE ==========\n")
