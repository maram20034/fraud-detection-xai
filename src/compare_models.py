
import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
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

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_style("whitegrid")

np.random.seed(42)
tf.keras.utils.set_random_seed(42)


# ==========================================
# 2. CHARGEMENT DES DONNEES
# ==========================================

print("\n" + "=" * 55)
print("1. CHARGEMENT DES DONNEES")
print("=" * 55)

X_test = pd.read_csv(
    os.path.join(DATA_DIR, "X_test.csv")
)

y_test = pd.read_csv(
    os.path.join(DATA_DIR, "y_test.csv")
).squeeze().to_numpy()

print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

print("\nDistribution des classes :")
print(pd.Series(y_test).value_counts())


# ==========================================
# 3. CHARGEMENT DES MODELES
# ==========================================

print("\n" + "=" * 55)
print("2. CHARGEMENT DES MODELES")
print("=" * 55)

# Baseline
baseline_model = joblib.load(
    os.path.join(MODELS_DIR, "baseline_model.pkl")
)

# MLP Class Weights
mlp_model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "mlp_model.keras"),
    compile=False
)

# MLP SMOTE
smote_model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "mlp_smote_model.keras"),
    compile=False
)

# MLP Undersampling
under_model = tf.keras.models.load_model(
    os.path.join(MODELS_DIR, "mlp_undersampling_model.keras"),
    compile=False
)

print("Baseline chargé !")
print("MLP Class Weights chargé !")
print("MLP SMOTE chargé !")
print("MLP Undersampling chargé !")


# ==========================================
# 4. CHARGEMENT DES SEUILS OPTIMAUX
# ==========================================

print("\n" + "=" * 55)
print("3. CHARGEMENT DES SEUILS")
print("=" * 55)


def load_threshold(filename):

    path = os.path.join(MODELS_DIR, filename)

    with open(path, "r") as f:
        data = json.load(f)

    return float(data["threshold"])


# Baseline : le seuil est récupéré depuis
# le fichier d'analyse des seuils de validation.

baseline_threshold_path = os.path.join(
    RESULTS_DIR,
    "baseline_threshold_analysis.csv"
)

if os.path.exists(baseline_threshold_path):

    threshold_df = pd.read_csv(baseline_threshold_path)

    best_index = threshold_df["F1-score"].idxmax()

    baseline_threshold = float(
        threshold_df.loc[best_index, "Threshold"]
    )

else:

    baseline_threshold = 0.5

    print("Attention : seuil Baseline introuvable.")
    print("Utilisation du seuil par défaut : 0.5")


mlp_threshold = load_threshold(
    "mlp_threshold.json"
)

smote_threshold = load_threshold(
    "mlp_smote_threshold.json"
)

under_threshold = load_threshold(
    "mlp_undersampling_threshold.json"
)


thresholds = {
    "Logistic Regression": baseline_threshold,
    "MLP + Class Weights": mlp_threshold,
    "MLP + SMOTE": smote_threshold,
    "MLP + Undersampling": under_threshold
}

for name, threshold in thresholds.items():
    print(f"{name} : {threshold:.4f}")


# ==========================================
# 5. PREDICTIONS
# ==========================================

print("\n" + "=" * 55)
print("4. PREDICTIONS SUR TEST")
print("=" * 55)

# Probabilités de fraude

probabilities = {

    "Logistic Regression":
        baseline_model.predict_proba(X_test)[:, 1],

    "MLP + Class Weights":
        mlp_model.predict(X_test, verbose=0).ravel(),

    "MLP + SMOTE":
        smote_model.predict(X_test, verbose=0).ravel(),

    "MLP + Undersampling":
        under_model.predict(X_test, verbose=0).ravel()
}
# Vérifier si les modèles produisent des probabilités identiques
class_weights_probs = probabilities["MLP + Class Weights"]
undersampling_probs = probabilities["MLP + Undersampling"]

print("\n--- Diagnostic Class Weights vs Undersampling ---")

print(
    "Probabilités identiques :",
    np.allclose(class_weights_probs, undersampling_probs)
)

print(
    "Différence absolue maximale :",
    np.max(np.abs(class_weights_probs - undersampling_probs))
)

print(
    "Corrélation :",
    np.corrcoef(class_weights_probs, undersampling_probs)[0, 1]
)

# Prédictions binaires avec les seuils optimaux

predictions = {}

for name, proba in probabilities.items():

    predictions[name] = (
        proba >= thresholds[name]
    ).astype(int)

    print(f"{name} : prédictions terminées")


# ==========================================
# 6. CALCUL DES METRIQUES
# ==========================================

print("\n" + "=" * 55)
print("5. EVALUATION DES MODELES")
print("=" * 55)

results = []

confusion_matrices = {}

for name in probabilities:

    y_pred = predictions[name]
    y_proba = probabilities[name]

    accuracy = accuracy_score(
        y_test, y_pred
    )

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    pr_auc = average_precision_score(
        y_test, y_proba
    )

    roc_auc = roc_auc_score(
        y_test, y_proba
    )

    cm = confusion_matrix(
        y_test, y_pred
    )

    tn, fp, fn, tp = cm.ravel()

    confusion_matrices[name] = cm

    results.append({

        "Model": name,

        "Threshold": thresholds[name],

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1-score": f1,

        "PR-AUC": pr_auc,

        "ROC-AUC": roc_auc,

        "TN": int(tn),

        "FP": int(fp),

        "FN": int(fn),

        "TP": int(tp)
    })

    print(f"\n{name}")
    print("-" * 40)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")


# ==========================================
# 7. TABLEAU COMPARATIF
# ==========================================

print("\n" + "=" * 55)
print("6. COMPARAISON GLOBALE")
print("=" * 55)

comparison_df = pd.DataFrame(results)

comparison_df = comparison_df.sort_values(
    by="F1-score",
    ascending=False
)

print(
    comparison_df.round(4).to_string(index=False)
)

# Sauvegarde CSV

comparison_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    ),
    index=False
)

# Sauvegarde JSON

comparison_df.to_json(
    os.path.join(
        RESULTS_DIR,
        "model_comparison.json"
    ),
    orient="records",
    indent=4
)


# ==========================================
# 8. IDENTIFICATION DU MEILLEUR MODELE
# ==========================================

print("\n" + "=" * 55)
print("7. MEILLEURS MODELES")
print("=" * 55)

best_f1_model = comparison_df.loc[
    comparison_df["F1-score"].idxmax()
]

best_pr_model = comparison_df.loc[
    comparison_df["PR-AUC"].idxmax()
]

best_roc_model = comparison_df.loc[
    comparison_df["ROC-AUC"].idxmax()
]

print(
    "Meilleur modèle selon F1-score :",
    best_f1_model["Model"]
)

print(
    "Meilleur F1-score :",
    round(best_f1_model["F1-score"], 4)
)

print(
    "\nMeilleur modèle selon PR-AUC :",
    best_pr_model["Model"]
)

print(
    "Meilleur PR-AUC :",
    round(best_pr_model["PR-AUC"], 4)
)

print(
    "\nMeilleur modèle selon ROC-AUC :",
    best_roc_model["Model"]
)

print(
    "Meilleur ROC-AUC :",
    round(best_roc_model["ROC-AUC"], 4)
)


# ==========================================
# 9. VISUALISATION DES METRIQUES
# ==========================================

print("\n" + "=" * 55)
print("8. VISUALISATION DES METRIQUES")
print("=" * 55)

metrics_to_plot = [
    "Precision",
    "Recall",
    "F1-score",
    "PR-AUC",
    "ROC-AUC"
]

plot_df = comparison_df.melt(
    id_vars="Model",
    value_vars=metrics_to_plot,
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(13, 7))

sns.barplot(
    data=plot_df,
    x="Metric",
    y="Score",
    hue="Model"
)

plt.title(
    "Global Comparison of Fraud Detection Models",
    fontsize=14
)

plt.ylabel("Score")
plt.xlabel("Evaluation Metrics")

plt.ylim(0, 1.05)

plt.legend(
    title="Models",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "global_model_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 10. COMPARAISON ROC CURVES
# ==========================================

print("\nGénération de la courbe ROC...")

plt.figure(figsize=(9, 7))

for name, y_proba in probabilities.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        y_proba
    )

    auc = roc_auc_score(
        y_test,
        y_proba
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{name} (AUC = {auc:.4f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - Model Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "model_comparison_roc.png"
    ),
    dpi=300
)

plt.close()


# ==========================================
# 11. COMPARAISON PRECISION-RECALL
# ==========================================

print("Génération de la courbe Precision-Recall...")

plt.figure(figsize=(9, 7))

for name, y_proba in probabilities.items():

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_test,
            y_proba
        )
    )

    ap = average_precision_score(
        y_test,
        y_proba
    )

    plt.plot(
        recall_curve,
        precision_curve,
        linewidth=2,
        label=f"{name} (AP = {ap:.4f})"
    )

plt.title("Precision-Recall Curve - Model Comparison")

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "model_comparison_pr.png"
    ),
    dpi=300
)

plt.close()


# ==========================================
# 12. MATRICES DE CONFUSION
# ==========================================

print("Génération des matrices de confusion...")

fig, axes = plt.subplots(
    2, 2,
    figsize=(13, 11)
)

axes = axes.flatten()

for i, (name, cm) in enumerate(
    confusion_matrices.items()
):

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"],
        ax=axes[i]
    )

    axes[i].set_title(name)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

plt.suptitle(
    "Confusion Matrices - Model Comparison",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "model_comparison_confusion_matrices.png"
    ),
    dpi=300
)

plt.close()


# ==========================================
# 13. SAUVEGARDE DES PREDICTIONS
# ==========================================

print("Sauvegarde des prédictions...")

prediction_df = pd.DataFrame({
    "Actual": y_test
})

for name in probabilities:

    safe_name = (
        name.lower()
        .replace(" ", "_")
        .replace("+", "plus")
    )

    prediction_df[f"{safe_name}_probability"] = (
        probabilities[name]
    )

    prediction_df[f"{safe_name}_prediction"] = (
        predictions[name]
    )

prediction_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "model_predictions.csv"
    ),
    index=False
)


# ==========================================
# 14. RESUME FINAL
# ==========================================

print("\n" + "=" * 55)
print("9. RESUME FINAL")
print("=" * 55)

print("\nNombre de modèles comparés :", len(results))

print(
    "\nMeilleur modèle selon F1-score :",
    best_f1_model["Model"]
)

print(
    "F1-score :",
    round(best_f1_model["F1-score"], 4)
)

print(
    "\nMeilleur modèle selon PR-AUC :",
    best_pr_model["Model"]
)

print(
    "PR-AUC :",
    round(best_pr_model["PR-AUC"], 4)
)

print("\nFichiers enregistrés :")

print(
    os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    )
)

print(
    os.path.join(
        RESULTS_DIR,
        "model_comparison.json"
    )
)

print("\nFigures enregistrées dans :")
print(FIGURES_DIR)

print("\n" + "=" * 55)
print("COMPARAISON DES MODELES TERMINEE")
print("=" * 55)
