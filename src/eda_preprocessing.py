
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. CONFIGURATION
# ==========================================

sns.set_style("whitegrid")
pd.set_option("display.max_columns", None)

# Chemins des fichiers
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "raw", "creditcard.csv"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results", "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ==========================================
# 2. CHARGEMENT DU DATASET
# ==========================================

print("\n========== CHARGEMENT DU DATASET ==========\n")

df = pd.read_csv(DATA_PATH)

print("Dataset chargé avec succès !")

print("Nombre de lignes :", df.shape[0])
print("Nombre de colonnes :", df.shape[1])


# ==========================================
# 3. AFFICHER LES PREMIERES LIGNES
# ==========================================

print("\n========== PREMIERES LIGNES ==========\n")

print(df.head())


# ==========================================
# 4. INFORMATIONS GENERALES
# ==========================================

print("\n========== INFORMATIONS GENERALES ==========\n")

df.info()


# ==========================================
# 5. STATISTIQUES DESCRIPTIVES
# ==========================================

print("\n========== STATISTIQUES DESCRIPTIVES ==========\n")

print(df.describe())


# ==========================================
# 6. VALEURS MANQUANTES
# ==========================================

print("\n========== VALEURS MANQUANTES ==========\n")

missing_values = df.isnull().sum()

print(missing_values)

print("\nNombre total de valeurs manquantes :",
      missing_values.sum())


# ==========================================
# 7. DOUBLONS
# ==========================================

print("\n========== DOUBLONS ==========\n")

duplicates = df.duplicated().sum()

print("Nombre de doublons :", duplicates)
# ==========================================
# 7.1 ANALYSE DETAILLEE DES DOUBLONS
# ==========================================

print("\n========== ANALYSE DES DOUBLONS ==========\n")

# Identifier toutes les lignes appartenant à un groupe de doublons
duplicate_rows = df[df.duplicated(keep=False)]

print("Nombre de lignes concernées :", len(duplicate_rows))

print("\nRépartition des doublons par classe :")
print(duplicate_rows["Class"].value_counts())

print("\nRépartition en pourcentage :")
print(duplicate_rows["Class"].value_counts(normalize=True) * 100)

# Vérifier si des transactions ayant les mêmes caractéristiques
# possèdent des classes différentes

feature_columns = df.drop(columns=["Class"]).columns

conflicting_duplicates = (
    df.groupby(list(feature_columns))["Class"]
    .nunique()
)

print(
    "\nGroupes de transactions identiques "
    "avec des classes différentes :",
    (conflicting_duplicates > 1).sum()
)


# ==========================================
# 8. DISTRIBUTION DES CLASSES
# ==========================================

print("\n========== DISTRIBUTION DES CLASSES ==========\n")

class_counts = df["Class"].value_counts()

class_percentages = df["Class"].value_counts(normalize=True) * 100

print("Nombre de transactions par classe :")
print(class_counts)

print("\nPourcentage par classe :")

for classe in class_counts.index:
    print(
        f"Classe {classe} : "
        f"{class_percentages[classe]:.4f}%"
    )


# ==========================================
# 9. VISUALISATION DU DESÉQUILIBRE
# ==========================================

plt.figure(figsize=(8, 5))

sns.countplot(data=df, x="Class")

plt.title("Distribution des transactions")
plt.xlabel("Classe (0 = Normal, 1 = Fraude)")
plt.ylabel("Nombre de transactions")

plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "class_distribution.png")
)

plt.show()


# ==========================================
# 10. VISUALISATION DES POURCENTAGES
# ==========================================

plt.figure(figsize=(7, 7))

plt.pie(
    class_counts,
    labels=["Transactions normales", "Fraudes"],
    autopct="%1.2f%%",
    startangle=90
)

plt.title("Répartition des transactions")

plt.savefig(
    os.path.join(RESULTS_DIR, "class_distribution_pie.png")
)

plt.show()


# ==========================================
# 11. MATRICE DE CORRELATION
# ==========================================

plt.figure(figsize=(18, 14))

correlation_matrix = df.corr()

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0,
    linewidths=0.1
)

plt.title("Matrice de corrélation")

plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "correlation_matrix.png")
)

plt.show()


# ==========================================
# 12. CORRELATION AVEC LA VARIABLE CLASS
# ==========================================

print("\n========== CORRELATION AVEC CLASS ==========\n")

class_correlation = correlation_matrix["Class"].drop("Class")

class_correlation = class_correlation.sort_values(
    key=abs,
    ascending=False
)

print(class_correlation)


# ==========================================
# 13. DISTRIBUTION DE LA VARIABLE AMOUNT
# ==========================================

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="Amount",
    bins=50,
    kde=True
)

plt.title("Distribution du montant des transactions")
plt.xlabel("Montant")
plt.ylabel("Fréquence")

plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "amount_distribution.png")
)

plt.show()


# ==========================================
# 14. COMPARAISON DES MONTANTS
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Class",
    y="Amount"
)

plt.title("Montants : transactions normales vs fraudes")
plt.xlabel("Classe (0 = Normal, 1 = Fraude)")
plt.ylabel("Montant")

plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "amount_by_class.png")
)

plt.show()


# ==========================================
# 15. CONCLUSION
# ==========================================

print("\n========== RESUME DE L'EXPLORATION ==========\n")

print("Nombre total de transactions :", len(df))
print("Nombre de variables :", df.shape[1])

print("Transactions normales :", class_counts[0])
print("Transactions frauduleuses :", class_counts[1])

print("Valeurs manquantes :", missing_values.sum())
print("Doublons :", duplicates)

print("\nToutes les visualisations sont enregistrées dans :")
print(RESULTS_DIR)

print("\n========== FIN DE L'EXPLORATION ==========\n")
