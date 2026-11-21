
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# ==========================================
# 1. CONFIGURATION DES CHEMINS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ==========================================
# 2. CHARGEMENT DU DATASET
# ==========================================

print("\n========== CHARGEMENT DES DONNEES ==========\n")

df = pd.read_csv(DATA_PATH)

print("Dimensions initiales :", df.shape)


# ==========================================
# 3. SUPPRESSION DES DOUBLONS
# ==========================================

print("\n========== SUPPRESSION DES DOUBLONS ==========\n")

duplicates_before = df.duplicated().sum()

print("Nombre de doublons :", duplicates_before)

df = df.drop_duplicates().copy()

print("Dimensions après suppression :", df.shape)
print("Nombre de doublons restants :", df.duplicated().sum())


# ==========================================
# 4. SEPARATION DE X ET Y
# ==========================================

print("\n========== SEPARATION DES VARIABLES ==========\n")

X = df.drop(columns=["Class"])
y = df["Class"]

print("Dimensions de X :", X.shape)
print("Dimensions de y :", y.shape)


# ==========================================
# 5. DIVISION TRAIN / VALIDATION / TEST
# ==========================================

print("\n========== DIVISION DES DONNEES ==========\n")

# Première division : 70% Training et 30% temporaire

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Deuxième division : 15% Validation et 15% Testing

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Training :", X_train.shape)
print("Validation :", X_val.shape)
print("Testing :", X_test.shape)


# ==========================================
# 6. VERIFICATION DE LA DISTRIBUTION
# ==========================================

print("\n========== DISTRIBUTION DES CLASSES ==========\n")

print("Training :")
print(y_train.value_counts())
print("\nPourcentages :")
print(y_train.value_counts(normalize=True) * 100)

print("\nValidation :")
print(y_val.value_counts())
print("\nPourcentages :")
print(y_val.value_counts(normalize=True) * 100)

print("\nTesting :")
print(y_test.value_counts())
print("\nPourcentages :")
print(y_test.value_counts(normalize=True) * 100)


# ==========================================
# 7. NORMALISATION DES DONNEES
# ==========================================

print("\n========== NORMALISATION ==========\n")

scaler = StandardScaler()

# Apprentissage du scaler uniquement sur Training
X_train_scaled = scaler.fit_transform(X_train)

# Transformation de Validation et Testing
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Conversion en DataFrames pour conserver les noms des colonnes

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X.columns
)

X_val_scaled = pd.DataFrame(
    X_val_scaled,
    columns=X.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X.columns
)

print("Normalisation terminée !")


# ==========================================
# 8. ENREGISTREMENT DES DONNEES
# ==========================================

print("\n========== ENREGISTREMENT ==========\n")

# Sauvegarde des variables explicatives

X_train_scaled.to_csv(
    os.path.join(PROCESSED_DIR, "X_train.csv"),
    index=False
)

X_val_scaled.to_csv(
    os.path.join(PROCESSED_DIR, "X_val.csv"),
    index=False
)

X_test_scaled.to_csv(
    os.path.join(PROCESSED_DIR, "X_test.csv"),
    index=False
)

# Sauvegarde des variables cibles

y_train.to_csv(
    os.path.join(PROCESSED_DIR, "y_train.csv"),
    index=False
)

y_val.to_csv(
    os.path.join(PROCESSED_DIR, "y_val.csv"),
    index=False
)

y_test.to_csv(
    os.path.join(PROCESSED_DIR, "y_test.csv"),
    index=False
)

# Sauvegarde du scaler pour les futures prédictions

joblib.dump(
    scaler,
    os.path.join(MODELS_DIR, "scaler.pkl")
)


# ==========================================
# 9. RESUME FINAL
# ==========================================

print("\n========== RESUME FINAL ==========\n")

print("Nombre total de transactions :", len(df))
print("Nombre de variables :", X.shape[1])

print("Training :", len(X_train))
print("Validation :", len(X_val))
print("Testing :", len(X_test))

print("\nNombre de fraudes :")
print("Training :", int(y_train.sum()))
print("Validation :", int(y_val.sum()))
print("Testing :", int(y_test.sum()))

print("\nFichiers enregistrés dans :", PROCESSED_DIR)
print("Scaler enregistré dans :", MODELS_DIR)

print("\n========== PREPROCESSING TERMINE ==========\n")
