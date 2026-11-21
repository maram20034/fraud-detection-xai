
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SHAP_DIR = RESULTS_DIR / "shap"
LIME_DIR = RESULTS_DIR / "lime"

st.set_page_config(
    page_title="Fraud Detection — XAI Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Fraud Detection Dashboard")
st.caption(
    "Comparaison des modèles, analyse des erreurs "
    "et explicabilité SHAP / LIME"
)

# ============================================================
# CHARGEMENT DES DONNEES ET MODELES
# ============================================================

@st.cache_data(show_spinner="Chargement des données de test...")
def load_data():
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0].astype(int)

    ae_predictions = pd.read_csv(
        RESULTS_DIR / "autoencoder" / "predictions.csv"
    )

    if len(X_test) != len(y_test) or len(X_test) != len(ae_predictions):
        raise ValueError(
            "Les fichiers de test et les prédictions "
            "Autoencoder n'ont pas la même taille."
        )

    return X_test, y_test, ae_predictions


@st.cache_resource(show_spinner="Chargement des modèles entraînés...")
def load_models():

    mlp = tf.keras.models.load_model(
        MODEL_DIR / "mlp_model.keras",
        compile=False
    )

    rf = joblib.load(
        MODEL_DIR / "random_forest_model.joblib"
    )

    scaler = joblib.load(
        MODEL_DIR / "autoencoder_scaler.pkl"
    )

    autoencoder = tf.keras.models.load_model(
        MODEL_DIR / "autoencoder.keras",
        compile=False
    )

    thresholds = {
        "MLP": float(
            json.loads(
                (MODEL_DIR / "mlp_threshold.json").read_text()
            )["threshold"]
        ),

        "Random Forest": float(
            json.loads(
                (MODEL_DIR / "random_forest_threshold.json").read_text()
            )["threshold"]
        ),

        "Autoencoder": float(
            json.loads(
                (MODEL_DIR / "autoencoder_threshold.json").read_text()
            )["threshold"]
        ),
    }

    return mlp, rf, autoencoder, scaler, thresholds


@st.cache_data(
    show_spinner="Calcul des prédictions des modèles sur le jeu de test..."
)
def calculate_predictions(_X_test, _y_test, _ae_predictions):

    mlp, rf, autoencoder, scaler, thresholds = load_models()

    X = _X_test.to_numpy(dtype=np.float32)

    mlp_scores = mlp.predict(
        X,
        batch_size=2048,
        verbose=0
    ).reshape(-1)

    rf_scores = rf.predict_proba(_X_test)[:, 1]

    # Erreurs de reconstruction sauvegardées
    # lors de l'évaluation officielle de l'Autoencoder.

    ae_scores = _ae_predictions[
        "Reconstruction_Error"
    ].to_numpy(dtype=float)

    scores = {
        "MLP": mlp_scores,
        "Random Forest": rf_scores,
        "Autoencoder": ae_scores,
    }

    predictions = {
        "MLP": (
            mlp_scores >= thresholds["MLP"]
        ).astype(int),

        "Random Forest": (
            rf_scores >= thresholds["Random Forest"]
        ).astype(int),

        "Autoencoder": (
            ae_scores > thresholds["Autoencoder"]
        ).astype(int),
    }

    return scores, predictions, thresholds


try:

    X_test, y_test, ae_predictions = load_data()

    scores, predictions, thresholds = calculate_predictions(
        X_test,
        y_test,
        ae_predictions
    )

except Exception as exc:

    st.error(f"Impossible de charger le projet : {exc}")

    st.info(
        "Vérifie que dashboard/app.py se trouve dans le dossier "
        "du projet et que les dossiers data/, models/ et results/ "
        "sont présents."
    )

    st.stop()


# ============================================================
# METRIQUES
# ============================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_score):

    return {
        "Accuracy": accuracy_score(y_true, y_pred),

        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),

        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),

        "F1-score": f1_score(
            y_true, y_pred, zero_division=0
        ),

        "PR-AUC": average_precision_score(
            y_true, y_score
        ),
    }


metrics = {
    model_name: compute_metrics(
        y_test,
        predictions[model_name],
        scores[model_name]
    )
    for model_name in predictions
}

metrics_df = pd.DataFrame(metrics).T
metrics_df.index.name = "Modèle"


# ============================================================
# NAVIGATION
# ============================================================

tab_compare, tab_errors, tab_xai, tab_transaction, tab_about = st.tabs([
    "📊 Comparaison",
    "🔎 Analyse des erreurs",
    "🧠 SHAP & LIME",
    "💳 Transaction individuelle",
    "ℹ️ À propos"
])


# ============================================================
# ONGLET 1 : COMPARAISON
# ============================================================

with tab_compare:

    st.subheader("Comparaison des trois modèles principaux")

    st.caption(
        "Les métriques sont recalculées sur le même jeu de test "
        "à partir des modèles sauvegardés et de leurs seuils enregistrés."
    )

    cols = st.columns(3)

    for col, model_name in zip(
        cols,
        ["MLP", "Random Forest", "Autoencoder"]
    ):

        with col:

            st.markdown(f"**{model_name}**")

            st.metric(
                "F1-score",
                f"{metrics[model_name]['F1-score']:.4f}"
            )

            st.metric(
                "PR-AUC",
                f"{metrics[model_name]['PR-AUC']:.4f}"
            )

            st.caption(
                f"Seuil : {thresholds[model_name]:.6g}"
            )

    st.markdown("#### Tableau comparatif")

    st.dataframe(
        metrics_df.style.format("{:.4f}"),
        use_container_width=True
    )

    metric_choice = st.selectbox(
        "Métrique à visualiser",
        [
            "Precision",
            "Recall",
            "F1-score",
            "PR-AUC",
            "Accuracy"
        ]
    )

    chart_data = metrics_df.reset_index().melt(
        id_vars="Modèle",
        var_name="Métrique",
        value_name="Valeur"
    )

    chart_data = chart_data[
        chart_data["Métrique"] == metric_choice
    ]

    fig = px.bar(
        chart_data,
        x="Modèle",
        y="Valeur",
        text_auto=".4f",
        range_y=[0, 1],
        title=f"Comparaison — {metric_choice}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("#### Matrices de confusion")

    cm_cols = st.columns(3)

    for col, model_name in zip(
        cm_cols,
        ["MLP", "Random Forest", "Autoencoder"]
    ):

        cm = confusion_matrix(
            y_test,
            predictions[model_name],
            labels=[0, 1]
        )

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues",
            x=["Prédit normal", "Prédit fraude"],
            y=["Réel normal", "Réel fraude"],
            title=model_name
        )

        fig_cm.update_layout(
            coloraxis_showscale=False,
            height=340
        )

        col.plotly_chart(
            fig_cm,
            use_container_width=True
        )

# ============================================================
# ONGLET 2 : ANALYSE DES ERREURS
# ============================================================

with tab_errors:

    st.subheader("Faux positifs et faux négatifs")

    selected_model = st.selectbox(
        "Choisir un modèle",
        list(predictions.keys()),
        key="error_model"
    )

    y_pred = predictions[selected_model]
    actual = y_test.to_numpy()

    fp_mask = (actual == 0) & (y_pred == 1)
    fn_mask = (actual == 1) & (y_pred == 0)

    tp_mask = (actual == 1) & (y_pred == 1)
    tn_mask = (actual == 0) & (y_pred == 0)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Vrais positifs (TP)", int(tp_mask.sum()))
    c2.metric("Faux positifs (FP)", int(fp_mask.sum()))
    c3.metric("Faux négatifs (FN)", int(fn_mask.sum()))
    c4.metric("Vrais négatifs (TN)", int(tn_mask.sum()))

    st.warning(
        "Un faux positif correspond à une transaction normale "
        "signalée comme fraude. Un faux négatif correspond à une "
        "fraude que le modèle n'a pas détectée."
    )

    error_type = st.radio(
        "Type d'erreur à examiner",
        ["Faux positifs", "Faux négatifs"],
        horizontal=True
    )

    mask = (
        fp_mask
        if error_type == "Faux positifs"
        else fn_mask
    )

    error_indices = np.flatnonzero(mask)

    st.write(
        f"**Nombre de {error_type.lower()} : {len(error_indices)}**"
    )

    if len(error_indices):

        error_table = X_test.iloc[error_indices].copy()

        error_table.insert(
            0, "Index_test", error_indices
        )

        error_table.insert(
            1, "Classe_réelle", actual[error_indices]
        )

        error_table.insert(
            2, "Classe_prédite", y_pred[error_indices]
        )

        error_table.insert(
            3,
            "Score_modèle",
            scores[selected_model][error_indices]
        )

        st.dataframe(
            error_table.head(200),
            use_container_width=True
        )

        st.download_button(
            "Télécharger les erreurs (CSV)",
            error_table.to_csv(index=False).encode("utf-8"),
            file_name=(
                f"{selected_model.lower().replace(' ', '_')}_"
                f"{error_type.lower().replace(' ', '_')}.csv"
            ),
            mime="text/csv"
        )

    else:

        st.success(
            "Aucune erreur de ce type pour ce modèle "
            "avec le seuil sélectionné."
        )

    st.markdown("#### Comparaison du nombre d'erreurs")

    error_comparison = []

    for name in predictions:

        pred = predictions[name]

        error_comparison.append({

            "Modèle": name,

            "Faux positifs": int(
                ((actual == 0) & (pred == 1)).sum()
            ),

            "Faux négatifs": int(
                ((actual == 1) & (pred == 0)).sum()
            ),

        })

    error_df = pd.DataFrame(
        error_comparison
    ).melt(
        id_vars="Modèle",
        var_name="Type d'erreur",
        value_name="Nombre"
    )

    st.plotly_chart(
        px.bar(
            error_df,
            x="Modèle",
            y="Nombre",
            color="Type d'erreur",
            barmode="group",
            text_auto=True
        ),
        use_container_width=True
    )

    st.markdown("#### Effet du seuil de décision")

    if selected_model == "Autoencoder":

        st.caption(
            "Pour l'Autoencoder, un score élevé d'erreur "
            "de reconstruction indique une anomalie."
        )

    else:

        st.caption(
            "Déplacer le seuil permet d'observer le compromis "
            "entre les fraudes détectées et les fausses alertes. "
            "Ce réglage est exploratoire et ne modifie pas "
            "le seuil sauvegardé."
        )

    default_threshold = float(
        thresholds[selected_model]
    )

    if selected_model == "Autoencoder":

        low, high = 0.0, max(
            float(np.nanmax(scores[selected_model])),
            default_threshold * 1.5
        )

    else:

        low, high = 0.0, 1.0

    exploratory_threshold = st.slider(
        "Seuil exploratoire",
        min_value=float(low),
        max_value=float(high),
        value=float(
            np.clip(default_threshold, low, high)
        ),
        key="threshold_slider"
    )

    if selected_model == "Autoencoder":

        exploratory_pred = (
            scores[selected_model] > exploratory_threshold
        ).astype(int)

    else:

        exploratory_pred = (
            scores[selected_model] >= exploratory_threshold
        ).astype(int)

    st.write({

        "Precision": round(
            precision_score(
                actual,
                exploratory_pred,
                zero_division=0
            ), 4
        ),

        "Recall": round(
            recall_score(
                actual,
                exploratory_pred,
                zero_division=0
            ), 4
        ),

        "F1-score": round(
            f1_score(
                actual,
                exploratory_pred,
                zero_division=0
            ), 4
        ),

        "Faux positifs": int(
            ((actual == 0) & (exploratory_pred == 1)).sum()
        ),

        "Faux négatifs": int(
            ((actual == 1) & (exploratory_pred == 0)).sum()
        ),

    })

# ============================================================
# ONGLET 3 : EXPLICABILITE
# ============================================================

with tab_xai:

    st.subheader("Explicabilité globale et locale")

    xai_model = st.selectbox(
        "Modèle à expliquer",
        ["MLP", "Random Forest", "Autoencoder"],
        key="xai_model"
    )

    shap_files = {

        "MLP": (
            "mlp_shap_bar.png",
            "mlp_shap_summary.png",
            "mlp_shap_waterfall_fraud.png",
            "mlp_shap_waterfall_normal.png"
        ),

        "Random Forest": (
            "random_forest_shap_bar.png",
            "random_forest_shap_summary.png",
            "random_forest_shap_waterfall_fraud.png",
            "random_forest_shap_waterfall_normal.png"
        ),

        "Autoencoder": (
            "autoencoder_shap_bar.png",
            "autoencoder_shap_summary.png",
            "autoencoder_shap_waterfall_fraud.png",
            "autoencoder_shap_waterfall_normal.png"
        ),

    }

    bar_file, summary_file, fraud_file, normal_file = (
        shap_files[xai_model]
    )

    st.markdown(
        "#### SHAP — importance globale des variables"
    )

    c1, c2 = st.columns(2)

    with c1:

        p = FIGURES_DIR / bar_file

        if p.exists():

            st.image(
                str(p),
                caption=f"Importance SHAP — {xai_model}",
                use_container_width=True
            )

        else:

            st.info("Graphique SHAP global non disponible.")

    with c2:

        p = FIGURES_DIR / summary_file

        if p.exists():

            st.image(
                str(p),
                caption=f"SHAP summary — {xai_model}",
                use_container_width=True
            )

        else:

            st.info("SHAP summary non disponible.")

    st.markdown(
        "#### SHAP — exemples d'explications locales pré-calculées"
    )

    c1, c2 = st.columns(2)

    with c1:

        p = FIGURES_DIR / fraud_file

        if p.exists():

            st.image(
                str(p),
                caption="Exemple de transaction frauduleuse",
                use_container_width=True
            )

    with c2:

        p = FIGURES_DIR / normal_file

        if p.exists():

            st.image(
                str(p),
                caption="Exemple de transaction normale",
                use_container_width=True
            )

    st.markdown(
        "#### LIME — exemples d'explications locales"
    )

    lime_df_path = LIME_DIR / "lime_all_explanations.csv"

    if lime_df_path.exists():

        lime_df = pd.read_csv(lime_df_path)

        lime_model_key = {
            "MLP": "mlp",
            "Random Forest": "random_forest",
            "Autoencoder": "autoencoder"
        }[xai_model]

        subset = lime_df[
            lime_df["model"] == lime_model_key
        ]

        if not subset.empty:

            st.dataframe(
                subset,
                use_container_width=True
            )

        else:

            st.info(
                "Aucune explication LIME tabulaire trouvée "
                "pour ce modèle."
            )

    for transaction_kind, caption in [
        ("fraud", "Exemple LIME — fraude"),
        ("normal", "Exemple LIME — transaction normale")
    ]:

        html_path = (
            LIME_DIR
            / f"{lime_model_key}_lime_{transaction_kind}.html"
        )

        if html_path.exists():

            with st.expander(caption):

                st.components.v1.html(
                    html_path.read_text(encoding="utf-8"),
                    height=520,
                    scrolling=True
                )

    st.info(
        "Important : les graphiques SHAP et les explications "
        "LIME affichés ici sont les exemples pré-calculés présents "
        "dans l'archive. Ils ne sont pas recalculés automatiquement "
        "pour chaque transaction sélectionnée."
    )


# ============================================================
# ONGLET 4 : TRANSACTION INDIVIDUELLE
# ============================================================

with tab_transaction:

    st.subheader(
        "Analyser une transaction du jeu de test"
    )

    max_index = len(X_test) - 1

    transaction_index = st.number_input(
        "Index de la transaction dans X_test",
        min_value=0,
        max_value=max_index,
        value=0,
        step=1
    )

    idx = int(transaction_index)

    actual_class = int(y_test.iloc[idx])

    st.write(
        f"**Classe réelle :** "
        f"{'Fraude' if actual_class == 1 else 'Normale'}"
    )

    transaction = X_test.iloc[[idx]]

    st.markdown(
        "#### Résultats des trois modèles"
    )

    result_rows = []

    for name in [
        "MLP",
        "Random Forest",
        "Autoencoder"
    ]:

        score = float(scores[name][idx])

        pred = int(predictions[name][idx])

        result_rows.append({

            "Modèle": name,

            "Score": score,

            "Seuil": thresholds[name],

            "Prédiction": (
                "Fraude" if pred else "Normale"
            ),

            "Résultat": (
                "Correct"
                if pred == actual_class
                else "Erreur"
            ),

        })

    st.dataframe(
        pd.DataFrame(result_rows).style.format({
            "Score": "{:.6f}",
            "Seuil": "{:.6f}"
        }),
        use_container_width=True
    )

    if actual_class == 1:

        st.warning(
            "Cette transaction est une fraude réelle."
        )

    else:

        st.info(
            "Cette transaction est une transaction normale."
        )

    st.markdown("#### Valeurs des variables")

    st.dataframe(
        transaction.T.rename(
            columns={idx: "Valeur"}
        ),
        use_container_width=True
    )

    st.markdown(
        "#### Explication locale disponible"
    )

    st.caption(
        "Les explications SHAP/LIME pré-calculées concernent "
        "des exemples choisis lors de l'étape d'explicabilité. "
        "Pour une explication locale de cet index précis, "
        "il faudra lancer le calcul SHAP/LIME correspondant."
    )


# ============================================================
# ONGLET 5 : INFORMATIONS
# ============================================================

with tab_about:

    st.subheader("Périmètre du dashboard")

    st.markdown(
        """
        - **MLP** : réseau de neurones entraîné avec class weights.
        - **Random Forest** : modèle de forêt aléatoire.
        - **Autoencoder** : détection d'anomalies par erreur de reconstruction.
        - **SHAP** : contribution des variables aux prédictions / scores
          selon les fichiers d'explication du projet.
        - **LIME** : explications locales sauvegardées pour des
          transactions exemples.

        Les métriques de comparaison sont recalculées sur le jeu de test
        commun. Le seuil de chaque modèle est lu depuis son fichier JSON.
        Les seuils exploratoires dans l'onglet d'erreurs ne changent pas
        les fichiers du projet.
        """
    )

    st.write(
        f"Nombre de transactions de test : **{len(X_test):,}**"
    )

    st.write(
        f"Nombre de fraudes réelles : **{int(y_test.sum()):,}**"
    )

    st.write(
        f"Nombre de transactions normales : "
        f"**{int((y_test == 0).sum()):,}**"
    )

    st.caption(
        "Les résultats dépendent des modèles, seuils "
        "et fichiers présents dans cette archive."
    )
