# app.py
# Streamlit — Dashboard + Prédicteur interactif (What-If) Facebook Ads
# -> ML prédit impressions/clicks/conversions à partir de spend + âge + genre
# -> calcule CTR, CPC, CPA, CPM sur la base des prédictions
#
# Lancer : streamlit run app.py
#
# Données attendues (au minimum) :
# age_group, gender, impressions, clicks, ad_spend (ou spent), approved_conversions (ou approved_conversion)
# + idéalement une colonne date (optionnelle)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

# -------------------------
# CONFIG PAGE
# -------------------------
st.set_page_config(
    page_title="Facebook Ads — Dashboard & Prédicteur What-If",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main > div {padding-top: 1.5rem;}
    h1 {font-weight: 800; margin-bottom: 0.3rem;}
    h2 {font-weight: 700; margin-top: 1.2rem;}
    .stPlotlyChart {width: 100% !important;}
</style>
""", unsafe_allow_html=True)


# -------------------------
# HELPERS
# -------------------------
def safe_div(n, d):
    if d is None or d == 0 or pd.isna(d):
        return np.nan
    return n / d

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Normaliser les noms selon ton pipeline
    df = df.copy()
    df.columns = df.columns.str.strip()

    rename_map = {
        "age": "age_group",
        "spent": "ad_spend",
        "total_conversion": "total_conversions",
        "approved_conversion": "approved_conversions",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Si approved_conversions absent mais approved_conversion existe déjà
    if "approved_conversions" not in df.columns and "approved_conversion" in df.columns:
        df["approved_conversions"] = df["approved_conversion"]

    # Dates (optionnel)
    for c in ["reporting_start", "reporting_end", "date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Forcer numeric
    for c in ["impressions", "clicks", "ad_spend", "total_conversions", "approved_conversions"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Nettoyage categories
    if "age_group" in df.columns:
        df["age_group"] = df["age_group"].astype(str)
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str)

    return df

def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # KPI ligne
    df["ctr_pct"] = (df["clicks"] / df["impressions"]) * 100
    df["cpc"] = df["ad_spend"] / df["clicks"]
    df["cpa"] = df["ad_spend"] / df["approved_conversions"]
    df["cpm"] = (df["ad_spend"] / df["impressions"]) * 1000

    # Eviter inf
    for c in ["ctr_pct", "cpc", "cpa", "cpm"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df = normalize_columns(df)
    # Colonnes minimales
    required = ["age_group", "gender", "impressions", "clicks", "ad_spend", "approved_conversions"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")
    return df

def build_model(df: pd.DataFrame, target: str):
    # Features: ad_spend + age_group + gender (et éventuellement interests si tu veux plus tard)
    X = df[["ad_spend", "age_group", "gender"]].copy()
    y = df[target].copy()

    # Retirer lignes sans target
    mask = y.notna() & X["ad_spend"].notna()
    X = X.loc[mask]
    y = y.loc[mask]

    cat_features = ["age_group", "gender"]
    num_features = ["ad_spend"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model)
    ])

    return pipe, X, y

def train_and_score(df: pd.DataFrame, target: str):
    pipe, X, y = build_model(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return pipe, {"MAE": mae, "RMSE": rmse, "R2": r2}

def predict_scenario(models: dict, age_group: str, gender: str, budget_total: float, days: int):
    # On simule une dépense journalière constante
    spend_per_day = safe_div(budget_total, days)
    spend_per_day = 0 if pd.isna(spend_per_day) else float(spend_per_day)

    X_one = pd.DataFrame([{
        "ad_spend": spend_per_day,
        "age_group": age_group,
        "gender": gender
    }])

    # prédictions "par jour"
    impr_day = max(0.0, float(models["impressions"].predict(X_one)[0]))
    clicks_day = max(0.0, float(models["clicks"].predict(X_one)[0]))
    conv_day = max(0.0, float(models["approved_conversions"].predict(X_one)[0]))

    # sur la période
    impr = impr_day * days
    clicks = clicks_day * days
    conv = conv_day * days

    # KPI sur base des prédictions (et budget total)
    ctr_pct = safe_div(clicks, impr) * 100 if impr else np.nan
    cpc = safe_div(budget_total, clicks)
    cpa = safe_div(budget_total, conv)
    cpm = safe_div(budget_total, impr) * 1000 if impr else np.nan

    return {
        "days": days,
        "budget_total": budget_total,
        "spend_per_day": spend_per_day,
        "impressions": impr,
        "clicks": clicks,
        "approved_conversions": conv,
        "ctr_pct": ctr_pct,
        "cpc": cpc,
        "cpa": cpa,
        "cpm": cpm
    }


# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("Navigation")
pages = [
    "Vue d'ensemble",
    "Analyse Exploratoire",
    "Corrélations",
    "Modélisation Prédictive",
    "Prédicteur Interactif"
]
page = st.sidebar.radio("Section", pages)

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Importer ton CSV Facebook Ads", type=["csv"])

if uploaded is None:
    st.info("⬅️ Importe ton fichier CSV dans la barre latérale pour commencer.")
    st.stop()

try:
    df = load_csv(uploaded)
except Exception as e:
    st.error(f"Erreur: {e}")
    st.stop()

df_kpi = add_kpis(df)

# Listes de filtres
age_options = sorted(df_kpi["age_group"].dropna().unique().tolist())
gender_options = sorted(df_kpi["gender"].dropna().unique().tolist())


# ========================
# 1) Vue d'ensemble
# ========================
if page == "Vue d'ensemble":
    st.title("Facebook Ads — Analyse & KPI")
    st.markdown("### Vue générale des performances (impressions, clics, dépenses, conversions)")

    col1, col2, col3, col4 = st.columns(4)

    total_impr = df_kpi["impressions"].sum()
    total_clicks = df_kpi["clicks"].sum()
    total_spend = df_kpi["ad_spend"].sum()
    total_conv = df_kpi["approved_conversions"].sum()

    with col1:
        st.metric("Impressions", f"{total_impr:,.0f}")
    with col2:
        st.metric("Clics", f"{total_clicks:,.0f}")
    with col3:
        st.metric("Dépenses (€)", f"{total_spend:,.2f}")
    with col4:
        st.metric("Conversions approuvées", f"{total_conv:,.0f}")

    st.markdown("---")

    k1, k2, k3, k4 = st.columns(4)
    ctr = safe_div(total_clicks, total_impr) * 100 if total_impr else np.nan
    cpc = safe_div(total_spend, total_clicks)
    cpa = safe_div(total_spend, total_conv)
    cpm = safe_div(total_spend, total_impr) * 1000 if total_impr else np.nan

    with k1:
        st.metric("CTR global (%)", f"{ctr:.2f}" if pd.notna(ctr) else "—")
    with k2:
        st.metric("CPC global (€)", f"{cpc:.2f}" if pd.notna(cpc) else "—")
    with k3:
        st.metric("CPA global (€)", f"{cpa:.2f}" if pd.notna(cpa) else "—")
    with k4:
        st.metric("CPM global (€)", f"{cpm:.2f}" if pd.notna(cpm) else "—")

    st.markdown("---")
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Aperçu des données (avec KPI)")
        st.dataframe(df_kpi.head(50), use_container_width=True)

    with right:
        st.subheader("Qualité")
        st.write("Lignes × Colonnes :", df_kpi.shape)
        st.write("Valeurs manquantes (top 10) :")
        st.dataframe(df_kpi.isna().sum().sort_values(ascending=False).head(10).to_frame("missing"))

# ========================
# 2) Analyse Exploratoire
# ========================
elif page == "Analyse Exploratoire":
    st.header("Analyse Exploratoire")

    numeric_cols = ["impressions", "clicks", "ad_spend", "approved_conversions", "ctr_pct", "cpc", "cpa", "cpm"]
    numeric_cols = [c for c in numeric_cols if c in df_kpi.columns]

    selected = st.selectbox("Variable à explorer", numeric_cols)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df_kpi, x=selected, nbins=40, title=f"Distribution — {selected}")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Statistiques descriptives")
        st.dataframe(df_kpi[selected].describe().to_frame().T)

        st.markdown("**Top 10 (valeurs les plus élevées)**")
        top10 = df_kpi.nlargest(10, selected)[["age_group","gender",selected,"ad_spend","clicks","impressions","approved_conversions"]].copy()
        top10 = top10.loc[:, ~top10.columns.duplicated()]
        st.dataframe(top10, use_container_width=True)

    st.markdown("---")
    st.subheader("Comparaison par tranche d’âge (moyennes)")
    by_age = (df_kpi.groupby("age_group")
                    .agg(ctr_pct=("ctr_pct","mean"),
                         cpc=("cpc","mean"),
                         cpa=("cpa","mean"),
                         cpm=("cpm","mean"))
                    .reset_index())
    fig2 = px.bar(by_age, x="age_group", y="cpa", title="CPA moyen (€) par tranche d’âge")
    st.plotly_chart(fig2, use_container_width=True)

# ========================
# 3) Corrélations
# ========================
elif page == "Corrélations":
    st.header("Corrélations (numériques)")

    corr_cols = ["impressions", "clicks", "ad_spend", "approved_conversions", "ctr_pct", "cpc", "cpa", "cpm"]
    corr_cols = [c for c in corr_cols if c in df_kpi.columns]
    corr = df_kpi[corr_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        zmid=0,
        text=corr.round(2).values,
        texttemplate="%{text}",
        hovertemplate="%{x} vs %{y}: <b>%{text}</b><extra></extra>"
    ))
    fig.update_layout(height=600, title="Matrice de corrélation")
    st.plotly_chart(fig, use_container_width=True)

    st.info("Astuce BI : regarde surtout les liens Spend → Impressions/Clicks/Conversions, puis l'impact sur CPA.")

# ========================
# 4) Modélisation Prédictive
# ========================
elif page == "Modélisation Prédictive":
    st.header("Modélisation Prédictive (ML)")

    st.markdown("""
On entraîne 3 modèles (Random Forest) pour prédire **par jour** :
- impressions à partir de spend + âge + genre
- clicks à partir de spend + âge + genre
- approved_conversions à partir de spend + âge + genre
""")

    # Données d'entraînement : on garde uniquement les lignes utiles
    df_model = df.copy()
    df_model = df_model.dropna(subset=["ad_spend", "age_group", "gender"])

    # Entraîner 3 modèles
    with st.spinner("Entraînement des modèles..."):
        model_impr, score_impr = train_and_score(df_model, "impressions")
        model_clicks, score_clicks = train_and_score(df_model, "clicks")
        model_conv, score_conv = train_and_score(df_model, "approved_conversions")

    res = pd.DataFrame({
        "impressions": score_impr,
        "clicks": score_clicks,
        "approved_conversions": score_conv
    }).T

    st.subheader("Scores (sur jeu de test)")
    st.dataframe(res.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","R2":"{:.3f}"}), use_container_width=True)

    st.markdown("---")
    st.subheader("Importance (approx.) — Spend est souvent dominant")
    st.caption("Pour RandomForest dans pipeline, l’importance directe est moins simple à sortir proprement sans extraire l’encodeur. "
               "Mais en général, le budget et les segments (âge/genre) expliquent beaucoup la variance.")

    st.success("✅ Modèles entraînés. Va dans 'Prédicteur Interactif' pour simuler un scénario.")

# ========================
# 5) Prédicteur Interactif
# ========================
else:
    st.header("🎛️ Prédicteur Interactif (What-If) — CTR / CPC / CPA / CPM")

    st.markdown("Tu modifies **jours / budget / âge / genre** → le modèle prédit **impressions, clics, conversions** → puis on calcule les KPI.")

    # Entraîner modèles sur tout (pour meilleure précision en prod)
    df_model = df.dropna(subset=["ad_spend", "age_group", "gender"]).copy()

    with st.spinner("Préparation du prédicteur..."):
        m_impr, _ = train_and_score(df_model, "impressions")
        m_clicks, _ = train_and_score(df_model, "clicks")
        m_conv, _ = train_and_score(df_model, "approved_conversions")

    models = {
        "impressions": m_impr,
        "clicks": m_clicks,
        "approved_conversions": m_conv
    }

    # UI entrées
    colA, colB, colC, colD = st.columns(4)

    with colA:
        days = st.slider("Nombre de jours", min_value=1, max_value=60, value=14, step=1)

    with colB:
        # bornes budget en fonction du dataset
        bmin = float(max(0.0, df_kpi["ad_spend"].min()))
        bmax = float(max(10.0, df_kpi["ad_spend"].quantile(0.99) * 30))  # budget total possible
        budget = st.slider("Budget total (€)", min_value=0.0, max_value=float(bmax), value=float(min(300.0, bmax)), step=10.0)

    with colC:
        age_choice = st.selectbox("Tranche d’âge", options=age_options, index=0)

    with colD:
        gender_choice = st.selectbox("Genre", options=gender_options, index=0)

    # prédiction
    sim = predict_scenario(models, age_choice, gender_choice, budget, days)

    st.markdown("---")
    st.subheader("✅ Résultats prédits (sur la période)")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("CTR (%)", f"{sim['ctr_pct']:.2f}" if pd.notna(sim["ctr_pct"]) else "—")
    k2.metric("CPC (€)", f"{sim['cpc']:.2f}" if pd.notna(sim["cpc"]) else "—")
    k3.metric("CPA (€)", f"{sim['cpa']:.2f}" if pd.notna(sim["cpa"]) else "—")
    k4.metric("CPM (€)", f"{sim['cpm']:.2f}" if pd.notna(sim["cpm"]) else "—")

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Impressions", f"{sim['impressions']:.0f}")
    v2.metric("Clics", f"{sim['clicks']:.0f}")
    v3.metric("Conversions approuvées", f"{sim['approved_conversions']:.0f}")
    v4.metric("Budget / jour", f"{sim['spend_per_day']:.2f} €")

    st.markdown("---")
    st.subheader("📈 Visualisation (projection cumulée)")
    days_range = np.arange(1, days + 1)
    proj = pd.DataFrame({
        "Jour": days_range,
        "Dépense cumulée (€)": (budget / days) * days_range if days else 0,
        "Impressions cumulées": (sim["impressions"] / days) * days_range if days else 0,
        "Clics cumulés": (sim["clicks"] / days) * days_range if days else 0,
        "Conversions cumulées": (sim["approved_conversions"] / days) * days_range if days else 0
    })

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.line(proj, x="Jour", y=["Impressions cumulées", "Clics cumulés"], title="Impressions & Clics cumulés")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.line(proj, x="Jour", y=["Dépense cumulée (€)", "Conversions cumulées"], title="Dépense & Conversions cumulées")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("🧾 Référence historique (segment sélectionné)")
    seg_hist = df_kpi[(df_kpi["age_group"] == age_choice) & (df_kpi["gender"] == gender_choice)].copy()
    if len(seg_hist) == 0:
        st.info("Pas de lignes historiques exactes pour ce segment. Le modèle général fait quand même la prédiction.")
    else:
        h_impr = seg_hist["impressions"].sum()
        h_clicks = seg_hist["clicks"].sum()
        h_spend = seg_hist["ad_spend"].sum()
        h_conv = seg_hist["approved_conversions"].sum()

        h_ctr = safe_div(h_clicks, h_impr) * 100 if h_impr else np.nan
        h_cpc = safe_div(h_spend, h_clicks)
        h_cpa = safe_div(h_spend, h_conv)
        h_cpm = safe_div(h_spend, h_impr) * 1000 if h_impr else np.nan

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("CTR hist (%)", f"{h_ctr:.2f}" if pd.notna(h_ctr) else "—")
        a2.metric("CPC hist (€)", f"{h_cpc:.2f}" if pd.notna(h_cpc) else "—")
        a3.metric("CPA hist (€)", f"{h_cpa:.2f}" if pd.notna(h_cpa) else "—")
        a4.metric("CPM hist (€)", f"{h_cpm:.2f}" if pd.notna(h_cpm) else "—")

    st.caption(
        "⚠️  **simulation prédictive (What-If)** basée sur l’historique : "
        "le modèle apprend la relation entre budget (ad_spend) et résultats (impressions/clics/conversions) "
        "en tenant compte de l’âge et du genre."
    )
