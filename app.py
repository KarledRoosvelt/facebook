# app.py
# Facebook Ads — Nexus Strategic Dashboard (Full)
# Dataset attendu (min) :
# age_group, gender, impressions, clicks, ad_spend, approved_conversions
# (date optionnelle : date ou reporting_start/reporting_end)

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

# ---------------------------------------------------
# CONFIG PAGE
# ---------------------------------------------------
st.set_page_config(
    page_title="Facebook Ads — Nexus Strategic Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main > div {padding-top: 1.2rem;}
h1 {font-weight: 800; margin-bottom: 0.2rem;}
h2, h3 {font-weight: 700;}
.stPlotlyChart {width: 100% !important;}
.small-note {color:#6b7280; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def safe_div(n, d):
    if d is None or d == 0 or pd.isna(d):
        return np.nan
    return n / d

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Renommages possibles (au cas où)
    rename_map = {
        "age": "age_group",
        "spent": "ad_spend",
        "approved_conversion": "approved_conversions",
        "approved_conversion_count": "approved_conversions",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Numeric coercion
    for c in ["impressions", "clicks", "ad_spend", "approved_conversions"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Date optionnelle
    for c in ["date", "reporting_start", "reporting_end"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # AGE clean : garder uniquement formats "18-24" etc
    if "age_group" in df.columns:
        df["age_group"] = df["age_group"].astype(str).str.strip()
        df = df[df["age_group"].str.contains(r"^\d+\s*-\s*\d+$", regex=True)]

    # GENDER clean : M/F -> Male/Female + filtrer
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str).str.strip()
        df["gender"] = df["gender"].replace({"M": "Male", "F": "Female", "m": "Male", "f": "Female"})
        df = df[df["gender"].isin(["Male", "Female"])]

    return df

def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # KPIs ligne
    df["ctr_pct"] = safe_div(df["clicks"], df["impressions"]) * 100
    df["cpc"] = safe_div(df["ad_spend"], df["clicks"])
    df["cpa"] = safe_div(df["ad_spend"], df["approved_conversions"])
    df["cpm"] = safe_div(df["ad_spend"], df["impressions"]) * 1000
    df["conversion_rate_pct"] = safe_div(df["approved_conversions"], df["clicks"]) * 100

    for c in ["ctr_pct","cpc","cpa","cpm","conversion_rate_pct"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df

@st.cache_data(show_spinner=False)
def load_csv(uploaded) -> pd.DataFrame:
    df = pd.read_csv(uploaded)
    df = normalize_columns(df)

    required = ["age_group","gender","impressions","clicks","ad_spend","approved_conversions"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    # Supprimer lignes complètement cassées
    df = df.dropna(subset=["age_group","gender","ad_spend"])
    return df

# ---------------------------------------------------
# MACHINE LEARNING
# ---------------------------------------------------
def build_model(df: pd.DataFrame, target: str):
    X = df[["ad_spend", "age_group", "gender"]].copy()
    y = df[target].copy()

    # retirer target NaN
    mask = X["ad_spend"].notna() & y.notna()
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
        n_estimators=350,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    return pipe, X, y

def train_and_score(df: pd.DataFrame, target: str):
    pipe, X, y = build_model(df, target)

    # Si dataset trop petit
    if len(X) < 20:
        pipe.fit(X, y)
        return pipe, {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    return pipe, {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds)
    }

@st.cache_resource(show_spinner=False)
def train_all_models(df_model: pd.DataFrame):
    # Entraîner 3 modèles (cache pour éviter de retrain à chaque slider)
    m_impr, s_impr = train_and_score(df_model, "impressions")
    m_clicks, s_clicks = train_and_score(df_model, "clicks")
    m_conv, s_conv = train_and_score(df_model, "approved_conversions")
    models = {"impressions": m_impr, "clicks": m_clicks, "approved_conversions": m_conv}
    scores = {"impressions": s_impr, "clicks": s_clicks, "approved_conversions": s_conv}
    return models, scores

def predict_scenario(models: dict, age_group: str, gender: str, budget_total: float, days: int):
    spend_day = safe_div(budget_total, days)
    spend_day = 0 if pd.isna(spend_day) else float(spend_day)

    X_one = pd.DataFrame([{"ad_spend": spend_day, "age_group": age_group, "gender": gender}])

    impr_day = max(0.0, float(models["impressions"].predict(X_one)[0]))
    clicks_day = max(0.0, float(models["clicks"].predict(X_one)[0]))
    conv_day = max(0.0, float(models["approved_conversions"].predict(X_one)[0]))

    impr = impr_day * days
    clicks = clicks_day * days
    conv = conv_day * days

    ctr = safe_div(clicks, impr) * 100 if impr else np.nan
    cpc = safe_div(budget_total, clicks)
    cpa = safe_div(budget_total, conv)
    cpm = safe_div(budget_total, impr) * 1000 if impr else np.nan

    return {
        "days": days,
        "budget_total": budget_total,
        "spend_per_day": spend_day,
        "impressions": impr,
        "clicks": clicks,
        "approved_conversions": conv,
        "ctr_pct": ctr,
        "cpc": cpc,
        "cpa": cpa,
        "cpm": cpm
    }

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.title("Navigation")
pages = [
    "Vue d'ensemble",
    "Analyse Exploratoire",
    "Corrélations",
    "Modélisation Prédictive",
    "Prédicteur Interactif"
]

page = st.sidebar.radio("Section", pages, key="nav_radio_unique_001")
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

# options dropdown propres
age_options = sorted(df_kpi["age_group"].dropna().unique().tolist())
gender_options = sorted(df_kpi["gender"].dropna().unique().tolist())

# Filtres globaux (Nexus style)
st.sidebar.markdown("### Filtres")
age_filter = st.sidebar.multiselect("Tranches d'âge", age_options, default=age_options, key="age_filter_ms")
gender_filter = st.sidebar.multiselect("Genre", gender_options, default=gender_options, key="gender_filter_ms")

df_f = df_kpi[df_kpi["age_group"].isin(age_filter) & df_kpi["gender"].isin(gender_filter)].copy()
if df_f.empty:
    st.warning("Aucune donnée ne correspond aux filtres.")
    st.stop()

# ---------------------------------------------------
# 1) Vue d'ensemble
# ---------------------------------------------------
if page == "Vue d'ensemble":
    st.title("Facebook Ads — Nexus Strategic Overview")
    st.markdown("<div class='small-note'>Vue globale + segmentation pour comprendre ce qui marche (et ce qui coûte cher).</div>", unsafe_allow_html=True)

    total_impr = df_f["impressions"].sum()
    total_clicks = df_f["clicks"].sum()
    total_spend = df_f["ad_spend"].sum()
    total_conv = df_f["approved_conversions"].sum()

    ctr = safe_div(total_clicks, total_impr) * 100 if total_impr else np.nan
    cpc = safe_div(total_spend, total_clicks)
    cpa = safe_div(total_spend, total_conv)
    cpm = safe_div(total_spend, total_impr) * 1000 if total_impr else np.nan
    cr = safe_div(total_conv, total_clicks) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Impressions", f"{total_impr:,.0f}")
    c2.metric("Clics", f"{total_clicks:,.0f}")
    c3.metric("Dépenses (€)", f"{total_spend:,.2f}")
    c4.metric("Conversions", f"{total_conv:,.0f}")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("CTR (%)", f"{ctr:.2f}" if pd.notna(ctr) else "—")
    k2.metric("CPC (€)", f"{cpc:.2f}" if pd.notna(cpc) else "—")
    k3.metric("CPA (€)", f"{cpa:.2f}" if pd.notna(cpa) else "—")
    k4.metric("CPM (€)", f"{cpm:.2f}" if pd.notna(cpm) else "—")
    k5.metric("Conv Rate (%)", f"{cr:.2f}" if pd.notna(cr) else "—")

    st.markdown("---")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Spend vs Résultats")
        fig = px.scatter(
            df_f,
            x="ad_spend",
            y="approved_conversions",
            size="impressions",
            color="age_group",
            hover_data=["gender", "clicks", "ctr_pct", "cpa"],
            title="Budget → Conversions (taille = impressions)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Qualité & manquants")
        st.write("Lignes × Colonnes :", df_f.shape)
        miss = df_f.isna().sum().sort_values(ascending=False).head(10)
        st.dataframe(miss.to_frame("missing"), use_container_width=True)

        st.markdown("**Top segments (CPA le plus bas)**")
        seg = (df_f.groupby(["age_group","gender"])
                 .agg(spend=("ad_spend","sum"),
                      conv=("approved_conversions","sum"),
                      clicks=("clicks","sum"),
                      impr=("impressions","sum"))
                 .reset_index())
        seg["cpa"] = seg["spend"] / seg["conv"].replace(0, np.nan)
        seg = seg.sort_values("cpa", ascending=True).head(8)
        st.dataframe(seg, use_container_width=True)

    st.markdown("---")
    st.subheader("Table rapide (données + KPI)")
    st.dataframe(df_f.head(60), use_container_width=True)

# ---------------------------------------------------
# 2) Analyse Exploratoire
# ---------------------------------------------------
elif page == "Analyse Exploratoire":
    st.header("Analyse Exploratoire (Nexus)")
    st.markdown("<div class='small-note'>On explore les distributions + comparaison segments.</div>", unsafe_allow_html=True)

    numeric_cols = ["impressions", "clicks", "ad_spend", "approved_conversions", "ctr_pct", "cpc", "cpa", "cpm", "conversion_rate_pct"]
    numeric_cols = [c for c in numeric_cols if c in df_f.columns]

    selected = st.selectbox("Variable à explorer", numeric_cols, index=numeric_cols.index("cpa") if "cpa" in numeric_cols else 0, key="explore_var")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df_f, x=selected, nbins=40, title=f"Distribution — {selected}")
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Stats")
        st.dataframe(df_f[selected].describe().to_frame().T, use_container_width=True)

        st.markdown("**Top 10 (valeurs élevées)**")
        cols_show = ["age_group","gender",selected,"ad_spend","clicks","impressions","approved_conversions"]
        cols_show = [c for c in cols_show if c in df_f.columns]
        top10 = df_f.nlargest(10, selected)[cols_show]
        st.dataframe(top10, use_container_width=True)

    st.markdown("---")
    st.subheader("Comparaison par âge (moyennes KPI)")
    by_age = (df_f.groupby("age_group")
                .agg(ctr_pct=("ctr_pct","mean"),
                     cpc=("cpc","mean"),
                     cpa=("cpa","mean"),
                     cpm=("cpm","mean"),
                     conversion_rate_pct=("conversion_rate_pct","mean"))
                .reset_index())

    cA, cB = st.columns(2)
    with cA:
        fig2 = px.bar(by_age, x="age_group", y="cpa", title="CPA moyen (€) par tranche d’âge")
        st.plotly_chart(fig2, use_container_width=True)

    with cB:
        fig3 = px.bar(by_age, x="age_group", y="ctr_pct", title="CTR moyen (%) par tranche d’âge")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("Comparaison par genre")
    by_gender = (df_f.groupby("gender")
                   .agg(ctr_pct=("ctr_pct","mean"),
                        cpa=("cpa","mean"),
                        cpm=("cpm","mean"),
                        spend=("ad_spend","sum"),
                        conv=("approved_conversions","sum"))
                   .reset_index())
    fig4 = px.bar(by_gender, x="gender", y=["cpa","cpm"], barmode="group", title="CPA & CPM par genre")
    st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------
# 3) Corrélations
# ---------------------------------------------------
elif page == "Corrélations":
    st.header("Corrélations (numériques)")
    st.markdown("<div class='small-note'>Regarde surtout Spend → résultats et l’impact indirect sur CPA.</div>", unsafe_allow_html=True)

    corr_cols = ["impressions", "clicks", "ad_spend", "approved_conversions", "ctr_pct", "cpc", "cpa", "cpm", "conversion_rate_pct"]
    corr_cols = [c for c in corr_cols if c in df_f.columns]
    corr = df_f[corr_cols].corr()

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
    fig.update_layout(height=650, title="Matrice de corrélation")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("2 relations rapides")
    c1, c2 = st.columns(2)
    with c1:
        fig_sc1 = px.scatter(df_f, x="ad_spend", y="impressions", color="age_group", title="Spend → Impressions")
        st.plotly_chart(fig_sc1, use_container_width=True)
    with c2:
        fig_sc2 = px.scatter(df_f, x="ad_spend", y="approved_conversions", color="gender", title="Spend → Conversions")
        st.plotly_chart(fig_sc2, use_container_width=True)

# ---------------------------------------------------
# 4) Modélisation Prédictive
# ---------------------------------------------------
elif page == "Modélisation Prédictive":
    st.header("Modélisation Prédictive (ML)")
    st.markdown("""
Ici on entraîne 3 modèles RandomForest pour prédire **par jour** :
- impressions à partir de spend + âge + genre  
- clicks à partir de spend + âge + genre  
- approved_conversions à partir de spend + âge + genre  
""")

    df_model = df.dropna(subset=["ad_spend", "age_group", "gender", "impressions", "clicks", "approved_conversions"]).copy()
    if len(df_model) < 20:
        st.warning("Dataset trop petit après nettoyage pour faire un scoring fiable. (Le prédicteur peut quand même tourner.)")

    with st.spinner("Entraînement des modèles (cache activé)..."):
        models, scores = train_all_models(df_model)

    res = pd.DataFrame(scores).T
    st.subheader("Scores (jeu de test)")
    st.dataframe(res.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","R2":"{:.3f}"}), use_container_width=True)

    st.markdown("---")
    st.subheader("Lecture rapide (Nexus)")
    colA, colB, colC = st.columns(3)
    r2_impr = scores["impressions"]["R2"]
    r2_clicks = scores["clicks"]["R2"]
    r2_conv = scores["approved_conversions"]["R2"]

    colA.metric("R2 Impressions", f"{r2_impr:.3f}" if pd.notna(r2_impr) else "—")
    colB.metric("R2 Clicks", f"{r2_clicks:.3f}" if pd.notna(r2_clicks) else "—")
    colC.metric("R2 Conversions", f"{r2_conv:.3f}" if pd.notna(r2_conv) else "—")

    st.info("Si R2 est faible : ton dataset est sûrement trop bruité (ou spend n’explique pas tout). Mais le What-If reste utile pour se faire une idée.")

# ---------------------------------------------------
# 5) Prédicteur Interactif
# ---------------------------------------------------
else:
    st.header("🎛️ Prédicteur Interactif (What-If)")
    st.markdown("Tu changes **jours / budget / âge / genre** → on prédit **impressions, clics, conversions** → puis on calcule **CTR, CPC, CPA, CPM**.")

    df_model = df.dropna(subset=["ad_spend", "age_group", "gender", "impressions", "clicks", "approved_conversions"]).copy()

    with st.spinner("Chargement des modèles (cache)..."):
        models, _ = train_all_models(df_model)

    colA, colB, colC, colD = st.columns(4)
    with colA:
        days = st.slider("Nombre de jours", 1, 60, 14, 1, key="pred_days")
    with colB:
        bmax = float(max(10.0, df_f["ad_spend"].quantile(0.99) * 30))
        budget = st.slider("Budget total (€)", 0.0, bmax, float(min(300.0, bmax)), 10.0, key="pred_budget")
    with colC:
        age_choice = st.selectbox("Tranche d’âge", age_options, index=0, key="pred_age")
    with colD:
        gender_choice = st.selectbox("Genre", gender_options, index=0, key="pred_gender")

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
    v3.metric("Conversions", f"{sim['approved_conversions']:.0f}")
    v4.metric("Budget / jour", f"{sim['spend_per_day']:.2f} €")

    st.markdown("---")
    st.subheader("📈 Projection cumulée")
    days_range = np.arange(1, sim["days"] + 1)
    proj = pd.DataFrame({
        "Jour": days_range,
        "Dépense cumulée (€)": (sim["budget_total"] / sim["days"]) * days_range if sim["days"] else 0,
        "Impressions cumulées": (sim["impressions"] / sim["days"]) * days_range if sim["days"] else 0,
        "Clics cumulés": (sim["clicks"] / sim["days"]) * days_range if sim["days"] else 0,
        "Conversions cumulées": (sim["approved_conversions"] / sim["days"]) * days_range if sim["days"] else 0
    })

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.line(proj, x="Jour", y=["Impressions cumulées", "Clics cumulés"], title="Impressions & Clics cumulés")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.line(proj, x="Jour", y=["Dépense cumulée (€)", "Conversions cumulées"], title="Dépense & Conversions cumulées")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("🧾 Benchmark historique (segment sélectionné)")
    seg_hist = df_f[(df_f["age_group"] == age_choice) & (df_f["gender"] == gender_choice)].copy()
    if len(seg_hist) == 0:
        st.info("Pas de lignes exactes pour ce segment. (Le modèle prédit quand même.)")
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

    st.caption("⚠️ Simulation prédictive basée sur ton historique : le modèle apprend la relation entre spend et résultats, modulée par l’âge et le genre.")