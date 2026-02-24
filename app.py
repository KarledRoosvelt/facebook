# ==========================================================
# Facebook Ads – Nexus Strategic Analytics (Clean Version)
# ==========================================================

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

# ==========================================================
# CONFIGURATION PAGE
# ==========================================================

st.set_page_config(
    page_title="Facebook Ads – Nexus Analytics",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# UTILITAIRES
# ==========================================================

def safe_div(a, b):
    if b == 0 or pd.isna(b):
        return np.nan
    return a / b

# ==========================================================
# NETTOYAGE DATASET
# ==========================================================

def clean_data(df):

    df = df.copy()
    df.columns = df.columns.str.strip()

    # Renommage standard
    rename_map = {
        "age": "age_group",
        "spent": "ad_spend",
        "approved_conversion": "approved_conversions"
    }
    df.rename(columns=rename_map, inplace=True)

    # Nettoyage âge → garder uniquement format 18-24 etc
    df["age_group"] = df["age_group"].astype(str).str.strip()
    df = df[df["age_group"].str.match(r"^\d{2}-\d{2}$")]

    # Nettoyage genre → garder uniquement Male / Female
    df["gender"] = df["gender"].astype(str).str.strip()
    df["gender"] = df["gender"].replace({"M": "Male", "F": "Female"})
    df = df[df["gender"].isin(["Male", "Female"])]

    # Conversion numérique
    numeric_cols = [
        "impressions",
        "clicks",
        "ad_spend",
        "approved_conversions"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    return df

# ==========================================================
# KPI
# ==========================================================

def add_kpis(df):

    df = df.copy()

    # Remplacer les 0 pour éviter division par 0
    impressions = df["impressions"].replace(0, np.nan)
    clicks = df["clicks"].replace(0, np.nan)
    conversions = df["approved_conversions"].replace(0, np.nan)

    df["ctr_pct"] = (df["clicks"] / impressions) * 100
    df["cpc"] = df["ad_spend"] / clicks
    df["cpa"] = df["ad_spend"] / conversions
    df["cpm"] = (df["ad_spend"] / impressions) * 1000
    df["conversion_rate_pct"] = (df["approved_conversions"] / clicks) * 100

    return df

# ==========================================================
# MACHINE LEARNING
# ==========================================================

def build_model(df, target):

    X = df[["ad_spend", "age_group", "gender"]]
    y = df[target]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["age_group", "gender"]),
        ("num", "passthrough", ["ad_spend"])
    ])

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    return pipe, X, y

def train_model(df, target):

    pipe, X, y = build_model(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    scores = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds)
    }

    return pipe, scores

def predict_scenario(models, age_group, gender, budget, days):

    spend_day = budget / days if days != 0 else 0

    X = pd.DataFrame([{
        "ad_spend": spend_day,
        "age_group": age_group,
        "gender": gender
    }])

    impr_day = models["impressions"].predict(X)[0]
    clicks_day = models["clicks"].predict(X)[0]
    conv_day = models["approved_conversions"].predict(X)[0]

    impr = impr_day * days
    clicks = clicks_day * days
    conv = conv_day * days

    ctr = safe_div(clicks, impr) * 100
    cpc = safe_div(budget, clicks)
    cpa = safe_div(budget, conv)
    cpm = safe_div(budget, impr) * 1000

    return impr, clicks, conv, ctr, cpc, cpa, cpm

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Section",
    [
        "Vue d'ensemble",
        "Analyse Exploratoire",
        "Corrélations",
        "Modélisation",
        "Simulateur What-If"
    ],
    key="navigation"
)

uploaded = st.sidebar.file_uploader("Importer CSV Facebook Ads", type=["csv"])

if uploaded is None:
    st.info("⬅️ Importer ton fichier CSV pour commencer.")
    st.stop()

# ==========================================================
# CHARGEMENT
# ==========================================================

df = pd.read_csv(uploaded)
df = clean_data(df)
df = add_kpis(df)

age_options = sorted(df["age_group"].unique(), key=lambda x: int(x.split("-")[0]))
gender_options = sorted(df["gender"].unique())

# ==========================================================
# 1️⃣ VUE D’ENSEMBLE
# ==========================================================

if page == "Vue d'ensemble":

    st.title("📊 Facebook Ads – Strategic Overview")

    col1, col2, col3, col4 = st.columns(4)

    total_impr = df["impressions"].sum()
    total_clicks = df["clicks"].sum()
    total_spend = df["ad_spend"].sum()
    total_conv = df["approved_conversions"].sum()

    col1.metric("Impressions", f"{total_impr:,.0f}")
    col2.metric("Clicks", f"{total_clicks:,.0f}")
    col3.metric("Spend (€)", f"{total_spend:,.2f}")
    col4.metric("Conversions", f"{total_conv:,.0f}")

    st.divider()

    ctr = safe_div(total_clicks, total_impr) * 100
    cpc = safe_div(total_spend, total_clicks)
    cpa = safe_div(total_spend, total_conv)
    cpm = safe_div(total_spend, total_impr) * 1000

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("CTR (%)", f"{ctr:.2f}")
    k2.metric("CPC (€)", f"{cpc:.2f}")
    k3.metric("CPA (€)", f"{cpa:.2f}")
    k4.metric("CPM (€)", f"{cpm:.2f}")

    st.dataframe(df.head(50), use_container_width=True)

# ==========================================================
# 2️⃣ ANALYSE EXPLORATOIRE
# ==========================================================

elif page == "Analyse Exploratoire":

    st.header("Distribution KPI")

    var = st.selectbox("Variable", ["ctr_pct", "cpa", "cpc", "cpm"])

    fig = px.histogram(df, x=var, nbins=40)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.scatter(
        df,
        x="ctr_pct",
        y="cpa",
        size="ad_spend",
        color="age_group"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================================
# 3️⃣ CORRÉLATIONS
# ==========================================================

elif page == "Corrélations":

    corr_cols = [
        "impressions",
        "clicks",
        "ad_spend",
        "approved_conversions",
        "ctr_pct",
        "cpa"
    ]

    corr = df[corr_cols].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale="RdBu",
        zmid=0
    ))

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 4️⃣ MODÉLISATION
# ==========================================================

elif page == "Modélisation":

    df_model = df.dropna()

    m1, s1 = train_model(df_model, "impressions")
    m2, s2 = train_model(df_model, "clicks")
    m3, s3 = train_model(df_model, "approved_conversions")

    results = pd.DataFrame({
        "Impressions": s1,
        "Clicks": s2,
        "Conversions": s3
    }).T

    st.dataframe(results)

# ==========================================================
# 5️⃣ SIMULATEUR
# ==========================================================

else:

    st.header("🎯 Simulateur What-If")

    df_model = df.dropna()

    m_impr, _ = train_model(df_model, "impressions")
    m_clicks, _ = train_model(df_model, "clicks")
    m_conv, _ = train_model(df_model, "approved_conversions")

    models = {
        "impressions": m_impr,
        "clicks": m_clicks,
        "approved_conversions": m_conv
    }

    age_choice = st.selectbox("Age Group", age_options)
    gender_choice = st.selectbox("Gender", gender_options)
    days = st.slider("Days", 1, 60, 14)
    budget = st.slider("Total Budget (€)", 0.0, float(df["ad_spend"].quantile(0.99)*30), 300.0)

    impr, clicks, conv, ctr, cpc, cpa, cpm = predict_scenario(
        models,
        age_choice,
        gender_choice,
        budget,
        days
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("CTR (%)", f"{ctr:.2f}")
    col2.metric("CPC (€)", f"{cpc:.2f}")
    col3.metric("CPA (€)", f"{cpa:.2f}")
    col4.metric("CPM (€)", f"{cpm:.2f}")

    st.caption("Simulation basée sur Random Forest + segmentation marketing.")