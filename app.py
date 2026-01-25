# app.py
# Streamlit — Analyse temporelle (sans prédiction) Facebook Ads
# Objectif: analyser l’évolution des KPI dans le temps (CTR, CPC, CPA, CPM)
# + segments simplifiés (âge regroupé + filtre genre)
# Dépendances: streamlit, pandas, numpy

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Facebook Ads — Analyse temporelle",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Helpers
# -------------------------
def safe_div(n, d):
    d = np.where(d == 0, np.nan, d)
    return n / d

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Harmoniser les noms possibles
    rename_map = {
        "age": "age_group",
        "spent": "ad_spend",
        "total_conversion": "total_conversions",
        "approved_conversion": "approved_conversions",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # dates: on privilégie "date" sinon reporting_start
    if "date" not in df.columns:
        if "reporting_start" in df.columns:
            df["date"] = df["reporting_start"]
        elif "reporting_end" in df.columns:
            df["date"] = df["reporting_end"]

    # convertir date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # convertir numériques
    for c in ["impressions", "clicks", "ad_spend", "total_conversions", "approved_conversions"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # catégories
    for c in ["age_group", "gender"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df

def make_age_bucket(age_group: str) -> str:
    """
    Transforme "30-34" -> 25-34 bucket, etc.
    Si format inconnu, retourne "Autre".
    """
    if age_group is None or pd.isna(age_group):
        return "Autre"

    s = str(age_group).strip()

    # formats attendus: "18-24", "25-34", etc.
    # on essaye d’extraire le début (18)
    try:
        start = int(s.split("-")[0])
    except Exception:
        return "Autre"

    if start <= 24:
        return "18–24"
    elif 25 <= start <= 34:
        return "25–34"
    elif 35 <= start <= 44:
        return "35–44"
    else:
        return "45+"

def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # KPI calculés à partir des colonnes brutes
    df["ctr_pct"] = safe_div(df["clicks"].to_numpy(), df["impressions"].to_numpy()) * 100
    df["cpc"] = safe_div(df["ad_spend"].to_numpy(), df["clicks"].to_numpy())
    df["cpa"] = safe_div(df["ad_spend"].to_numpy(), df["approved_conversions"].to_numpy())
    df["cpm"] = safe_div(df["ad_spend"].to_numpy(), df["impressions"].to_numpy()) * 1000

    # nettoyer inf
    for c in ["ctr_pct", "cpc", "cpa", "cpm"]:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    return df

def agg_time(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Agrège par date selon freq:
    - "D" jour
    - "W" semaine
    """
    out = df.dropna(subset=["date"]).copy()
    out["period"] = out["date"].dt.to_period(freq).dt.start_time

    g = (out.groupby("period", dropna=False)
            .agg(
                impressions=("impressions", "sum"),
                clicks=("clicks", "sum"),
                spend=("ad_spend", "sum"),
                approved_conversions=("approved_conversions", "sum"),
            )
            .reset_index()
            .sort_values("period")
        )

    # KPI niveau période (recalculés sur les sommes)
    g["ctr_pct"] = safe_div(g["clicks"].to_numpy(), g["impressions"].to_numpy()) * 100
    g["cpc"] = safe_div(g["spend"].to_numpy(), g["clicks"].to_numpy())
    g["cpa"] = safe_div(g["spend"].to_numpy(), g["approved_conversions"].to_numpy())
    g["cpm"] = safe_div(g["spend"].to_numpy(), g["impressions"].to_numpy()) * 1000

    return g

def top_segments(df: pd.DataFrame, min_conversions: int = 5) -> pd.DataFrame:
    """
    Classe les segments (age_bucket x gender) avec des règles simples:
    - conversions >= min_conversions
    - tri CPA croissant (meilleur)
    """
    seg = (df.groupby(["age_bucket", "gender"], dropna=False)
             .agg(
                 impressions=("impressions", "sum"),
                 clicks=("clicks", "sum"),
                 spend=("ad_spend", "sum"),
                 approved_conversions=("approved_conversions", "sum"),
             )
             .reset_index()
          )
    seg["ctr_pct"] = safe_div(seg["clicks"].to_numpy(), seg["impressions"].to_numpy()) * 100
    seg["cpc"] = safe_div(seg["spend"].to_numpy(), seg["clicks"].to_numpy())
    seg["cpa"] = safe_div(seg["spend"].to_numpy(), seg["approved_conversions"].to_numpy())
    seg["cpm"] = safe_div(seg["spend"].to_numpy(), seg["impressions"].to_numpy()) * 1000

    seg = seg[(seg["approved_conversions"] >= min_conversions) & seg["cpa"].notna()].copy()
    seg = seg.sort_values(["cpa", "spend"], ascending=[True, False])
    return seg

# -------------------------
# UI / Sidebar
# -------------------------
st.title("📊 Facebook Ads — Analyse temporelle (basée sur l’existant)")

with st.sidebar:
    st.header("1) Charger les données")
    uploaded = st.file_uploader("Importer ton CSV nettoyé", type=["csv"])

    st.header("2) Réglages d’analyse")
    granularity = st.radio("Granularité", ["Jour", "Semaine"], index=1)
    rolling = st.slider("Moyenne mobile (lissage)", min_value=1, max_value=8, value=1, step=1)
    min_conv = st.slider("Seuil min conversions (Top segments)", min_value=0, max_value=50, value=5, step=1)

    st.markdown("---")
    st.caption("✅ Pas de prédiction ici : on analyse les tendances réelles dans le temps.")

if uploaded is None:
    st.info("⬅️ Importe ton fichier CSV dans la barre latérale pour commencer.")
    st.stop()

# -------------------------
# Load & prep
# -------------------------
df = pd.read_csv(uploaded)
df = normalize_columns(df)

required = ["age_group", "gender", "impressions", "clicks", "ad_spend", "approved_conversions", "date"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Colonnes manquantes: {missing}\n\nVérifie ton CSV (ou renomme: spent→ad_spend, approved_conversion→approved_conversions, etc.)")
    st.stop()

df["age_bucket"] = df["age_group"].apply(make_age_bucket)
df = add_kpis(df)

# Date range filter
min_date = df["date"].min()
max_date = df["date"].max()
if pd.isna(min_date) or pd.isna(max_date):
    st.error("La colonne 'date' est vide ou invalide. Vérifie le format date dans ton CSV.")
    st.stop()

colf1, colf2, colf3 = st.columns([1.2, 1, 1])
with colf1:
    page = st.selectbox("Page", ["Vue d’ensemble", "Analyse temporelle", "Segments utiles"], index=1)
with colf2:
    age_choice = st.selectbox("Tranche d’âge (regroupée)", ["Tous", "18–24", "25–34", "35–44", "45+", "Autre"], index=0)
with colf3:
    gender_choice = st.selectbox("Genre", ["Tous"] + sorted(df["gender"].dropna().unique().tolist()), index=0)

date_col1, date_col2 = st.columns([1, 1])
with date_col1:
    start_date = st.date_input("Date début", value=min_date.date(), min_value=min_date.date(), max_value=max_date.date())
with date_col2:
    end_date = st.date_input("Date fin", value=max_date.date(), min_value=min_date.date(), max_value=max_date.date())

# Apply filters
mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
dff = df.loc[mask].copy()

if age_choice != "Tous":
    dff = dff[dff["age_bucket"] == age_choice]
if gender_choice != "Tous":
    dff = dff[dff["gender"] == gender_choice]

if dff.empty:
    st.warning("Aucune donnée après filtres. Change la période / âge / genre.")
    st.stop()

# -------------------------
# Page 1 — Vue d’ensemble
# -------------------------
if page == "Vue d’ensemble":
    st.subheader("✅ Vue d’ensemble (après filtres)")

    total_impr = dff["impressions"].sum()
    total_clicks = dff["clicks"].sum()
    total_spend = dff["ad_spend"].sum()
    total_conv = dff["approved_conversions"].sum()

    ctr = safe_div(total_clicks, total_impr) * 100 if total_impr else np.nan
    cpc = safe_div(total_spend, total_clicks)
    cpa = safe_div(total_spend, total_conv)
    cpm = safe_div(total_spend, total_impr) * 1000 if total_impr else np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Impressions", f"{total_impr:,.0f}")
    m2.metric("Clics", f"{total_clicks:,.0f}")
    m3.metric("Dépenses (€)", f"{total_spend:,.2f}")
    m4.metric("Conversions approuvées", f"{total_conv:,.0f}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("CTR (%)", f"{ctr:.2f}" if pd.notna(ctr) else "—")
    k2.metric("CPC (€)", f"{cpc:.2f}" if pd.notna(cpc) else "—")
    k3.metric("CPA (€)", f"{cpa:.2f}" if pd.notna(cpa) else "—")
    k4.metric("CPM (€)", f"{cpm:.2f}" if pd.notna(cpm) else "—")

    st.markdown("---")
    st.subheader("Aperçu des lignes (top 50)")
    show_cols = ["date", "age_group", "age_bucket", "gender", "impressions", "clicks", "ad_spend", "approved_conversions", "ctr_pct", "cpc", "cpa", "cpm"]
    show_cols = [c for c in show_cols if c in dff.columns]
    st.dataframe(dff[show_cols].head(50), use_container_width=True)

# -------------------------
# Page 2 — Analyse temporelle
# -------------------------
elif page == "Analyse temporelle":
    st.subheader("📈 Analyse temporelle (tendances réelles)")

    freq = "D" if granularity == "Jour" else "W"
    ts = agg_time(dff, freq=freq)

    # Lissage (moyenne mobile)
    if rolling > 1:
        for col in ["impressions", "clicks", "spend", "approved_conversions", "ctr_pct", "cpc", "cpa", "cpm"]:
            ts[f"{col}_ma"] = ts[col].rolling(rolling, min_periods=1).mean()
        view = ts.set_index("period")[[f"{c}_ma" for c in ["spend","approved_conversions","ctr_pct","cpa"]]]
        view = view.rename(columns={
            "spend_ma": "Dépenses (MA)",
            "approved_conversions_ma": "Conversions (MA)",
            "ctr_pct_ma": "CTR % (MA)",
            "cpa_ma": "CPA € (MA)",
        })
    else:
        view = ts.set_index("period")[["spend","approved_conversions","ctr_pct","cpa"]].rename(columns={
            "spend": "Dépenses",
            "approved_conversions": "Conversions",
            "ctr_pct": "CTR %",
            "cpa": "CPA €",
        })

    # KPIs de la dernière période vs précédente (petit “trend” simple)
    if len(ts) >= 2:
        last = ts.iloc[-1]
        prev = ts.iloc[-2]
        delta_cpa = last["cpa"] - prev["cpa"]
        delta_ctr = last["ctr_pct"] - prev["ctr_pct"]
    else:
        delta_cpa = np.nan
        delta_ctr = np.nan

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Dernier CTR (%)", f"{ts['ctr_pct'].iloc[-1]:.2f}", f"{delta_ctr:+.2f}" if pd.notna(delta_ctr) else None)
    a2.metric("Dernier CPA (€)", f"{ts['cpa'].iloc[-1]:.2f}" if pd.notna(ts['cpa'].iloc[-1]) else "—", f"{delta_cpa:+.2f}" if pd.notna(delta_cpa) else None)
    a3.metric("Dernières dépenses (€)", f"{ts['spend'].iloc[-1]:.2f}")
    a4.metric("Dernières conversions", f"{ts['approved_conversions'].iloc[-1]:.0f}")

    st.markdown("---")
    st.subheader("Courbes (dépenses, conversions, CTR, CPA)")
    st.line_chart(view, use_container_width=True)

    st.markdown("---")
    st.subheader("Table récap par période")
    ts_display = ts.copy()
    ts_display = ts_display.rename(columns={"period":"période","spend":"dépenses"})
    st.dataframe(ts_display.round(4), use_container_width=True)

    st.info(
        "Lecture simple : si le **CPA monte** alors que le **CTR reste stable**, le souci vient souvent de la **conversion** (landing page/offre). "
        "Si le **CTR baisse**, c’est plutôt un problème de **créatif / ciblage**."
    )

# -------------------------
# Page 3 — Segments utiles (simplifiés)
# -------------------------
else:
    st.subheader("👥 Segments utiles (âge regroupé + genre)")
    st.caption("On évite les segments trop fins. On garde des groupes lisibles avec un minimum de conversions.")

    seg = top_segments(dff, min_conversions=min_conv)

    if seg.empty:
        st.warning("Aucun segment ne passe le seuil de conversions. Diminue le seuil (sidebar) ou élargis la période.")
        st.stop()

    # Top 10 segments
    st.subheader("Top segments (CPA bas, conversions suffisantes)")
    st.dataframe(
        seg[["age_bucket","gender","approved_conversions","cpa","ctr_pct","cpc","cpm","spend","impressions","clicks"]]
          .head(15)
          .round(4),
        use_container_width=True
    )

    # Comparaison par age_bucket (tous genres confondus)
    by_age = (dff.groupby("age_bucket", dropna=False)
                .agg(impressions=("impressions","sum"),
                     clicks=("clicks","sum"),
                     spend=("ad_spend","sum"),
                     approved_conversions=("approved_conversions","sum"))
                .reset_index())
    by_age["ctr_pct"] = safe_div(by_age["clicks"].to_numpy(), by_age["impressions"].to_numpy()) * 100
    by_age["cpa"] = safe_div(by_age["spend"].to_numpy(), by_age["approved_conversions"].to_numpy())
    by_age = by_age.sort_values("age_bucket")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("CPA par âge (regroupé)")
        chart = by_age.set_index("age_bucket")[["cpa"]].rename(columns={"cpa":"CPA (€)"})
        st.bar_chart(chart, use_container_width=True)

    with c2:
        st.subheader("CTR par âge (regroupé)")
        chart2 = by_age.set_index("age_bucket")[["ctr_pct"]].rename(columns={"ctr_pct":"CTR (%)"})
        st.bar_chart(chart2, use_container_width=True)

    st.info(
        "Conseil : commence par **Tous** (âge/genre) pour voir la tendance globale, puis filtre seulement si tu vois une vraie différence."
    )
