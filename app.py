# app.py
# Dashboard Streamlit PRO — Facebook Ads (Analyse temporelle, sans prédiction)
# Pages: Vue d'ensemble / Analyse temporelle / Segments (âge simplifié + genre) / Qualité & colonnes
#
# Dépendances:
#   pip install streamlit pandas numpy
#
# Lancer:
#   streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# =========================
# CONFIG + STYLE
# =========================
st.set_page_config(
    page_title="Facebook Ads — Dashboard (Time Analysis)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main > div {padding-top: 1.5rem;}
    h1 {font-weight: 800; margin-bottom: 0.2rem;}
    h2 {font-weight: 700; margin-top: 1.2rem;}
    .stMetric {background: rgba(255,255,255,0.5); padding: 10px; border-radius: 10px;}
    .block-container {padding-top: 1.2rem;}
    :root { --bg: #E1E5F2; --text: #1E3A5F; }
    .stApp { background-color: var(--bg); color: var(--text); font-family: Inter, sans-serif; }
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def safe_div(n, d):
    d = np.where(d == 0, np.nan, d)
    return n / d

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Harmoniser noms
    rename_map = {
        "age": "age_group",
        "spent": "ad_spend",
        "total_conversion": "total_conversions",
        "approved_conversion": "approved_conversions",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Date: priorité à "date", sinon reporting_start, sinon reporting_end
    if "date" not in df.columns:
        if "reporting_start" in df.columns:
            df["date"] = df["reporting_start"]
        elif "reporting_end" in df.columns:
            df["date"] = df["reporting_end"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Numériques
    for c in ["impressions", "clicks", "ad_spend", "total_conversions", "approved_conversions"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Catégories
    if "age_group" in df.columns:
        df["age_group"] = df["age_group"].astype(str)
    if "gender" in df.columns:
        df["gender"] = df["gender"].astype(str)

    return df

def age_bucket(age_group: str) -> str:
    """Regroupe les âges en 4 segments lisibles."""
    if age_group is None or pd.isna(age_group):
        return "Autre"
    s = str(age_group).strip()
    try:
        start = int(s.split("-")[0])
    except Exception:
        return "Autre"

    if start <= 24:
        return "18–24"
    if 25 <= start <= 34:
        return "25–34"
    if 35 <= start <= 44:
        return "35–44"
    return "45+"

def compute_kpis_from_totals(impr, clicks, spend, conv):
    ctr = safe_div(np.array([clicks], dtype=float), np.array([impr], dtype=float))[0] * 100 if impr else np.nan
    cpc = safe_div(np.array([spend], dtype=float), np.array([clicks], dtype=float))[0]
    cpa = safe_div(np.array([spend], dtype=float), np.array([conv], dtype=float))[0]
    cpm = safe_div(np.array([spend], dtype=float), np.array([impr], dtype=float))[0] * 1000 if impr else np.nan
    return ctr, cpc, cpa, cpm

def agg_time(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Agrège par jour ou semaine et recalcule KPI sur les sommes."""
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

    g["ctr_pct"] = safe_div(g["clicks"].to_numpy(), g["impressions"].to_numpy()) * 100
    g["cpc"]     = safe_div(g["spend"].to_numpy(), g["clicks"].to_numpy())
    g["cpa"]     = safe_div(g["spend"].to_numpy(), g["approved_conversions"].to_numpy())
    g["cpm"]     = safe_div(g["spend"].to_numpy(), g["impressions"].to_numpy()) * 1000

    # Nettoyage inf
    for c in ["ctr_pct", "cpc", "cpa", "cpm"]:
        g[c] = g[c].replace([np.inf, -np.inf], np.nan)

    return g

def corr_matrix(df: pd.DataFrame, cols):
    d = df[cols].copy()
    return d.corr(numeric_only=True)

@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df = normalize_columns(df)

    required = ["age_group", "gender", "impressions", "clicks", "ad_spend", "approved_conversions", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    df["age_bucket"] = df["age_group"].apply(age_bucket)
    return df

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navigation")

pages = [
    "Vue d'ensemble",
    "Analyse temporelle",
    "Segments (âge simplifié + genre)",
    "Qualité & colonnes"
]
page = st.sidebar.radio("Sélectionner une section", pages)

st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Importer ton CSV Facebook Ads", type=["csv"])

if uploaded is None:
    st.title("📊 Facebook Ads — Dashboard (Analyse temporelle)")
    st.info("⬅️ Importe ton fichier CSV dans la barre latérale pour commencer.")
    st.stop()

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Erreur de chargement: {e}")
    st.stop()

# Filtres globaux
min_date = df["date"].min()
max_date = df["date"].max()
if pd.isna(min_date) or pd.isna(max_date):
    st.error("La colonne 'date' est invalide (format date). Vérifie ton CSV.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Filtres globaux")

start_date = st.sidebar.date_input("Date début", value=min_date.date(), min_value=min_date.date(), max_value=max_date.date())
end_date   = st.sidebar.date_input("Date fin", value=max_date.date(), min_value=min_date.date(), max_value=max_date.date())

age_choice = st.sidebar.selectbox("Âge (regroupé)", ["Tous","18–24","25–34","35–44","45+","Autre"], index=0)
gender_list = ["Tous"] + sorted(df["gender"].dropna().unique().tolist())
gender_choice = st.sidebar.selectbox("Genre", gender_list, index=0)

granularity = st.sidebar.radio("Granularité temps", ["Jour", "Semaine"], index=1)
rolling = st.sidebar.slider("Moyenne mobile (lissage)", min_value=1, max_value=8, value=2, step=1)

mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
dff = df.loc[mask].copy()
if age_choice != "Tous":
    dff = dff[dff["age_bucket"] == age_choice]
if gender_choice != "Tous":
    dff = dff[dff["gender"] == gender_choice]

if dff.empty:
    st.warning("Aucune donnée après filtres. Change la période / âge / genre.")
    st.stop()

# Totaux filtrés
total_impr = float(dff["impressions"].sum())
total_clicks = float(dff["clicks"].sum())
total_spend = float(dff["ad_spend"].sum())
total_conv = float(dff["approved_conversions"].sum())

ctr, cpc, cpa, cpm = compute_kpis_from_totals(total_impr, total_clicks, total_spend, total_conv)

# =========================
# PAGE 1 — VUE D’ENSEMBLE
# =========================
if page == "Vue d'ensemble":
    st.title("Facebook Ads — Vue d’ensemble")
    st.markdown("### Résumé global des performances (après filtres)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Impressions", f"{total_impr:,.0f}")
    c2.metric("Clics", f"{total_clicks:,.0f}")
    c3.metric("Dépenses (€)", f"{total_spend:,.2f}")
    c4.metric("Conversions approuvées", f"{total_conv:,.0f}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("CTR (%)", f"{ctr:.2f}" if pd.notna(ctr) else "—")
    k2.metric("CPC (€)", f"{cpc:.2f}" if pd.notna(cpc) else "—")
    k3.metric("CPA (€)", f"{cpa:.2f}" if pd.notna(cpa) else "—")
    k4.metric("CPM (€)", f"{cpm:.2f}" if pd.notna(cpm) else "—")

    st.markdown("---")
    left, right = st.columns([3, 1])
    with left:
        st.subheader("Aperçu des données")
        show_cols = ["date","age_group","age_bucket","gender","impressions","clicks","ad_spend","approved_conversions"]
        st.dataframe(dff[show_cols].sort_values("date").head(80), use_container_width=True)
    with right:
        st.subheader("Structure")
        st.write("Lignes × Colonnes :", dff.shape)
        st.write("Types de données :")
        st.dataframe(dff.dtypes.rename("Type").to_frame(), use_container_width=True)

# =========================
# PAGE 2 — ANALYSE TEMPORELLE
# =========================
elif page == "Analyse temporelle":
    st.title("Facebook Ads — Analyse temporelle")
    st.markdown("### Évolution des résultats dans le temps (basée sur l’existant)")

    freq = "D" if granularity == "Jour" else "W"
    ts = agg_time(dff, freq=freq)

    # Moyenne mobile (lissage)
    ts_plot = ts.copy()
    if rolling > 1:
        for col in ["impressions","clicks","spend","approved_conversions","ctr_pct","cpa","cpc","cpm"]:
            ts_plot[col] = ts_plot[col].rolling(rolling, min_periods=1).mean()

    # Petits “trends” dernière période vs précédente
    if len(ts) >= 2:
        last = ts.iloc[-1]
        prev = ts.iloc[-2]
        delta_ctr = (last["ctr_pct"] - prev["ctr_pct"]) if pd.notna(last["ctr_pct"]) and pd.notna(prev["ctr_pct"]) else np.nan
        delta_cpa = (last["cpa"] - prev["cpa"]) if pd.notna(last["cpa"]) and pd.notna(prev["cpa"]) else np.nan
    else:
        delta_ctr, delta_cpa = np.nan, np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dernier CTR (%)", f"{ts['ctr_pct'].iloc[-1]:.2f}" if pd.notna(ts['ctr_pct'].iloc[-1]) else "—",
              f"{delta_ctr:+.2f}" if pd.notna(delta_ctr) else None)
    m2.metric("Dernier CPA (€)", f"{ts['cpa'].iloc[-1]:.2f}" if pd.notna(ts['cpa'].iloc[-1]) else "—",
              f"{delta_cpa:+.2f}" if pd.notna(delta_cpa) else None)
    m3.metric("Dernières dépenses (€)", f"{ts['spend'].iloc[-1]:,.2f}")
    m4.metric("Dernières conversions", f"{ts['approved_conversions'].iloc[-1]:,.0f}")

    st.markdown("---")

    # Graphiques (Streamlit)
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Dépenses vs Conversions (dans le temps)")
        chart_df = ts_plot.set_index("period")[["spend","approved_conversions"]].rename(
            columns={"spend":"Dépenses (€)","approved_conversions":"Conversions"}
        )
        st.line_chart(chart_df, use_container_width=True)

    with colB:
        st.subheader("CTR (%) vs CPA (€) (dans le temps)")
        chart_df2 = ts_plot.set_index("period")[["ctr_pct","cpa"]].rename(
            columns={"ctr_pct":"CTR (%)","cpa":"CPA (€)"}
        )
        st.line_chart(chart_df2, use_container_width=True)

    st.markdown("---")
    st.subheader("Tableau récapitulatif par période")
    st.dataframe(ts.round(4), use_container_width=True)

    st.info(
        "Lecture simple :\n"
        "- Si **CPA augmente** et **CTR stable** → souci côté conversion (landing page/offre).\n"
        "- Si **CTR baisse** → problème d’audience/visuel/message.\n"
        "- Si **dépenses augmentent** mais conversions stagnent → budget mal distribué."
    )

# =========================
# PAGE 3 — SEGMENTS SIMPLIFIÉS
# =========================
elif page == "Segments (âge simplifié + genre)":
    st.title("Facebook Ads — Segments utiles")
    st.markdown("### On simplifie (4 âges max) pour éviter les segments vides/inutiles")

    # Agrégation par âge_bucket x genre
    seg = (dff.groupby(["age_bucket","gender"], dropna=False)
             .agg(
                 impressions=("impressions","sum"),
                 clicks=("clicks","sum"),
                 spend=("ad_spend","sum"),
                 approved_conversions=("approved_conversions","sum"),
             )
             .reset_index())

    seg["ctr_pct"] = safe_div(seg["clicks"].to_numpy(), seg["impressions"].to_numpy()) * 100
    seg["cpc"] = safe_div(seg["spend"].to_numpy(), seg["clicks"].to_numpy())
    seg["cpa"] = safe_div(seg["spend"].to_numpy(), seg["approved_conversions"].to_numpy())
    seg["cpm"] = safe_div(seg["spend"].to_numpy(), seg["impressions"].to_numpy()) * 1000
    for c in ["ctr_pct","cpc","cpa","cpm"]:
        seg[c] = seg[c].replace([np.inf,-np.inf], np.nan)

    # Option pour éviter les segments non significatifs
    st.markdown("---")
    colx, coly = st.columns([1,2])
    with colx:
        min_conv = st.slider("Seuil min conversions (affichage)", 0, 50, 5, 1)
    with coly:
        st.caption("Astuce : monte le seuil pour enlever les segments trop petits (qui faussent CPA).")

    seg_show = seg.copy()
    if min_conv > 0:
        seg_show = seg_show[seg_show["approved_conversions"] >= min_conv]

    # Top segments (CPA bas)
    st.subheader("Top segments (CPA le plus bas)")
    top = seg_show.dropna(subset=["cpa"]).sort_values(["cpa","spend"], ascending=[True, False]).head(15)
    st.dataframe(
        top[["age_bucket","gender","approved_conversions","cpa","ctr_pct","cpc","cpm","spend","impressions","clicks"]].round(4),
        use_container_width=True
    )

    st.markdown("---")
    # Bar charts par âge (tous genres confondus)
    by_age = (dff.groupby("age_bucket", dropna=False)
                .agg(impressions=("impressions","sum"),
                     clicks=("clicks","sum"),
                     spend=("ad_spend","sum"),
                     approved_conversions=("approved_conversions","sum"))
                .reset_index())
    by_age["ctr_pct"] = safe_div(by_age["clicks"].to_numpy(), by_age["impressions"].to_numpy()) * 100
    by_age["cpa"] = safe_div(by_age["spend"].to_numpy(), by_age["approved_conversions"].to_numpy())
    by_age = by_age.sort_values("age_bucket")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("CPA (€) par âge (regroupé)")
        st.bar_chart(by_age.set_index("age_bucket")[["cpa"]].rename(columns={"cpa":"CPA (€)"}), use_container_width=True)
    with c2:
        st.subheader("CTR (%) par âge (regroupé)")
        st.bar_chart(by_age.set_index("age_bucket")[["ctr_pct"]].rename(columns={"ctr_pct":"CTR (%)"}), use_container_width=True)

# =========================
# PAGE 4 — QUALITÉ & COLONNES
# =========================
else:
    st.title("Qualité des données & colonnes")
    st.markdown("### Vérification rapide: manquants, types, colonnes utiles")

    st.subheader("Colonnes présentes")
    st.write(list(df.columns))

    st.subheader("Valeurs manquantes (Top 15)")
    st.dataframe(df.isna().sum().sort_values(ascending=False).head(15).to_frame("missing"), use_container_width=True)

    st.subheader("Types de données")
    st.dataframe(df.dtypes.rename("Type").to_frame(), use_container_width=True)

    st.markdown("---")
    st.subheader("Corrélations (numériques)")
    corr_cols = ["impressions","clicks","ad_spend","approved_conversions"]
    corr = corr_matrix(dff, corr_cols)
    st.dataframe(corr.round(3), use_container_width=True)

    st.info(
        "Conseil : pour BI propre, garde surtout\n"
        "- date, age_bucket, gender\n"
        "- impressions, clicks, ad_spend, approved_conversions\n"
        "et recalcule CTR/CPC/CPA/CPM au besoin."
    )
