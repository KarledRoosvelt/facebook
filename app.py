# Streamlit — Simulation prédictive (What-If) Facebook Ads
# Entrées: nombre de jours, tranche d’âge, genre, budget
# Sorties: CTR, CPC, CPA, CPM + volumes estimés (impressions, clics, conversions)

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="What-If Facebook Ads", layout="wide")

# -------------------------
# Helpers
# -------------------------
def safe_div(n, d):
    return np.nan if d == 0 or pd.isna(d) else n / d

def ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    # Normalisation des noms possibles
    rename_map = {
        "age": "age_group",
        "spent": "ad_spend",
        "total_conversion": "total_conversions",
        "approved_conversion": "approved_conversions",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Types
    for c in ["reporting_start", "reporting_end", "date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    df = ensure_numeric(df, ["impressions", "clicks", "ad_spend", "total_conversions", "approved_conversions"])

    # Colonnes minimales requises
    required = ["age_group", "gender", "impressions", "clicks", "ad_spend", "approved_conversions"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    # Nettoyage simple
    df["age_group"] = df["age_group"].astype(str)
    df["gender"] = df["gender"].astype(str)

    # KPI ligne (optionnel)
    df["ctr"] = df.apply(lambda r: safe_div(r["clicks"], r["impressions"]), axis=1)
    df["cpm"] = df.apply(lambda r: safe_div(r["ad_spend"], r["impressions"]) * 1000 if r["impressions"] else np.nan, axis=1)
    df["conv_rate"] = df.apply(lambda r: safe_div(r["approved_conversions"], r["clicks"]), axis=1)

    return df

def segment_table(df: pd.DataFrame) -> pd.DataFrame:
    # Agrégation par âge + genre
    seg = (df.groupby(["age_group", "gender"], dropna=False)
             .agg(impressions=("impressions", "sum"),
                  clicks=("clicks", "sum"),
                  spend=("ad_spend", "sum"),
                  approved_conversions=("approved_conversions", "sum"))
             .reset_index())

    seg["ctr"] = seg.apply(lambda r: safe_div(r["clicks"], r["impressions"]), axis=1)
    seg["ctr_pct"] = seg["ctr"] * 100
    seg["cpm"] = seg.apply(lambda r: safe_div(r["spend"], r["impressions"]) * 1000 if r["impressions"] else np.nan, axis=1)
    seg["cpc"] = seg.apply(lambda r: safe_div(r["spend"], r["clicks"]), axis=1)
    seg["cpa"] = seg.apply(lambda r: safe_div(r["spend"], r["approved_conversions"]), axis=1)
    seg["conv_rate"] = seg.apply(lambda r: safe_div(r["approved_conversions"], r["clicks"]), axis=1)
    seg["conv_rate_pct"] = seg["conv_rate"] * 100

    return seg

def filter_segment(seg: pd.DataFrame, age_choice: str, gender_choice: str) -> pd.DataFrame:
    out = seg.copy()
    if age_choice != "Tous":
        out = out[out["age_group"] == age_choice]
    if gender_choice != "Tous":
        out = out[out["gender"] == gender_choice]
    return out

def weighted_kpis(seg_f: pd.DataFrame) -> dict:
    # Pondération par impressions / spend selon l’indicateur
    impr = seg_f["impressions"].sum()
    clicks = seg_f["clicks"].sum()
    spend = seg_f["spend"].sum()
    conv = seg_f["approved_conversions"].sum()

    ctr = safe_div(clicks, impr)
    cpm = safe_div(spend, impr) * 1000 if impr else np.nan
    cpc = safe_div(spend, clicks)
    cpa = safe_div(spend, conv)
    conv_rate = safe_div(conv, clicks)

    return {
        "impressions": impr,
        "clicks": clicks,
        "spend": spend,
        "approved_conversions": conv,
        "ctr": ctr,
        "ctr_pct": ctr * 100 if ctr is not None else np.nan,
        "cpm": cpm,
        "cpc": cpc,
        "cpa": cpa,
        "conv_rate": conv_rate,
        "conv_rate_pct": conv_rate * 100 if conv_rate is not None else np.nan,
    }

def simulate(days: int, budget: float, base: dict) -> dict:
    # Simulation basée sur les ratios historiques du segment
    # Hypothèse: CPM & CTR & taux de conversion restent similaires sur la période.
    cpm = base["cpm"]
    ctr = base["ctr"]
    conv_rate = base["conv_rate"]

    if pd.isna(cpm) or cpm == 0:
        # fallback: si pas de CPM exploitable, on utilise le ratio impressions/spend historique
        # impressions_per_euro = impressions / spend
        impressions_per_euro = safe_div(base["impressions"], base["spend"])
        if impressions_per_euro is None or pd.isna(impressions_per_euro):
            impressions_per_euro = 0
        impressions = budget * impressions_per_euro
    else:
        impressions = budget / (cpm / 1000)

    clicks = impressions * (0 if pd.isna(ctr) else ctr)
    conversions = clicks * (0 if pd.isna(conv_rate) else conv_rate)

    # KPIs simulés
    ctr_pct = safe_div(clicks, impressions) * 100 if impressions else np.nan
    cpc = safe_div(budget, clicks)
    cpa = safe_div(budget, conversions)
    cpm_sim = safe_div(budget, impressions) * 1000 if impressions else np.nan

    return {
        "days": days,
        "budget": budget,
        "budget_per_day": safe_div(budget, days),
        "impressions": impressions,
        "clicks": clicks,
        "approved_conversions": conversions,
        "ctr_pct": ctr_pct,
        "cpc": cpc,
        "cpa": cpa,
        "cpm": cpm_sim,
    }

# -------------------------
# UI
# -------------------------
st.title("📈 Simulation prédictive (What-If) — Facebook Ads (CTR / CPC / CPA / CPM)")

with st.sidebar:
    st.header("1) Charger les données")
    uploaded = st.file_uploader("Upload ton CSV nettoyé", type=["csv"])
    st.caption("Le fichier doit contenir au minimum : age_group, gender, impressions, clicks, ad_spend (ou spent), approved_conversions.")

    st.header("2) Paramètres du scénario")
    days = st.slider("Nombre de jours", min_value=1, max_value=60, value=14, step=1)
    budget = st.slider("Budget total (€)", min_value=0.0, max_value=5000.0, value=300.0, step=10.0)

# Chargement dataset
if uploaded is None:
    st.info("⬅️ Charge un CSV dans la barre latérale pour démarrer.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
    df = prep_df(raw)
except Exception as e:
    st.error(f"Erreur de lecture/préparation : {e}")
    st.stop()

seg = segment_table(df)

# Choix segments
age_list = ["Tous"] + sorted(seg["age_group"].unique().tolist())
gender_list = ["Tous"] + sorted(seg["gender"].unique().tolist())

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    age_choice = st.selectbox("Tranche d’âge", age_list, index=0)
with colB:
    gender_choice = st.selectbox("Genre", gender_list, index=0)
with colC:
    st.caption("Astuce: commence avec 'Tous' puis filtre (âge/genre) pour voir l’impact sur CPA/CTR.")

seg_f = filter_segment(seg, age_choice, gender_choice)

if seg_f.empty:
    st.warning("Aucune donnée pour ce segment. Choisis un autre âge/genre.")
    st.stop()

base = weighted_kpis(seg_f)
sim = simulate(days=days, budget=budget, base=base)

# -------------------------
# Résultats
# -------------------------
st.subheader("✅ Résultats du scénario (simulation)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("CTR (%) estimé", f"{sim['ctr_pct']:.2f}" if pd.notna(sim["ctr_pct"]) else "—")
m2.metric("CPC (€) estimé", f"{sim['cpc']:.2f}" if pd.notna(sim["cpc"]) else "—")
m3.metric("CPA (€) estimé", f"{sim['cpa']:.2f}" if pd.notna(sim["cpa"]) else "—")
m4.metric("CPM (€) estimé", f"{sim['cpm']:.2f}" if pd.notna(sim["cpm"]) else "—")

v1, v2, v3, v4 = st.columns(4)
v1.metric("Impressions estimées", f"{sim['impressions']:.0f}")
v2.metric("Clics estimés", f"{sim['clicks']:.0f}")
v3.metric("Conversions estimées", f"{sim['approved_conversions']:.0f}")
v4.metric("Budget / jour", f"{sim['budget_per_day']:.2f} €" if pd.notna(sim["budget_per_day"]) else "—")

st.divider()

st.subheader("📌 Référence historique (sur le segment sélectionné)")
b1, b2, b3, b4 = st.columns(4)
b1.metric("CTR (%) historique", f"{base['ctr_pct']:.2f}" if pd.notna(base["ctr_pct"]) else "—")
b2.metric("CPC (€) historique", f"{base['cpc']:.2f}" if pd.notna(base["cpc"]) else "—")
b3.metric("CPA (€) historique", f"{base['cpa']:.2f}" if pd.notna(base["cpa"]) else "—")
b4.metric("CPM (€) historique", f"{base['cpm']:.2f}" if pd.notna(base["cpm"]) else "—")

# Table segment
st.subheader("🔎 Détails des segments (âge × genre)")
show_cols = ["age_group", "gender", "impressions", "clicks", "spend", "approved_conversions", "ctr_pct", "cpc", "cpa", "cpm"]
st.dataframe(seg[show_cols].round(4), use_container_width=True)

# Mini “courbe” cumulée par jour (projection linéaire)
st.subheader("📅 Projection cumulée sur la période")
daily = pd.DataFrame({"day": np.arange(1, days + 1)})
daily["cum_spend"] = (budget / days) * daily["day"]
daily["cum_impressions"] = (sim["impressions"] / days) * daily["day"]
daily["cum_clicks"] = (sim["clicks"] / days) * daily["day"]
daily["cum_conversions"] = (sim["approved_conversions"] / days) * daily["day"]

c1, c2 = st.columns(2)
with c1:
    st.line_chart(daily.set_index("day")[["cum_impressions", "cum_clicks"]])
with c2:
    st.line_chart(daily.set_index("day")[["cum_spend", "cum_conversions"]])

st.caption(
    "⚠️ Important : c’est une simulation (What-If) basée sur les ratios historiques du segment (CTR, CPM, taux de conversion). "
    "Elle répond à : “si on met X€ pendant Y jours sur ce segment, à quoi s’attendre ?”"
)
