# ------------------------------------------------------------------
#  Urban Fuel – Robust, Beautiful Dashboard (FIXED VERSION)
# ------------------------------------------------------------------

"""Streamlit application for analysing the synthetic Urban Fuel survey.
Key fixes vs. previous version:
- Column‑renaming layer to reconcile lowercase CSV headers with PascalCase
  references used in the UI.
- Defensive handling of numeric conversions (Income / Spend) so sliders do
  not crash when values are missing or non‑numeric.
- Helper utilities consolidated for clarity; numeric column detection
  excludes any object / boolean types.
"""

import os
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules

# ------------------------------------------------------------------
# PAGE CONFIG & GLOBAL STYLES
# ------------------------------------------------------------------
st.set_page_config(page_title="Urban Fuel Analytics", page_icon="🍱", layout="wide")
CUSTOM_CSS = """
<style>
#MainMenu, footer {visibility:hidden;}
[data-testid="stAppViewContainer"] > .main {background:#F5F8FF;}
.metric {padding:1rem;border-radius:12px;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,0.08);} 
.metric h3 {font-size:0.85rem;margin:0;color:#666}
.metric h2 {font-size:1.8rem;margin:0;color:#223D62}

/* download link */
a.dl {background:#4F8BF9;color:#fff!important;padding:6px 14px;border-radius:8px;text-decoration:none;font-weight:600;}
a.dl:hover{background:#3c6fe0}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------
DATA_FILE = "UrbanFuelSyntheticSurvey.csv"

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------

def _to_num(series: pd.Series) -> pd.Series:
    """Strip any non‑numeric characters and coerce to float."""
    cleaned = series.astype(str).str.replace(r"[^\d.]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def num_cols(df: pd.DataFrame):
    """Return strictly numeric feature names (exclude bool/object)."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def dl_link(df: pd.DataFrame, name: str) -> str:
    """Create an inline download link for a dataframe as CSV."""
    b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return f'<a class="dl" href="data:file/csv;base64,{b64}" download="{name}">Download ▼</a>'

# ------------------------------------------------------------------
# DATA LOADER WITH COLUMN‑MAPPING & SANITISATION
# ------------------------------------------------------------------
RENAME = {
    "age": "Age",
    "city": "City",
    "income_inr": "Income",          # will create IncomeValue numeric clone
    "willing_to_pay_mealkit_inr": "SpendMealKit"  # numeric already
}

@st.cache_data
def load_df() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        st.error(f"{DATA_FILE} not found – please upload the CSV first.")
        st.stop()

    df = pd.read_csv(DATA_FILE)
    # Harmonise headers
    df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns}, inplace=True)

    # ------------------------------------------------------------------
    # Create explicit numeric helper columns
    # ------------------------------------------------------------------
    if "Income" in df:
        if pd.api.types.is_numeric_dtype(df["Income"]):
            df["IncomeValue"] = df["Income"]
        else:
            df["IncomeValue"] = _to_num(df["Income"])

    if "SpendMealKit" in df:
        if pd.api.types.is_numeric_dtype(df["SpendMealKit"]):
            df["SpendMealKitValue"] = df["SpendMealKit"]
        else:
            df["SpendMealKitValue"] = _to_num(df["SpendMealKit"])

    return df

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
df = load_df()

# ------------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------------
c1, c2, c3 = st.columns(3)

# Respondent count
c1.markdown(f'<div class="metric"><h3>Respondents</h3><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)

# Average age
if "Age" in df:
    c2.markdown(f'<div class="metric"><h3>Avg Age</h3><h2>{df["Age"].mean():.1f}</h2></div>', unsafe_allow_html=True)
else:
    c2.markdown('<div class="metric"><h3>Avg Age</h3><h2>N/A</h2></div>', unsafe_allow_html=True)

# Average spend
if "SpendMealKitValue" in df and df["SpendMealKitValue"].notna().any():
    c3.markdown(f'<div class="metric"><h3>Avg Spend</h3><h2>₹{df["SpendMealKitValue"].mean():,.0f}</h2></div>', unsafe_allow_html=True)
else:
    c3.markdown('<div class="metric"><h3>Avg Spend</h3><h2>₹–</h2></div>', unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------------
# FILTERS
# ------------------------------------------------------------------
with st.expander("🎛 Filters", expanded=False):
    # City filter
    if "City" in df:
        city_sel = st.multiselect("City", sorted(df["City"].dropna().unique()), default=list(df["City"].dropna().unique()))
    else:
        city_sel = []

    # Income slider (robust)
    if "IncomeValue" in df and df["IncomeValue"].notna().any():
        inc_vals = df["IncomeValue"].dropna()
        inc_min, inc_max = int(inc_vals.min()), int(inc_vals.max())
        inc_sel = st.slider("Income range (₹)", inc_min, inc_max, (inc_min, inc_max), step=10_000)
    else:
        st.info("No numeric income column detected; skipping income filter.")
        inc_sel = None

# Apply filters
filtered = df.copy()
if city_sel:
    filtered = filtered[filtered["City"].isin(city_sel)]
if inc_sel is not None:
    filtered = filtered.query("IncomeValue >= @inc_sel[0] & IncomeValue <= @inc_sel[1]")

# ------------------------------------------------------------------
# TABS LAYOUT
# ------------------------------------------------------------------
visual_tab, class_tab, cluster_tab, rules_tab, reg_tab = st.tabs(
    ["📊 Visuals", "🤖 Classification", "📦 Clustering", "🔗 Rules", "📈 Regression"])

# ------------------------------------------------------------------
# VISUALS TAB
# ------------------------------------------------------------------
with visual_tab:
    if "Age" in filtered:
        st.plotly_chart(px.histogram(filtered, x="Age", nbins=25), use_container_width=True)

    if "IncomeValue" in filtered:
        st.plotly_chart(px.histogram(filtered, x="IncomeValue", nbins=30, color_discrete_sequence=["#4F8BF9"]), use_container_width=True)

    st.markdown(dl_link(filtered, "filtered.csv"), unsafe_allow_html=True)

# ------------------------------------------------------------------
# CLASSIFICATION TAB
# ------------------------------------------------------------------
with class_tab:
    cat_targets = [c for c in filtered.columns if filtered[c].dtype == "object" and filtered[c].nunique() <= 10]
    if not cat_targets:
        st.info("No categorical targets with ≤10 unique classes detected.")
    else:
        target = st.selectbox("Target", cat_targets)
        X, y = filtered.drop(columns=[target]), filtered[target]
        pre = ColumnTransformer([
            ("num", StandardScaler(), num_cols(X)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in X.columns if c not in num_cols(X)])
        ])
        Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        results = []
        models = {
            "KNN": KNeighborsClassifier(),
            "DT": DecisionTreeClassifier(random_state=42),
            "RF": RandomForestClassifier(n_estimators=300, random_state=42),
            "GB": GradientBoostingClassifier(random_state=42)
        }
        for name, clf in models.items():
            pipe = Pipeline([("prep", pre), ("model", clf)])
            pipe.fit(Xtr, ytr)
            yhat = pipe.predict(Xte)
            results.append({"Model": name, "Accuracy": accuracy_score(yte, yhat)})
        st.dataframe(pd.DataFrame(results).set_index("Model").round(3))

# ------------------------------------------------------------------
# CLUSTERING TAB
# ------------------------------------------------------------------
with cluster_tab:
    numeric_features = num_cols(filtered)
    if len(numeric_features) < 2:
        st.info("Need at least two numeric columns for K‑Means clustering.")
    else:
        k = st.slider("Choose k", 2, 10, 4)
        scaled = StandardScaler().fit_transform(filtered[numeric_features])
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        clusters = km.fit_predict(scaled)
        clus_df = filtered.assign(Cluster=clusters)
        st.dataframe(clus_df.groupby("Cluster")[numeric_features].mean().round(1))
        st.markdown(dl_link(clus_df, "clusters.csv"), unsafe_allow_html=True)

# ------------------------------------------------------------------
# ASSOCIATION RULE MINING TAB
# ------------------------------------------------------------------
with rules_tab:
    multi_cols = [c for c in filtered.columns if filtered[c].dtype == "object" and filtered[c].str.contains(",").any()]
    if not multi_cols:
        st.info("No multi‑select columns found for Apriori analysis.")
    else:
        chosen_cols = st.multiselect("Columns for Apriori", multi_cols, default=multi_cols[:3])
        if st.button("Run Apriori"):
            basket = pd.concat([filtered[c].str.get_dummies(sep=", ") for c in chosen_cols], axis=1)
            freq = apriori(basket, min_support=0.05, use_colnames=True)
            rules = association_rules(freq, metric="confidence", min_threshold=0.3)
            st.dataframe(rules.head(15))

# ------------------------------------------------------------------
# REGRESSION TAB
# ------------------------------------------------------------------
with reg_tab:
    numeric_features = num_cols(filtered)
    if len(numeric_features) < 2:
        st.info("Need at least two numeric columns to run regression.")
    else:
        target_num = st.selectbox("Regression target", numeric_features)
        Xr = filtered[numeric_features].drop(columns=[target_num])
        yr = filtered[target_num]
        Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
        reg = LinearRegression().fit(Xtr, ytr)
        r2 = reg.score(Xte, yte)
        st.metric(label="R² on holdout", value=f"{r2:.3f}")
