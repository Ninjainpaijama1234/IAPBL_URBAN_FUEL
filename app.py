# urban_fuel_dashboard/app.py
# ------------------------------------------------------------------
#  Urban Fuel – Streamlit Analytics Dashboard (robust v2) · July 2025
# ------------------------------------------------------------------
#  ✓ Loads **UrbanFuelSyntheticSurvey.csv** with given lowercase headers.
#  ✓ Central RENAME map ➜ human‑readable names.
#  ✓ Fully defensive Income slider (never casts NaN to int).
#  ✓ Gradient KPI cards + modern Plotly theme.
# ------------------------------------------------------------------

import os, base64, numpy as np, pandas as pd, streamlit as st
import plotly.express as px, plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules

# ------------------------------------------------------------------
#  CONFIG & STYLE
# ------------------------------------------------------------------
st.set_page_config(page_title="Urban Fuel Analytics", page_icon="🍱", layout="wide")

st.markdown(
    """
    <style>
        /* Hide default Streamlit extras */
        #MainMenu, footer {visibility:hidden;}
        [data-testid="stAppViewContainer"]>.main {background:#F5F8FF;}
        /* KPI cards */
        .metric{padding:1rem;border-radius:14px;background:linear-gradient(135deg,#ffffff 0%,#f0f4ff 100%);box-shadow:0 2px 6px rgba(0,0,0,.06)}
        .metric h3{font-size:.8rem;font-weight:600;color:#5a5a5a;margin:0}
        .metric h2{font-size:1.8rem;font-weight:700;color:#1E3A8A;margin:0}
        /* Download link */
        a.dl{background:#4F8BF9;color:#fff!important;padding:6px 14px;border-radius:8px;text-decoration:none;font-weight:600}
        a.dl:hover{background:#3c6fe0}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_FILE = "UrbanFuelSyntheticSurvey.csv"

# ------------------------------------------------------------------
#  HELPERS
# ------------------------------------------------------------------
RENAME = {
    "age": "Age",
    "gender": "Gender",
    "income_inr": "Income",            # numeric already
    "city": "City",
    "employment_type": "EmploymentType",
    "willing_to_pay_mealkit_inr": "SpendMealKit",
}

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")

@st.cache_data
def load_df() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        st.error(f"{DATA_FILE} not found."); st.stop()
    df = pd.read_csv(DATA_FILE)
    df.rename(columns={k:v for k,v in RENAME.items() if k in df.columns}, inplace=True)
    # helper numeric cols
    if "Income" in df:
        df["IncomeValue"] = df["Income"]
    if "SpendMealKit" in df:
        df["SpendMealKitValue"] = df["SpendMealKit"]
    return df

def num_cols(df):
    return df.select_dtypes("number").columns.tolist()

def dl_link(df, name):
    b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return f'<a class="dl" download="{name}" href="data:file/csv;base64,{b64}">Download ▼</a>'

# ------------------------------------------------------------------
#  LOAD DATA
# ------------------------------------------------------------------
df = load_df()

# ------------------------------------------------------------------
#  KPI CARDS
# ------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="metric"><h3>Respondents</h3><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)

if "Age" in df:
    c2.markdown(f'<div class="metric"><h3>Avg Age</h3><h2>{df["Age"].mean():.1f}</h2></div>', unsafe_allow_html=True)
else:
    c2.markdown('<div class="metric"><h3>Avg Age</h3><h2>–</h2></div>', unsafe_allow_html=True)

if "SpendMealKitValue" in df:
    c3.markdown(f'<div class="metric"><h3>Avg Kit Spend</h3><h2>₹{df["SpendMealKitValue"].mean():,.0f}</h2></div>', unsafe_allow_html=True)
else:
    c3.markdown('<div class="metric"><h3>Avg Kit Spend</h3><h2>–</h2></div>', unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------------
#  FILTER PANEL
# ------------------------------------------------------------------
with st.expander("🎛 Filters", expanded=False):
    # City filter
    city_sel = st.multiselect("City", sorted(df["City"].unique()) if "City" in df else [],
                              default=sorted(df["City"].unique()) if "City" in df else [])

    # Income slider – fully defensive
    if "IncomeValue" in df:
        inc_vals = df["IncomeValue"].dropna()
    else:
        inc_vals = pd.Series(dtype=float)
    if not inc_vals.empty:
        inc_min, inc_max = map(int, [inc_vals.min(), inc_vals.max()])
        inc_sel = st.slider("Income range (₹)", inc_min, inc_max, (inc_min, inc_max), 10000)
    else:
        inc_sel = None

# Apply filters
filtered = df.copy()
if city_sel:
    filtered = filtered[filtered["City"].isin(city_sel)]
if inc_sel is not None:
    filtered = filtered.query("IncomeValue >= @inc_sel[0] & IncomeValue <= @inc_sel[1]")

# ------------------------------------------------------------------
#  TABS
# ------------------------------------------------------------------
visual_tab, cls_tab, clu_tab, rule_tab, reg_tab = st.tabs([
    "📊 Visuals", "🤖 Classification", "📦 Clustering", "🔗 Rules", "📈 Regression"])

# --------------------------- VISUALS ----------------------------------------
with visual_tab:
    st.subheader("Age & Income Distributions")
    col_v1, col_v2 = st.columns(2)
    if "Age" in filtered:
        col_v1.plotly_chart(px.histogram(filtered, x="Age", nbins=25, color_discrete_sequence=["#4F8BF9"]), use_container_width=True)
    if "IncomeValue" in filtered:
        col_v2.plotly_chart(px.histogram(filtered, x="IncomeValue", nbins=30, color_discrete_sequence=["#FF6B6B"]), use_container_width=True)

    st.markdown(dl_link(filtered, "filtered.csv"), unsafe_allow_html=True)

# --------------------------- CLASSIFICATION ---------------------------------
with cls_tab:
    st.subheader("Choose Target (categorical)")
    cat_targets = [c for c in filtered if filtered[c].dtype == "object"]
    if cat_targets:
        target = st.selectbox("Target", cat_targets)
        X, y = filtered.drop(columns=[target]), filtered[target]
        pre = ColumnTransformer([
            ("num", StandardScaler(), num_cols(filtered)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in filtered.columns if c not in num_cols(filtered)+[target]])
        ])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        res = []
        for name, clf in {"KNN":KNeighborsClassifier(),"DT":DecisionTreeClassifier(random_state=42),"RF":RandomForestClassifier(n_estimators=200,random_state=42),"GB":GradientBoostingClassifier(random_state=42)}.items():
            pipe = Pipeline([("p",pre),("c",clf)]).fit(Xtr,ytr)
            res.append({"Model":name,"Acc":accuracy_score(yte,pipe.predict(Xte))})
        st.dataframe(pd.DataFrame(res).round(3))
    else:
        st.info("No categorical columns in data.")

# ------------------------------ CLUSTERING ----------------------------------
with clu_tab:
    st.subheader("K‑Means Quick Segments")
    nums = num_cols(filtered)
    if len(nums)>=2:
        k = st.slider("k",2,10,4)
        km = KMeans(n_clusters=k,random_state=42,n_init=10).fit(StandardScaler().fit_transform(filtered[nums]))
        cludf = filtered.assign(Cluster=km.labels_)
        st.dataframe(cludf.groupby("Cluster")[nums].mean().round(1))
        st.markdown(dl_link(cludf,"clusters.csv"), unsafe_allow_html=True)
    else:
        st.info("Need at least 2 numeric columns.")

# -------------------------- ASSOCIATION RULES -------------------------------
with rule_tab:
    st.subheader("Apriori Quick Rules")
    multi = [c for c in filtered if filtered[c].dtype=="object" and filtered[c].str.contains(",").any()]
    if multi:
        cols = st.multiselect("Select comma‑separated columns", multi, default=multi[:2])
        if st.button("Run Apriori"):
            basket = pd.concat([filtered[c].str.get_dummies(sep=", ") for c in cols],axis=1)
            rules = association_rules(apriori(basket,min_support=0.05,use_colnames=True),metric="confidence",min_threshold=0.3)
            st.dataframe(rules.head(10))
    else:
        st.info("No suitable multi‑select columns detected.")

# ------------------------------- REGRESSION ---------------------------------
with reg_tab:
    st.subheader("Numeric Regression")
    nums = num_cols(filtered)
    if nums:
        tgt = st.selectbox("Target", nums)
        Xr, yr = filtered[nums].drop(columns=[tgt]), filtered[tgt]
        if len(Xr.columns)==0:
            st.info("Select a target with at least one other numeric predictor.")
        else:
            Xtr,Xte,ytr,yte = train_test_split(Xr,yr,test_size=0.2,random_state=42)
            mdl = LinearRegression().fit(Xtr,ytr)
            st.write("R²:", mdl.score(Xte,yte).round(3))
    else:
        st.info("Dataset has no numeric columns.")
