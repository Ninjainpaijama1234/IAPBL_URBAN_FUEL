# urban_fuel_dashboard/app.py
# ------------------------------------------------------------------
#  Urban Fuel – Robust, Beautiful Dashboard
# ------------------------------------------------------------------

import os, base64, numpy as np, pandas as pd, streamlit as st
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

# ====== CONFIG & STYLE ========================================================
st.set_page_config(page_title="Urban Fuel Analytics", page_icon="🍱", layout="wide")
st.markdown(
    """
    <style>
        #MainMenu, footer {visibility:hidden;}
        [data-testid="stAppViewContainer"] > .main {background:#F5F8FF;}
        .metric {padding:1rem;border-radius:12px;background:#fff;
                 box-shadow:0 1px 3px rgba(0,0,0,0.08);}
        .metric h3 {font-size:0.85rem;margin:0;color:#666}
        .metric h2 {font-size:1.8rem;margin:0;color:#223D62}
        a.dl {background:#4F8BF9;color:#fff!important;padding:6px 14px;
              border-radius:8px;text-decoration:none;font-weight:600;}
        a.dl:hover{background:#3c6fe0}
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_FILE = "urban_fuel_survey_synthetic.csv"

# ------------------------------------------------------------------ #
#                       HELPERS & LOAD DATA                          #
# ------------------------------------------------------------------ #

# master rename map (synthetic → dashboard convention)
RENAME = {
    "age": "Age",
    "city": "City",
    "income_inr": "Income",                         # still numeric
    "willing_to_pay_mealkit_inr": "SpendMealKit",
}

def _to_num(s: pd.Series) -> pd.Series:
    """Strip non-numeric chars and coerce to float."""
    return (
        s.astype(str)
        .str.replace(r"[^\d.]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )

@st.cache_data
def load_df() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        st.error(f"{DATA_FILE} missing."); st.stop()

    df = pd.read_csv(DATA_FILE)

    # harmonise column names
    df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns},
              inplace=True)

    # ---- Create numeric helper fields --------------------------------------
    # Income
    if "Income" in df:
        if pd.api.types.is_numeric_dtype(df["Income"]):
            df["IncomeValue"] = df["Income"]
        else:
            df["IncomeValue"] = _to_num(df["Income"])

    # Spend per meal kit
    if "SpendMealKit" in df:
        if pd.api.types.is_numeric_dtype(df["SpendMealKit"]):
            df["SpendMealKitValue"] = df["SpendMealKit"]
        else:
            df["SpendMealKitValue"] = _to_num(df["SpendMealKit"])

    return df

def num_cols(df):      # convenient numeric selector
    return df.select_dtypes("number").columns.tolist()

def dl_link(df, name):
    b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return f'<a class="dl" href="data:file/csv;base64,{b64}" download="{name}">Download ▼</a>'

# ------------------------------------------------------------------ #
#                              DATASET                               #
# ------------------------------------------------------------------ #
df = load_df()

# ------------------------------------------------------------------ #
#                            KPI CARDS                               #
# ------------------------------------------------------------------ #
c1, c2, c3 = st.columns(3)

c1.markdown(f'<div class="metric"><h3>Respondents</h3><h2>{len(df):,}</h2></div>',
            unsafe_allow_html=True)

age_mean = df["Age"].mean() if "Age" in df else np.nan
c2.markdown(f'<div class="metric"><h3>Avg Age</h3><h2>{age_mean:.1f}</h2></div>',
            unsafe_allow_html=True)

spend = df["SpendMealKitValue"].mean() if "SpendMealKitValue" in df else np.nan
c3.markdown(f'<div class="metric"><h3>Avg Spend / Kit</h3><h2>₹{spend:,.0f}</h2></div>',
            unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------------ #
#                               FILTERS                              #
# ------------------------------------------------------------------ #
with st.expander("🎛 Filters", expanded=False):
    # -- City filter
    if "City" in df:
        city_sel = st.multiselect("City", sorted(df["City"].unique()),
                                  default=list(df["City"].unique()))
    else:
        city_sel = []

    # -- Income filter (robust, works with text or numeric) ----------
    if "IncomeValue" in df:
        inc_vals = df["IncomeValue"].dropna()
    else:
        inc_vals = pd.to_numeric(df.get("Income"), errors="coerce").dropna()

    if not inc_vals.empty:
        inc_min, inc_max = int(inc_vals.min()), int(inc_vals.max())
        inc_sel = st.slider("Income range (₹)", inc_min, inc_max,
                            (inc_min, inc_max), step=10_000)
    else:
        st.info("No numeric income data found.")
        inc_sel = None

# -- Apply filters
df_f = df.copy()
if city_sel:
    df_f = df_f[df_f["City"].isin(city_sel)]
if inc_sel is not None:
    df_f = df_f.query("IncomeValue >= @inc_sel[0] & IncomeValue <= @inc_sel[1]")

# ------------------------------------------------------------------ #
#                                TABS                                #
# ------------------------------------------------------------------ #
tab_v, tab_c, tab_k, tab_a, tab_r = st.tabs(
    ["📊 Visuals", "🤖 Classification", "📦 Clustering", "🔗 Rules", "📈 Regression"]
)

# ------------------------------- VISUALS --------------------------- #
with tab_v:
    if "Age" in df_f:
        st.plotly_chart(px.histogram(df_f, x="Age", nbins=25), use_container_width=True)

    if "IncomeValue" in df_f:
        st.plotly_chart(
            px.histogram(df_f, x="IncomeValue", nbins=30,
                         color_discrete_sequence=["#4F8BF9"]),
            use_container_width=True,
        )
    st.markdown(dl_link(df_f, "filtered.csv"), unsafe_allow_html=True)

# --------------------------- CLASSIFICATION ----------------------- #
with tab_c:
    cat_targets = [c for c in df_f if df_f[c].dtype == "object"]
    if not cat_targets:
        st.info("No categorical targets available."); st.stop()

    target = st.selectbox("Target", cat_targets)
    X, y = df_f.drop(columns=[target]), df_f[target]

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols(df_f)),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             [c for c in df_f.columns if c not in num_cols(df_f) + [target]]),
        ]
    )

    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    res = []
    for name, clf in {
        "KNN": KNeighborsClassifier(),
        "DT": DecisionTreeClassifier(random_state=42),
        "RF": RandomForestClassifier(n_estimators=200, random_state=42),
        "GB": GradientBoostingClassifier(random_state=42),
    }.items():
        pipe = Pipeline([("p", pre), ("c", clf)])
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)
        res.append({"Model": name, "Acc": accuracy_score(yte, yhat)})

    st.dataframe(pd.DataFrame(res).round(3))

# ----------------------------- CLUSTERING ------------------------- #
with tab_k:
    numeric = num_cols(df_f)
    if len(numeric) >= 2:
        k = st.slider("k", 2, 10, 4)
        mod = KMeans(n_clusters=k, n_init=10, random_state=42).fit(
            StandardScaler().fit_transform(df_f[numeric])
        )
        df_k = df_f.assign(Cluster=mod.labels_)
        st.dataframe(df_k.groupby("Cluster")[numeric].mean().round(1))
        st.markdown(dl_link(df_k, "clusters.csv"), unsafe_allow_html=True)
    else:
        st.info("Need at least two numeric columns for K-means.")

# ----------------------- ASSOCIATION RULES ------------------------ #
with tab_a:
    multi = [
        c for c in df_f
        if df_f[c].dtype == "object" and df_f[c].str.contains(",").any()
    ]
    if multi:
        sel = st.multiselect("Columns", multi, default=multi[:3])
        if st.button("Run Apriori"):
            basket = pd.concat(
                [df_f[c].str.get_dummies(sep=", ") for c in sel], axis=1
            )
            rules = association_rules(
                apriori(basket, min_support=0.05, use_colnames=True),
                metric="confidence", min_threshold=0.3,
            ).head(10)
            st.dataframe(rules)
    else:
        st.info("No multi-select columns detected for association mining.")

# ----------------------------- REGRESSION ------------------------- #
with tab_r:
    nums = num_cols(df_f)
    if nums:
        tgt = st.selectbox("Target", nums)
        Xr, yr = df_f[nums].drop(columns=[tgt]), df_f[tgt]
        Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
        reg = LinearRegression().fit(Xtr, ytr)
        pred = reg.predict(Xte)
        st.write("R²:", reg.score(Xte, yte).round(3))
    else:
        st.info("No numeric columns available for regression.")
