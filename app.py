# urban_fuel_dashboard/app.py
# ------------------------------------------------------------------
#  Urban Fuel – Streamlit Dashboard  (v3, column-safe)
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

# ------------------------------------------------------------------#
#  1.  UI CONFIG
# ------------------------------------------------------------------#
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

DATA_FILE = "UrbanFuelSyntheticSurvey.csv"

# ------------------------------------------------------------------#
#  2.  HELPERS
# ------------------------------------------------------------------#
def numerics(df):  # list of numeric columns
    return df.select_dtypes(include="number").columns.tolist()


def download_link(df, name):
    b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return (
        f'<a class="dl" href="data:file/csv;base64,{b64}" '
        f'download="{name}">Download ▼</a>'
    )

# ------------------------------------------------------------------#
#  3.  LOAD + NORMALISE CSV
# ------------------------------------------------------------------#
@st.cache_data
def load_df():
    if not os.path.exists(DATA_FILE):
        st.error(f"`{DATA_FILE}` not found");  st.stop()

    df = pd.read_csv(DATA_FILE)
    # normalise column names to lower case
    df.columns = [c.lower() for c in df.columns]

    # rename two critical fields so downstream code is simple
    df = df.rename(
        columns={
            "income_inr": "income",                      # numeric
            "willing_to_pay_mealkit_inr": "spend_mealkit"  # numeric
        }
    )

    # add helper columns (identical numeric copy, useful for sliders)
    if "income" in df:
        df["income_value"] = df["income"]
    if "spend_mealkit" in df:
        df["spend_mealkit_value"] = df["spend_mealkit"]

    return df


df = load_df()

# ------------------------------------------------------------------#
#  4.  KPI CARDS
# ------------------------------------------------------------------#
c1, c2, c3 = st.columns(3)
c1.markdown(
    f'<div class="metric"><h3>Respondents</h3><h2>{len(df):,}</h2></div>',
    unsafe_allow_html=True,
)
c2.markdown(
    f'<div class="metric"><h3>Avg Age</h3><h2>{df["age"].mean():.1f}</h2></div>',
    unsafe_allow_html=True,
)
c3.markdown(
    f'<div class="metric"><h3>Avg Spend/Meal Kit</h3>'
    f'<h2>₹{df["spend_mealkit_value"].mean():,.0f}</h2></div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ------------------------------------------------------------------#
#  5.  FILTERS
# ------------------------------------------------------------------#
with st.expander("🎛 Filters", expanded=False):
    if "city" in df:
        city_sel = st.multiselect("City", sorted(df["city"].unique()),
                                  default=list(df["city"].unique()))
    else:
        city_sel = []

    if "income_value" in df and not df["income_value"].isna().all():
        inc = df["income_value"].dropna()
        inc_min, inc_max = int(inc.min()), int(inc.max())
        inc_sel = st.slider("Income range (₹)", inc_min, inc_max,
                            (inc_min, inc_max), step=10_000)
    else:
        st.info("Income slider hidden – no numeric income data"); inc_sel = None

# apply filters
df_f = df.copy()
if city_sel:
    df_f = df_f[df_f["city"].isin(city_sel)]
if inc_sel is not None:
    df_f = df_f.query("@inc_sel[0] <= income_value <= @inc_sel[1]")

# ------------------------------------------------------------------#
#  6.  TABS
# ------------------------------------------------------------------#
tab_v, tab_c, tab_k, tab_a, tab_r = st.tabs(
    ["📊 Visuals", "🤖 Classification", "📦 Clustering", "🔗 Rules", "📈 Regression"]
)

# ---- Visuals ------------------------------------------------------
with tab_v:
    if "age" in df_f:
        st.plotly_chart(px.histogram(df_f, x="age", nbins=30), use_container_width=True)

    if "income_value" in df_f:
        st.plotly_chart(
            px.histogram(df_f, x="income_value", nbins=30,
                         color_discrete_sequence=["#4F8BF9"]),
            use_container_width=True,
        )

    st.markdown(download_link(df_f, "filtered.csv"), unsafe_allow_html=True)

# ---- Classification ----------------------------------------------
with tab_c:
    targets = [c for c in df_f.columns if df_f[c].dtype == "object"]
    if not targets:
        st.info("No categorical columns found.")
    else:
        target = st.selectbox("Target variable", targets)
        X, y = df_f.drop(columns=[target]), df_f[target]

        pre = ColumnTransformer([
            ("num", StandardScaler(), numerics(df_f)),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             [c for c in df_f.columns if c not in numerics(df_f) + [target]])
        ])

        Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2,
                                              random_state=42)
        res = []
        for name, clf in {
            "KNN": KNeighborsClassifier(),
            "DT": DecisionTreeClassifier(random_state=42),
            "RF": RandomForestClassifier(n_estimators=200, random_state=42),
            "GB": GradientBoostingClassifier(random_state=42)
        }.items():
            pipe = Pipeline([("prep", pre), ("clf", clf)])
            pipe.fit(Xtr, ytr)
            res.append({"Model": name, "Acc": accuracy_score(yte, pipe.predict(Xte))})

        st.dataframe(pd.DataFrame(res).round(3))

# ---- Clustering ---------------------------------------------------
with tab_k:
    num = numerics(df_f)
    if len(num) >= 2:
        k = st.slider("k (clusters)", 2, 10, 4)
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(
            StandardScaler().fit_transform(df_f[num]))
        df_k = df_f.assign(cluster=km.labels_)
        st.dataframe(df_k.groupby("cluster")[num].mean().round(1))
        st.markdown(download_link(df_k, "clusters.csv"), unsafe_allow_html=True)
    else:
        st.info("Need ≥2 numeric columns for K-Means")

# ---- Association Rules --------------------------------------------
with tab_a:
    multi = [c for c in df_f if df_f[c].dtype == "object" and df_f[c].str.contains(",").any()]
    if multi:
        sel = st.multiselect("Columns for Apriori", multi, default=multi[:3])
        if st.button("Run Apriori"):
            basket = pd.concat([df_f[c].str.get_dummies(sep=",") for c in sel], axis=1)
            rules = association_rules(apriori(basket, min_support=0.05,
                                              use_colnames=True),
                                      metric="confidence",
                                      min_threshold=0.3).head(10)
            st.dataframe(rules)
    else:
        st.info("No multi-select columns detected.")

# ---- Regression ---------------------------------------------------
with tab_r:
    num = numerics(df_f)
    if num:
        tgt = st.selectbox("Target", num)
        Xr, yr = df_f[num].drop(columns=[tgt]), df_f[tgt]
        Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
        reg = LinearRegression().fit(Xtr, ytr)
        st.write("R²:", round(reg.score(Xte, yte), 3))
    else:
        st.info("No numeric columns available.")
