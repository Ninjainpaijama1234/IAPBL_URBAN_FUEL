# urban_fuel_dashboard/app.py
# ------------------------------------------------------------------
#  Urban Fuel – Streamlit Dashboard  ·  July 2025
#  Robust against text-range Income values
# ------------------------------------------------------------------

import os, base64, numpy as np, pandas as pd, streamlit as st
import plotly.express as px, plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules

# ------------------------------------------------------------------
# Page styling
# ------------------------------------------------------------------
st.set_page_config(page_title="Urban Fuel Analytics",
                   page_icon="🍱", layout="wide")
st.markdown("""
<style>
#MainMenu, footer {visibility:hidden;}
[data-testid="stAppViewContainer"] > .main {background:#F5F8FF;}
.metric-box {padding:1rem;border-radius:12px;background:#fff;
box-shadow:0 1px 3px rgba(0,0,0,0.08);}
.metric-title{font-size:0.85rem;color:#666}
.metric-val{font-size:1.8rem;font-weight:600;color:#223D62}
.dl-btn{background:#4F8BF9;padding:0.5rem 1rem;color:#fff!important;
text-decoration:none;border-radius:8px;font-weight:600;}
.dl-btn:hover{background:#3c6fe0;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------
DATA_FILE = "Urban_Fuel_Synthetic_Dataset.csv"

def _to_numeric(series: pd.Series) -> pd.Series:
    """Strip non-digits (₹, commas, k) and return float; NaN if not possible."""
    cleaned = (series.astype(str)
                     .str.replace(r"[^\d.]", "", regex=True)
                     .replace("", np.nan)
                     .astype(float))
    return cleaned

@st.cache_data
def load_df() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        st.error(f"{DATA_FILE} not found next to app.py — upload & restart.")
        st.stop()
    df = pd.read_csv(DATA_FILE)

    # Spend conversion
    if "SpendMealKit" in df.columns and df["SpendMealKit"].dtype == "object":
        df["SpendMealKitValue"] = _to_numeric(df["SpendMealKit"])

    # ▶ FIX — Attempt numeric cast of Income; keep both if textual
    if "Income" in df.columns and df["Income"].dtype == "object":
        numeric_income = _to_numeric(df["Income"])
        if numeric_income.notna().sum() > 0:
            df["IncomeValue"] = numeric_income  # add numeric proxy
    return df

def _numeric_cols(data: pd.DataFrame):
    return data.select_dtypes(include=["int64", "float64"]).columns.tolist()

def _download_link(df: pd.DataFrame, name: str):
    b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return f'<a class="dl-btn" href="data:file/csv;base64,{b64}" download="{name}">Download ▼</a>'

def preprocess(df, target):
    num = _numeric_cols(df)
    cat = [c for c in df.columns if c not in num + [target]]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

def cls_models():
    return {
        "K-NN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

def reg_models():
    return {
        "Linear": LinearRegression(),
        "Ridge":  Ridge(alpha=1.0),
        "Lasso":  Lasso(alpha=0.01),
        "Decision Tree Reg": DecisionTreeRegressor(max_depth=6, random_state=42)
    }

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
df = load_df()

# ------------------------------------------------------------------
# Sidebar – theme
# ------------------------------------------------------------------
st.sidebar.image("https://i.imgur.com/aDKW5J1.png", width=160)
plot_theme = st.sidebar.selectbox(
    "Plotly theme", ["plotly", "plotly_dark", "ggplot2", "seaborn"], 0)

# ------------------------------------------------------------------
# KPI cards
# ------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="metric-box"><div class="metric-title">Respondents</div>'
            f'<div class="metric-val">{len(df):,}</div></div>', unsafe_allow_html=True)
avg_age = df["Age"].mean() if "Age" in df.columns else np.nan
c2.markdown(f'<div class="metric-box"><div class="metric-title">Avg Age</div>'
            f'<div class="metric-val">{avg_age:.1f}</div></div>', unsafe_allow_html=True)
spend_col = "SpendMealKitValue" if "SpendMealKitValue" in df else None
avg_spend = f'₹{df[spend_col].mean():,.0f}' if spend_col else "—"
c3.markdown(f'<div class="metric-box"><div class="metric-title">Avg Spend</div>'
            f'<div class="metric-val">{avg_spend}</div></div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------
# Filters
# ------------------------------------------------------------------
with st.expander("🎛️ Filters", expanded=False):
    # City filter
    if "City" in df.columns:
        sel_city = st.multiselect("City", df["City"].unique(),
                                  default=list(df["City"].unique()))
    else:
        sel_city = None

    # ▶ FIX — Income filter: numeric slider if possible else category multiselect
    if "IncomeValue" in df:
        inc_min, inc_max = int(df["IncomeValue"].min()), int(df["IncomeValue"].max())
        sel_inc = st.slider("Income range (₹)", inc_min, inc_max, (inc_min, inc_max),
                            step=10000)
    elif df["Income"].dtype != "object":
        inc_min, inc_max = int(df["Income"].min()), int(df["Income"].max())
        sel_inc = st.slider("Income range (₹)", inc_min, inc_max, (inc_min, inc_max),
                            step=10000)
    else:
        inc_categories = df["Income"].unique()
        sel_inc_cat = st.multiselect("Income categories", inc_categories,
                                     default=list(inc_categories))
        sel_inc = None

# Apply filters
df_f = df.copy()
if sel_city: df_f = df_f[df_f["City"].isin(sel_city)]
if sel_inc is not None and "IncomeValue" in df_f:
    df_f = df_f.query("IncomeValue >= @sel_inc[0] & IncomeValue <= @sel_inc[1]")
elif sel_inc is not None and "Income" in df_f and df_f["Income"].dtype != "object":
    df_f = df_f.query("Income >= @sel_inc[0] & Income <= @sel_inc[1]")
elif "sel_inc_cat" in locals():
    df_f = df_f[df_f["Income"].isin(sel_inc_cat)]

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Visuals", "🤖 Classification", "📦 Clustering",
     "🔗 Assoc Rules", "📈 Regression"])

# ---------------- Visuals ----------------
with tab1:
    st.subheader("Distribution Visuals")
    if "Age" in df_f:
        st.plotly_chart(px.histogram(df_f, x="Age", nbins=25,
                                     template=plot_theme), use_container_width=True)
    if "IncomeValue" in df_f:
        st.plotly_chart(px.histogram(df_f, x="IncomeValue", nbins=30,
                                     template=plot_theme,
                                     color_discrete_sequence=["#4F8BF9"]),
                        use_container_width=True)
    elif "Income" in df_f and df_f["Income"].dtype == "object":
        st.plotly_chart(px.histogram(df_f, x="Income", template=plot_theme,
                                     color_discrete_sequence=["#4F8BF9"]),
                        use_container_width=True)
    if {"City", "TryMealKit"}.issubset(df_f.columns):
        st.plotly_chart(px.histogram(df_f, x="City", color="TryMealKit",
                                     barmode="group", template=plot_theme),
                        use_container_width=True)
    st.markdown(_download_link(df_f, "filtered_data.csv"), unsafe_allow_html=True)

# ---------------- Classification ----------------
with tab2:
    st.subheader("Model Performance")
    cat_targets = [c for c in df_f if df_f[c].dtype == "object"]
    target = st.selectbox("Target", cat_targets)
    X, y = df_f.drop(columns=[target]), df_f[target]
    pre = preprocess(df_f, target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2,
                                          random_state=42)
    metrics, models = [], {}
    for n, clf in cls_models().items():
        pipe = Pipeline([("prep", pre), ("clf", clf)])
        pipe.fit(Xtr, ytr); yhat = pipe.predict(Xte)
        metrics.append({"Model": n, "Acc": accuracy_score(yte, yhat),
                        "Prec": precision_score(yte, yhat, average="weighted",
                                                zero_division=0),
                        "Rec": recall_score(yte, yhat, average="weighted",
                                            zero_division=0),
                        "F1": f1_score(yte, yhat, average="weighted",
                                       zero_division=0)})
        models[n] = pipe
    st.dataframe(pd.DataFrame(metrics).round(3))

    with st.expander("🔍 Feature Importance (Trees)"):
        m_choice = st.selectbox("Tree Model", ["Decision Tree", "Random Forest"])
        tree_clf = models[m_choice].named_steps["clf"]
        fn = models[m_choice].named_steps["prep"].get_feature_names_out()
        imp = pd.DataFrame({"Feature": fn, "Importance": tree_clf.feature_importances_}
                           ).sort_values("Importance", ascending=False).head(20)
        st.plotly_chart(px.bar(imp, y="Feature", x="Importance",
                               orientation="h", template=plot_theme,
                               color_discrete_sequence=["#4F8BF9"]),
                        use_container_width=True)

# ---------------- Clustering ----------------
with tab3:
    st.subheader("K-Means Explorer")
    num_cols = _numeric_cols(df_f)
    if len(num_cols) < 2:
        st.info("Need ≥2 numeric cols."); st.stop()
    k = st.slider("k", 2, 10, 4)
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42
                    ).fit(StandardScaler().fit_transform(df_f[num_cols]))
    df_k = df_f.assign(Cluster=kmeans.labels_)
    st.plotly_chart(px.scatter(df_k, x=num_cols[0], y=num_cols[1], color="Cluster",
                               template=plot_theme, symbol="Cluster"),
                    use_container_width=True)
    st.dataframe(df_k.groupby("Cluster")[num_cols].mean().round(1))
    st.markdown(_download_link(df_k, "clustered.csv"), unsafe_allow_html=True)

# ---------------- Assoc Rules ----------------
with tab4:
    st.subheader("Apriori")
    multis = [c for c in df_f if df_f[c].dtype == "object" and
              df_f[c].str.contains(",").any()]
    if not multis:
        st.info("No comma-sep cols.")
    else:
        cols = st.multiselect("Columns", multis, default=multis[:3])
        sup = st.slider("Min support", 0.01, 0.3, 0.05, 0.01)
        conf = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)
        if st.button("Run"):
            trans = pd.concat([df_f[c].str.get_dummies(sep=", ") for c in cols], axis=1)
            rules = (association_rules(apriori(trans, min_support=sup,
                                               use_colnames=True),
                                       metric="confidence", min_threshold=conf)
                     .sort_values("confidence", ascending=False).head(10))
            st.dataframe(rules[["antecedents", "consequents",
                                "support", "confidence", "lift"]])

# ---------------- Regression ----------------
with tab5:
    st.subheader("Regression & Forecast")
    nums = _numeric_cols(df_f)
    tgt = st.selectbox("Target", nums)
    preds = st.multiselect("Predictors", [c for c in nums if c != tgt],
                           default=nums[:min(3, len(nums))])
    if not preds:
        st.warning("Select ≥1 predictor."); st.stop()
    Xr, yr = df_f[preds], df_f[tgt]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    rows, reg_f = [], {}
    for n, m in reg_models().items():
        m.fit(Xtr, ytr); pred = m.predict(Xte)
        rows.append({"Model": n,
                     "MAE": mean_absolute_error(yte, pred),
                     "RMSE": np.sqrt(mean_squared_error(yte, pred)),
                     "R²": r2_score(yte, pred)})
        reg_f[n] = m
    st.dataframe(pd.DataFrame(rows).round(3))

    if {"City", "SpendMealKitValue"}.issubset(df_f):
        mdl = reg_f[st.selectbox("Model for forecast", list(reg_f))]
        months = np.arange(1, 13).reshape(-1, 1)
        fc = []
        for city, g in df_f.groupby("City"):
            base = g["SpendMealKitValue"].mean()
            y = base * (1 + 0.02*months.flatten()) + np.random.normal(0, base*0.05, 12)
            mdl.fit(months, y)
            fc.append(pd.DataFrame({"Month": months.flatten(),
                                    "Forecast": mdl.predict(months),
                                    "City": city}))
        fc_df = pd.concat(fc)
        st.plotly_chart(px.line(fc_df, x="Month", y="Forecast", color="City",
                                template=plot_theme), use_container_width=True)
        st.markdown(_download_link(fc_df, "forecast.csv"),
                    unsafe_allow_html=True)
