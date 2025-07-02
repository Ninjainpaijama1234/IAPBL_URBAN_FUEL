# urban_fuel_dashboard/app.py
# ------------------------------------------------------------------
#  Urban Fuel – Beautiful Streamlit Dashboard
#  July 2025   ·  Theme-aware, interactive, aesthetic
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
# ░ GLOBAL STYLING  ░
# ------------------------------------------------------------------
st.set_page_config(page_title="Urban Fuel Analytics",
                   page_icon="🍱", layout="wide")

CUSTOM_CSS = """
<style>
/* Hide default footer & menu */
#MainMenu, footer {visibility:hidden;}
/* Page background */
[data-testid="stAppViewContainer"] > .main {
    background: #F5F8FF;
}
/* Card-style metric boxes */
.metric-container {padding:1rem 1rem 1rem 1rem; border-radius:12px;
background:#FFFFFF; box-shadow:0 1px 3px rgba(0,0,0,0.08);}
.metric-title {font-size:0.9rem; color:#666666;}
.metric-value {font-size:1.8rem; font-weight:600; color:#223D62;}
/* Pretty download buttons */
.download-btn {
    background:#4F8BF9; padding:0.5rem 1rem; color:#fff !important;
    text-decoration:none; border-radius:8px; font-weight:600;}
.download-btn:hover {background:#3c6fe0;}
/* Tabs */
.stTabs [data-baseweb="tab"] {font-size:1.05rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------------
# ░ LOAD DATA & PREP  ░
# ------------------------------------------------------------------
DATA_FILE = "Urban_Fuel_Synthetic_Dataset.csv"

@st.cache_data
def load_df() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        st.error(f"**{DATA_FILE}** not found next to app.py – upload & restart.")
        st.stop()
    df = pd.read_csv(DATA_FILE)
    if "SpendMealKit" in df.columns and df["SpendMealKit"].dtype == "object":
        df["SpendMealKitValue"] = (
            df["SpendMealKit"].astype(str).str.replace(r"[^\d.]", "", regex=True)
            .astype(float)
        )
    return df

df = load_df()

# ------------------------------------------------------------------
# ░ SIDEBAR  ░
# ------------------------------------------------------------------
st.sidebar.image(
    "https://i.imgur.com/aDKW5J1.png",  # replace with your logo URL
    width=160
)
st.sidebar.markdown("### Urban Fuel Dashboard")
plotly_theme = st.sidebar.selectbox(
    "Chart Theme",
    ["plotly", "plotly_dark", "ggplot2", "seaborn"],
    index=0
)

# ------------------------------------------------------------------
# ░ HELPER FUNCTIONS  ░
# ------------------------------------------------------------------
def _download_link(data: pd.DataFrame, filename: str) -> str:
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return (
        f'<a class="download-btn" href="data:file/csv;base64,{b64}" '
        f'download="{filename}">Download ▼</a>'
    )

def _numeric_cols(data: pd.DataFrame):
    return data.select_dtypes(include=["int64", "float64"]).columns.tolist()

def build_preprocessor(df_, target):
    num = _numeric_cols(df_)
    cat = [c for c in df_.columns if c not in num + [target]]
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

def cls_models():  # Classification models
    return {
        "K-NN":               KNeighborsClassifier(),
        "Decision Tree":      DecisionTreeClassifier(random_state=42),
        "Random Forest":      RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting":  GradientBoostingClassifier(random_state=42)
    }

def reg_models():  # Regression models
    return {
        "Linear":             LinearRegression(),
        "Ridge":              Ridge(alpha=1.0),
        "Lasso":              Lasso(alpha=0.01),
        "Decision Tree Reg":  DecisionTreeRegressor(max_depth=6, random_state=42)
    }

# ------------------------------------------------------------------
# ░ KPI CARDS  ░
# ------------------------------------------------------------------
with st.container():
    col1, col2, col3 = st.columns(3)
    col1.markdown('<div class="metric-container">'
                  f'<div class="metric-title">Total Respondents</div>'
                  f'<div class="metric-value">{len(df):,}</div></div>',
                  unsafe_allow_html=True)
    avg_age = round(df["Age"].mean(), 1) if "Age" in df else "—"
    col2.markdown('<div class="metric-container">'
                  f'<div class="metric-title">Average Age</div>'
                  f'<div class="metric-value">{avg_age}</div></div>',
                  unsafe_allow_html=True)
    spend_col = "SpendMealKitValue" if "SpendMealKitValue" in df else None
    avg_spend = f"₹{df[spend_col].mean():,.0f}" if spend_col else "—"
    col3.markdown('<div class="metric-container">'
                  f'<div class="metric-title">Avg. Meal Spend</div>'
                  f'<div class="metric-value">{avg_spend}</div></div>',
                  unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------------
# ░ GLOBAL FILTERS  ░
# ------------------------------------------------------------------
with st.expander("🎛️  Filters", expanded=False):
    cities = df["City"].unique() if "City" in df.columns else []
    sel_city = st.multiselect("City filter", cities, default=list(cities))
    if "Income" in df.columns:
        inc_min, inc_max = int(df["Income"].min()), int(df["Income"].max())
        sel_income = st.slider("Income range", inc_min, inc_max,
                               (inc_min, inc_max), step=10000)
    else:
        sel_income = None

# Apply filters
df_filt = df.copy()
if sel_city: df_filt = df_filt[df_filt["City"].isin(sel_city)]
if sel_income and "Income" in df_filt.columns:
    df_filt = df_filt.query("Income >= @sel_income[0] & Income <= @sel_income[1]")

# ------------------------------------------------------------------
# ░ TABS  ░
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Visuals", "🤖 Classification", "📦 Clustering",
     "🔗 Assoc Rules", "📈 Regression"]
)

# ================================================================
# 📊 Visuals
# ================================================================
with tab1:
    st.subheader("Distribution Insights")
    # Age / Income hist
    if "Age" in df_filt.columns:
        st.plotly_chart(
            px.histogram(df_filt, x="Age", nbins=25, template=plotly_theme),
            use_container_width=True
        )
    if "Income" in df_filt.columns:
        st.plotly_chart(
            px.histogram(df_filt, x="Income", nbins=30,
                         template=plotly_theme, color_discrete_sequence=["#4F8BF9"]),
            use_container_width=True
        )

    if {"City", "TryMealKit"}.issubset(df_filt.columns):
        st.plotly_chart(
            px.histogram(df_filt, x="City", color="TryMealKit",
                         barmode="group", template=plotly_theme),
            use_container_width=True
        )

    st.markdown(_download_link(df_filt, "filtered_data.csv"), unsafe_allow_html=True)

# ================================================================
# 🤖 Classification
# ================================================================
with tab2:
    st.subheader("Train & Evaluate")

    target_opts = [c for c in df_filt.columns if df_filt[c].dtype == "object"]
    if not target_opts:
        st.info("No categorical columns for classification."); st.stop()
    target = st.selectbox("Target column", target_opts)

    X, y = df_filt.drop(columns=[target]), df_filt[target]
    prep = build_preprocessor(df_filt, target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y,
                                          test_size=0.2, random_state=42)

    # -------- Train models
    results, pipes = [], {}
    for name, clf in cls_models().items():
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)
        results.append({"Model": name,
                        "Acc":  accuracy_score(yte, yhat),
                        "Prec": precision_score(yte, yhat, average="weighted",
                                                zero_division=0),
                        "Rec":  recall_score(yte, yhat, average="weighted",
                                             zero_division=0),
                        "F1":   f1_score(yte, yhat, average="weighted",
                                         zero_division=0)})
        pipes[name] = pipe

    st.dataframe(pd.DataFrame(results).round(3))

    # -------- Feature Importance
    with st.expander("🔍 Feature Importance (Trees)"):
        fi_model = st.selectbox("Choose model", ["Decision Tree", "Random Forest"])
        tree_pipe = pipes[fi_model]
        clf = tree_pipe.named_steps["clf"]
        feat_names = tree_pipe.named_steps["prep"].get_feature_names_out()
        fi_df = (pd.DataFrame({"Feature": feat_names,
                               "Importance": clf.feature_importances_})
                 .sort_values("Importance", ascending=False)
                 .head(20))
        st.plotly_chart(
            px.bar(fi_df, y="Feature", x="Importance", orientation="h",
                   template=plotly_theme,
                   color_discrete_sequence=["#4F8BF9"]),
            use_container_width=True
        )

# ================================================================
# 📦 Clustering
# ================================================================
with tab3:
    st.subheader("K-Means Explorer")

    num_cols = _numeric_cols(df_filt)
    if len(num_cols) < 2:
        st.info("Need at least 2 numeric cols."); st.stop()

    k = st.slider("Number of clusters", 2, 10, 4)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) \
        .fit(StandardScaler().fit_transform(df_filt[num_cols]))
    df_k = df_filt.assign(Cluster=kmeans.labels_)

    st.plotly_chart(
        px.scatter(df_k, x=num_cols[0], y=num_cols[1], color="Cluster",
                   template=plotly_theme, symbol="Cluster"),
        use_container_width=True
    )
    st.dataframe(df_k.groupby("Cluster")[num_cols].mean().round(1))
    st.markdown(_download_link(df_k, "clustered_data.csv"), unsafe_allow_html=True)

# ================================================================
# 🔗 Association Rules
# ================================================================
with tab4:
    st.subheader("Apriori Insights")

    multi_cols = [c for c in df_filt.columns
                  if df_filt[c].dtype == "object" and df_filt[c].str.contains(",").any()]
    if not multi_cols:
        st.info("No multi-select columns detected.")
    else:
        cols = st.multiselect("Columns", multi_cols, default=multi_cols[:3])
        sup = st.slider("Min support", 0.01, 0.5, 0.05, 0.01)
        conf = st.slider("Min confidence", 0.1, 1.0, 0.3, 0.05)

        if st.button("Run"):
            trans = pd.concat([df_filt[c].str.get_dummies(sep=", ") for c in cols], axis=1)
            freq = apriori(trans, min_support=sup, use_colnames=True)
            rules = association_rules(freq, metric="confidence",
                                      min_threshold=conf)\
                    .sort_values("confidence", ascending=False).head(10)
            st.dataframe(rules[["antecedents", "consequents",
                                "support", "confidence", "lift"]])

# ================================================================
# 📈 Regression & Forecast
# ================================================================
with tab5:
    st.subheader("Predict & Forecast")

    num_all = _numeric_cols(df_filt)
    tgt = st.selectbox("Target (numeric)", num_all)
    preds = st.multiselect("Predictors",
                           [c for c in num_all if c != tgt],
                           default=num_all[:min(3, len(num_all))])
    if not preds:
        st.warning("Select ≥1 predictor."); st.stop()

    Xr, yr = df_filt[preds], df_filt[tgt]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    reg_out, reg_fitted = [], {}
    for n, m in reg_models().items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        reg_out.append({"Model": n,
                        "MAE":  mean_absolute_error(yte, pred),
                        "RMSE": np.sqrt(mean_squared_error(yte, pred)),
                        "R²":   r2_score(yte, pred)})
        reg_fitted[n] = m

    st.dataframe(pd.DataFrame(reg_out).round(3))

    # ⬇ Forecast
    if {"City", "SpendMealKitValue"}.issubset(df_filt.columns):
        st.markdown("#### 12-Month City Forecast")
        model_sel = st.selectbox("Regressor", list(reg_fitted))
        mdl = reg_fitted[model_sel]
        months = np.arange(1, 13).reshape(-1, 1)

        dfs = []
        for city, grp in df_filt.groupby("City"):
            base = grp["SpendMealKitValue"].mean()
            series = base * (1 + 0.02 * months.flatten()) \
                     + np.random.normal(0, base * 0.05, 12)
            mdl.fit(months, series)
            dfs.append(pd.DataFrame({"Month": months.flatten(),
                                     "Forecast": mdl.predict(months),
                                     "City": city}))
        fc_df = pd.concat(dfs)
        st.plotly_chart(
            px.line(fc_df, x="Month", y="Forecast", color="City",
                    template=plotly_theme), use_container_width=True
        )
        st.markdown(_download_link(fc_df, "forecast.csv"), unsafe_allow_html=True)
