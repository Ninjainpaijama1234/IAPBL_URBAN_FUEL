# urban_fuel_dashboard/app.py
# ------------------------------------------------------------------
#  Urban Fuel – Beautiful & Robust Streamlit Dashboard
#  July 2025  ·  Income column now auto-detects numeric vs. categorical
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

# ------------------------------  GLOBAL STYLE  ------------------------------
st.set_page_config(page_title="Urban Fuel Analytics", page_icon="🍱", layout="wide")

CUSTOM_CSS = """
<style>
#MainMenu, footer {visibility:hidden;}
[data-testid="stAppViewContainer"] > .main {background:#F5F8FF;}
.metric-container {padding:1rem;border-radius:12px;background:#fff;
box-shadow:0 1px 3px rgba(0,0,0,0.08);}
.metric-title {font-size:0.9rem;color:#666;}
.metric-value {font-size:1.8rem;font-weight:600;color:#223D62;}
.download-btn {background:#4F8BF9;padding:0.5rem 1rem;color:#fff!important;
text-decoration:none;border-radius:8px;font-weight:600;}
.download-btn:hover{background:#3c6fe0;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------  LOAD DATA  ------------------------------
DATA_FILE = "Urban_Fuel_Synthetic_Dataset.csv"

@st.cache_data
def load_df() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        st.error(f"**{DATA_FILE}** not found next to *app.py*."); st.stop()
    df0 = pd.read_csv(DATA_FILE)

    # ▶ FIX — numeric coercion helpers
    def to_num(s):
        return pd.to_numeric(
            s.astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce"
        )

    if "SpendMealKit" in df0 and df0["SpendMealKit"].dtype == "object":
        df0["SpendMealKitValue"] = to_num(df0["SpendMealKit"])

    # Try to build numeric Income column if possible
    if "Income" in df0:
        if pd.api.types.is_numeric_dtype(df0["Income"]):
            df0["IncomeValue"] = df0["Income"]
        else:
            num_conv = to_num(df0["Income"])
            if num_conv.notna().sum() > 0:
                df0["IncomeValue"] = num_conv  # partial numeric, allow slider
    return df0

df = load_df()

# ------------------------------  SIDEBAR  ------------------------------
st.sidebar.image("https://i.imgur.com/aDKW5J1.png", width=160)
st.sidebar.markdown("### Urban Fuel Dashboard")
plotly_theme = st.sidebar.selectbox(
    "Chart Theme",
    ["plotly", "plotly_dark", "ggplot2", "seaborn"],
    index=0
)

# ------------------------------  HELPERS  ------------------------------
def download_link(data: pd.DataFrame, filename: str) -> str:
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a class="download-btn" href="data:file/csv;base64,{b64}" download="{filename}">Download ▼</a>'

def numeric_cols(d): return d.select_dtypes(include=["int64", "float64"]).columns.tolist()

def preprocessor(df_, target):
    num = numeric_cols(df_)
    cat = [c for c in df_.columns if c not in num + [target]]
    return ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

def cls_models():
    return {
        "K-NN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boost": GradientBoostingClassifier(random_state=42),
    }

def reg_models():
    return {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "DT Regressor": DecisionTreeRegressor(max_depth=6, random_state=42),
    }

# ------------------------------  KPI CARDS  ------------------------------
col1, col2, col3 = st.columns(3)
col1.markdown(f"""<div class="metric-container">
<div class="metric-title">Total Respondents</div>
<div class="metric-value">{len(df):,}</div></div>""", unsafe_allow_html=True)
avg_age = round(df["Age"].mean(), 1) if "Age" in df else "—"
col2.markdown(f"""<div class="metric-container">
<div class="metric-title">Average Age</div>
<div class="metric-value">{avg_age}</div></div>""", unsafe_allow_html=True)
spend_col = "SpendMealKitValue" if "SpendMealKitValue" in df else None
avg_spend = f"₹{df[spend_col].mean():,.0f}" if spend_col else "—"
col3.markdown(f"""<div class="metric-container">
<div class="metric-title">Avg. Meal Spend</div>
<div class="metric-value">{avg_spend}</div></div>""", unsafe_allow_html=True)
st.markdown("---")

# ------------------------------  GLOBAL FILTERS  ------------------------------
with st.expander("🎛️  Filters", expanded=False):
    if "City" in df:
        sel_city = st.multiselect("City", df["City"].unique(), default=list(df["City"].unique()))
    else:
        sel_city = []
    # ▶ FIX — Income slider only if numeric
    if "IncomeValue" in df:
        inc_min, inc_max = int(df["IncomeValue"].min()), int(df["IncomeValue"].max())
        sel_income = st.slider("Income range (numeric)", inc_min, inc_max,
                               (inc_min, inc_max), step=10000)
    else:
        sel_income = None
        if "Income" in df:
            sel_inc_cat = st.multiselect("Income category", df["Income"].unique(),
                                         default=list(df["Income"].unique()))
        else:
            sel_inc_cat = []

# Apply filters
df_f = df.copy()
if sel_city: df_f = df_f[df_f["City"].isin(sel_city)]
if sel_income:
    df_f = df_f.query("IncomeValue >= @sel_income[0] & IncomeValue <= @sel_income[1]")
elif "Income" in df and 'sel_inc_cat' in locals():
    df_f = df_f[df_f["Income"].isin(sel_inc_cat)]

# ------------------------------  TABS  ------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Visuals", "🤖 Classify", "📦 Cluster", "🔗 Rules", "📈 Regress"]
)

# ------------------------------------------------------------------
# 📊 VISUALS
# ------------------------------------------------------------------
with tab1:
    st.subheader("Distribution Insights")
    if "Age" in df_f:
        st.plotly_chart(px.histogram(df_f, x="Age", nbins=25, template=plotly_theme),
                        use_container_width=True)
    # ▶ FIX — choose Income numeric vs categorical bar
    if "IncomeValue" in df_f:
        st.plotly_chart(px.histogram(df_f, x="IncomeValue", nbins=30,
                                     template=plotly_theme,
                                     color_discrete_sequence=["#4F8BF9"]),
                        use_container_width=True)
    elif "Income" in df_f:
        inc_counts = df_f["Income"].value_counts().reset_index()
        st.plotly_chart(px.bar(inc_counts, x="index", y="Income",
                               template=plotly_theme,
                               labels={"index": "Income Bracket", "Income": "Count"},
                               color_discrete_sequence=["#4F8BF9"]),
                        use_container_width=True)

    if {"City", "TryMealKit"}.issubset(df_f.columns):
        st.plotly_chart(px.histogram(df_f, x="City", color="TryMealKit",
                                     barmode="group", template=plotly_theme),
                        use_container_width=True)
    st.markdown(download_link(df_f, "filtered_data.csv"), unsafe_allow_html=True)

# ------------------------------------------------------------------
# 🤖 CLASSIFICATION
# ------------------------------------------------------------------
with tab2:
    st.subheader("Train & Evaluate")
    cat_targets = [c for c in df_f.columns if df_f[c].dtype == "object"]
    if not cat_targets: st.info("No categorical target available."); st.stop()
    target = st.selectbox("Target", cat_targets)

    X, y = df_f.drop(columns=[target]), df_f[target]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)
    prep = preprocessor(df_f, target)

    res, pipes = [], {}
    for n, clf in cls_models().items():
        p = Pipeline([("prep", prep), ("clf", clf)]).fit(Xtr, ytr)
        yhat = p.predict(Xte)
        res.append({"Model": n, "Acc": accuracy_score(yte, yhat),
                    "Prec": precision_score(yte, yhat, average="weighted",
                                            zero_division=0),
                    "Rec": recall_score(yte, yhat, average="weighted",
                                        zero_division=0),
                    "F1": f1_score(yte, yhat, average="weighted",
                                   zero_division=0)})
        pipes[n] = p
    st.dataframe(pd.DataFrame(res).round(3))

    with st.expander("🔍 Feature Importance"):
        fi_mdl = st.selectbox("Tree model", ["Decision Tree", "Random Forest"])
        tr_pipe = pipes[fi_mdl]
        fi_vals = tr_pipe.named_steps["clf"].feature_importances_
        fi_names = tr_pipe.named_steps["prep"].get_feature_names_out()
        fi_df = pd.DataFrame({"Feature": fi_names, "Importance": fi_vals})\
                .sort_values("Importance", ascending=False).head(20)
        st.plotly_chart(px.bar(fi_df, y="Feature", x="Importance",
                               orientation="h", template=plotly_theme,
                               color_discrete_sequence=["#4F8BF9"]),
                        use_container_width=True)

# ------------------------------------------------------------------
# 📦 CLUSTERING
# ------------------------------------------------------------------
with tab3:
    st.subheader("K-Means Explorer")
    num_cols = numeric_cols(df_f)
    if len(num_cols) < 2: st.info("Need ≥2 numeric cols."); st.stop()
    k = st.slider("Clusters (k)", 2, 10, 4)
    labels = KMeans(n_clusters=k, n_init=10, random_state=42)\
             .fit(StandardScaler().fit_transform(df_f[num_cols])).labels_
    df_k = df_f.assign(Cluster=labels)
    st.plotly_chart(px.scatter(df_k, x=num_cols[0], y=num_cols[1],
                               color="Cluster", template=plotly_theme),
                    use_container_width=True)
    st.dataframe(df_k.groupby("Cluster")[num_cols].mean().round(1))
    st.markdown(download_link(df_k, "clusters.csv"), unsafe_allow_html=True)

# ------------------------------------------------------------------
# 🔗 ASSOCIATION RULES
# ------------------------------------------------------------------
with tab4:
    st.subheader("Apriori Rules")
    multi = [c for c in df_f.columns if df_f[c].dtype == "object"
             and df_f[c].str.contains(",").any()]
    if not multi:
        st.info("No multi-choice cols.")
    else:
        sel = st.multiselect("Columns", multi, default=multi[:3])
        sup = st.slider("Support", 0.01, 0.5, 0.05, 0.01)
        conf = st.slider("Confidence", 0.1, 1.0, 0.3, 0.05)
        if st.button("Run"):
            basket = pd.concat([df_f[c].str.get_dummies(sep=", ") for c in sel], axis=1)
            freq = apriori(basket, min_support=sup, use_colnames=True)
            rules = association_rules(freq, metric="confidence",
                                      min_threshold=conf).sort_values(
                        "confidence", ascending=False).head(10)
            st.dataframe(rules[["antecedents", "consequents",
                                "support", "confidence", "lift"]])

# ------------------------------------------------------------------
# 📈 REGRESSION & FORECAST
# ------------------------------------------------------------------
with tab5:
    st.subheader("Regression")
    num_all = numeric_cols(df_f)
    tgt = st.selectbox("Target", num_all)
    preds = st.multiselect("Predictors", [c for c in num_all if c != tgt],
                           default=num_all[:min(3, len(num_all))])
    if not preds: st.warning("Select ≥1 predictor."); st.stop()

    Xr, yr = df_f[preds], df_f[tgt]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    rows, fits = [], {}
    for n, m in reg_models().items():
        m.fit(Xtr, ytr); pred = m.predict(Xte)
        rows.append({"Model": n, "MAE": mean_absolute_error(yte, pred),
                     "RMSE": np.sqrt(mean_squared_error(yte, pred)),
                     "R²": r2_score(yte, pred)})
        fits[n] = m
    st.dataframe(pd.DataFrame(rows).round(3))

    # Forecast by City (if numeric spend exists)
    if {"City", "SpendMealKitValue"}.issubset(df_f.columns):
        st.markdown("#### 12-Month Forecast")
        mdl_sel = st.selectbox("Regressor", list(fits))
        m = fits[mdl_sel]
        months = np.arange(1, 13).reshape(-1, 1)
        out = []
        for city, grp in df_f.groupby("City"):
            base = grp["SpendMealKitValue"].mean()
            series = base * (1 + 0.02 * months.flatten())\
                     + np.random.normal(0, base*0.05, 12)
            m.fit(months, series)
            out.append(pd.DataFrame({"Month": months.flatten(),
                                     "Forecast": m.predict(months),
                                     "City": city}))
        fc = pd.concat(out)
        st.plotly_chart(px.line(fc, x="Month", y="Forecast", color="City",
                                template=plotly_theme), use_container_width=True)
        st.markdown(download_link(fc, "forecast.csv"), unsafe_allow_html=True)
