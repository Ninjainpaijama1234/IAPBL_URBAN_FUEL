# urban_fuel_dashboard/app.py
# ------------------------------------------------------------------
#  Urban Fuel – Streamlit Analytics Dashboard  ·  July 2025
#  ► Fixed: currency strings converted to numeric (SpendMealKitValue)
# ------------------------------------------------------------------

import os, base64, io, re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(page_title="Urban Fuel Analytics",
                   layout="wide",
                   initial_sidebar_state="collapsed")

DATA_FILE = "Urban_Fuel_Synthetic_Dataset.csv"

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """Load CSV and coerce currency columns to numeric."""
    if not os.path.exists(path):
        st.error(f"**{path}** not found. Place it next to *app.py* and restart.")
        st.stop()
    df_ = pd.read_csv(path)

    # Detect and convert 'SpendMealKit' (₹xx) → numeric
    if "SpendMealKit" in df_.columns and df_["SpendMealKit"].dtype == "object":
        df_["SpendMealKitValue"] = _to_numeric(df_["SpendMealKit"])
    return df_

def _to_numeric(series: pd.Series) -> pd.Series:
    """Remove non-digits (₹, commas, spaces) and convert to float."""
    cleaned = series.astype(str).str.replace(r"[^\d.]+", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")

def download_link(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download ▶</a>'

def build_preprocessor(df: pd.DataFrame, target: str):
    num = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat = [c for c in df.columns if c not in num + [target]]

    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

def cls_models():
    return {
        "K-NN":              KNeighborsClassifier(),
        "Decision Tree":     DecisionTreeClassifier(random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

def reg_models():
    return {
        "Linear":             LinearRegression(),
        "Ridge":              Ridge(alpha=1.0),
        "Lasso":              Lasso(alpha=0.01),
        "Decision Tree Reg":  DecisionTreeRegressor(max_depth=6, random_state=42)
    }

# ------------------------------------------------------------------
# Load fixed dataset
# ------------------------------------------------------------------
df = load_dataset(DATA_FILE)
st.sidebar.success("✅ Urban_Fuel_Synthetic_Dataset loaded")

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tabs = st.tabs([
    "Data Visualisation",
    "Classification",
    "Clustering",
    "Association Rules",
    "Regression & Forecast"
])

# ================================================================
# 1 · DATA VISUALISATION
# ================================================================
with tabs[0]:
    st.header("📊 Exploratory Dashboards")
    st.dataframe(df.head())

    # Age filter
    if "Age" in df.columns:
        a_min, a_max = int(df["Age"].min()), int(df["Age"].max())
        a_rng = st.slider("Filter by age", a_min, a_max, (a_min, a_max))
        dfx = df.query("Age >= @a_rng[0] & Age <= @a_rng[1]")
    else:
        dfx = df.copy()

    # Income distribution
    if "Income" in dfx.columns:
        st.subheader("Income Distribution")
        st.plotly_chart(px.histogram(dfx, x="Income", nbins=30),
                        use_container_width=True)

    # Intent by city
    if {"City", "TryMealKit"}.issubset(dfx.columns):
        st.subheader("Meal-Kit Intent by City")
        st.plotly_chart(px.histogram(dfx, x="City", color="TryMealKit",
                                     barmode="group"), use_container_width=True)

# ================================================================
# 2 · CLASSIFICATION  +  Feature Importance
# ================================================================
with tabs[1]:
    st.header("🤖 Classification")

    cat_targets = [c for c in df.columns if df[c].dtype == "object"]
    if not cat_targets:
        st.error("No categorical column available as target."); st.stop()
    target = st.selectbox("Target variable", cat_targets)

    X, y = df.drop(columns=[target]), df[target]
    prep = build_preprocessor(df, target)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)

    metrics, models = [], {}
    for name, clf in cls_models().items():
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)
        metrics.append({
            "Model": name,
            "Accuracy":  accuracy_score(yte, yhat),
            "Precision": precision_score(yte, yhat, average="weighted", zero_division=0),
            "Recall":    recall_score(yte, yhat, average="weighted", zero_division=0),
            "F1":        f1_score(yte, yhat, average="weighted", zero_division=0)
        })
        models[name] = pipe

    st.dataframe(pd.DataFrame(metrics).round(3))

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm_model = st.selectbox("Choose model", list(models))
    if st.checkbox("Show matrix"):
        cm = confusion_matrix(yte, models[cm_model].predict(Xte),
                              labels=models[cm_model].classes_)
        st.plotly_chart(px.imshow(cm, x=models[cm_model].classes_,
                                  y=models[cm_model].classes_,
                                  text_auto=True,
                                  color_continuous_scale="Blues"),
                        use_container_width=True)

    # ROC (binary)
    if y.nunique() == 2:
        st.subheader("ROC Curves")
        bin_map = {c: i for i, c in enumerate(sorted(y.unique()))}
        y_bin = yte.map(bin_map)
        fig = go.Figure()
        for n, p in models.items():
            if hasattr(p.named_steps["clf"], "predict_proba"):
                prob = p.predict_proba(Xte)[:, 1]
                fpr, tpr, _ = roc_curve(y_bin, prob)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=n))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color="grey", dash="dash"))
        fig.update_layout(xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    st.subheader("🔍 Feature Importance")
    fi_choice = st.selectbox("Tree model", ["Decision Tree", "Random Forest"])
    pipe_tree = models[fi_choice]
    clf_tree  = pipe_tree.named_steps["clf"]

    if hasattr(pipe_tree.named_steps["prep"], "get_feature_names_out"):
        fn = pipe_tree.named_steps["prep"].get_feature_names_out()
    else:
        fn = [f"f{i}" for i in range(len(clf_tree.feature_importances_))]

    fi_df = (pd.DataFrame({
                "Feature": fn,
                "Importance": clf_tree.feature_importances_})
             .sort_values("Importance", ascending=False)
             .head(20))
    st.plotly_chart(px.bar(fi_df, x="Importance", y="Feature",
                           orientation="h"), use_container_width=True)

# ================================================================
# 3 · CLUSTERING
# ================================================================
with tabs[2]:
    st.header("📦 K-Means Clustering")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns"); st.stop()

    k = st.slider("Clusters (k)", 2, 10, 4)
    scaler = StandardScaler()
    df["Cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10) \
        .fit(scaler.fit_transform(df[numeric_cols])) \
        .labels_

    st.subheader("Cluster Personas (mean numeric values)")
    st.dataframe(df.groupby("Cluster")[numeric_cols].mean().round(1))
    st.markdown(download_link(df, "clustered_data.csv"), unsafe_allow_html=True)

# ================================================================
# 4 · ASSOCIATION RULES
# ================================================================
with tabs[3]:
    st.header("🔗 Association Rules (Apriori)")

    multi = [c for c in df.columns if df[c].dtype == "object" and df[c].str.contains(",").any()]
    if not multi:
        st.info("No multi-option comma-separated columns detected.")
    else:
        sel_cols = st.multiselect("Columns", multi, default=multi[:3])
        sup = st.number_input("Min support", 0.01, 1.0, 0.05, 0.01)
        conf = st.number_input("Min confidence", 0.01, 1.0, 0.20, 0.01)

        if st.button("Run Apriori"):
            basket = pd.concat([df[c].str.get_dummies(sep=", ") for c in sel_cols], axis=1)
            freq   = apriori(basket, min_support=sup, use_colnames=True)
            rules  = association_rules(freq, metric="confidence", min_threshold=conf) \
                     .sort_values("confidence", ascending=False).head(10)
            st.dataframe(rules[["antecedents", "consequents", "support",
                                "confidence", "lift"]])

# ================================================================
# 5 · REGRESSION & FORECAST
# ================================================================
with tabs[4]:
    st.header("📈 Regression & 12-Month Forecast")

    numeric_all = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if not numeric_all:
        st.error("No numeric columns available."); st.stop()

    target_reg = st.selectbox("Target (numeric)", numeric_all)
    predictors_all = [c for c in numeric_all if c != target_reg]
    default_pred = predictors_all[:min(3, len(predictors_all))]
    predictors = st.multiselect("Predictors", predictors_all, default=default_pred)
    if not predictors:
        st.warning("Select at least one predictor."); st.stop()

    Xr, yr = df[predictors], df[target_reg]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    reg_rows, reg_fitted = [], {}
    for n, m in reg_models().items():
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        reg_rows.append({
            "Model": n,
            "MAE":  mean_absolute_error(yte, pred),
            "RMSE": np.sqrt(mean_squared_error(yte, pred)),
            "R²":   r2_score(yte, pred)
        })
        reg_fitted[n] = m

    st.dataframe(pd.DataFrame(reg_rows).round(3))

    # 12-month revenue forecast (requires SpendMealKitValue + City)
    if {"City", "SpendMealKitValue"}.issubset(df.columns):
        st.subheader("12-Month Revenue Forecast by City")
        fore_reg = st.selectbox("Regressor", list(reg_fitted))
        mdl = reg_fitted[fore_reg]
        months = np.arange(1, 13).reshape(-1, 1)

        fc_out = []
        for city, grp in df.groupby("City"):
            base = grp["SpendMealKitValue"].mean()
            trend = base * (1 + 0.02 * months.flatten()) \
                    + np.random.normal(0, base * 0.05, 12)
            mdl.fit(months, trend)
            fc_out.append(pd.DataFrame({
                "Month": months.flatten(),
                "ForecastRevenue": mdl.predict(months),
                "City": city
            }))

        fc_df = pd.concat(fc_out)
        st.plotly_chart(px.line(fc_df, x="Month", y="ForecastRevenue",
                                color="City"), use_container_width=True)
        st.markdown(download_link(fc_df, "revenue_forecast.csv"),
                    unsafe_allow_html=True)
