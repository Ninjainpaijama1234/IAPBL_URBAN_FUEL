# urban_fuel_dashboard/app.py
# ---------------------------------------------------------------
#  Urban Fuel – Analytics Dashboard (Streamlit)
#  Fixed-dataset version  |  July 2025
# ---------------------------------------------------------------

import base64, io, os, numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go, streamlit as st

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

# ---------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------
st.set_page_config(page_title="Urban Fuel Analytics",
                   layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------
DATA_FILE = "Urban_Fuel_Synthetic_Dataset.csv"

def load_local_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Dataset **{path}** not found. "
                 "Place the CSV in the same folder as *app.py* and restart.")
        st.stop()
    return pd.read_csv(path)

def download_link(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download ▶</a>'

def build_preprocessor(df: pd.DataFrame, target: str):
    nums = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cats = [c for c in df.columns if c not in nums + [target]]

    num_pipe = Pipeline([("scale", StandardScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer([("num", num_pipe, nums),
                              ("cat", cat_pipe, cats)])

def cls_models():
    return {
        "K-NN":               KNeighborsClassifier(),
        "Decision Tree":      DecisionTreeClassifier(random_state=42),
        "Random Forest":      RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting":  GradientBoostingClassifier(random_state=42)
    }

def reg_models():
    return {
        "Linear":             LinearRegression(),
        "Ridge":              Ridge(alpha=1.0),
        "Lasso":              Lasso(alpha=0.01),
        "Decision Tree Reg":  DecisionTreeRegressor(max_depth=6, random_state=42)
    }

# ---------------------------------------------------------------
# 1 · Load fixed dataset
# ---------------------------------------------------------------
df = load_local_dataset(DATA_FILE)
st.sidebar.success("Loaded Urban_Fuel_Synthetic_Dataset.csv")

# ---------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------
tabs = st.tabs([
    "Data Visualisation",
    "Classification",
    "Clustering",
    "Association Rules",
    "Regression & Forecast"
])

# ================================================================
# TAB 1 · DATA VISUALISATION
# ================================================================
with tabs[0]:
    st.header("📊 Exploratory Dashboards")
    st.dataframe(df.head())

    # Age filter demo
    if "Age" in df.columns:
        amin, amax = int(df["Age"].min()), int(df["Age"].max())
        arange = st.slider("Filter by Age", amin, amax, (amin, amax))
        dfx = df.query("Age >= @arange[0] & Age <= @arange[1]")
    else:
        dfx = df.copy()

    if "Income" in dfx.columns:
        st.subheader("Income Distribution")
        st.plotly_chart(px.histogram(dfx, x="Income", nbins=30),
                        use_container_width=True)

    if {"City", "TryMealKit"}.issubset(dfx.columns):
        st.subheader("Meal-Kit Intent by City")
        st.plotly_chart(px.histogram(dfx, x="City", color="TryMealKit",
                                     barmode="group"), use_container_width=True)

# ================================================================
# TAB 2 · CLASSIFICATION   + Feature Importance
# ================================================================
with tabs[1]:
    st.header("🤖 Classification")

    # --- choose target ---
    cat_targets = [c for c in df.columns if df[c].dtype == "object"]
    if not cat_targets:
        st.error("No categorical target column found."); st.stop()
    target = st.selectbox("Target variable", cat_targets)

    X, y = df.drop(columns=[target]), df[target]
    prep  = build_preprocessor(df, target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2,
                                          random_state=42)

    results, pipes = [], {}
    for name, clf in cls_models().items():
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)
        results.append({
            "Model":     name,
            "Accuracy":  accuracy_score(yte, yhat),
            "Precision": precision_score(yte, yhat, average="weighted",
                                         zero_division=0),
            "Recall":    recall_score(yte, yhat, average="weighted",
                                      zero_division=0),
            "F1":        f1_score(yte, yhat, average="weighted",
                                  zero_division=0)
        })
        pipes[name] = pipe

    st.dataframe(pd.DataFrame(results).round(3))

    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix")
    cm_choice = st.selectbox("Choose model", list(pipes))
    if st.checkbox("Show matrix"):
        cm = confusion_matrix(yte, pipes[cm_choice].predict(Xte),
                              labels=pipes[cm_choice].classes_)
        st.plotly_chart(
            px.imshow(cm, x=pipes[cm_choice].classes_,
                      y=pipes[cm_choice].classes_,
                      text_auto=True, color_continuous_scale="Blues"),
            use_container_width=True
        )

    # --- ROC (binary) ---
    if len(y.unique()) == 2:
        st.subheader("ROC Curves")
        label_map = {c: i for i, c in enumerate(sorted(y.unique()))}
        y_bin = yte.map(label_map)
        fig_roc = go.Figure()
        for n, p in pipes.items():
            if hasattr(p.named_steps["clf"], "predict_proba"):
                prob = p.predict_proba(Xte)[:, 1]
                fpr, tpr, _ = roc_curve(y_bin, prob)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=n))
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="grey"))
        fig_roc.update_layout(xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig_roc, use_container_width=True)

    # --- Feature Importance (Tree models) ---
    st.subheader("🔍 Feature Importance (Decision Tree & Random Forest)")
    fi_model = st.selectbox("Select tree model",
                            ["Decision Tree", "Random Forest"])
    tree_pipe = pipes[fi_model]
    clf = tree_pipe.named_steps["clf"]
    if hasattr(tree_pipe.named_steps["prep"], "get_feature_names_out"):
        feat_names = tree_pipe.named_steps["prep"].get_feature_names_out()
    else:  # fallback
        feat_names = [f"f{i}" for i in range(clf.feature_importances_.shape[0])]

    imp_df = (pd.DataFrame({"Feature": feat_names,
                            "Importance": clf.feature_importances_})
                .sort_values("Importance", ascending=False)
                .head(20))

    st.plotly_chart(px.bar(imp_df, x="Importance", y="Feature",
                           orientation="h"), use_container_width=True)

# ================================================================
# TAB 3 · CLUSTERING
# ================================================================
with tabs[2]:
    st.header("📦 K-Means Clustering")

    numeric = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if len(numeric) < 2:
        st.error("Need ≥2 numeric columns for clustering."); st.stop()

    k = st.slider("Clusters (k)", 2, 10, 4)
    scale = StandardScaler()
    df["Cluster"] = KMeans(n_clusters=k, n_init=10, random_state=42) \
        .fit(scale.fit_transform(df[numeric])).labels_

    st.subheader("Cluster personas")
    st.dataframe(df.groupby("Cluster")[numeric].mean().round(1))
    st.markdown(download_link(df, "clustered_data.csv"), unsafe_allow_html=True)

# ================================================================
# TAB 4 · ASSOCIATION RULES
# ================================================================
with tabs[3]:
    st.header("🔗 Association Rules (Apriori)")

    multi_cols = [c for c in df.columns
                  if df[c].dtype == "object" and df[c].str.contains(",").any()]
    if not multi_cols:
        st.info("No comma-separated columns detected.")
    else:
        sel = st.multiselect("Columns", multi_cols, default=multi_cols[:3])
        sup  = st.number_input("Min support", 0.01, 1.0, 0.05, 0.01)
        conf = st.number_input("Min confidence", 0.01, 1.0, 0.20, 0.01)

        if st.button("Run Apriori"):
            basket = pd.concat([df[c].str.get_dummies(sep=", ") for c in sel], axis=1)
            freq   = apriori(basket, min_support=sup, use_colnames=True)
            rules  = association_rules(freq, metric="confidence",
                                       min_threshold=conf)\
                     .sort_values("confidence", ascending=False).head(10)
            st.dataframe(rules[["antecedents", "consequents", "support",
                                "confidence", "lift"]])

# ================================================================
# TAB 5 · REGRESSION & FORECAST
# ================================================================
with tabs[4]:
    st.header("📈 Regression & 12-Month Forecast")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    target_reg = st.selectbox("Target (numeric)", numeric_cols)
    predictors = [c for c in numeric_cols if c != target_reg]
    default = predictors[:min(3, len(predictors))]
    feats = st.multiselect("Predictors", predictors, default=default)
    if not feats:
        st.warning("Select ≥1 predictor."); st.stop()

    Xr, yr = df[feats], df[target_reg]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)

    reg_summary, reg_fit = [], {}
    for nm, mdl in reg_models().items():
        mdl.fit(Xtr, ytr)
        pred = mdl.predict(Xte)
        rmse = np.sqrt(mean_squared_error(yte, pred))
        reg_summary.append({"Model": nm,
                            "MAE": mean_absolute_error(yte, pred),
                            "RMSE": rmse,
                            "R²": r2_score(yte, pred)})
        reg_fit[nm] = mdl

    st.dataframe(pd.DataFrame(reg_summary).round(3))

    # Simple Forecast by City
    if {"City", "SpendMealKit"}.issubset(df.columns):
        st.subheader("12-Month Revenue Forecast by City")
        sel_reg = st.selectbox("Regressor for forecast", list(reg_fit))
        mdl = reg_fit[sel_reg]
        months = np.arange(1, 13).reshape(-1, 1)

        out = []
        for city, grp in df.groupby("City"):
            base = grp["SpendMealKit"].mean()
            series = base * (1 + 0.02 * months.flatten()) \
                     + np.random.normal(0, base * 0.05, 12)
            mdl.fit(months, series)
            preds = mdl.predict(months)
            out.append(pd.DataFrame({"Month": months.flatten(),
                                     "ForecastRevenue": preds,
                                     "City": city}))
        fc = pd.concat(out)
        st.plotly_chart(px.line(fc, x="Month", y="ForecastRevenue", color="City"),
                        use_container_width=True)
        st.markdown(download_link(fc, "revenue_forecast.csv"), unsafe_allow_html=True)
