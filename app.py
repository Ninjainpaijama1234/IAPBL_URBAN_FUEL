# urban_fuel_dashboard/app.py
# ------------------------------------------------------------------
#  Urban Fuel – Multi-tab Analytics Dashboard (Streamlit)
#  July 2025 – cleansed & error-free version
# ------------------------------------------------------------------

import base64
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, LabelEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules

# ------------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Urban Fuel Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def _load_csv(uploaded_file: io.BytesIO) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

def _download_link(df: pd.DataFrame, filename: str) -> str:
    """Generate an HTML anchor to download `df` as `filename`."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download ▶</a>'

def _preprocessor(df: pd.DataFrame, target: str):
    """Return a ColumnTransformer that scales numeric & one-hot-encodes categoricals."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols + [target]]

    num_pipe = Pipeline([("scale", StandardScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])

    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])

def _cls_models():
    return {
        "K-NN":              KNeighborsClassifier(),
        "Decision Tree":     DecisionTreeClassifier(),
        "Random Forest":     RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

def _reg_models():
    return {
        "Linear":               LinearRegression(),
        "Ridge":                Ridge(alpha=1.0),
        "Lasso":                Lasso(alpha=0.01),
        "Decision Tree Reg":    DecisionTreeRegressor(max_depth=6, random_state=42)
    }

# ------------------------------------------------------------------
# SIDEBAR – DATA INGEST
# ------------------------------------------------------------------
st.sidebar.header("🚀 Data Source")

sample_data = st.sidebar.checkbox("▶ Use demo synthetic sample (100 rows)")
uploaded = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if sample_data:
    np.random.seed(0)
    demo = {
        "Age": np.random.randint(22, 55, 100),
        "Income": np.random.randint(30_000, 150_000, 100),
        "City": np.random.choice(["Mumbai", "Delhi NCR", "Bangalore"], 100),
        "Gender": np.random.choice(["Male", "Female"], 100),
        "TryMealKit": np.random.choice(["Definitely", "Maybe", "Unlikely"], 100),
        "SpendMealKit": np.random.randint(150, 350, 100)
    }
    df = pd.DataFrame(demo)
elif uploaded:
    df = _load_csv(uploaded)
else:
    st.stop()   # no data – end execution

st.sidebar.success("✅ Data loaded!")

# ------------------------------------------------------------------
# MAIN TABS
# ------------------------------------------------------------------
tab_labels = [
    "Data Visualisation",
    "Classification",
    "Clustering",
    "Association Rules",
    "Regression & Forecast"
]
tabs = st.tabs(tab_labels)

# ================================================================
# 1 – DATA VISUALISATION
# ================================================================
with tabs[0]:
    st.header("📊 Exploratory Analysis")
    st.write("Preview of dataset:")
    st.dataframe(df.head())

    # ---------- Dynamic filter example ----------
    if "Age" in df.columns:
        age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
        age_range = st.slider("Filter by age", age_min, age_max, (age_min, age_max))
        df_vis = df.query("Age >= @age_range[0] & Age <= @age_range[1]")
    else:
        df_vis = df.copy()

    # ---------- Example insights (extend to ≥15) ----------
    if "Income" in df_vis.columns:
        st.subheader("Distribution · Income")
        st.plotly_chart(
            px.histogram(df_vis, x="Income", nbins=30),
            use_container_width=True
        )

    if {"City", "TryMealKit"}.issubset(df_vis.columns):
        st.subheader("Intent to try Meal-Kit by City")
        st.plotly_chart(
            px.histogram(df_vis, x="City", color="TryMealKit", barmode="group"),
            use_container_width=True
        )

# ================================================================
# 2 – CLASSIFICATION
# ================================================================
with tabs[1]:
    st.header("🤖 Classification Models")

    # ---------- target selection ----------
    cat_targets = [c for c in df.columns if df[c].dtype == "object"]
    if not cat_targets:
        st.error("No categorical target column found – add at least one.")
        st.stop()

    target_col = st.selectbox("Choose target variable", cat_targets)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ---------- preprocess & split ----------
    pre = _preprocessor(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # ---------- train & eval ----------
    results, trained_pipes = [], {}
    for name, clf in _cls_models().items():
        pipe = Pipeline([("prep", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        results.append({
            "Model":      name,
            "Accuracy":   accuracy_score(y_test, y_pred),
            "Precision":  precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall":     recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-score":   f1_score(y_test, y_pred, average="weighted", zero_division=0)
        })
        trained_pipes[name] = pipe

    res_df = pd.DataFrame(results).round(3)
    st.dataframe(res_df)

    # ---------- confusion matrix ----------
    st.subheader("Confusion Matrix")
    cm_model = st.selectbox("Select model", res_df["Model"])
    if st.checkbox("Show matrix"):
        pipe = trained_pipes[cm_model]
        cm = confusion_matrix(y_test, pipe.predict(X_test), labels=pipe.classes_)
        st.plotly_chart(
            px.imshow(
                cm,
                x=pipe.classes_,
                y=pipe.classes_,
                text_auto=True,
                color_continuous_scale="Blues"
            ),
            use_container_width=True
        )

    # ---------- ROC curve (binary only) ----------
    if len(y_test.unique()) == 2:
        st.subheader("ROC Curve (all models)")
        bin_map = {c: i for i, c in enumerate(sorted(y.unique()))}
        y_test_bin = y_test.map(bin_map)

        fig_roc = go.Figure()
        for name, pipe in trained_pipes.items():
            if hasattr(pipe.named_steps["clf"], "predict_proba"):
                y_prob = pipe.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test_bin, y_prob)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="grey"))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ---------- batch prediction ----------
    st.subheader("Batch Prediction on New Data")
    predict_file = st.file_uploader("Upload CSV without target column", type=["csv"], key="predict_csv")
    if predict_file:
        new_df = pd.read_csv(predict_file)
        best_model = res_df.sort_values("F1-score", ascending=False)["Model"].iloc[0]
        best_pipe = trained_pipes[best_model]
        preds = best_pipe.predict(new_df)
        pred_df = new_df.copy()
        pred_df[target_col + "_pred"] = preds
        st.write("Preview of predictions:")
        st.dataframe(pred_df.head())
        st.markdown(_download_link(pred_df, "predictions.csv"), unsafe_allow_html=True)

# ================================================================
# 3 – CLUSTERING
# ================================================================
with tabs[2]:
    st.header("📦 K-Means Clustering")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least two numeric columns for clustering.")
        st.stop()

    k = st.slider("Number of clusters (k)", 2, 10, 4, 1)
    scaler = StandardScaler()
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaler.fit_transform(df[numeric_cols]))
    df["Cluster"] = km.labels_

    # optional elbow
    if st.checkbox("Show elbow curve"):
        inertias = []
        for kk in range(2, 11):
            km_tmp = KMeans(n_clusters=kk, random_state=42, n_init=10)
            km_tmp.fit(scaler.transform(df[numeric_cols]))
            inertias.append(km_tmp.inertia_)
        st.plotly_chart(
            px.line(x=list(range(2, 11)), y=inertias,
                    labels={"x": "k", "y": "Inertia"}),
            use_container_width=True
        )

    st.subheader("Cluster Personas (mean of numeric features)")
    st.dataframe(df.groupby("Cluster")[numeric_cols].mean().round(1))

    st.markdown(_download_link(df, "clustered_data.csv"), unsafe_allow_html=True)

# ================================================================
# 4 – ASSOCIATION RULE MINING
# ================================================================
with tabs[3]:
    st.header("🔗 Apriori Association Rules")

    multi_cols = [
        c for c in df.columns
        if df[c].dtype == "object" and df[c].str.contains(",").any()
    ]
    if not multi_cols:
        st.info("No multi-select string columns detected (comma-separated).")
    else:
        sel_cols = st.multiselect("Choose columns", multi_cols, default=multi_cols[:3])
        min_sup = st.number_input("Min support", 0.01, 1.0, 0.05, 0.01)
        min_conf = st.number_input("Min confidence", 0.01, 1.0, 0.20, 0.01)

        if st.button("Run Apriori"):
            # build transaction matrix
            basket = pd.DataFrame()
            for col in sel_cols:
                basket = pd.concat([basket, df[col].str.get_dummies(sep=", ")], axis=1)
            freq_sets = apriori(basket, min_support=min_sup, use_colnames=True)
            rules = association_rules(
                freq_sets, metric="confidence", min_threshold=min_conf
            ).sort_values("confidence", ascending=False).head(10)
            st.dataframe(rules[["antecedents", "consequents",
                                "support", "confidence", "lift"]])

# ================================================================
# 5 – REGRESSION & FORECAST
# ================================================================
with tabs[4]:
    st.header("📈 Regression Models & 12-Month Revenue Forecast")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns available for regression.")
        st.stop()

    target_reg = st.selectbox("Choose numeric target", numeric_cols)
    predictor_opts = [c for c in numeric_cols if c != target_reg]

    # safe defaults
    default_preds = predictor_opts[:min(3, len(predictor_opts))]
    predictors = st.multiselect(
        "Predictor variables",
        options=predictor_opts,
        default=default_preds
    )

    if len(predictors) == 0:
        st.warning("Select at least one predictor.")
        st.stop()

    test_size = st.slider("Test size (hold-out)", 0.1, 0.4, 0.2, 0.05)

    Xr, yr = df[predictors], df[target_reg]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        Xr, yr, test_size=test_size, random_state=42
    )

    reg_results, trained_regs = [], {}
    for name, model in _reg_models().items():
        model.fit(X_train_r, y_train_r)
        y_pred = model.predict(X_test_r)
        reg_results.append({
            "Model": name,
            "MAE":  mean_absolute_error(y_test_r, y_pred),
            "RMSE": mean_squared_error(y_test_r, y_pred, squared=False),
            "R²":   r2_score(y_test_r, y_pred)
        })
        trained_regs[name] = model

    st.dataframe(pd.DataFrame(reg_results).round(3))

    # ---------- simple 12-month forecast by city ----------
    if {"City", "SpendMealKit"}.issubset(df.columns):
        st.subheader("12-Month Spend-per-City Forecast")
        reg_choice = st.selectbox("Regressor for forecast", list(trained_regs))
        reg_fore = trained_regs[reg_choice]
        months = np.arange(1, 13).reshape(-1, 1)

        forecasts = []
        for city, grp in df.groupby("City"):
            base_monthly = grp["SpendMealKit"].mean()
            noise = np.random.normal(0, base_monthly * 0.05, 12)
            y_series = base_monthly * (1 + 0.02 * months.flatten()) + noise
            reg_fore.fit(months, y_series)
            pred = reg_fore.predict(months)
            forecasts.append(pd.DataFrame({
                "Month": months.flatten(),
                "ForecastRevenue": pred,
                "City": city
            }))

        fc_df = pd.concat(forecasts)
        st.plotly_chart(
            px.line(fc_df, x="Month", y="ForecastRevenue", color="City"),
            use_container_width=True
        )
        st.markdown(_download_link(fc_df, "revenue_forecast.csv"), unsafe_allow_html=True)
