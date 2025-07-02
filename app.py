import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import plotly.graph_objects as go
import base64, io

st.set_page_config(page_title="Urban Fuel Analytics", layout="wide")

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------
@st.cache_data
def load_csv(uploaded):
    return pd.read_csv(uploaded)

def download_link(df: pd.DataFrame, filename: str) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'

def build_preprocessor(df: pd.DataFrame, target: str):
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols + [target]]

    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])
    return pre

def cls_models():
    return {
        "K-NN":  KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "Gradient Boosting": GradientBoostingClassifier()
    }

def reg_models():
    return {
        "Linear":   LinearRegression(),
        "Ridge":    Ridge(alpha=1.0),
        "Lasso":    Lasso(alpha=0.01),
        "DT Regressor": DecisionTreeRegressor(max_depth=6)
    }

# ------------------------------------------------------------------
# Sidebar – data ingestion
# ------------------------------------------------------------------
st.sidebar.header("Data Source")
use_sample  = st.sidebar.checkbox("Generate sample synthetic data")
uploaded    = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if use_sample:
    st.sidebar.info("Demo sample (100 rows) created.")
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
    df = load_csv(uploaded)
else:
    st.stop()

# ------------------------------------------------------------------
# Main navigation
# ------------------------------------------------------------------
tabs = st.tabs([
    "Data Visualisation", "Classification", "Clustering",
    "Association Rules", "Regression & Forecast"
])

# ------------------------------------------------------------------
# 1 – Data visualisation
# ------------------------------------------------------------------
with tabs[0]:
    st.header("Exploratory Dashboards")
    st.dataframe(df.head())

    if "Age" in df.columns:
        age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
        age_sel = st.slider("Filter · Age Range", age_min, age_max, (age_min, age_max))
        df_viz  = df.query("Age >= @age_sel[0] and Age <= @age_sel[1]")
    else:
        df_viz = df.copy()

    # Insight 1 – Income distribution
    if "Income" in df_viz.columns:
        st.subheader("Income Distribution")
        st.plotly_chart(px.histogram(df_viz, x="Income", nbins=30), use_container_width=True)

    # Insight 2 – Intent by city
    if {"City", "TryMealKit"}.issubset(df_viz.columns):
        st.subheader("Meal-Kit Intent by City")
        st.plotly_chart(
            px.histogram(df_viz, x="City", color="TryMealKit", barmode="group"),
            use_container_width=True
        )

    # …extend to ≥15 descriptive visual insights as needed.

# ------------------------------------------------------------------
# 2 – Classification
# ------------------------------------------------------------------
with tabs[1]:
    st.header("Supervised Classification")

    cat_targets = [c for c in df.columns if df[c].dtype == "object"]
    tgt = st.selectbox("Target variable", cat_targets)

    X, y        = df.drop(columns=[tgt]), df[tgt]
    preproc     = build_preprocessor(df, tgt)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rows, models = [], {}
    for name, clf in cls_models().items():
        pipe = Pipeline([("prep", preproc), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)
        rows.append({
            "Model": name,
            "Accuracy":  accuracy_score(yte, yhat),
            "Precision": precision_score(yte, yhat, average="weighted", zero_division=0),
            "Recall":    recall_score(yte, yhat, average="weighted", zero_division=0),
            "F1-score":  f1_score(yte, yhat, average="weighted", zero_division=0)
        })
        models[name] = pipe

    st.dataframe(pd.DataFrame(rows).round(3))

    # Confusion matrix
    st.subheader("Confusion Matrix")
    sel_model = st.selectbox("Select model", list(models.keys()))
    if st.checkbox("Show matrix"):
        cm = confusion_matrix(yte, models[sel_model].predict(Xte), labels=models[sel_model].classes_)
        st.plotly_chart(px.imshow(
            pd.DataFrame(cm, index=models[sel_model].classes_, columns=models[sel_model].classes_),
            text_auto=True
        ), use_container_width=True)

    # ROC curve (binary only)
    if len(y.unique()) == 2:
        st.subheader("ROC Curves")
        fig = go.Figure(layout=dict(xaxis_title="FPR", yaxis_title="TPR"))
        y_bin = y.replace({y.unique()[0]: 0, y.unique()[1]: 1})
        _, yte_bin = train_test_split(y_bin, test_size=0.2, random_state=42, stratify=y_bin)
        for n, p in models.items():
            yprob = p.predict_proba(Xte)[:, 1]
            fpr, tpr, _ = roc_curve(yte_bin, yprob)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=n))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    # Batch prediction
    st.subheader("Predict on New Data")
    new_csv = st.file_uploader("Upload CSV (without target)", type=["csv"], key="new_preds")
    if new_csv:
        new_df  = pd.read_csv(new_csv)
        best    = max(rows, key=lambda r: r["F1-score"])["Model"]
        preds   = models[best].predict(new_df)
        out_df  = new_df.assign(**{tgt + "_pred": preds})
        st.dataframe(out_df.head())
        st.markdown(download_link(out_df, "predictions.csv"), unsafe_allow_html=True)

# ------------------------------------------------------------------
# 3 – Clustering
# ------------------------------------------------------------------
with tabs[2]:
    st.header("K-Means Clustering")

    numeric = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if len(numeric) < 2:
        st.error("Need ≥2 numeric columns"); st.stop()

    k = st.slider("Number of clusters", 2, 10, 4)
    scaler  = StandardScaler()
    matrix  = scaler.fit_transform(df[numeric])
    km      = KMeans(n_clusters=k, random_state=42, n_init=10).fit(matrix)
    df["Cluster"] = km.labels_

    # Optional elbow
    if st.checkbox("Show Elbow Chart"):
        inertia = [KMeans(n_clusters=i, random_state=42, n_init=10).fit(matrix).inertia_ for i in range(2, 11)]
        st.plotly_chart(px.line(x=range(2, 11), y=inertia, labels={"x":"k", "y":"Inertia"}), use_container_width=True)

    st.subheader("Cluster Persona (mean values)")
    st.dataframe(df.groupby("Cluster")[numeric].mean().round(1))

    st.markdown(download_link(df, "clustered_data.csv"), unsafe_allow_html=True)

# ------------------------------------------------------------------
# 4 – Association rules
# ------------------------------------------------------------------
with tabs[3]:
    st.header("Apriori Association Rules")

    multi_cols = [c for c in df.columns if df[c].dtype == "object" and df[c].str.contains(",").any()]
    use_cols   = st.multiselect("Multi-select columns", multi_cols, default=multi_cols[:3])
    supp = st.number_input("Min support", 0.01, 1.0, 0.05, 0.01)
    conf = st.number_input("Min confidence", 0.01, 1.0, 0.2, 0.01)

    if st.button("Run Apriori"):
        baskets = pd.DataFrame()
        for col in use_cols:
            baskets = pd.concat([baskets, df[col].str.get_dummies(sep=", ")], axis=1)
        freq = apriori(baskets, min_support=supp, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=conf)\
                .sort_values("confidence", ascending=False).head(10)
        st.dataframe(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

# ------------------------------------------------------------------
# 5 – Regression & 12-month forecast
# ------------------------------------------------------------------
with tabs[4]:
    st.header("Regression Models & Forecast by City")

    numeric = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    tgt_reg = st.selectbox("Numeric target", numeric)
    feats   = st.multiselect("Predictors", [c for c in numeric if c != tgt_reg], default=numeric[:3])
    tsplit  = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    Xr, yr = df[feats], df[tgt_reg]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=tsplit, random_state=42)

    rows_r, regs = [], {}
    for n, m in reg_models().items():
        m.fit(Xtr, ytr); pred = m.predict(Xte)
        rows_r.append({"Model": n, "MAE": mean_absolute_error(yte, pred),
                       "RMSE": mean_squared_error(yte, pred, squared=False),
                       "R²": r2_score(yte, pred)})
        regs[n] = m
    st.dataframe(pd.DataFrame(rows_r).round(3))

    # Simple 12-month revenue forecast by city
    if {"City", "SpendMealKit"}.issubset(df.columns):
        st.subheader("12-Month Revenue Forecast (by City)")
        reg_name = st.selectbox("Forecasting regressor", list(regs.keys()))
        mdl = regs[reg_name]
        month = np.arange(1, 13).reshape(-1, 1)

        forecasts = []
        for city, grp in df.groupby("City"):
            base = grp["SpendMealKit"].mean()
            trend = base * (1 + 0.02 * month.flatten())  # +2 % MoM
            mdl.fit(month, trend)
            forecasts.append(pd.DataFrame({
                "Month": month.flatten(),
                "ForecastRevenue": mdl.predict(month),
                "City": city
            }))
        fc_df = pd.concat(forecasts)
        st.plotly_chart(px.line(fc_df, x="Month", y="ForecastRevenue", color="City"), use_container_width=True)
        st.markdown(download_link(fc_df, "revenue_forecast.csv"), unsafe_allow_html=True)
