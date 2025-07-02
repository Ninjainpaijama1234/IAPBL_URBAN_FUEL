# urban_fuel_dashboard/app.py
# ------------------------------------------------------------------
#  Urban Fuel – Streamlit Analytics Dashboard  ·  July 2025 (refreshed)
# ------------------------------------------------------------------
#  ✓ Uses **UrbanFuelSyntheticSurvey.csv** with the provided lowercase headers
#  ✓ One RENAME map transforms raw columns ➜ readable camel‑case names once
#  ✓ KPI cards get a gradient style, subtle shadows, and icons
#  ✓ Numeric handling is defensive (no int(NaN) crashes)
#  ✓ Ready for deployment on Streamlit Cloud
# ------------------------------------------------------------------

import os, base64, numpy as np, pandas as pd, streamlit as st
import plotly.express as px, plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from mlxtend.frequent_patterns import apriori, association_rules

# ============================ CONFIG & STYLE ============================
st.set_page_config(page_title="Urban Fuel Analytics", page_icon="🍱", layout="wide", initial_sidebar_state="collapsed")

# ——— CSS polish (gradient KPI, smoother fonts) ———
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"]  {font-family:'Inter', sans-serif;}
#MainMenu, footer {visibility:hidden;}
[data-testid="stAppViewContainer"]>.main {background:#F4F7FF;}
.metric-card {padding:1.2rem 1rem;border-radius:16px;background:linear-gradient(135deg,#fff 0%,#ecf2ff 100%);box-shadow:0 2px 6px rgba(0,0,0,.06);} 
.metric-card h4 {margin:0;font-size:.85rem;color:#6b7280;font-weight:600;}
.metric-card h2 {margin:2px 0 0;font-size:2rem;color:#1f2937;font-weight:700;}
a.dl-btn {background:#4F8BF9;color:#fff!important;padding:6px 16px;border-radius:8px;text-decoration:none;font-weight:600;}
a.dl-btn:hover{background:#3d73e0}
</style>
""", unsafe_allow_html=True)

DATA_FILE = "UrbanFuelSyntheticSurvey.csv"

# ============================ HELPERS ==================================
RENAME = {
    "age": "Age", "gender": "Gender", "income_inr": "Income", "city": "City",
    "employment_type": "EmploymentType", "work_hours_per_day": "WorkHours", "commute_minutes": "CommuteMins",
    "dinners_cooked_per_week": "DinnersCooked", "enjoy_cooking": "EnjoyCooking", "cooking_skill_rating": "CookingSkill",
    "cooking_challenges": "CookingChallenges", "dinner_time_hour": "DinnerTime", "primary_cook": "PrimaryCook",
    "favorite_cuisines": "FavoriteCuisines", "meal_type_pref": "MealType", "non_veg_freq_per_week": "NonVegFreq",
    "dietary_goals": "DietaryGoals", "allergies": "Allergies", "healthy_importance_rating": "HealthyImportance",
    "track_macros": "TrackMacros", "orders_outside_per_week": "OrdersOutside", "spend_outside_per_meal_inr": "SpendOutside",
    "willing_to_pay_mealkit_inr": "SpendMealKit", "payment_mode": "PaymentMode", "priority_taste": "PriorityTaste",
    "priority_price": "PriorityPrice", "priority_nutrition": "PriorityNutrition", "priority_ease": "PriorityEase",
    "priority_time": "PriorityTime", "subscribe_try": "TryMealKit", "delivery_pref": "DeliveryPref",
    "meals_per_week_subscription": "MealsPerWeek", "desired_features": "DesiredFeatures", "priority_health": "PriorityHealth",
    "priority_time_savings": "PriorityTimeSavings", "priority_convenience": "PriorityConvenience", "priority_variety": "PriorityVariety",
    "priority_affordability": "PriorityAffordability", "continue_service": "ContinueService", "refer_service": "ReferService",
    "incentive_pref": "IncentivePref", "healthy_brand_association": "HealthyBrand", "additional_needs": "AdditionalNeeds"
}

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[^\d.]", "", regex=True), errors="coerce")

@st.cache_data
def load_df() -> pd.DataFrame:
    if not os.path.exists(DATA_FILE):
        st.error(f"**{DATA_FILE}** not found in working directory."); st.stop()
    df = pd.read_csv(DATA_FILE)
    # rename once → internal consistency
    df.rename(columns={k:v for k,v in RENAME.items() if k in df.columns}, inplace=True)

    # helper numeric fields
    if "SpendMealKit" in df:
        df["SpendMealKitValue"] = _to_num(df["SpendMealKit"]) if df["SpendMealKit"].dtype=="object" else df["SpendMealKit"]
    if "Income" in df:
        df["IncomeValue"] = df["Income"] if pd.api.types.is_numeric_dtype(df["Income"]) else _to_num(df["Income"])
    return df

def num_cols(df):
    return df.select_dtypes("number").columns.tolist()

def dl_link(df, name):
    b64 = base64.b64encode(df.to_csv(index=False).encode()).decode()
    return f'<a class="dl-btn" href="data:file/csv;base64,{b64}" download="{name}">Download ▼</a>'

# ============================ LOAD DATA =================================
df = load_df()

# ============================ KPI CARDS =================================
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-card'><h4>Respondents</h4><h2>{len(df):,}</h2></div>", unsafe_allow_html=True)
with col2:
    age_avg = df["Age"].mean() if "Age" in df else np.nan
    st.markdown(f"<div class='metric-card'><h4>Average Age</h4><h2>{age_avg:.1f}</h2></div>", unsafe_allow_html=True)
with col3:
    spend_avg = df["SpendMealKitValue"].mean() if "SpendMealKitValue" in df else np.nan
    st.markdown(f"<div class='metric-card'><h4>Avg Spend / Kit</h4><h2>₹{spend_avg:,.0f}</h2></div>", unsafe_allow_html=True)

st.markdown("---")

# ============================ FILTER PANEL ===============================
with st.expander("🎚️ Filters", expanded=False):
    # City
    city_sel = st.multiselect("City", sorted(df["City"].unique()), default=list(df["City"].unique())) if "City" in df else []

    # Income slider resilient
    inc_vals = df["IncomeValue"].dropna() if "IncomeValue" in df else pd.Series(dtype=float)
    if not inc_vals.empty:
        inc_min, inc_max = int(inc_vals.min()), int(inc_vals.max())
        inc_sel = st.slider("Income (₹)", inc_min, inc_max, (inc_min, inc_max), step=10_000)
    else:
        inc_sel = None

# apply filters
filtered = df.copy()
if city_sel:
    filtered = filtered[filtered["City"].isin(city_sel)]
if inc_sel is not None:
    filtered = filtered.query("IncomeValue >= @inc_sel[0] & IncomeValue <= @inc_sel[1]")

# ============================ TABS =======================================
vis_tab, cls_tab, clu_tab, ar_tab, reg_tab = st.tabs(["📊 Visuals", "🤖 Classification", "📦 Clustering", "🔗 Rules", "📈 Regression"])

# ----------------------------- VISUALS ----------------------------------
with vis_tab:
    st.subheader("Key Distributions & Relationships")
    if "Age" in filtered:
        st.plotly_chart(px.histogram(filtered, x="Age", nbins=25, color_discrete_sequence=["#636efa"]), use_container_width=True)
    if "IncomeValue" in filtered:
        st.plotly_chart(px.histogram(filtered, x="IncomeValue", nbins=30, color_discrete_sequence=["#ef553b"]), use_container_width=True)
    if {"City", "TryMealKit"}.issubset(filtered.columns):
        st.plotly_chart(px.histogram(filtered, x="City", color="TryMealKit", barmode="group", color_discrete_sequence=px.colors.qualitative.Set2), use_container_width=True)
    st.markdown(dl_link(filtered, "filtered.csv"), unsafe_allow_html=True)

# ------------------------- CLASSIFICATION ------------------------------
with cls_tab:
    st.subheader("Build & Compare Classifiers")
    cat_targets = [c for c in filtered.columns if filtered[c].dtype=="object" and filtered[c].nunique()<=10]
    if not cat_targets:
        st.info("No suitable categorical target.")
    else:
        target = st.selectbox("Target", cat_targets)
        X, y = filtered.drop(columns=[target]), filtered[target]
        pre = ColumnTransformer([("num", StandardScaler(), num_cols(filtered)), ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in filtered.columns if c not in num_cols(filtered)+[target]])])
        Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        results = []
        for name, clf in {"KNN":KNeighborsClassifier(), "DT":DecisionTreeClassifier(random_state=42), "RF":RandomForestClassifier(n_estimators=250,random_state=42), "GB":GradientBoostingClassifier(random_state=42)}.items():
            pipe = Pipeline([("pre", pre),("clf", clf)]); pipe.fit(Xtr, ytr)
            yhat = pipe.predict(Xte)
            results.append({"Model":name,"Acc":accuracy_score(yte,yhat),"F1":f1_score(yte,yhat,average="weighted",zero_division=0)})
        st.dataframe(pd.DataFrame(results).round(3))

# --------------------------- CLUSTERING --------------------------------
with clu_tab:
    st.subheader("K‑Means Segmentation")
    numeric = num_cols(filtered)
    if len(numeric)>=2:
        k = st.slider("Choose k", 2, 10, 4)
        model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(StandardScaler().fit_transform(filtered[numeric]))
        seg = filtered.assign(Cluster=model.labels_)
        st.dataframe(seg.groupby("Cluster")[numeric].mean().round(1))
        st.markdown(dl_link(seg, "clusters.csv"), unsafe_allow_html=True)
    else:
        st.info("Need ≥2 numeric columns")

# ------------------------ ASSOCIATION RULES ----------------------------
with ar_tab:
    st.subheader("Market‑Basket Insights (Apriori)")
    multi_cols = [c for c in filtered if filtered[c].dtype=="object" and filtered[c].str.contains(",").any()]
    if multi_cols:
        cols = st.multiselect("Multi‑select columns", multi_cols, default=multi_cols[:3])
        if st.button("Run Apriori"):
            basket = pd.concat([filtered[c].str.get_dummies(sep=", ") for c in cols], axis=1)
            rules = association_rules(apriori(basket, min_support=0.05, use_colnames=True), metric="confidence", min_threshold=0.3).head(15)
            st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])
    else:
        st.info("No comma‑separated columns found.")

# ------------------------------ REGRESSION -----------------------------
with reg_tab:
    st.subheader("Numerical Predictions")
    nums = num_cols(filtered)
    if nums:
        tgt = st.selectbox("Target", nums)
        pred_cols = [c for c in nums if c!=tgt]
        Xr, yr = filtered[pred_cols], filtered[tgt]
        Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
        mdl = Linear
