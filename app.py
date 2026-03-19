import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Pricing Dashboard", layout="wide")

# -----------------------------
# MODERN UI CSS
# -----------------------------
st.markdown("""
<style>
.stApp { background-color: #0A0F1C; color: #E5E7EB; }

.block-container {
    padding-top: 1rem;
    max-width: 1100px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.04);
    padding: 20px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
}

/* Buttons */
div.stButton > button {
    background: #2563EB;
    color: white !important;
    border-radius: 8px;
    height: 42px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<h1 style='font-size: 24px;'>Dynamic Pricing Dashboard</h1>
<p style='color: #9CA3AF;'>AI-powered revenue optimization engine</p>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Upload", "Analysis", "Results"])

# -----------------------------
# TAB 1 → UPLOAD
# -----------------------------
with tab1:

    st.markdown("### Upload Dataset")

    uploaded_file = st.file_uploader("Upload CSV")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.success("Dataset loaded successfully")

        st.dataframe(df.head())

        target_column = st.selectbox("Select Target Column", df.columns)

        st.session_state["df"] = df
        st.session_state["target"] = target_column

# -----------------------------
# FUNCTIONS
# -----------------------------
def preprocess_data(df, target_column):

    df = df.drop_duplicates()
    df = df.ffill()

    # -----------------------------
    # HANDLE DATE COLUMN
    # -----------------------------
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')

            df[col + "_year"] = df[col].dt.year
            df[col + "_month"] = df[col].dt.month
            df[col + "_day"] = df[col].dt.day
            df[col + "_weekday"] = df[col].dt.weekday

            df = df.drop(columns=[col])

    # -----------------------------
    # SPLIT TARGET
    # -----------------------------
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # -----------------------------
    # ENCODE CATEGORICAL
    # -----------------------------
    X = pd.get_dummies(X, drop_first=True)

    return X, y


def train_best_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100),
        "XGBoost": XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)    }

    best_model = None
    best_score = -np.inf
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = r2_score(y_test, preds)

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    return best_model, X.columns, best_name, best_score

# -----------------------------
# TAB 2 → ANALYSIS
# -----------------------------
with tab2:

    st.markdown("### Run Pricing Analysis")

    price = st.number_input("Base Price", value=100.0)

    train_btn = st.button("Train Model")

if train_btn:

    if "df" not in st.session_state:
        st.error("Upload dataset first")
    else:
        df = st.session_state["df"]
        target_column = st.session_state["target"]

        st.info("Training model...")

        X, y = preprocess_data(df, target_column)

        model, feature_cols, model_name, model_score = train_best_model(X, y)

        # Save everything
        st.session_state["model"] = model
        st.session_state["features"] = feature_cols
        st.session_state["price"] = price
        st.session_state["trained"] = True

        st.success(f"Best Model: {model_name} | R²: {model_score:.3f}")

# -----------------------------
# TAB 3 → RESULTS
# -----------------------------
with tab3:

    st.markdown("### Results Dashboard")

    if "model" not in st.session_state:
        st.warning("Run analysis first")
    else:
        model = st.session_state["model"]
        feature_cols = st.session_state["features"]
        price = st.session_state["price"]

        df = st.session_state["df"]
        X, y = preprocess_data(df, st.session_state["target"])

        sample_row = X.iloc[0:1]

        prices = np.linspace(price * 0.8, price * 1.2, 50)
        revenues = []

        for p in prices:
            temp = sample_row.copy()

            if "price" in temp.columns:
                temp["price"] = p

            temp = temp.reindex(columns=feature_cols, fill_value=0)

            pred = max(0, model.predict(temp)[0])
            revenues.append(pred * p)

        prices = np.array(prices)
        revenues = np.array(revenues)

        best_idx = np.argmax(revenues)
        optimal_price = prices[best_idx]
        max_revenue = revenues[best_idx]

        base_sales = max(0, model.predict(sample_row)[0])
        base_revenue = base_sales * price

        improvement = ((max_revenue - base_revenue) / base_revenue) * 100

        # -----------------------------
        # KPI CARDS
        # -----------------------------
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Sales", f"{base_sales:.2f}")
        col2.metric("Revenue", f"{base_revenue:.2f}")
        col3.metric("Optimal Price", f"{optimal_price:.2f}")
        col4.metric("Max Revenue", f"{max_revenue:.2f}")

        st.metric("Revenue Improvement", f"{improvement:.2f}%")

        # -----------------------------
        # GRAPH
        # -----------------------------
        st.markdown("### Price Optimization Curve")

        fig, ax = plt.subplots(figsize=(9,4))

        ax.plot(prices, revenues, color="#3B82F6", linewidth=2)
        ax.axvline(optimal_price, linestyle='--', color="#22C55E")

        ax.set_facecolor("#0A0F1C")
        fig.patch.set_facecolor("#0A0F1C")

        ax.tick_params(colors='white')
        ax.set_xlabel("Price", color='white')
        ax.set_ylabel("Revenue", color='white')

        st.pyplot(fig)