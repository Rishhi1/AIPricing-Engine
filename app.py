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
# UI CSS
# -----------------------------
st.markdown("""
<style>
.stApp { background-color: #0A0F1C; color: #E5E7EB; }
[data-testid="stSidebar"] { background-color: #020617; }
.block-container { padding-top: 1.5rem; max-width: 1200px; }
h1 { font-size: 30px !important; font-weight: 600; }

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    padding: 15px;
    border-radius: 12px;
}

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
# SIDEBAR
# -----------------------------
st.sidebar.title("Pricing Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV")
price = st.sidebar.number_input("Product Price", value=100.0)
run = st.sidebar.button("Run Analysis")

# -----------------------------
# HEADER
# -----------------------------
st.title("Dynamic Pricing Dashboard")
st.caption("Machine learning powered revenue optimization")
st.markdown("---")

# -----------------------------
# DATA HANDLING
# -----------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Select Target Column", df.columns)
else:
    df = None
    target_column = None

# -----------------------------
# PREPROCESSING
# -----------------------------
def preprocess_data(df, target_column):
    df = df.drop_duplicates()
    df = df.ffill()

    y = df[target_column]
    X = df.drop(columns=[target_column])

    X = pd.get_dummies(X, drop_first=True)

    return X, y

# -----------------------------
# MODEL TRAINING
# -----------------------------
def train_best_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100),
        "XGBoost": XGBRegressor(n_estimators=200, max_depth=5)
    }

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
# RESULTS
# -----------------------------
if run:

    if df is None:
        st.error("Please upload a dataset to run the model.")
    
    else:
        st.info("Training model...")

        X, y = preprocess_data(df, target_column)

        model, feature_cols, model_name, model_score = train_best_model(X, y)

        st.success(f"Best Model: {model_name} (R²: {model_score:.3f})")

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

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Sales", f"{base_sales:.2f}")
        col2.metric("Revenue", f"{base_revenue:.2f}")
        col3.metric("Optimal Price", f"{optimal_price:.2f}")
        col4.metric("Max Revenue", f"{max_revenue:.2f}")

        st.metric("Improvement (%)", f"{improvement:.2f}%")

        # Graph
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