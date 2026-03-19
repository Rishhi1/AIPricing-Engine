import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

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

input {
    background-color: #020617 !important;
    color: #E5E7EB !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("best_model.pkl")
feature_columns = model.feature_names_in_

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Pricing Controls")

# Upload dataset (Version 2 prep)
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
# FUNCTIONS
# -----------------------------
def predict_sales(price):
    df_input = pd.DataFrame([{"price": price}])
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    return max(0, model.predict(df_input)[0])

def optimize_price(price):
    prices = np.linspace(price * 0.8, price * 1.2, 50)
    revenues = []

    for p in prices:
        pred = predict_sales(p)
        revenues.append(pred * p)

    prices = np.array(prices)
    revenues = np.array(revenues)

    best_idx = np.argmax(revenues)

    return prices, revenues, prices[best_idx], revenues[best_idx]

# -----------------------------
# RESULTS
# -----------------------------
if run:

    # Dataset mode (future feature)
    if df is not None and target_column is not None:
        st.warning("Dataset mode will be implemented in Version 2")

    # Default pricing model
    else:
        sales = predict_sales(price)
        revenue = sales * price

        prices, revenues, optimal_price, max_revenue = optimize_price(price)

        improvement = ((max_revenue - revenue) / revenue) * 100 if revenue != 0 else 0

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Sales", f"{sales:.2f}")
        col2.metric("Revenue", f"{revenue:.2f}")
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
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        ax.set_xlabel("Price")
        ax.set_ylabel("Revenue")

        st.pyplot(fig)
