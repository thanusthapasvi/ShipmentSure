"""
Title:  ShipmentSure Predicting On-Time Delivery Using Supplier Data
Author: Thummala Thanus Thapasvi
GitHub: https://github.com/thanusthapasvi/shipment_sure

Model Artifacts:
    - shipment_best_model.pkl
    - scaler.joblib
    - feature_order.pkl

Dependencies:
    streamlit, pandas, numpy, joblib, scikit-learn, xgboost

"""

import streamlit as st
import pandas as pd
import joblib

st.markdown("""
<style>
    #root {
        --ac: #118888;
        --acb: #88FFFF;
        --acd: #113333;
        --acb2: #FF88FF;
        box-sizing: border-box;
    }
    .stMain {
        background-color: #112222;
        background: linear-gradient(135deg, #112222 0%, #112222 45%, #113333 50%, #112222 55%, #112222 100%);
        padding: 20px;
        background-size: 300% 300%;
        animation: bgMove 3s linear infinite;
    }
    @keyframes bgMove {
        0% { background-position: 0% 50%; }
        100% { background-position: 110% 50%; }
    }
    .stAppHeader {
        background: var(--acd);
        box-shadow: 0 0.5px white;
    }
    .st-emotion-cache-1anq8dj {
        /* predict Button */
        background: black;
        border-radius: 999px;
        border: 2px solid var(--ac);
        transition: all 0.3s ease-in-out;
        & p {
            color: var(--acb);
        }
    }
    .st-emotion-cache-1anq8dj:hover {
        background: var(--ac);
        box-shadow: 0 0 6px 2px var(--ac);
        & p {
            color: white;
        }
    }
    .st-emotion-cache-1anq8dj:active {
        transform: scale(0.95);
    }
    .stSidebar {
        background: var(--acd);
        box-shadow: 0.5px 0 white;
    }
    .stAlertContainer {
        /* Results containers */
        box-shadow: 0 2px 6px black;
    }
    .stAlertContainer:hover {
        box-shadow: 0 0 12px 2px var(--ac);
    }
    .st-emotion-cache-ujm5ma, .stMainMenu, span {
        /* Sidebar close button, main menu button, header text */
        color: var(--acb);
    }
    .st-emotion-cache-11xx4re {
        /* Slider thumb */
        background: var(--acb);
    }
    .st-emotion-cache-jigjfz {
        /* Slider thumb number */
        color: var(--acb);
    }
    div[data-baseweb="select"], div[data-testid="stNumberInputContainer"]  {
        /* select box and number input */
        background: #001a1a !important;
        border: 2px solid var(--ac) !important;
        border-radius: 10px !important;
        transition: all 0.3s ease-in-out;
    }
    div[data-baseweb="select"]:hover, div[data-testid="stNumberInputContainer"]:hover {
        border: 2px solid var(--acb) !important;
        box-shadow: 0 0 8px 1px var(--ac);
    }
    div[data-baseweb="select"] > div, input[type="number"] {
        /* text in select box and number input*/
        color: var(--acb) !important;
        font-weight: 500;
    }
    ul[data-testid="stSelectboxVirtualDropdown"]{
        /* dropdown for select */
        background: var(--acd) !important;
        border: 1px solid var(--ac) !important;
        border-radius: 10px !important;
        animation: fadeSlide 0.5s ease;
        color: var(--acb);
    }
    ul[data-testid="stSelectboxVirtualDropdown"]:hover {
        border: 1px solid var(--acb) !important;
        box-shadow: 0 0 8px 1px var(--acb);
    }
    ul[data-testid="stSelectboxVirtualDropdown"] > div > div > li {
        /* list in dropdown */
        color: var(--acb) !important;
        padding: 6px 10px !important;
        transition: all 0.2s ease-in-out;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] > div > div > li:hover {
        background: rgba(0,200,200,0.2) !important;
    }
    @keyframes fadeSlide {
        0% {
            opacity: 0;
            transform: translateY(-20px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }
}
</style>
""", unsafe_allow_html=True)


# Load Model, Scaler, and Feature Order
model = joblib.load("shipment_best_model.pkl")
scaler = joblib.load("scaler.joblib")
final_features = joblib.load("feature_order.pkl")  # From training

st.set_page_config(page_title="Shipment On-Time Prediction", page_icon="truck", layout="centered")
st.title("Shipment Delivery Prediction")
st.markdown("### Predicts whether a shipment will reach **on time** or be **delayed**")

# Input fields
st.sidebar.header("Enter Shipment Details")
# Categorical Inputs
Warehouse_block = st.sidebar.selectbox("Warehouse Block", ["A", "B", "C", "D", "E", "F"])
Product_importance = st.sidebar.selectbox("Product Importance", ["low", "medium", "high"])
Mode_of_Shipment = st.sidebar.selectbox("Mode of Shipment", ["Road", "Ship", "Flight"])

Customer_care_calls = st.sidebar.slider("Customer Care Calls", 1, 7, 3)
Customer_rating = st.sidebar.slider("Customer Rating", 1, 5, 3)
Cost_of_the_Product = st.sidebar.number_input("Cost of Product ($)", 10, 500, 200)
Prior_purchases = st.sidebar.slider("Prior Purchases", 0, 10, 4)
Discount_offered = st.sidebar.number_input("Discount Offered (%)", 0, 80, 10)
Weight_in_gms = st.sidebar.number_input("Weight (gms)", 100, 8000, 2000)

# Manual Encoding (same as training)
# Product_importance: low=0, medium=1, high=2
importance_map = {"low": 0, "medium": 1, "high": 2}
Product_importance = importance_map[Product_importance]

# Warehouse_block: A=1, B=2, ..., F=6
warehouse_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
Warehouse_block = warehouse_map[Warehouse_block]

# Mode_of_Shipment: Road=0, Ship=1, Flight=2
mode_map = {'Road': 0, 'Ship': 1, 'Flight': 2}
Mode_of_Shipment = mode_map[Mode_of_Shipment]

# Feature Engineering
Cost_to_Weight_ratio = Cost_of_the_Product / Weight_in_gms if Weight_in_gms > 0 else 0

# Prepare input in EXACT order as training
input_data = pd.DataFrame([[
    Customer_care_calls,
    Customer_rating,
    Cost_of_the_Product,
    Prior_purchases,
    Discount_offered,
    Weight_in_gms,
    Product_importance,
    Warehouse_block,
    Mode_of_Shipment,  # Already encoded as 0,1,2
    Cost_to_Weight_ratio
]], columns=[
    'Customer_care_calls',
    'Customer_rating',
    'Cost_of_the_Product',
    'Prior_purchases',
    'Discount_offered',
    'Weight_in_gms',
    'Product_importance',
    'Warehouse_block',
    'Mode_of_Shipment',
    'Cost_to_Weight_ratio'
])

# Ensure columns are in correct order (safety)
input_data = input_data[final_features]

# Scale only the numeric + ratio columns (same as training)
scale_cols = [
    'Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
    'Prior_purchases', 'Discount_offered', 'Weight_in_gms', 'Cost_to_Weight_ratio'
]
input_scaled = input_data.copy()
input_scaled[scale_cols] = scaler.transform(input_data[scale_cols])
# Prediction
if st.button("Predict Delivery Status"):
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]
    if pred == 1:
        st.success("Shipment is likely to reach **on time!**")
        confidence = proba
    else:
        st.error("Shipment is likely to be **delayed.**")
        confidence = 1 - proba

    if confidence > 0.7:
        bar_color = "var(--acb)"
    elif confidence > 0.5:
        bar_color = "#FFDD33"
    else:
        bar_color = "#FF4444"

    st.markdown(f"""
    <style>
    [data-testid="stProgress"] > div > div > div > div {{
        background-color: {bar_color};
        background: linear-gradient(to right, {bar_color} 0%, {bar_color} 90%, var(--acb2) 95%, var(--acb2) 100%);
        transition: all 0.6s ease-in-out;
    }}
    </style>
    """, unsafe_allow_html=True)

    # Show progress bar and confidence
    st.markdown(f"### Confidence: **{confidence:.2%}**")
    st.progress(float(confidence))

