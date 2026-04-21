# =============================
# streamlit_app.py (UI - RUN THIS FOR FRONTEND)
# =============================

import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open('car_price_model.pkl', 'rb'))

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Enter car details to predict selling price")

# Inputs
present_price = st.number_input("Present Price (in lakhs)", min_value=0.0)
kms_driven = st.number_input("Kilometers Driven", min_value=0)
owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])
car_age = st.slider("Car Age (years)", 0, 20)

fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# Encoding
fuel_diesel = 1 if fuel_type == "Diesel" else 0
fuel_petrol = 1 if fuel_type == "Petrol" else 0
seller_individual = 1 if seller_type == "Individual" else 0
trans_manual = 1 if transmission == "Manual" else 0

# Predict
if st.button("Predict Price"):
    input_data = pd.DataFrame([[present_price, kms_driven, owner, car_age,
                                fuel_diesel, fuel_petrol,
                                seller_individual, trans_manual]],
                              columns=['Present_Price', 'Kms_Driven', 'Owner', 'car_age',
                                       'Fuel_Type_Diesel', 'Fuel_Type_Petrol',
                                       'Seller_Type_Individual', 'Transmission_Manual'])

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Selling Price: ₹ {round(prediction, 2)} lakhs")
