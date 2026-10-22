
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("house_price_model.pkl")

st.title("House Price Prediction")
st.write("Enter house details to predict its price.")

# User inputs
area = st.number_input("Area (sq ft)", min_value=1, value=1200)
bedrooms = st.number_input("Bedrooms", min_value=1, value=2)
bathrooms = st.number_input("Bathrooms", min_value=1, value=2)
age = st.number_input("House Age (years)", min_value=0, value=5)

location = st.selectbox(
    "Location",
    ["Budget", "Mid-range", "Premium"]
)

parking = st.selectbox("Parking", ["Yes", "No"])

if st.button("Predict House Price"):
    new_house = pd.DataFrame({
        "Area_sqft": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Age_years": [age],
        "Location": [location],
        "Parking": [parking]
    })

    prediction = model.predict(new_house)

    st.success(f"Predicted House Price: {prediction[0]:,.2f}")
