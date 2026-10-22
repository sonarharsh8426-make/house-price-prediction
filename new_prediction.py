import pandas as pd  
import joblib
model=joblib.load("house_price_model.pkl")
new_house = pd.DataFrame([{
    "Area_sqft": 1200,
    "Bedrooms": 2,
    "Bathrooms": 2,
    "Age_years": 5,
    "Location": "Mid-range",
    "Parking": "Yes"
}])

# Price predict karo
prediction = model.predict(new_house)

print("Predicted House Price: ₹", round(prediction[0], 2))