import pandas as pd
sheet=pd.read_csv(r"house_price_practice_dataset.csv")
print(sheet.isnull().sum())
print(sheet.head())
print(sheet.shape)
print(sheet.info())
print(sheet.describe()) 
print(sheet["Price_INR"].isnull().sum())