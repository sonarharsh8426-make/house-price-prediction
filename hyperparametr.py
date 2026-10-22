
from sheet1 import sheet
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV

# 1. Features and target
x = sheet.drop("Price_INR", axis=1)
y = sheet["Price_INR"]

# 2. Numerical and categorical columns
number_columns = x.select_dtypes(include=["number"]).columns
text_columns = x.select_dtypes(include=["object", "str"]).columns

# 3. Preprocessing
process = ColumnTransformer([
    ("numerical", StandardScaler(), number_columns),
    ("category", OneHotEncoder(handle_unknown="ignore"), text_columns)
])

# 4. Pipeline
random = Pipeline([
    ("preprocessing", process),
    ("model", RandomForestRegressor(random_state=42))
])

# 5. Train-test split: 80% training, 20% testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=10
)

# 6. Parameters
param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [5, 10, None],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

# 7. Hyperparameter tuning
grid = GridSearchCV(
    estimator=random,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

# 8. Train and tune
grid.fit(x_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV MAE:", -grid.best_score_)

# 9. Best model
best_model = grid.best_estimator_

# 10. Prediction
y_predict = best_model.predict(x_test)

# 11. Evaluation
print("MAE:", mean_absolute_error(y_test, y_predict))
print("RMSE:", mean_squared_error(y_test, y_predict) ** 0.5)
print("R2 Score:", r2_score(y_test, y_predict))

joblib.dump(best_model,"house_price_model.pkl")

print("Model saved successfully!")