from sheet1 import sheet
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import  LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV 
x=sheet.drop("Price_INR",axis=1)
y=sheet["Price_INR"]

number_columns=x.select_dtypes(include=['int64']).columns
text_columns = x.select_dtypes(include=['object', 'str']).columns

# now create preprocessing 
process=ColumnTransformer([
    ("numerical",StandardScaler(),number_columns),
    ("category",OneHotEncoder(handle_unknown='ignore'),text_columns)
])


#model-1:-linerregression
linear=Pipeline([
    ("preprocessing",process),
    ("linear",LinearRegression())
])

#model-2:-desicion tree 
decision=Pipeline([
    ("preprocessing",process),
    ("linear",DecisionTreeRegressor())
])

#model-3:-random forest 
random=Pipeline([
    ("preprocessing",process),
    ("linear",RandomForestRegressor())
])

#train split 
x_train,x_test,y_train,y_test=train_test_split(
    x,y,train_size=0.2,random_state=10
)

# here all process of  LinearRegression  and matrix 
linear.fit(x_train,y_train)
y_predict_liner=linear.predict(x_test)

print("mae:",mean_absolute_error(y_test,y_predict_liner))

mse = mean_squared_error(y_test,y_predict_liner)
print("RMSE:", mse ** 0.5)

print("R2 Score:", r2_score(y_test,y_predict_liner)) 

print("decision tress")
# here all process of  desicion tree  and matrix 
decision.fit(x_train,y_train)
y_predict_desicion=decision.predict(x_test)

print("mae:",mean_absolute_error(y_test,y_predict_desicion))

mse = mean_squared_error(y_test,y_predict_desicion)
print("RMSE:", mse ** 0.5)

print("R2 Score:", r2_score(y_test,y_predict_desicion)) 

print("random forest")
# here all process of  random forest and matrix 
random.fit(x_train,y_train)
y_predict_random=random.predict(x_test)

print("mae:",mean_absolute_error(y_test,y_predict_random))

mse = mean_squared_error(y_test,y_predict_random)
print("RMSE:", mse ** 0.5)

print("R2 Score:", r2_score(y_test,y_predict_random)) 


