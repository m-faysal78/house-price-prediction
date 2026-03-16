import joblib
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import preprocess

X_train, X_test, y_train, y_test = preprocess("../data/housing.csv")

model = joblib.load("../model/house_price_model.pkl")

preds = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, preds))
print("R2:", r2_score(y_test, preds))
