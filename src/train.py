import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from preprocess import preprocess

X_train, X_test, y_train, y_test = preprocess("../data/housing.csv")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

os.makedirs("../model", exist_ok=True)

joblib.dump(model, "../model/house_price_model.pkl")

print("House price prediction model trained")
