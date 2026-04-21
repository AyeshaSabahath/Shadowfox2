# =============================
# train.py (RUN THIS FIRST)
# =============================

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load data

df = pd.read_csv('car.csv')

# Feature Engineering

df['car_age'] = 2026 - df['Year']

df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# Encoding

df = pd.get_dummies(df, drop_first=True)

# Split
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

search = RandomizedSearchCV(xgb, param_dist, n_iter=20, cv=5, scoring='r2', n_jobs=-1)
search.fit(X_train, y_train)

best_model = search.best_estimator_

# Evaluate
preds = best_model.predict(X_test)
print("R2 Score:", r2_score(y_test, preds))
print("MSE:", mean_squared_error(y_test, preds))

# Save model
with open('car_price_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Model saved successfully!")