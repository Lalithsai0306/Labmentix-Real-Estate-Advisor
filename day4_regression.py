import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib

print("Loading ML-ready dataset...")
df = pd.read_csv('ml_ready_housing_data.csv')

# Simulate future price
np.random.seed(42)
appreciation_rates = np.random.uniform(0.04, 0.09, size=len(df))
df['Future_Price_5_Years'] = df['Price_in_Lakhs'] * ((1 + appreciation_rates) ** 5)

# THE FIX: We keep 'Price_in_Lakhs' as a feature!
# We only drop the target itself, the per-sqft price, and our other target
drop_cols = ['Future_Price_5_Years', 'Price_per_SqFt', 'Good_Investment']
X = df.drop(columns=drop_cols)
y = df['Future_Price_5_Years']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nRetraining XGBoost with Current Price included...")
model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- NEW Regression Model Performance ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} Lakhs")
print(f"R-Squared (R2) Score: {r2:.3f} (Closer to 1.0 is better!)")

joblib.dump(model, 'xgboost_price_model.pkl')
joblib.dump(list(X.columns), 'model_features.pkl')
print("Model saved successfully.")