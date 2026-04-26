import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("Loading ML-ready dataset...")
df = pd.read_csv('ml_ready_housing_data.csv')

# 1. SELECT FEATURES (X) AND TARGET (y)
# We want to predict 'Good_Investment'. 
# We must drop the target itself, and the future price (since we wouldn't know that yet)
X = df.drop(columns=['Good_Investment', 'Future_Price_5_Years'], errors='ignore')
y = df['Good_Investment']

# 2. SPLIT THE DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Classification Model on {len(X_train)} properties...")

# 3. INITIALIZE AND TRAIN RANDOM FOREST
# We limit the depth to prevent the model from memorizing the data (overfitting)
clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
clf.fit(X_train, y_train)

print("Classification Model Training Complete!")

# 4. EVALUATE THE MODEL
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n--- Classification Model Performance ---")
print(f"Accuracy Score: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 5. EXTRACT FEATURE IMPORTANCE
# Let's see what the AI thinks makes a "Good Investment"
importances = pd.DataFrame(
    {'Feature': X.columns, 'Importance': clf.feature_importances_}
).sort_values(by='Importance', ascending=False)

print("\n--- Top 5 Most Important Features ---")
print(importances.head(5))

# 6. SAVE THE MODEL FOR STREAMLIT
model_filename = 'rf_classifier_model.pkl'
joblib.dump(clf, model_filename)

print(f"\nDay 5 Complete! Model successfully saved as: {model_filename}")