import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading ML-ready dataset for MLflow tracking...")
df = pd.read_csv('ml_ready_housing_data.csv')

X = df.drop(columns=['Good_Investment', 'Future_Price_5_Years'], errors='ignore')
y = df['Good_Investment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the MLflow Experiment
experiment_name = "Real_Estate_Investment_Classifier"
mlflow.set_experiment(experiment_name)

print("Starting MLflow run...")
with mlflow.start_run(run_name="Random_Forest_Run_1"):
    
    # 1. Define the parameters we want to track
    n_estimators = 100
    max_depth = 10
    
    # Log parameters to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # 2. Train the model
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    
    # 3. Evaluate and Log Metrics
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    mlflow.log_metric("accuracy", accuracy)
    
    # 4. Log the model itself
    mlflow.sklearn.log_model(clf, "random_forest_model")

    print(f"Successfully logged experiment to MLflow!")
    print(f"Tracked Accuracy: {accuracy:.4f}")