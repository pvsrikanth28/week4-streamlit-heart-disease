import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


# UCI Heart Disease dataset
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target"
]

# Load dataset
df = pd.read_csv(
    URL,
    names=columns,
    na_values="?"
)

# Convert target to binary
df["target"] = (df["target"] > 0).astype(int)

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -------------------------
# Logistic Regression
# -------------------------

logistic_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)

logistic_f1 = f1_score(y_test, logistic_pred)
logistic_accuracy = accuracy_score(y_test, logistic_pred)
logistic_precision = precision_score(y_test, logistic_pred)
logistic_recall = recall_score(y_test, logistic_pred)


# -------------------------
# Random Forest
# -------------------------

random_forest_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

random_forest_model.fit(X_train, y_train)

rf_pred = random_forest_model.predict(X_test)

rf_f1 = f1_score(y_test, rf_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)


# -------------------------
# Choose best model using F1
# -------------------------

if rf_f1 >= logistic_f1:
    best_model = random_forest_model
    best_model_name = "Random Forest"

    best_metrics = {
        "accuracy": rf_accuracy,
        "precision": rf_precision,
        "recall": rf_recall,
        "f1": rf_f1
    }

else:
    best_model = logistic_model
    best_model_name = "Logistic Regression"

    best_metrics = {
        "accuracy": logistic_accuracy,
        "precision": logistic_precision,
        "recall": logistic_recall,
        "f1": logistic_f1
    }


# Save model and metadata
model_data = {
    "model": best_model,
    "model_name": best_model_name,
    "features": list(X.columns),
    "metrics": best_metrics
}

joblib.dump(
    model_data,
    "heart_disease_model.joblib"
)


print("=" * 50)
print("MODEL TRAINING COMPLETED")
print("=" * 50)

print(f"Selected Model: {best_model_name}")
print(f"Accuracy:  {best_metrics['accuracy']:.4f}")
print(f"Precision: {best_metrics['precision']:.4f}")
print(f"Recall:    {best_metrics['recall']:.4f}")
print(f"F1 Score:  {best_metrics['f1']:.4f}")

print("\nSaved model as:")
print("heart_disease_model.joblib")