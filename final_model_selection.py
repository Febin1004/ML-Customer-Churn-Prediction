
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(os.path.join(BASE_DIR, "bank_customer_churn.csv")):
    BASE_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "bank_customer_churn.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "final_model_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Exited", "CustomerId"])
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Dataset loaded successfully.")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 3. BUILD PREPROCESSING PIPELINE
# --------------------------------------------------

numerical_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "EstimatedSalary"
]

categorical_features = [
    "Geography",
    "Gender"
]

passthrough_features = [
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember"
]

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features),
    ("pass", "passthrough", passthrough_features)
])


# --------------------------------------------------
# 4. SPLIT TRAINING DATA FOR VALIDATION
# --------------------------------------------------

X_subtrain, X_validation, y_subtrain, y_validation = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("\nTraining subset:", len(X_subtrain))
print("Validation subset:", len(X_validation))


# --------------------------------------------------
# 5. TRAIN TUNED LOGISTIC REGRESSION
# --------------------------------------------------

validation_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        C=0.01,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

validation_model.fit(X_subtrain, y_subtrain)

validation_probabilities = validation_model.predict_proba(
    X_validation
)[:, 1]


# --------------------------------------------------
# 6. FIND BEST CLASSIFICATION THRESHOLD
# --------------------------------------------------

thresholds = np.arange(0.10, 0.91, 0.01)

threshold_results = []

for threshold in thresholds:

    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_validation, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_validation, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_validation, predictions, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

best_row = threshold_df.loc[
    threshold_df["F1 Score"].idxmax()
]

best_threshold = float(best_row["Threshold"])

print("\nBEST VALIDATION THRESHOLD")
print(best_row.to_string())


# Save threshold analysis
threshold_df.to_csv(
    os.path.join(OUTPUT_DIR, "threshold_analysis.csv"),
    index=False
)

plt.figure(figsize=(9, 5))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1 Score"],
    label="F1 Score"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best threshold: {best_threshold:.2f}"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Validation Threshold Analysis")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "threshold_analysis.png"),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 7. RETRAIN MODEL ON FULL TRAINING DATA
# --------------------------------------------------

final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        C=0.01,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

final_model.fit(X_train, y_train)

test_probabilities = final_model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 8. COMPARE DEFAULT AND SELECTED THRESHOLDS
# --------------------------------------------------

threshold_comparison = {}

for name, threshold in {
    "Default (0.50)": 0.50,
    "Selected": best_threshold
}.items():

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    threshold_comparison[name] = predictions

    print(f"\n{name} - Threshold: {threshold:.2f}")

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print("Recall:", recall_score(y_test, predictions))
    print("F1 Score:", f1_score(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))


# --------------------------------------------------
# 9. SAVE FINAL MODEL AND THRESHOLD
# --------------------------------------------------

with open(
    os.path.join(MODEL_DIR, "final_churn_model.pkl"), "wb"
) as file:
    pickle.dump(final_model, file)

with open(
    os.path.join(OUTPUT_DIR, "best_threshold.txt"), "w"
) as file:
    file.write(str(best_threshold))


# --------------------------------------------------
# 10. SAVE MODEL COMPARISON
# --------------------------------------------------

comparison_rows = []

for name, predictions in threshold_comparison.items():

    comparison_rows.append({
        "Threshold Type": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions)
    })

comparison_df = pd.DataFrame(comparison_rows)

comparison_df.to_csv(
    os.path.join(OUTPUT_DIR, "final_model_comparison.csv"),
    index=False
)

print("\nFINAL MODEL SAVED SUCCESSFULLY.")
print("Selected threshold:", round(best_threshold, 2))
print("Output folder:", OUTPUT_DIR)
