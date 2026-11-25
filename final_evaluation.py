
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(os.path.join(BASE_DIR, "bank_customer_churn.csv")):
    BASE_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "bank_customer_churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "tuning_outputs", "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "final_evaluation_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
print("Test records:", len(X_test))


# --------------------------------------------------
# 3. LOAD TUNED MODELS
# --------------------------------------------------

numerical_features = [
    "CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"
]
categorical_features = ["Geography", "Gender"]
passthrough_features = ["NumOfProducts", "HasCrCard", "IsActiveMember"]

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

models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            C=0.01, max_iter=1000, class_weight="balanced", random_state=42
        ))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ]),
    "Gradient Boosting": Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=2,
            min_samples_leaf=4, random_state=42
        ))
    ])
}

for model_name, model in models.items():
    model.fit(X_train, y_train)

print("\nAll tuned model configurations retrained successfully.")


# --------------------------------------------------
# 4. MODEL EVALUATION
# --------------------------------------------------

results = []

for model_name, model in models.items():

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })

    print(f"\n{model_name}")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))


results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))


# --------------------------------------------------
# 5. CONFUSION MATRICES
# --------------------------------------------------

for model_name, model in models.items():

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"]
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    filename = (
        model_name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        os.path.join(OUTPUT_DIR, filename),
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 6. ROC CURVE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(8, 6))

for model_name, model in models.items():

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        name=model_name,
        ax=plt.gca()
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "roc_curve_comparison.png"),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 7. METRIC COMPARISON CHART
# --------------------------------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Tuned Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "metric_comparison.png"),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 8. COMPLETION
# --------------------------------------------------

print("\nFinal evaluation completed.")
print("Results saved to:", OUTPUT_DIR)
