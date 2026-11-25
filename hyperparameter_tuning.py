
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

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
    classification_report
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(
    PROJECT_DIR, "bank_customer_churn.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR, "tuning_outputs"
)

MODEL_DIR = os.path.join(
    OUTPUT_DIR, "models"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD RAW DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["CustomerId", "Exited"])
y = df["Exited"]

print("\nDATA LOADED")
print("-----------")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDATA SPLIT")
print("----------")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 4. DEFINE PREPROCESSING
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
# 5. DEFINE CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# --------------------------------------------------
# 6. DEFINE MODELS AND PARAMETER SEARCH SPACES
# --------------------------------------------------

model_configurations = {

    "Logistic Regression": {
        "model": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "parameters": {
            "model__C": [0.01, 0.1, 1, 10]
        },

        "iterations": 4
    },


    "Random Forest": {
        "model": RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "parameters": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 10],
            "model__min_samples_leaf": [1, 4],
            "model__max_features": ["sqrt"]
        },

        "iterations": 6
    },


    "Gradient Boosting": {
        "model": GradientBoostingClassifier(
            random_state=42
        ),

        "parameters": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__min_samples_leaf": [1, 4]
        },

        "iterations": 6
    }
}


# --------------------------------------------------
# 7. HYPERPARAMETER TUNING
# --------------------------------------------------

tuning_results = []
best_models = {}

for name, configuration in model_configurations.items():

    print("\n" + "=" * 60)
    print("TUNING:", name)
    print("=" * 60)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", configuration["model"])
    ])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=configuration["parameters"],
        n_iter=configuration["iterations"],
        scoring="f1",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    best_models[name] = best_model

    print("\nBest Parameters:")
    print(search.best_params_)

    print(
        "\nBest Cross-Validation F1:",
        round(search.best_score_, 4)
    )

    tuning_results.append({
        "Model": name,
        "Best CV F1": search.best_score_,
        "Best Parameters": str(search.best_params_)
    })

    filename = (
        name.lower().replace(" ", "_") + "_tuned.pkl"
    )

    joblib.dump(
        best_model,
        os.path.join(MODEL_DIR, filename)
    )


# --------------------------------------------------
# 8. SAVE CROSS-VALIDATION RESULTS
# --------------------------------------------------

tuning_df = pd.DataFrame(tuning_results)

tuning_df = tuning_df.sort_values(
    by="Best CV F1",
    ascending=False
)

print("\n" + "=" * 70)
print("CROSS-VALIDATION COMPARISON")
print("=" * 70)

print(
    tuning_df[
        ["Model", "Best CV F1"]
    ].to_string(
        index=False,
        formatters={
            "Best CV F1": "{:.4f}".format
        }
    )
)

tuning_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cross_validation_results.csv"
    ),
    index=False
)


# --------------------------------------------------
# 9. EVALUATE TUNED MODELS ON TEST DATA
# --------------------------------------------------

test_results = []

print("\n" + "=" * 70)
print("TUNED MODEL TEST EVALUATION")
print("=" * 70)

for name, model in best_models.items():

    y_pred = model.predict(X_test)

    y_probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test, y_probability
    )

    test_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    print("\n", name)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Retained", "Churned"],
            zero_division=0
        )
    )


# --------------------------------------------------
# 10. SAVE TEST RESULTS
# --------------------------------------------------

test_df = pd.DataFrame(test_results)

test_df = test_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\nFINAL TUNED MODEL COMPARISON")
print("----------------------------")

print(
    test_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format,
            "ROC-AUC": "{:.4f}".format
        }
    )
)

test_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "tuned_model_test_results.csv"
    ),
    index=False
)


# --------------------------------------------------
# 11. COMPLETION
# --------------------------------------------------

print("\nHYPERPARAMETER TUNING COMPLETED")
print("--------------------------------")
print("Output folder:", OUTPUT_DIR)
print("Tuned models saved:", MODEL_DIR)
