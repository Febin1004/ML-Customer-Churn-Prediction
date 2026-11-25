
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# If this script is inside a subfolder, locate the project root
if not os.path.exists(os.path.join(BASE_DIR, "bank_customer_churn.csv")):
    BASE_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "bank_customer_churn.csv")

MODEL_DIR = os.path.join(BASE_DIR, "tuning_outputs", "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "feature_analysis_outputs")

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
print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 3. LOAD TUNED MODELS
# --------------------------------------------------

rf_model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest_tuned.pkl")
)

lr_model = joblib.load(
    os.path.join(MODEL_DIR, "logistic_regression_tuned.pkl")
)

print("\nTuned models loaded successfully.")


# --------------------------------------------------
# 4. GET FEATURE NAMES
# --------------------------------------------------

preprocessor = rf_model.named_steps["preprocessor"]
feature_names = preprocessor.get_feature_names_out()

display_names = [
    name.replace("num__", "")
        .replace("cat__", "")
        .replace("pass__", "")
    for name in feature_names
]


# --------------------------------------------------
# 5. RANDOM FOREST FEATURE IMPORTANCE
# --------------------------------------------------

rf_classifier = rf_model.named_steps["model"]

rf_importance = pd.DataFrame({
    "Feature": display_names,
    "Importance": rf_classifier.feature_importances_
})

rf_importance = rf_importance.sort_values(
    by="Importance",
    ascending=False
)

rf_importance.to_csv(
    os.path.join(OUTPUT_DIR, "random_forest_importance.csv"),
    index=False
)

plt.figure(figsize=(10, 6))
plt.barh(
    rf_importance["Feature"][::-1],
    rf_importance["Importance"][::-1]
)
plt.title("Random Forest - Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "random_forest_importance.png"),
    dpi=300
)
plt.close()

print("\nRandom Forest Feature Importance:")
print(rf_importance.to_string(index=False))


# --------------------------------------------------
# 6. LOGISTIC REGRESSION COEFFICIENTS
# --------------------------------------------------

lr_classifier = lr_model.named_steps["model"]

coefficients = pd.DataFrame({
    "Feature": display_names,
    "Coefficient": lr_classifier.coef_[0]
})

coefficients["Absolute_Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)

coefficients.to_csv(
    os.path.join(OUTPUT_DIR, "logistic_regression_coefficients.csv"),
    index=False
)

plot_data = coefficients.sort_values("Coefficient")

plt.figure(figsize=(10, 6))
plt.barh(
    plot_data["Feature"],
    plot_data["Coefficient"]
)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Logistic Regression - Feature Coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "logistic_regression_coefficients.png"),
    dpi=300
)
plt.close()

print("\nLogistic Regression Coefficients:")
print(coefficients.to_string(index=False))


# --------------------------------------------------
# 7. PERMUTATION IMPORTANCE
# --------------------------------------------------

print("\nCalculating permutation importance...")

permutation_result = permutation_importance(
    rf_model,
    X_test,
    y_test,
    scoring="f1",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

permutation_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance_Mean": permutation_result.importances_mean,
    "Importance_Std": permutation_result.importances_std
})

permutation_df = permutation_df.sort_values(
    by="Importance_Mean",
    ascending=False
)

permutation_df.to_csv(
    os.path.join(OUTPUT_DIR, "permutation_importance.csv"),
    index=False
)

plt.figure(figsize=(10, 6))
plt.barh(
    permutation_df["Feature"][::-1],
    permutation_df["Importance_Mean"][::-1],
    xerr=permutation_df["Importance_Std"][::-1]
)
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Permutation Importance - Random Forest")
plt.xlabel("Mean decrease in F1 score")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "permutation_importance.png"),
    dpi=300
)
plt.close()


# --------------------------------------------------
# 8. COMPLETION
# --------------------------------------------------

print("\nPermutation Importance:")
print(permutation_df.to_string(index=False))

print("\nFeature analysis completed.")
print("Output folder:", OUTPUT_DIR)
