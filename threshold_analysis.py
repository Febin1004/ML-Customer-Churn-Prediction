
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# --------------------------------------------------
# 1. SET PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)

PREPROCESS_DIR = os.path.join(
    BASE_DIR, "preprocessing_outputs"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "threshold_outputs"
)

MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# 2. LOAD PREPROCESSED DATA
# --------------------------------------------------

X_train = np.load(
    os.path.join(PREPROCESS_DIR, "X_train_processed.npy")
)

X_test = np.load(
    os.path.join(PREPROCESS_DIR, "X_test_processed.npy")
)

y_train = np.load(
    os.path.join(PREPROCESS_DIR, "y_train.npy")
)

y_test = np.load(
    os.path.join(PREPROCESS_DIR, "y_test.npy")
)

print("\nDATA LOADED SUCCESSFULLY")
print("------------------------")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# --------------------------------------------------
# 3. CREATE TRAINING AND VALIDATION SPLIT
# --------------------------------------------------

X_subtrain, X_validation, y_subtrain, y_validation = (
    train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )
)

print("\nVALIDATION SPLIT")
print("----------------")
print("Subtraining samples:", len(X_subtrain))
print("Validation samples:", len(X_validation))


# --------------------------------------------------
# 4. TRAIN LOGISTIC REGRESSION
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_subtrain, y_subtrain)

print("\nLogistic Regression trained successfully.")


# --------------------------------------------------
# 5. GET VALIDATION PROBABILITIES
# --------------------------------------------------

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]


# --------------------------------------------------
# 6. TEST DIFFERENT CLASSIFICATION THRESHOLDS
# --------------------------------------------------

thresholds = np.arange(0.10, 0.91, 0.05)

results = []

for threshold in thresholds:

    y_validation_pred = (
        validation_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_validation, y_validation_pred
    )

    precision = precision_score(
        y_validation,
        y_validation_pred,
        zero_division=0
    )

    recall = recall_score(
        y_validation, y_validation_pred
    )

    f1 = f1_score(
        y_validation,
        y_validation_pred,
        zero_division=0
    )

    results.append({
        "Threshold": round(threshold, 2),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


# --------------------------------------------------
# 7. DISPLAY THRESHOLD COMPARISON
# --------------------------------------------------

threshold_df = pd.DataFrame(results)

print("\nTHRESHOLD PERFORMANCE COMPARISON")
print("---------------------------------")

print(
    threshold_df.to_string(
        index=False,
        formatters={
            "Threshold": "{:.2f}".format,
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format
        }
    )
)

threshold_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "threshold_comparison.csv"
    ),
    index=False
)


# --------------------------------------------------
# 8. SELECT BEST THRESHOLD USING VALIDATION F1
# --------------------------------------------------

best_index = threshold_df["F1 Score"].idxmax()

best_threshold = threshold_df.loc[
    best_index, "Threshold"
]

print("\nBEST VALIDATION THRESHOLD")
print("-------------------------")
print("Selected threshold:", best_threshold)
print(
    "Validation F1 Score:",
    round(threshold_df.loc[best_index, "F1 Score"], 4)
)


# --------------------------------------------------
# 9. PLOT THRESHOLD PERFORMANCE
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1 Score"],
    marker="o",
    label="F1 Score"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Selected threshold = {best_threshold:.2f}"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Classification Performance at Different Thresholds")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "threshold_performance.png"
    ),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 10. RETRAIN MODEL ON FULL TRAINING DATA
# --------------------------------------------------

final_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

final_model.fit(X_train, y_train)


# --------------------------------------------------
# 11. FINAL EVALUATION ON UNTOUCHED TEST DATA
# --------------------------------------------------

test_probabilities = final_model.predict_proba(
    X_test
)[:, 1]

# Default threshold
default_predictions = (
    test_probabilities >= 0.50
).astype(int)

# Optimized threshold
optimized_predictions = (
    test_probabilities >= best_threshold
).astype(int)


print("\nFINAL TEST SET COMPARISON")
print("-------------------------")

for name, predictions in [
    ("Default Threshold (0.50)", default_predictions),
    ("Optimized Threshold", optimized_predictions)
]:

    print("\n", name)

    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions), 4)
    )

    print(
        "Precision:",
        round(
            precision_score(
                y_test, predictions, zero_division=0
            ), 4
        )
    )

    print(
        "Recall:",
        round(recall_score(y_test, predictions), 4)
    )

    print(
        "F1 Score:",
        round(
            f1_score(
                y_test, predictions, zero_division=0
            ), 4
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Retained", "Churned"],
            zero_division=0
        )
    )


# --------------------------------------------------
# 12. SAVE OPTIMIZED MODEL AND THRESHOLD
# --------------------------------------------------

joblib.dump(
    final_model,
    os.path.join(
        MODEL_DIR,
        "threshold_tuned_logistic_regression.pkl"
    )
)

with open(
    os.path.join(OUTPUT_DIR, "best_threshold.txt"),
    "w"
) as file:

    file.write(str(best_threshold))


# --------------------------------------------------
# 13. SAVE FINAL CONFUSION MATRIX
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    optimized_predictions,
    display_labels=["Retained", "Churned"],
    cmap="Blues",
    ax=ax
)

ax.set_title(
    f"Optimized Logistic Regression (Threshold={best_threshold:.2f})"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "optimized_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 14. COMPLETION
# --------------------------------------------------

print("\nTHRESHOLD ANALYSIS COMPLETED")
print("----------------------------")
print("Output folder:", OUTPUT_DIR)
print("Selected threshold:", best_threshold)
print("Optimized model saved successfully.")
