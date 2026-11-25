
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve
)


# --------------------------------------------------
# 1. SET PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

BASE_DIR = os.path.dirname(BASE_DIR)
PREPROCESS_DIR = os.path.join(BASE_DIR, "preprocessing_outputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "model_outputs")
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
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)


# --------------------------------------------------
# 3. DEFINE MACHINE LEARNING MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# --------------------------------------------------
# 4. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

results = []
trained_models = {}
roc_data = {}

for name, model in models.items():

    print("\n" + "=" * 55)
    print("TRAINING:", name)
    print("=" * 55)

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Predicted probability of churn
    y_probability = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
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

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[name] = model

    roc_data[name] = roc_curve(
        y_test, y_probability
    )

    # Save trained model
    model_filename = (
        name.lower().replace(" ", "_") + ".pkl"
    )

    joblib.dump(
        model,
        os.path.join(MODEL_DIR, model_filename)
    )

    # Display classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Retained", "Churned"],
            zero_division=0
        )
    )

    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Retained", "Churned"],
        cmap="Blues",
        ax=ax
    )

    ax.set_title(f"{name} - Confusion Matrix")

    plt.tight_layout()

    confusion_filename = (
        name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        os.path.join(OUTPUT_DIR, confusion_filename),
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# 5. MODEL COMPARISON TABLE
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\n" + "=" * 75)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 75)

print(
    results_df.to_string(
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

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison.csv"),
    index=False
)


# --------------------------------------------------
# 6. ROC CURVE COMPARISON
# --------------------------------------------------

plt.figure(figsize=(9, 7))

for name, (fpr, tpr, thresholds) in roc_data.items():

    auc_value = results_df.loc[
        results_df["Model"] == name,
        "ROC-AUC"
    ].iloc[0]

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc_value:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Model Comparison")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "roc_curve_comparison.png"),
    dpi=300
)

plt.close()


# --------------------------------------------------
# 7. COMPLETION MESSAGE
# --------------------------------------------------

print("\nMODEL TRAINING COMPLETED")
print("------------------------")
print("Saved model comparison:", OUTPUT_DIR)
print("Saved trained models:", MODEL_DIR)
print("Saved confusion matrices and ROC curve.")
