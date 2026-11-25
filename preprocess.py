
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "bank_customer_churn.csv"
OUTPUT_DIR = PROJECT_DIR / "preprocessing_outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

print("=" * 60)
print("       BANK CUSTOMER CHURN - DATA PREPROCESSING")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 3. FEATURE SELECTION
# --------------------------------------------------

print("\n--- Feature Selection ---")

# Remove customer identifier
df = df.drop(columns=["CustomerId"])

# Separate input features and target
X = df.drop(columns=["Exited"])
y = df["Exited"]

print("Input features shape:", X.shape)
print("Target shape:", y.shape)

print("\nSelected input features:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())


# --------------------------------------------------
# 4. IDENTIFY FEATURE TYPES
# --------------------------------------------------

print("\n--- Identifying Feature Types ---")

# Continuous numerical features
numerical_features = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "EstimatedSalary"
]

# Categorical features
categorical_features = [
    "Geography",
    "Gender"
]

# Binary and discrete numerical features
passthrough_features = [
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember"
]

print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)
print("Passthrough features:", passthrough_features)


# --------------------------------------------------
# 5. TRAIN-TEST SPLIT
# --------------------------------------------------

print("\n--- Train-Test Split ---")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(4) * 100)

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(4) * 100)


# --------------------------------------------------
# 6. NUMERICAL PREPROCESSING
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# --------------------------------------------------
# 7. CATEGORICAL PREPROCESSING
# --------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


# --------------------------------------------------
# 8. COMBINE PREPROCESSING STEPS
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
        ("pass", "passthrough", passthrough_features)
    ]
)


# --------------------------------------------------
# 9. FIT AND TRANSFORM DATA
# --------------------------------------------------

print("\n--- Applying Preprocessing ---")

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using training-fitted transformations
X_test_processed = preprocessor.transform(X_test)

print("Training data after preprocessing:", X_train_processed.shape)
print("Testing data after preprocessing:", X_test_processed.shape)


# --------------------------------------------------
# 10. FEATURE NAMES
# --------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

print("\nProcessed feature names:")
for feature in feature_names:
    print(feature)


# --------------------------------------------------
# 11. SAVE PREPROCESSING ARTIFACTS
# --------------------------------------------------

print("\n--- Saving Preprocessing Artifacts ---")

joblib.dump(
    preprocessor,
    OUTPUT_DIR / "preprocessor.pkl"
)

np.save(
    OUTPUT_DIR / "X_train_processed.npy",
    X_train_processed
)

np.save(
    OUTPUT_DIR / "X_test_processed.npy",
    X_test_processed
)

np.save(
    OUTPUT_DIR / "y_train.npy",
    y_train.to_numpy()
)

np.save(
    OUTPUT_DIR / "y_test.npy",
    y_test.to_numpy()
)

# Save feature names for later interpretation
pd.Series(feature_names, name="Feature").to_csv(
    OUTPUT_DIR / "feature_names.csv",
    index=False
)

print("Saved preprocessor.pkl")
print("Saved X_train_processed.npy")
print("Saved X_test_processed.npy")
print("Saved y_train.npy")
print("Saved y_test.npy")
print("Saved feature_names.csv")


# --------------------------------------------------
# 12. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("       PREPROCESSING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("Original dataset records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Original input features:", X.shape[1])
print("Processed input features:", X_train_processed.shape[1])

print("\nOutput directory:", OUTPUT_DIR.resolve())

print("=" * 60)
