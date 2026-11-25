
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. LOAD FINAL MODEL AND THRESHOLD
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(os.path.join(BASE_DIR, "bank_customer_churn.csv")):
    BASE_DIR = os.path.dirname(BASE_DIR)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "final_model_outputs",
    "best_threshold.txt"
)

with open(THRESHOLD_PATH, "r") as file:
    threshold = float(file.read().strip())


DATA_PATH = os.path.join(BASE_DIR, "bank_customer_churn.csv")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["CustomerId", "Exited"])
y = df["Exited"]

X_train, _, y_train, _ = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

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

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        C=0.01,
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])
model.fit(X_train, y_train)


# --------------------------------------------------
# 2. PREDICTION FUNCTION
# --------------------------------------------------

def predict_churn(customer_data):

    customer_df = pd.DataFrame([customer_data])

    probability = model.predict_proba(customer_df)[0][1]

    prediction = int(probability >= threshold)

    if prediction == 1:
        status = "Likely to Churn"
    else:
        status = "Likely to Stay"

    return {
        "Churn Probability": round(probability * 100, 2),
        "Prediction": status,
        "Classification Threshold": threshold
    }


# --------------------------------------------------
# 3. TEST WITH A SAMPLE CUSTOMER
# --------------------------------------------------

sample_customer = {
    "CreditScore": 600,
    "Geography": "Germany",
    "Gender": "Male",
    "Age": 55,
    "Tenure": 5,
    "Balance": 120000,
    "NumOfProducts": 3,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 80000
}

result = predict_churn(sample_customer)

print("\nCUSTOMER CHURN PREDICTION")
print("-------------------------")

for key, value in result.items():
    print(f"{key}: {value}")
