
import pandas as pd
import numpy as np
from pathlib import Path


def validate_dataset():

    file_path = Path("bank_customer_churn.csv")

    # Check whether dataset exists
    if not file_path.exists():
        print("ERROR: Dataset file not found!")
        return

    df = pd.read_csv(file_path)

    print("=" * 55)
    print("       BANK CUSTOMER CHURN - DATA VALIDATION")
    print("=" * 55)

    errors = 0

    def check(description, condition):
        nonlocal errors

        if condition:
            print(f"[PASS] {description}")
        else:
            print(f"[FAIL] {description}")
            errors += 1

    # 1. Dataset dimensions
    check("Dataset contains 10,000 records", len(df) == 10000)
    check("Dataset contains 12 columns", len(df.columns) == 12)

    # 2. Required columns
    expected_columns = [
        'CustomerId', 'CreditScore', 'Geography', 'Gender',
        'Age', 'Tenure', 'Balance', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
        'Exited'
    ]

    check(
        "All required columns are present",
        set(expected_columns).issubset(df.columns)
    )

    if not set(expected_columns).issubset(df.columns):
        print("\nCannot continue: required columns are missing.")
        return

    # 3. Missing values
    check("No missing values", df.isnull().sum().sum() == 0)

    # 4. Duplicate records and IDs
    check("No duplicate records", not df.duplicated().any())
    check("No duplicate customer IDs",
          not df['CustomerId'].duplicated().any())

    # 5. Data types
    numeric_columns = [
        'CustomerId', 'CreditScore', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard',
        'IsActiveMember', 'EstimatedSalary', 'Exited'
    ]

    check(
        "Numerical columns have numeric data types",
        all(
            pd.api.types.is_numeric_dtype(df[col])
            for col in numeric_columns
        )
    )

    # 6. Numerical range validation
    range_checks = {
        'Age': (18, 90),
        'CreditScore': (300, 850),
        'Tenure': (0, 10),
        'Balance': (0, 250000),
        'NumOfProducts': (1, 4),
        'EstimatedSalary': (10000, 200000)
    }

    print("\n--- Numerical Range Checks ---")

    for column, (minimum, maximum) in range_checks.items():

        valid = df[column].between(minimum, maximum).all()

        check(
            f"{column} is between {minimum} and {maximum}",
            valid
        )

    # 7. Categorical validation
    print("\n--- Categorical Checks ---")

    check(
        "Geography contains valid categories",
        df['Geography'].isin(
            ['France', 'Germany', 'Spain']
        ).all()
    )

    check(
        "Gender contains valid categories",
        df['Gender'].isin(['Male', 'Female']).all()
    )

    # 8. Binary column validation
    print("\n--- Binary Column Checks ---")

    binary_columns = [
        'HasCrCard', 'IsActiveMember', 'Exited'
    ]

    for column in binary_columns:

        check(
            f"{column} contains only 0 and 1",
            df[column].isin([0, 1]).all()
        )

    # 9. Target distribution
    print("\n--- Churn Distribution ---")

    churn_counts = df['Exited'].value_counts()

    churn_rate = df['Exited'].mean() * 100

    print(f"Retained customers: {(df['Exited'] == 0).sum()}")
    print(f"Churned customers: {(df['Exited'] == 1).sum()}")
    print(f"Churn rate: {churn_rate:.2f}%")

    # 10. Zero balance analysis
    zero_balance = (df['Balance'] == 0).sum()

    print(f"\nZero-balance customers: {zero_balance}")

    # 11. Final report
    print("\n" + "=" * 55)
    print("                VALIDATION SUMMARY")
    print("=" * 55)

    print(f"Total checks failed: {errors}")

    if errors == 0:
        print("RESULT: ALL VALIDATION CHECKS PASSED!")
    else:
        print("RESULT: DATASET NEEDS CORRECTION.")

    print("=" * 55)


if __name__ == "__main__":
    validate_dataset()
