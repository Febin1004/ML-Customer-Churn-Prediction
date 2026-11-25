
import pandas as pd
import numpy as np


def generate_bank_data(n_samples=10000, random_state=42):

    # Initialize random number generator
    rng = np.random.default_rng(random_state)

    data = []

    geographies = ['France', 'Germany', 'Spain']
    genders = ['Male', 'Female']

    for i in range(n_samples):

        # Customer identification
        customer_id = 15500000 + i

        # Customer information
        credit_score = int(
            np.clip(rng.normal(650, 60), 300, 850)
        )

        geography = rng.choice(geographies)
        gender = rng.choice(genders)

        age = int(
            np.clip(np.rint(rng.normal(38, 10)), 18, 90)
        )

        # Banking information
        tenure = int(rng.integers(0, 11))

        # Generate balance, including zero-balance customers
        if rng.random() < 0.10:
            balance = 0.0
        else:
            balance = round(rng.uniform(100, 250000), 2)

        num_of_products = int(
            rng.choice(
                [1, 2, 3, 4],
                p=[0.50, 0.45, 0.04, 0.01]
            )
        )

        has_cr_card = int(
            rng.choice([1, 0], p=[0.70, 0.30])
        )

        is_active_member = int(
            rng.choice([1, 0], p=[0.50, 0.50])
        )

        estimated_salary = round(
            rng.uniform(10000, 200000), 2
        )

        # Determine churn probability
        churn_prob = 0.10

        if age > 50:
            churn_prob += 0.20

        if num_of_products >= 3:
            churn_prob += 0.40

        if balance == 0:
            churn_prob -= 0.10

        if not is_active_member:
            churn_prob += 0.15

        if geography == 'Germany':
            churn_prob += 0.10

        # Keep probability within valid limits
        churn_prob = np.clip(churn_prob, 0.01, 0.95)

        # Generate target variable
        exited = int(rng.random() < churn_prob)

        # Append customer record
        data.append([
            customer_id,
            credit_score,
            geography,
            gender,
            age,
            tenure,
            balance,
            num_of_products,
            has_cr_card,
            is_active_member,
            estimated_salary,
            exited
        ])

    # Define column names
    columns = [
        'CustomerId',
        'CreditScore',
        'Geography',
        'Gender',
        'Age',
        'Tenure',
        'Balance',
        'NumOfProducts',
        'HasCrCard',
        'IsActiveMember',
        'EstimatedSalary',
        'Exited'
    ]

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Save dataset
    df.to_csv('bank_customer_churn.csv', index=False)

    print("Dataset generated successfully!")
    print(f"Total records: {len(df)}")
    print(f"Churned customers: {df['Exited'].sum()}")
    print(f"Retained customers: {(df['Exited'] == 0).sum()}")

    return df


if __name__ == "__main__":
    generate_bank_data()
