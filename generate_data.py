import pandas as pd
import numpy as np
import random

def generate_bank_data(n_samples=10000):
    np.random.seed(42)
    random.seed(42)
    
    data = []
    geographies = ['France', 'Germany', 'Spain']
    genders = ['Male', 'Female']
    
    for i in range(n_samples):
        customer_id = 15500000 + i
        credit_score = int(np.random.normal(650, 60))
        geography = random.choice(geographies)
        gender = random.choice(genders)
        age = int(np.random.normal(38, 10))
        tenure = random.randint(0, 10)
        balance = round(np.random.uniform(0, 250000), 2)
        num_of_products = random.choices([1, 2, 3, 4], weights=[0.5, 0.45, 0.04, 0.01])[0]
        has_cr_card = random.choices([1, 0], weights=[0.7, 0.3])[0]
        is_active_member = random.choices([1, 0], weights=[0.5, 0.5])[0]
        estimated_salary = round(np.random.uniform(10000, 200000), 2)
        
        # Determine churn logic (Exited) based on some hidden rules
        churn_prob = 0.1
        if age > 50: churn_prob += 0.2
        if num_of_products >= 3: churn_prob += 0.4
        if balance == 0: churn_prob -= 0.1
        if not is_active_member: churn_prob += 0.15
        if geography == 'Germany': churn_prob += 0.1
        
        exited = 1 if random.random() < churn_prob else 0
        
        data.append([
            customer_id, credit_score, geography, gender, age, 
            tenure, balance, num_of_products, has_cr_card, 
            is_active_member, estimated_salary, exited
        ])
        
    columns = [
        'CustomerId', 'CreditScore', 'Geography', 'Gender', 'Age', 
        'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 
        'IsActiveMember', 'EstimatedSalary', 'Exited'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('bank_customer_churn.csv', index=False)
    print("Successfully generated bank_customer_churn.csv with 10,000 records.")

if __name__ == "__main__":
    generate_bank_data()
