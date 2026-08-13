import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_model():
    print("Loading data...")
    try:
        df = pd.read_csv('bank_customer_churn.csv')
    except FileNotFoundError:
        print("Please run generate_data.py first.")
        return

    # Features and Target
    X = df.drop(columns=['CustomerId', 'Exited'])
    y = df['Exited']

    # Identify categorical and numerical columns
    categorical_cols = ['Geography', 'Gender']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ])

    # Create the pipeline with a Random Forest Classifier
    # We use class_weight='balanced' because churn data is usually imbalanced
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model_pipeline.fit(X_train, y_train)

    # Predictions and Evaluation
    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save the model
    print("Saving model to churn_model.pkl...")
    joblib.dump(model_pipeline, 'churn_model.pkl')
    print("Training complete! Model is ready for deployment.")

if __name__ == "__main__":
    train_model()
