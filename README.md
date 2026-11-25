# Bank Customer Churn Prediction

An end-to-end Machine Learning project that predicts the likelihood of bank customers leaving a service. The project covers synthetic data generation, data validation, exploratory data analysis, preprocessing, model training, hyperparameter tuning, model evaluation, and an interactive Streamlit application.

## Project Overview

The objective is to identify customers who may be at risk of churning using customer demographic, financial, and account activity information.

The project uses a synthetic dataset of 10,000 bank customers with 12 features. It demonstrates a complete machine-learning workflow, from data preparation to model deployment.

**Note:** The dataset is synthetically generated for educational and demonstration purposes. Results should not be interpreted as findings from real banking customers.

## Project Workflow

### 1. Data Generation and Validation
- `generate_data.py` – Generates the synthetic bank customer dataset.
- `validate_data.py` – Checks data quality, missing values, duplicates, and valid ranges.

### 2. Exploratory Data Analysis
- `eda.py` – Examines customer characteristics and churn patterns.
- Generates visualizations for churn distribution, age, geography, activity, products, numerical distributions, outliers, and correlations.

### 3. Data Preprocessing
- `preprocess.py` – Prepares the dataset for machine learning.
- Handles numerical scaling and categorical encoding.
- Uses a stratified train-test split.
- Saves the fitted preprocessing pipeline and processed datasets.

### 4. Model Training and Comparison
- `model_training.py` – Trains baseline Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting models.
- `threshold_analysis.py` – Evaluates classification thresholds.
- `hyperparameter_tuning.py` – Tunes model hyperparameters using cross-validation.

### 5. Feature Analysis and Evaluation
- `feature_analysis.py` – Examines feature importance using model-based and permutation methods.
- `final_evaluation.py` – Compares the tuned models using classification metrics and confusion matrices.
- `final_model_selection.py` – Selects and saves the final prediction model and classification threshold.

### 6. Prediction and Deployment
- `predict.py` – Provides a prediction pipeline using the saved model.
- `app.py` – Provides an interactive Streamlit interface for entering customer details and viewing churn probability and predicted status.

## Model Performance

The final model comparison showed:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6360 | 0.3518 | 0.6247 | **0.4502** | 0.6768 |
| Random Forest | 0.6745 | 0.3739 | 0.5409 | 0.4422 | 0.6890 |
| Gradient Boosting | **0.7820** | **0.6541** | 0.1824 | 0.2852 | **0.6977** |

Logistic Regression was selected for the final application based on its F1 score and churn detection performance.

The application uses a classification threshold of 0.47, selected using validation data. This threshold prioritizes identifying more potential churners at the cost of additional false positives.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Streamlit

## How to Run Locally

### 1. Clone the Repository

```bash

git clone https://github.com/Febin1004/ML-Customer-Churn-Prediction.git
cd ML-Customer-Churn-Prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate and Validate Data

```bash
python generate_data.py
python validate_data.py
```

### 5. Run Exploratory Data Analysis and Preprocessing

```bash
python eda.py
python preprocess.py
```

### 6. Train and Evaluate Models

```bash
python model_training.py
python threshold_analysis.py
python hyperparameter_tuning.py
python feature_analysis.py
python final_evaluation.py
python final_model_selection.py
```

### 7. Launch the Streamlit Application

```bash
streamlit run app.py
```

**Important:** Run the model preparation scripts before launching the application. The app requires the saved final model and threshold files.

## Project Outputs

The project generates:
- Exploratory data analysis visualizations.
- Preprocessed datasets and feature information.
- Baseline and tuned model evaluation results.
- Confusion matrices and ROC curves.
- Feature importance analysis.
- A saved final Logistic Regression model.
- An interactive Streamlit application.

## Business Application

Customer churn prediction can help banks identify customers who may be at risk of leaving. Such predictions can support proactive retention strategies, including personalized offers, improved customer engagement, and targeted customer support.

This project demonstrates how machine learning can support data-driven customer retention decisions.