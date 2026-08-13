# Bank Customer Churn Predictor

This repository demonstrates a complete end-to-end Machine Learning pipeline for predicting customer churn in the banking sector. The project showcases data synthesis, preprocessing, model training (Random Forest), and a deployed web application (Streamlit).

## Project Architecture
1. **Data Engineering:** `generate_data.py` creates a synthetic but highly realistic dataset of 10,000 bank customers with complex hidden rules for churn probability.
2. **Machine Learning Pipeline:** `train_model.py` handles:
   - Data scaling and One-Hot Encoding via `sklearn.compose.ColumnTransformer`.
   - Training a `RandomForestClassifier` with balanced class weights to handle imbalanced churn data.
   - Saving the trained model using `joblib`.
3. **Deployment:** `app.py` provides an interactive web interface using `Streamlit`. Users can input customer details and instantly receive a churn probability score.

## How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the Data
```bash
python generate_data.py
```

### 3. Train the Model
```bash
python train_model.py
```

### 4. Run the Web Application
```bash
streamlit run app.py
```

## Business Value
By accurately identifying customers at high risk of churning, banks can proactively deploy targeted retention strategies (e.g., fee waivers, customized financial products), significantly reducing customer acquisition costs and stabilizing revenue streams.
