# Credit Risk Assisment

## Project Overview

This project focuses on predicting whether a credit card applicant is likely to become a bad customer (high risk of default) or a good customer based on historical credit behavior and demographic information. The solution is built using machine learning techniques and deployed using Streamlit.

## Problem Statement

Financial institutions need to assess credit risk before approving loans or credit cards. The objective of this project is to build a predictive model that classifies customers into:

- Good Customer (0): Low risk of default  
- Bad Customer (1): High risk of serious payment delay or default

## Dataset Description

The project uses two datasets:

1. application_record.csv  
   Contains customer demographic and financial information such as income, education, occupation, and ownership details.

2. credit_record.csv  
   Contains monthly credit history including payment status for each customer.

## Target Variable Definition

The target variable is created based on credit behavior:

- 1 (Bad Customer): Customer has serious delays (STATUS = 2, 3, 4, 5)
- 0 (Good Customer): No serious delay

## Feature Engineering

The following features were created and used for training:

- Past delay count (number of times customer had delayed payments)
- Past record duration (number of months of credit history)
- Age (derived from birth date)
- Employment years (derived from employment data)
- Encoded categorical variables (income type, education, occupation, etc.)

## Data Preprocessing

- Missing values handled using imputation
- Irrelevant columns removed
- Categorical variables encoded using Label Encoding and binary mapping
- Outliers handled in employment data
- Data imbalance handled using SMOTE

## Machine Learning Models Used

- Logistic Regression
- Random Forest Classifier

## Model Evaluation Metrics

The models were evaluated using:

- Accuracy Score
- Precision, Recall, F1-score
- ROC-AUC Score
- Confusion Matrix

## Project Workflow

1. Data collection and loading  
2. Data cleaning and preprocessing  
3. Feature engineering  
4. Train-test split  
5. Handling class imbalance using SMOTE  
6. Model training  
7. Model evaluation  
8. Deployment using Streamlit

## Deployment

The model is deployed using Streamlit, allowing users to input customer details and get real-time credit risk predictions.

To run the application locally:

pip install -r requirements.txt
python -m streamlit run app.py

## Author
## Muskan Solanki
Data Science and Machine Learning Enthusiast
