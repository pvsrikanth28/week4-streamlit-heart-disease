# week4-streamlit-heart-disease
Week 4 internship project converting a heart disease classification model into a local Streamlit application.
# Week 4 - Local Heart Disease Prediction App

## Overview

This project converts a machine learning classification model into a local Streamlit web application.

The application uses the UCI Heart Disease dataset and allows a user to enter patient features and receive a machine learning prediction.

## Features

- User-friendly Streamlit interface
- Heart disease prediction
- Saved machine learning model
- Probability estimate
- Model performance metrics
- Local execution

## Machine Learning

Two models from the previous classification project were considered:

- Logistic Regression
- Random Forest

The model with the better F1-score was automatically selected and saved.

## Features Used

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Induced Angina
- ST Depression
- Slope
- Number of Major Vessels
- Thal

## Technologies

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
