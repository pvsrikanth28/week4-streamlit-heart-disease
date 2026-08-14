import streamlit as st
import pandas as pd
import joblib


# -------------------------
# Page configuration
# -------------------------

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# -------------------------
# Load trained model
# -------------------------

model_data = joblib.load("heart_disease_model.joblib")

model = model_data["model"]
model_name = model_data["model_name"]
metrics = model_data["metrics"]


# -------------------------
# Title
# -------------------------

st.title("❤️ Heart Disease Prediction")

st.write(
    "Enter the patient's information below to predict "
    "whether heart disease is likely to be present."
)

st.info(f"Model used: {model_name}")


# -------------------------
# Input fields
# -------------------------

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=50
)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.selectbox(
    "Chest Pain Type",
    options=[1, 2, 3, 4],
    help="1 = Typical angina, 2 = Atypical angina, "
         "3 = Non-anginal pain, 4 = Asymptomatic"
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol",
    min_value=50,
    max_value=700,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

restecg = st.selectbox(
    "Resting ECG",
    options=[0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

oldpeak = st.number_input(
    "ST Depression (Oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope",
    options=[1, 2, 3]
)

ca = st.selectbox(
    "Number of Major Vessels",
    options=[0, 1, 2, 3]
)

thal = st.selectbox(
    "Thal",
    options=[3, 6, 7]
)


# -------------------------
# Prediction
# -------------------------

if st.button("Predict Heart Disease"):

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Prediction: Heart Disease")
    else:
        st.success("✅ Prediction: No Heart Disease")

    st.write(
        f"Estimated probability of heart disease: "
        f"{probability:.2%}"
    )


# -------------------------
# Model information
# -------------------------

st.divider()

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    st.metric("Recall", f"{metrics['recall']:.2%}")

with col2:
    st.metric("Precision", f"{metrics['precision']:.2%}")
    st.metric("F1 Score", f"{metrics['f1']:.2%}")

st.caption(
    "This application is an educational machine learning demonstration "
    "and should not be used for medical diagnosis."
)