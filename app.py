import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st

# Load the saved model and scaler
model = joblib.load('breast_cancer_model.joblib')
scaler = joblib.load('scaler.joblib')

# Streamlit page configuration
st.set_page_config(page_title="Breast Cancer Detection")
st.title("Breast Cancer Detection Using Classification")
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://4kwallpapers.com/images/wallpapers/macos-monterey-stock-green-dark-mode-layers-5k-6016x3384-5890.jpg");
        background-attachment: fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Accept user inputs for the 6 feature columns
radius_mean = st.number_input('Radius Mean')
texture_mean = st.number_input('Texture Mean')
perimeter_mean = st.number_input('Perimeter Mean')
area_mean = st.number_input('Area Mean')
smoothness_mean = st.number_input('Smoothness Mean')
compactness_mean = st.number_input('Compactness Mean')

# Prepare the input for prediction (make sure the input follows the order of the features)
user_data = [[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean]]

# Scale the input using the same scaler used during training
user_data_scaled = scaler.transform(user_data)

# Make predictions using the pre-trained model
prediction = model.predict(user_data_scaled)

# Display the prediction result to the user
if prediction[0] == 0:
    st.write('Prediction: Benign')
else:
    st.write('Prediction: Malignant')
