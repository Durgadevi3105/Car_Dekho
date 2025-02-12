import streamlit as st
import pandas as pd
import base64
import numpy as np
import pickle
import os
import boto3

# Securely load AWS credentials from environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")

s3 = boto3.client("s3",
                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                  region_name=AWS_REGION)

# AWS S3 Configuration
BUCKET_NAME = "my-car-model-bucket"
MODEL_FILE = "model.pkl"
LOCAL_MODEL_PATH = "/mnt/data/model.pkl"

# Function to download model from S3
def download_model_from_s3():
    s3.download_file(BUCKET_NAME, MODEL_FILE, LOCAL_MODEL_PATH)
    print("Model downloaded successfully from S3.")

# Download model if not exists
if not os.path.exists(LOCAL_MODEL_PATH):
    st.info("Downloading model from S3...")
    download_model_from_s3()

# Load model
with open(LOCAL_MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

st.title("Car Resale Prediction & Chatbot 🚗")

option = st.sidebar.selectbox("Choose a Feature", ["Predict Car Resale Value", "Chatbot"])

if option == "Predict Car Resale Value":
    st.header("Enter Car Specifications")

    model_year = st.number_input("Year of Manufacture", min_value=2000, max_value=2025, step=1)
    kms_driven = st.number_input("Kilometers Driven", min_value=0, step=500, value=10000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])

    if st.button("Predict Resale Value"):
        input_data = np.array([[model_year, kms_driven]]).reshape(1, -1)
        prediction = model.predict(input_data)
        st.success(f"The predicted resale value of the car is: ₹{prediction[0]:,.2f}")

elif option == "Chatbot":
    st.header("Car Chatbot Assistant 💬")
    st.write("Ask about car brands and models!")
