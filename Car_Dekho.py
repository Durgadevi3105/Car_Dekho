import streamlit as st
import pandas as pd
import base64
import numpy as np
import pickle
import os
import boto3
session = boto3.Session(
    aws_access_key_id="AKIA4SDNVSW6KCENG27V",
    aws_secret_access_key="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",
    region_name="ap-south-1"
)

s3 = session.client("s3")
# AWS S3 Configuration
BUCKET_NAME = "my-car-model-bucket"  # Change this to your S3 bucket name
MODEL_FILE = "model.pkl"
LOCAL_MODEL_PATH = "/mnt/data/model.pkl"

# Function to download the model from S3
def download_model_from_s3():
    s3 = boto3.client("s3")
    s3.download_file(BUCKET_NAME, MODEL_FILE, LOCAL_MODEL_PATH)
    print("Model downloaded successfully from S3.")

# Download model if it doesn't exist locally
if not os.path.exists(LOCAL_MODEL_PATH):
    st.info("Downloading model from S3...")
    download_model_from_s3()

# Load the model
with open(LOCAL_MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Car Resale Prediction & Chatbot 🚗")

option = st.sidebar.selectbox("Choose a Feature", ["Predict Car Resale Value", "Chatbot"])

if option == "Predict Car Resale Value":
    st.header("Enter Car Specifications")
    
    # Input fields
    model_year = st.number_input("Year of Manufacture", min_value=2000, max_value=2025, step=1)
    kms_driven = st.number_input("Kilometers Driven", min_value=0, step=500, value=10000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "Hybrid"])
    
    # Predict button
    if st.button("Predict Resale Value"):
        input_data = np.array([[model_year, kms_driven]]).reshape(1, -1)
        prediction = model.predict(input_data)
        st.success(f"The predicted resale value of the car is: ₹{prediction[0]:,.2f}")

elif option == "Chatbot":
    st.header("Car Chatbot Assistant 💬")
    st.write("Ask about car brands and models!")
