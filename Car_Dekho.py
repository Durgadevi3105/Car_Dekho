import streamlit as st
import pandas as pd
import base64
import numpy as np
import pickle
import os
import boto3



image_path = r"C:\Users\HP\Downloads\carimg.jpg"

if os.path.exists(image_path):
    with open(image_path, "rb") as file:
        img = file.read()
    base64_image = base64.b64encode(img).decode("utf-8")

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.error("Image file not found! Please check the file path.")



# AWS S3 Configuration
BUCKET_NAME = "my-car-model-bucket"
MODEL_FILE = "model.pkl"
LOCAL_MODEL_PATH = "/mnt/data/model.pkl"
CSV_FILE = "car_dekho_final.csv"

AWS_ACCESS_KEY = "AKIA4SDNVSW6KCENG27V"
AWS_SECRET_KEY = "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"
AWS_REGION = "ap-south-1"

# Initialize Boto3 Client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def download_model_from_s3():
    try:
        s3.download_file(BUCKET_NAME, MODEL_FILE, LOCAL_MODEL_PATH)
        print("✅ Model downloaded successfully from S3.")
    except Exception as e:
        print(f"❌ Error downloading model: {e}")

# Download model if not exists
if not os.path.exists(LOCAL_MODEL_PATH): 
    st.info("Downloading model from S3...")
    download_model_from_s3()
if os.path.exists(LOCAL_MODEL_PATH):
    with open(LOCAL_MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    print("✅ Model loaded successfully!")
else:
    print("❌ Model file not found. Please check your S3 bucket.")

# Load the car dataset
@st.cache_data
def load_car_data():
   file_path = "car_dekho_final.csv"

   if not os.path.exists(file_path):
        st.error(f"❌ '{file_path}' not found! Please upload it.")
        return pd.DataFrame()  # Return empty DataFrame to prevent crashes

   return pd.read_csv(file_path)  
def get_car_details_by_brand(brand_name, df):
    df = df.dropna(subset=['oem'])
    filtered_cars = df[df['oem'].str.lower() == brand_name.lower()]
    if filtered_cars.empty:
        return [{"message": f"No cars found for brand: {brand_name}"}]
    return filtered_cars.head(5)[['oem', 'model', 'price', 'Fuel Type', 'Transmission', 'Mileage']].to_dict('records')

# Display Streamlit page
st.title("Car Resale Prediction & Chatbot 🚗")

# Sidebar Navigation
option = st.sidebar.selectbox("Choose a Feature", ["Predict Car Resale Value", "Chatbot"])
if option == "Predict Car Resale Value":
    
    st.markdown(
    """
    ## Welcome to the Car Resale Value Predictor! 🚗💰  
    Curious about how much your car is worth in the resale market?  
    Enter your car details, and our AI-powered model will predict its estimated resale value.  
    This tool helps car buyers and sellers make informed decisions effortlessly.  
    Get started now! 🎯  
    """, 
    unsafe_allow_html=True  
    )

# Input features for prediction
    st.header("Enter Car Specifications")
    col1, col2, col3, col4 = st.columns(4)
    # Categorical feature mappings
    city_map = {"Banglore": 0, "Chennai": 1, "Delhi": 2, "Hyderabad": 3, "Jaipur": 4, "kolkata": 5}
    fuel_type_map = {"Petrol": 0, "Diesel": 1, "Electric": 2, "Hybrid": 3}
    ownership_map = {"First Owner": 0, "Second Owner": 1, "Third Owner": 2, "Fourth or More": 3}
    transmission_map = {"Manual": 0, "Automatic": 1}

    # Inputs
    with col1:
        city = st.selectbox("City", options=city_map.keys())
        oem = st.text_input("OEM (Manufacturer)", placeholder="Enter manufacturer name")
        model_name = st.text_input("Model", placeholder="Enter car model")
        model_year = st.number_input("Year of Manufacture", min_value=2000, max_value=2025, step=1)
        kms_driven = st.number_input("Kilometers Driven", min_value=0, step=500, value=10000)
    with col2:
        fuel_type = st.selectbox("Fuel Type", options=fuel_type_map.keys())
        ownership = st.selectbox("Ownership Type", options=ownership_map.keys())
        transmission = st.selectbox("Transmission Type", options=transmission_map.keys())
        max_power = st.number_input("Max Power (in BHP)", min_value=50, step=1)
    with col3:
        engine_type = st.text_input("Engine Type", placeholder="Enter engine type")
        mileage = st.number_input("Mileage (in km/l)", min_value=0.0, step=0.1)
        seating_capacity = st.number_input("Seating Capacity", min_value=2, max_value=10, step=1)
        engine_displacement = st.number_input("Engine Displacement (in cc)", min_value=500, step=50)
    with col4:
        body_type = st.text_input("Body Type", placeholder="Enter body type (e.g., Sedan, SUV)")
        acceleration = st.number_input("Acceleration (0-100 km/h in seconds)", min_value=1.0, step=0.1)

# Preprocess the input
    def preprocess_input(features):
        return np.array(features).reshape(1, -1)

# Collect all inputs
    input_features = [
            city_map[city],            # Encoded city
            len(oem),                  # Example: Encode OEM as the length of the input string
            len(model_name),           # Example: Encode model as the length of the input string
            model_year, 
            kms_driven, 
            fuel_type_map[fuel_type],  # Encoded fuel type
            ownership_map[ownership],  # Encoded ownership
            transmission_map[transmission],  # Encoded transmission
            max_power,
            len(engine_type),          # Example: Encode engine type as the length of the input string
            mileage,
            seating_capacity,
            engine_displacement,
            len(body_type),            # Example: Encode body type as the length of the input string
            acceleration
        ]

    # Predict button
    if st.button("Predict Resale Value"):
            input_data = preprocess_input(input_features)
            prediction = model.predict(input_data)
            st.success(f"The predicted resale value of the car is: ₹{prediction[0]:,.2f}")
    
elif option == "Chatbot":
    st.header("Car Chatbot Assistant 💬")
    df = load_car_data()
            
    user_query = st.text_input("Ask me about cars!", "")

    if user_query:
                if "tell me about" in user_query.lower():
                    brand_name = user_query.lower().replace("tell me about", "").strip()
                    details = get_car_details_by_brand(brand_name, df)
                    st.write("### Car Details")
                    st.json(details)
                else:
                    st.write("I'm still learning to answer more queries!")

    #except Exception as e:
     #   st.error(f"Error: {e}")
