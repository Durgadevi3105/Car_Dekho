import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
import re

# Load the trained model
with open(car_model.pkl, 'rb') as file
    model = pickle.load(file)                             

# Load the car dataset
@st.cache_data
def load_car_data():
   return pd.read_csv("car_dheko_Final.csv")
def get_car_details_by_brand(brand_name, df):
    df = df.dropna(subset=['oem'])
    filtered_cars = df[df['oem'].str.lower() == brand_name.lower()]
    if filtered_cars.empty:
        return [{"message": f"No cars found for brand: {brand_name}"}]
    return filtered_cars.head(5)[['oem', 'model', 'price', 'Fuel Type', 'Transmission', 'Mileage']].to_dict('records')
# Clean and preprocess dataset
fl['Kms_Driven'] = fl['Kms_Driven'].fillna(0).astype(str).str.replace(',', '').astype(int)
fl['Max Power'] = fl['Max Power'].str.extract(r'(\d+\.?\d*)').astype(float)

# Function to predict car prices
def predict_car_price(brand, model_name, year, kms_driven, fuel_type, transmission, engine_displacement):
    try:
        if brand not in fl['oem'].unique():
            return {"error": f"Brand '{brand}' not found."}
        if model_name not in fl['model'].unique():
            return {"error": f"Model '{model_name}' not found."}
        
        body_type_encoded = fl['oem'].astype('category').cat.categories.get_loc(brand)
        model_name_encoded = fl['model'].astype('category').cat.categories.get_loc(model_name)
        input_data = np.array([[body_type_encoded, model_name_encoded, year, kms_driven, fuel_type, transmission, engine_displacement]])
        predicted_price = model.predict(input_data)[0]
        return {"predicted_price": f"₹{predicted_price:,.2f}"}
    except Exception as e:
        return {"error": str(e)}


def set_image_local(image_path):
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
            #background-position: center;
            #background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_image_local("resale.jpg")

# Custom CSS for Sidebar Styling with White, Black, and Blue
sidebar_style = """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #ffffff, #4f83cc, #000000); /* Gradient from White to Blue to Black */
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label {
        color: white;
        font-weight: bold;
    }
    [data-testid="stSidebar"] .css-1e4r1h6 {
        background-color: transparent;
    }
    </style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)


# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Function to switch pages
def go_to_page(page_name):
    st.session_state.page = page_name

# 🌟 Home Page
if st.session_state.page == "Home":
    st.title("🚗 Welcome to Car Price Prediction")
    st.write("""
        🚗 Car Price Prediction App
In the ever-evolving automobile market, determining the right price for a car is crucial for both buyers and sellers. The Car Price Prediction App simplifies this process by using machine learning to estimate car prices accurately based on various factors such as brand, model, year, mileage, fuel type, and transmission. With this tool, users can make data-driven decisions and ensure they get the best value.

🎯 The Challenge
Pricing a car manually is complex, as multiple factors influence its value. Traditional methods are often time-consuming and may not reflect the latest market trends. This app eliminates guesswork and provides instant predictions with high accuracy.

🚀 Key Features
🔹 User-Friendly Interface – A sleek and intuitive Streamlit-based UI.
🔹 Machine Learning Model – Powered by RandomForestRegressor, trained on a rich dataset.
🔹 Instant Predictions – Enter car details to get a real-time price estimate.
🔹 Comparison Tool – Compare multiple cars for better decision-making.
🔹 Transparent & Data-Driven – Helps users make informed buying and selling decisions.

👥 Who Can Benefit?
✅ Car Buyers & Sellers – Get a fair market value for used cars.
✅ Dealerships & Resellers – Easily estimate vehicle prices.
✅ Financial Institutions – Banks and insurance companies can assess car values.

🛠️ Technologies Used
🚀 Frontend: Built with Streamlit for an interactive experience.
📊 Backend: Uses Python, Scikit-learn, RandomForestRegressor, and Pandas.
📈 ML Tracking: Integrated with MLflow for model logging and performance monitoring.

🔥 Why This App?
By leveraging machine learning, the Car Price Prediction App provides fast, accurate, and transparent car price estimates, making the buying and selling process smoother and more efficient. 🚘💡
    """)

    

    st.write("🔍 Use the **Prediction Page** to get a price estimate for your car.")


    # Sidebar for chatbot
    if st.sidebar.header("Chatbot 🤖"):
        
        user_query = st.sidebar.text_input("Ask about car prices and details:", "")
        if user_query:
            price_pattern = r'predict price for (.+?) (.+?) (\d{4})'
            detail_pattern = r'show details for (.+?) (\d{4})'
                
            if re.search(price_pattern, user_query.lower()):
                match = re.search(price_pattern, user_query.lower())
                brand, model_name, year = match.groups()
                response = predict_car_price(brand.capitalize(), model_name.capitalize(), int(year), 10000, 'Petrol', 'Manual', 1200)
                st.sidebar.json(response)
            elif re.search(detail_pattern, user_query.lower()):
                match = re.search(detail_pattern, user_query.lower())
                brand, year = match.groups()
                filtered_data = fl[(fl['oem'].str.lower() == brand.lower()) & (fl['modelYear'] == int(year))]
                if filtered_data.empty:
                    st.sidebar.json({"error": f"No details found for {brand} ({year})."})
                else:
                    st.sidebar.json(filtered_data.head().to_dict(orient="records"))
            else:
                st.sidebar.json({"error": "Ask about price predictions or car details."})


    # Button to switch to Prediction Page
    if st.button("Go to Prediction Page ➡️"):
        go_to_page("Prediction")

# 🚀 Prediction Page
elif st.session_state.page == "Prediction":
     # Main app title
    st.title("Car Price Prediction App")
    st.write("Enter the car details to predict its price.")

    # Sidebar for filtering options
    st.sidebar.header('Filter Options')
    city = st.sidebar.selectbox("Select the city:", options=fl['city'].unique())
    filtered_data = fl[fl['city'] == city]

    body_type = st.sidebar.selectbox("Select body type:", options=filtered_data['body_type'].unique())
    filtered_data = filtered_data[filtered_data['body_type'] == body_type]

    year = st.sidebar.selectbox("Select the year:", options=sorted(filtered_data['modelYear'].unique()))
    filtered_data = filtered_data[filtered_data['modelYear'] == year]

    oem = st.sidebar.selectbox("Select OEM (Manufacturer):", options=filtered_data['oem'].unique())
    filtered_data = filtered_data[filtered_data['oem'] == oem]

    model_name = st.sidebar.selectbox("Select Car Model:", options=filtered_data['model'].unique())
    filtered_data = filtered_data[filtered_data['model'] == model_name]

    Mileage = st.sidebar.selectbox("Select Mileage (in km/l):", options=sorted(filtered_data['Mileage'].unique()))
    filtered_data = filtered_data[filtered_data['Mileage'] == Mileage]

    engine_type = st.sidebar.selectbox("Select the engine type:", options=sorted(filtered_data['Engine Type'].unique()))
    filtered_data = filtered_data[filtered_data['Engine Type'] == engine_type]

    engine_displacement = st.sidebar.selectbox("Select Engine Displacement (in CC):", options=sorted(filtered_data['Engine_Displacement'].unique()))
    filtered_data = filtered_data[filtered_data['Engine_Displacement'] == engine_displacement]

    seating_capacity = st.sidebar.selectbox("Select seating capacity:", options=sorted(filtered_data['Seating_Capacity'].unique()))
    filtered_data = filtered_data[filtered_data['Seating_Capacity'] == seating_capacity]

    ownership = st.sidebar.selectbox("Select ownership type:", options=filtered_data['Ownership'].unique())
    filtered_data = filtered_data[filtered_data['Ownership'] == ownership]

    kms_driven = st.sidebar.number_input("Enter kilometers driven:", min_value=0, value=10000)

    fuel_type = st.sidebar.selectbox("Select fuel type:", options=filtered_data['Fuel Type'].unique())
    filtered_data = filtered_data[filtered_data['Fuel Type'] == fuel_type]

    transmission = st.sidebar.selectbox("Select transmission type:", options=filtered_data['Transmission'].unique())
    filtered_data = filtered_data[filtered_data['Transmission'] == transmission]

    max_power = st.sidebar.selectbox("Select Max Power (in BHP):", options=sorted(filtered_data['Max Power'].unique()))
    filtered_data = filtered_data[filtered_data['Max Power'] == max_power]

    Acceleration = st.sidebar.selectbox("Select Acceleration type:", options=sorted(filtered_data['Acceleration'].unique()))
    filtered_data = filtered_data[filtered_data['Acceleration'] == Acceleration]

    # Predict button and price display
    if st.button("Predict Price"):
        try:
            # Encode categorical variables
            body_type_encoded = fl['body_type'].astype('category').cat.categories.get_loc(body_type)
            city_encoded = fl['city'].astype('category').cat.categories.get_loc(city)
            engine_type_encoded = fl['Engine Type'].astype('category').cat.categories.get_loc(engine_type)
            fuel_type_encoded = fl['Fuel Type'].astype('category').cat.categories.get_loc(fuel_type)
            ownership_encoded = fl['Ownership'].astype('category').cat.categories.get_loc(ownership)
            transmission_encoded = fl['Transmission'].astype('category').cat.categories.get_loc(transmission)
            oem_encoded = fl['oem'].astype('category').cat.categories.get_loc(oem)
            model_name_encoded = fl['model'].astype('category').cat.categories.get_loc(model_name)

            # Prepare input data in the correct order
            input_data = np.array([[city_encoded, body_type_encoded, oem_encoded, model_name_encoded, year, 
                                    fuel_type_encoded, ownership_encoded, transmission_encoded, engine_type_encoded, 
                                    seating_capacity, Mileage, max_power, engine_displacement, kms_driven, Acceleration]])

            # Predict price
            predicted_price = model.predict(input_data)[0]

            # Display the predicted price, styled and in the same order
            st.markdown(f"<h3 style='text-align: center; color: darkblue;'>The predicted price of the car is: ₹{predicted_price:,.2f}</h3>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

st.write("Note: This is a demo app. The predictions depend on the model's training data.")

# Button to go back to Home Page
if st.button("⬅️ Back to Home"):
    go_to_page("Home")
