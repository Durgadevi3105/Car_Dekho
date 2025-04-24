import streamlit as st 
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


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

# Load the pickled model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

car_model=load_model('car_model1.pkl')

encoder_car_name=load_model('encoder_car_name.pkl')
encoder_fuel_type=load_model('encoder_fuel_type.pkl')
encoder_location=load_model('encoder_location.pkl')
encoder_transmission=load_model('encoder_transmission.pkl')

dataset=pd.read_csv('cleaned_car_data.csv')
st.title('Used Car Price Prediction')
st.write('This is a web app to predict the price of a used car based on its features. Please adjust the values of the features below and click on the "Predict" button to see the predicted price of the used car.')

categories = ['Car Name','Fuel Type','Location','Transmission',]
dropdown_options = {feature: dataset[feature].unique().tolist() for feature in categories}


tab1, tab2 = st.tabs(["Predict", "🚗Chatbot"])
with tab1:
    with st.form("Used Car Price Prediction Form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            year = st.number_input("enter year",min_value=1985,max_value=2025)

        with col2:
            carname_select = st.selectbox('Car Name', dropdown_options['Car Name'])
            carname=encoder_car_name.transform([[carname_select]])[0][0]    
        with col3:
            km=st.number_input("Enter KM driven",min_value=10)

        with col1:
            fueltype_select = st.selectbox('Fuel Type', dropdown_options['Fuel Type'])
            fueltype=encoder_fuel_type.transform([[fueltype_select]])[0][0]

        col5, col6= st.columns(2)
        with col2:
            transmission_select = st.selectbox('Transmission', dropdown_options['Transmission'])
            transmission=encoder_transmission.transform([[transmission_select]])[0][0]
        with col3:
            location_select = st.selectbox('Location', dropdown_options['Location'])
            location=encoder_location.transform([[location_select]])[0][0]
    
        if st.form_submit_button(label='Predict'):
        # Debugging: Print user input before DataFrame creation
            print(f"Year: {year}, Type: {type(year)}")
            print(f"Car Name: {carname}, Type: {type(carname)}")
            print(f"KM Driven: {km}, Type: {type(km)}")
            print(f"Fuel Type: {fueltype}, Type: {type(fueltype)}")
            print(f"Transmission: {transmission}, Type: {type(transmission)}")
            print(f"Location: {location}, Type: {type(location)}")

        # Create DataFrame with actual numerical values
        input_data = pd.DataFrame([[year, carname, km, fueltype, transmission, location]], 
                                  columns=["Year", "Car Name", "Mileage (kms)", "Fuel Type", "Transmission", "Location"])

        # Debugging: Print DataFrame before conversion
        print(input_data)
        print(input_data.dtypes)

        # Convert DataFrame to float if necessary
        input_data = input_data.astype(float)
 

    try:
        prediction = car_model.predict(input_data)
    
        st.subheader("Predicted Car Price")
        st.success(f"### :green[₹ {prediction[0]:,.2f}]")
    except Exception as e:
        st.error(f"Error in Prediction: {str(e)}")




with tab2:
    st.write("Welcome to the Car Dealership Chatbot! ")
    



    # Define the chatbot's behavior
    car_dealership_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant at a car dealership. "
                         "You can provide information about car models, pricing, and financing options. "
                         "Be friendly and informative."),
            ("placeholder", "{messages}"),
        ]
    )

    # Instantiate the chat model
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key="AIzaSyBpufebS9kziyw2coPefRL0wtXDs5wKbjM")

    # Streamlit UI
    st.title("🚗 Car Dealership Chatbot")
    st.write("Ask me anything about car models, pricing, and financing options!")

    # Chat history
    if "messages" not in st.session_state:
       st.session_state["messages"] = []

    # Display previous chat messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask me a question...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
    
        with st.chat_message("user"):
            st.markdown(user_input)
    
        # Get response from chatbot
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state["messages"]]
        response = llm.invoke(messages)
    
        with st.chat_message("assistant"):
            st.markdown(response.content)
    
       # Store assistant response
        st.session_state["messages"].append({"role": "assistant", "content": response.content})

    
    

