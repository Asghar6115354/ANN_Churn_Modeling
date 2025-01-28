import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle 

# Load the  model
model = tf.keras.models.load_model('model.h5')

# Load the encoder and scaler

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


## Streamlit app


    
    
    

st.title("Bank Customer Churn Prediction")

# Input fields
credit_score = st.number_input("Credit Score", 300, 900, 619)
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.number_input("Age", 18, 100, 42)
tenure = st.number_input("Tenure", 0, 10, 2)
balance = st.number_input("Balance", 0.0, 250000.0, 0.0)
num_products = st.number_input("Number of Products", 1, 4, 1)
has_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active = st.selectbox("Is Active Member", ["Yes", "No"])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 101348.88)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

if st.button("Predict"):
    # Convert categorical inputs
    gender_encoded = 1 if gender == "Male" else 0
    has_card_encoded = 1 if has_card == "Yes" else 0
    is_active_encoded = 1 if is_active == "Yes" else 0
    
    # One-hot encode geography
    geo_france = 1 if geography == "France" else 0
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    # Create input array
    input_data = np.array([[
        credit_score, gender_encoded, age, tenure, balance, num_products,
        has_card_encoded, is_active_encoded, salary,
        geo_france, geo_germany, geo_spain
    ]])

    # Scale and predict
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    
    # Show results
    churn_prob = prediction[0][0] * 100
    st.write(f"Churn Probability: {churn_prob:.2f}%")
    
    if churn_prob > 50:
        st.error("High risk of customer leaving!")
    else:
        st.success("Customer likely to stay!")