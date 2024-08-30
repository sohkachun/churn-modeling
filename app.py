import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

#Load model
model = tf.keras.models.load_model('Notebook\model.h5')

#Load the encoder and scaler


import os
def load_pickle_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, "rb") as file:
            return pickle.load(file)
    else:
        raise FileNotFoundError(f"The file at {filepath} does not exist.")

label_encoder_gender = load_pickle_file('artifact\label_encoder_gender.pkl')

geo_encoder = load_pickle_file('artifact\geo_encoder.pkl')


scaler = load_pickle_file("artifact/scaler.pkl")

## streamlit app
st.title("Customer Churn Prediction")

#User Input
with st.form(key='input_form'):
    credit_score = st.number_input('Credit Score')
    geography = st.selectbox('Geography', geo_encoder.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.number_input('Age', 18,29)
    tenure = st.number_input('Tenure', 0,10)
    balance = st.number_input('Balance')
    num_of_products = st.number_input('Number of Products')
    # Mapping Yes/No to 1/0
    has_cr_card = st.selectbox('Has Credit Card', options=['Yes', 'No'])
    has_cr_card = 1 if has_cr_card == 'Yes' else 0
    
    is_active_member = st.selectbox('Is Active Member', options=['Yes', 'No'])
    is_active_member = 1 if is_active_member == 'Yes' else 0
    estimated_salary = st.number_input('Estimated Salary')

    # Submit button
    submit_button = st.form_submit_button(label='Submit')


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure':[tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

#Encode geography data
geo_encoded = geo_encoder.transform([[geography]])
geography_data = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out())
input_data =pd.concat([input_data.drop('Geography', axis=1), geography_data], axis=1)


#Scale data
data_scaled = scaler.transform(input_data)

prediction = model.predict(data_scaled)
prediction_prob = prediction[0][0]

if prediction_prob > 0.5:
    st.write('The customer is likely to churn')
else:
   
    st.write('The customer is not likely to churn')
