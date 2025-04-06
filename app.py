import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle
import tensorflow as tf

# Loading the model
model=tf.keras.models.load_model('model.h5')
# Loading the preprocessors
with open('scaler.pickle','rb') as file:
    scaler=pickle.load(file)
with open('label_encoder_gender.pkl','rb') as file:
    label_endcoder_gender=pickle.load(file)
with open('oh_encoder_country.pkl','rb') as file:
    oh_encoder_country=pickle.load(file)

# Streamlit App
st.title("Customer Chrun Prediction")

# User Input
with st.form("Churn Prediction"):
    country=st.selectbox('Country',oh_encoder_country.categories_[0])
    gender=st.selectbox('Gender',label_endcoder_gender.classes_)
    age=st.slider('Age',18,92)
    balance=st.number_input('Balance')
    credit_score=st.number_input("Credit Score")
    estimated_salary=st.number_input('Estimated Salary')
    tenure=st.slider('Tenure',0,10)
    num_of_products=st.selectbox('Number of Products',[1,2,3,4])
    has_credit_card=st.selectbox("Has Credit Card",[0,1])
    is_active_member=st.selectbox('Is Active Memeber',[0,1])
    submit_button=st.form_submit_button("Predict")

# Preparing Input Data for tensorflow
input_data=pd.DataFrame({
    'credit_score': [credit_score], 
    'gender':[label_endcoder_gender.transform([gender])[[0]]], 
    'age':[age], 
    'tenure':[tenure], 
    'balance':[balance],
    'products_number':[num_of_products], 
    'credit_card':[has_credit_card],
    'active_member':[is_active_member], 
    'estimated_salary':[estimated_salary]
})

# One Hot Encoding of country
country_encoded=oh_encoder_country.transform([[country]]).toarray()
country_encoded_df=pd.DataFrame(country_encoded,columns=oh_encoder_country.categories_[0])

input_data=input_data.join(country_encoded_df)

# scaling the input data
input_data_scaled=scaler.transform(input_data)

# Prediction churn 
prediction=model.predict(input_data_scaled)
prediction_proba=prediction[0][0]

# Output to user.
if submit_button:
    if prediction_proba>0.5:
        st.write("The Customer is likely to churn.")
    else:
        st.write("The Customer is not likely to chrun.")
