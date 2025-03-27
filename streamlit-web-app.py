#import libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

#load our model pipeline
model=joblib.load("house_price_model.joblib")

# addd title and istructions
st.title("House Price Prediction Model(Surat-Gujarat-India)")

st.subheader("This app uses Machine Learning to predict the house price with given features. For using this app you can enter the inputs from the user interface and then use predict button")

Address=st.text_input(
    label="01. Enter the address")    

areaWithType=st.radio(
    label="02.Choose the area type",
    options=['Carpet Area', 'Super Area'])

square_feet=st.slider(
    label="3.select the sqft",
    min_value=500,
    max_value=6000,
    value=1500)
    
price_per_sqft=st.slider(
    label="4.select the price per sqft(Rs)",
    min_value=500,
    max_value=10000,
    value=2500)

status=st.radio(
    label="5.Choose the availability of the property",
    options=['Ready to Move', 'Still in construction'])

transaction=st.radio(
    label="6.Choose the property type",
    options=['New Property', 'Resale'])

furnishing=st.radio(
    label="7.Furnishing type",
    options=['Unfurnished', 'Semi-furnished', 'furnished'])

Bedrooms=st.number_input(
    label="8.Enter the number of bedrooms",
    min_value=0,
    max_value=6,
    value=2)

#submit inputs to model
if st.button("submit for prediction"):
    
    #store our data in dataframe for prediction
    new_data=pd.DataFrame({"Address":[Address],
                      "areaWithType" : [areaWithType],
                      "square_feet":[square_feet],
                      "transaction":[transaction],
                      "status":[status],
                      "furnishing":[furnishing],
                      "price_per_sqft":[price_per_sqft],
                      "BHK":[Bedrooms]})
        
    #apply our pipeline to this input data
    pred_price=model.predict(new_data)[0]
    
    #output prediction
    st.subheader(f"Based on the given input, predicted house price is {round((pred_price),2)} Lacs")   


