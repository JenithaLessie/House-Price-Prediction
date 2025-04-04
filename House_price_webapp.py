import streamlit as st
import pandas as pd
import numpy as np
import joblib

page_element="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://tse1.mm.bing.net/th?id=OIP.k872ix6azMUrYtG0okon3wHaDt&pid=Api&P=0&h=180");
  background-size: cover;
}
</style>
"""
st.set_page_config(layout="wide")
st.markdown(page_element, unsafe_allow_html=True)
st.header("House Price Prediction")

model=joblib.load("house_price_model.joblib")
 
col1, col2, col3 = st.columns([1,2,1])

with col1:
      
    Address = st.selectbox(" Address :house:",
                       ("Adajan", "Bhagal ", "Dindoli", "Jahangirabad", "Mota Varachha", "Palanpur", "Piplod", "Vesu"))
    Bedrooms=st.selectbox("Bedrooms :bed:",
                          (1,2,3,4,5,6))
    square_feet=st.slider(label="Square Feet",
                                min_value=500, max_value=8000, value=2000)
    price_per_sqft=st.slider(label="Price(sqft) in ₹ ",
                                   min_value=500, max_value=8000, value=2500)
    

with col2:
               
    areaWithType=st.radio(label="Area type",
                          options=['Carpet Area', 'Super Area'])
    transaction=st.radio(label=" Property type :receipt:", options=['New Property', 'Resale'])
    status=st.radio(label="Availability of the property",
                    options=['Ready to Move', 'Still in construction'])
    furnishing=st.radio(label="Furnishing type :chair:",
                        options=['Unfurnished', 'Semi-furnished', 'furnished'])      
   
#submit inputs to model
if st.button("Submit", type="primary"):
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
    pred=model.predict(new_data)[0]        
    st.text(f"House price will be Rs {pred:,.2f} Lacs")
        