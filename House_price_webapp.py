import streamlit as st
import pandas as pd
import numpy as np
import joblib

page_element="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://static.vecteezy.com/system/resources/thumbnails/005/327/845/small/hand-holding-house-model-in-blue-background-for-refinance-plan-and-real-estate-concept-free-photo.jpg");
  background-size: cover;
}
</style>
"""
st.set_page_config(layout="wide")
st.markdown(page_element, unsafe_allow_html=True)

st.markdown(
"""
<style>
.stAppHeader {
background-color: rgba(255, 255, 255, 0.0); /* Transparent background */
visibility: visible; /* Ensure the header is visible */
}

.block-container {
padding-top: 1rem;
padding-bottom: 0rem;
padding-left: 2rem;
padding-right: 2rem;

}
</style>
""",
unsafe_allow_html=True,
)

st.header("House Price Prediction")
model=joblib.load("house_price_model.joblib")

col1, col2, col3 = st.columns([1,2,1])

with col1:
          
    Address = st.selectbox(" **1.Address** :house:",
                           ("Adajan", "Bhagal ", "Dindoli", "Jahangirabad", "Mota Varachha", "Palanpur", "Piplod", "Vesu"))
    Bedrooms=st.selectbox("**2.Bedrooms** :bed:",
                          (1,2,3,4,5,6))
    square_feet=st.slider(label="**3.Square Feet**",
                                min_value=500, max_value=8000, value=2000)
    price_per_sqft=st.slider(label="**4.Price(sqft) in ₹** ",
                                   min_value=500, max_value=8000, value=2500)
    

with col2:
    
    areaWithType=st.radio(label="**5.Area type**",
                          options=['Carpet Area', 'Super Area'], horizontal=True)
    transaction=st.radio(label="**6.Property type** :receipt:", options=['New Property', 'Resale'], horizontal=True)
    status=st.radio(label="**7.Availability of the property**",
                    options=['Ready to Move', 'Still in construction'], horizontal=True)
    furnishing=st.radio(label="**8.Furnishing type** :chair:",
                        options=['Unfurnished', 'Semi-furnished', 'furnished'], horizontal=True)      
   
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
    st.subheader(f" **House price will be Rs {pred:,.2f} Lacs**")
        
