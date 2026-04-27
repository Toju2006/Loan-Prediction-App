import streamlit as st
import pandas as pd
import joblib

model = joblib.load('loan_model.pkl')
st.title('Loan Approval Predictor')

income = st.number_input('Annual Income', min_value=0, value=100)
loan_amount = st.number_input('Requested Loan Amount', min_value=0, value=1000)
age = st.number_input('Age', min_value=18, max_value=100,value=30)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850,value=650)
import numpy as np
if st.button('Check Eligibility'):
    input_data = np.zeros(len(model.feature_names_in_))
    input_df = pd.DataFrame([input_data], columns=model.feature_names_in_)
    
    input_df['person_income'] = income
    input_df['loan_amnt'] = loan_amount
    input_df['person_age'] = age
    input_df['credit_score'] = credit_score
    

    prediction = model.predict(input_df)
    if prediction[0] == 0: 
         st.success('Congratulations! Loan is likely to be APPROVED.')
    else:
         st.error('Sorry, the loan is likely to be Rejected.')
        
        

