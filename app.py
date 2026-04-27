import streamlit as st
import pandas as pd
import joblib

model = joblib.load('loan_model.pkl')
model_columns = joblib.load('model_column.pkl')
st.title('Loan Approval Predictor')

income = st.number_input('Annual Income', min_value=0)
loan_amount = st.number_input('Requested Loan Amount', min_value=0)

if st.button('Check Eligibility'):
    data = {
        'person_age': [25],
        'person_income': [income],
        'person_emp_exp':[5],
        'loan_amnt':[loan_amount],
        'loan_int_rate':[10.5],
        'loan_percent_income':[0.1],
        'cb_person_cred_hist_length': [3],
        'credit_score': [700]
    }
    df_input = pd.DataFrame(data)
    df_input = df_input.reindex(columns=model_columns,fill_value=0)

    prediction = model.predict(df_input)
    if prediction[0] == 1: 
         st.success('Congratulations! Loan is likely to be APPROVED.')
    else:
         st.error('Sorry, the loan is likely to be Rejected.')
        
        

