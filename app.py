#Website file
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import base64
import warnings

st.set_page_config(
            page_title="Gimme Some Credit",
            page_icon="💰",
            layout="centered")


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)


FONT_SIZE_CSS = f"""
<style>
h1 {{
    color: white;
}}
h2 {{
    color: white;
}}
p {{
    color: white;
}}
body {{
    background-color: black;
}}
label {{
    font-size: 25px !important;
    color: white !important;
}}
</style>
"""
st.write(FONT_SIZE_CSS, unsafe_allow_html=True)


'# Gimme some Credit!'

"## Credit for you?"

image = Image.open('images/money.jpg')
st.image(image, use_column_width=True)

"## Please fill out the form below"
'# '

###Variables that will fill in the DataFrame that will go into the model
name = st.text_input('Full name')

phone = st.text_input('Phone number')

'# '

loan_amount = st.slider('Loan amount', min_value=0, max_value=35000, step=100)

term = st.selectbox('Loan term',
				   ('','Short Term', 'Long Term'))

credit_profile = st.selectbox('Your credit score',
				   ('','Good (730 and above)', 'Average (703 - 729)', 'Below average (702 and below)'))

if credit_profile == 'Good (730 and above)':
	credit_profile = 731
elif credit_profile == 'Average (703 - 729)':
	credit_profile = 710
elif credit_profile == 'Below average (702 and below)':
	credit_profile = 650


years_job = st.selectbox('Years in current job',
				   ('','less than 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'))


# SOLANO: ELIMINAR O BLOCO DE CODIGO ABAIXO QUANDO ESTIVER TRABALHANDO COM O MODELO ATUALIZADO
home_ownership = 'Own Home'
# home_ownership = st.selectbox('Home ownership status',
# 				   ('','Home Mortgage', 'Rent', 'Own Home'))

annual_income = st.slider('Annual income', min_value=0, max_value=200_000, step=1000)

purpose = st.selectbox('Purpose of the loan',
				   ('','Debt Consolidation', 'Home Improvements', 'Business Loan', 'Buy a Car', 'Medical Bills', 'Buy House', 'Other'))

month_debt = st.slider('Monthly debt payment', min_value=0, max_value=10000, step=100)

cred_history = st.selectbox('Years of credit history',
				           ('','0 - 10', '10.1 - 20', '20.1 - 30', '30+'))

if cred_history == '0 - 10':
	cred_history = 5
elif cred_history == '10.1 - 20':
	cred_history = 15
elif cred_history == '20.1 - 30':
	cred_history = 25
elif cred_history == '30+':
	cred_history = 35

delinquent = st.selectbox('Delinquent in the past 3 years?',
				   ('','Yes', 'No'))
if delinquent == 'Yes':
	delinquent = 21
elif delinquent == 'No':
	delinquent = 42


open_accounts = 8
# open_accounts = st.selectbox('Number of open accounts',
# 				   ('','Less than 10', '10 - 20', 'More than 20'))

# if open_accounts == 'Less than 10':
# 	open_accounts = 8
# elif open_accounts == '10 - 20':
# 	open_accounts = 15
# elif open_accounts == 'More than 20':
#     open_accounts = 25

credit_problems = st.selectbox('Any credit problems?',
				   ('','No', 'Yes, 1', 'Yes, more than 1'))

if credit_problems == 'No':
	credit_problems = 0
elif credit_problems == 'Yes, 1':
	credit_problems = 1
elif credit_problems == 'Yes, more than 1':
	credit_problems = 2

credit_balance = st.slider('Current credit balance', min_value=0, max_value=100_000, step=1000)

# SOLANO: ELIMINAR O BLOCO DE CODIGO ABAIXO QUANDO ESTIVER TRABALHANDO COM O MODELO ATUALIZADO
open_credit = 1
# open_credit = st.slider('Maximum open credit', min_value=0, max_value=2000000, step=1000)

# SOLANO: ELIMINAR O BLOCO DE CODIGO ABAIXO QUANDO ESTIVER TRABALHANDO COM O MODELO ATUALIZADO
bankruptcy = 0
# bankruptcy = st.selectbox('Have you ever declared bankruptcy?',
# 				   ('','No', 'Yes'))
# if bankruptcy == 'No':
# 	bankruptcy = 0
# elif bankruptcy == 'Yes':
# 	bankruptcy = 1

# SOLANO: ELIMINAR O BLOCO DE CODIGO ABAIXO QUANDO ESTIVER TRABALHANDO COM O MODELO ATUALIZADO
tax_liens = 0
# tax_liens = st.selectbox('Have you ever been imposed a tax lien?',
# 				   ('','No', 'Yes'))
# if tax_liens == 'No':
# 	tax_liens = 0
# elif tax_liens == 'Yes':
# 	tax_liens = 1

##############################################################

input_teste = pd.DataFrame({'Loan.ID': [0],
						 'Current.Loan.Amount': [17879],
						 'Term': ['Short Term'],
						 'Credit.Score': [739.0],
						 'Years.in.current.job': ['6 years'],
						 'Home.Ownership': ['Home Mortgage'], # SOLANO: ELIMINAR ESTE ITEM QUANDO ESTIVER TRABALHANDO COM O MODELO ATUALIZADO
						 'Annual.Income': [95357.0],
						 'Purpose': ['Debt Consolidation'],
						 'Monthly.Debt': [1509.82],
						 'Years.of.Credit.History': [34.4],
						 'Months.since.last.delinquent': [5.0],
						 'Number.of.Open.Accounts': [26],
						 'Number.of.Credit.Problems': [0],
						 'Current.Credit.Balance': [23986],
						 'Maximum.Open.Credit': [40313], # SOLANO: ELIMINAR ESTE ITEM QUANDO ESTIVER TRABALHANDO COM O MODELO ATUALIZADO
						 'Bankruptcies': [0.0], # SOLANO: ELIMINAR ESTE ITEM QUANDO ESTIVER TRABALHANDO COM O MODELO ATUALIZADO
						 'Tax.Liens': [0.0]}) # SOLANO: ELIMINAR ESTE ITEM QUANDO ESTIVER TRABALHANDO COM O MODELO ATUALIZADO


input_customer = pd.DataFrame({'Loan.ID': [0],
						 'Current.Loan.Amount': [loan_amount],
						 'Term': [term],
						 'Credit.Score': [credit_profile],
						 'Years.in.current.job': [years_job],
						 'Home.Ownership': [home_ownership], # SOLANO: ELIMINAR ESTE ITEM QUANDO ESTIVER TRABALHANDO COM O MODELO ATUALIZADO
						 'Annual.Income': [annual_income],
						 'Purpose': [purpose],
						 'Monthly.Debt': [month_debt],
						 'Years.of.Credit.History': [cred_history],
						 'Months.since.last.delinquent': [delinquent],
						 'Number.of.Open.Accounts': [open_accounts],
						 'Number.of.Credit.Problems': [credit_problems],
						 'Current.Credit.Balance': [credit_balance],
						 'Maximum.Open.Credit': [open_credit], # SOLANO: ELIMINAR ESTE ITEM QUANDO ESTIVER TRABALHANDO COM O MODELO ATUALIZADO
						 'Bankruptcies': [bankruptcy], # SOLANO: ELIMINAR ESTE ITEM QUANDO ESTIVER TRABALHANDO COM O MODELO ATUALIZADO
						 'Tax.Liens': [tax_liens] # SOLANO: ELIMINAR ESTE ITEM QUANDO ESTIVER TRABALHANDO COM O MODELO ATUALIZADO
						 }) 

'#'

pipeline = joblib.load('model.joblib')

if st.button('Am I eligible?'):
    if (loan_amount != 0) and (term != '') and (credit_profile != '') and (years_job != '') and (annual_income != 0) and \
        (purpose != '') and (cred_history != '') and (delinquent != '') and (open_accounts != '') and (credit_problems != ''):
        result = pipeline.predict_proba(input_customer)[0]
        if result[1] >= 0.6277:
            '#'
            st.success('CONGRATULATIONS, YOUR CREDIT WAS APPROVED!!')
            st.balloons()
        elif result[1] <= 0.2713:
            '#'
            st.error('Unfortunately your credit was disapproved.')
        else:
            '#'
            st.warning('Thanks for your time. We will contact you soon!')
    else:
        '#'
        st.info('MISSING DATA: Please fill in all the fields above.')
