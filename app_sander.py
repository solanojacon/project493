#Website file
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import base64

st.set_page_config(
            page_title="Gimme Some Credit", # => Quick reference - Streamlit
            page_icon="💰",
            layout="centered")

'# Gimme some Credit???!'

# @st.cache
# def load_image(path):
#     with open(path, 'rb') as f:
#         data = f.read()
#     encoded = base64.b64encode(data).decode()
#     return encoded

# def image_tag(path):
#     encoded = load_image(path)
#     tag = f'<img src="data:image/jpg;base64,{encoded}">'
#     return tag

# def background_image_style(path):
#     encoded = load_image(path)
#     style = f'''
#     <style>
#     body {{
#         background-image: url("data:image/jpg;base64,{encoded}");
#         background-size: cover;
#     }}
#     </style>
#     '''
#     return style

# image_path = 'images/money.jpg'
# image_link = 'https://docs.python.org/3/'

# st.write('*Hey*, click me I\'m a button!')

# st.write(f'<a href="{image_link}">{image_tag(image_path)}</a>', unsafe_allow_html=True)

# if st.checkbox('Show background image', False):
#     st.write(background_image_style(image_path), unsafe_allow_html=True)



# our_html = f'''
#     <p>Moneyyy</p>
#     <img src="images/money.jpg">
# '''

# st.write(our_html, unsafe_allow_html=True)

# '-----'

st.header('Are you good enough???')

image = Image.open('images/money.jpg')
st.image(image, use_column_width=True)

"## Fill out the form bellow, please ✍🏼 !!!"


loan_amount = st.number_input('Loan Amount')

term = st.selectbox('Loan Term',
				   ('','Short Term', 'Long Term'))

credit_profile = st.selectbox('Your Credit Score',
				   ('','Good (730 and above)', 'Average (703 - 729)', 'Below Average (702 and bellow)'))

if credit_profile == 'Good (730 and above)':
	credit_profile = 731
elif credit_profile == 'Average (703 - 729)':
	credit_profile = 710
elif credit_profile == 'Below Average (702 and bellow)':
	credit_profile = 650

years_job = st.selectbox('Years in Current Job',
				   ('','less than 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'))

home_ownership = st.selectbox('Home Ownership Status',
				   ('','Home Mortgage', 'Rent', 'Own Home'))

annual_income = st.number_input('Annual Income')

purpose = st.selectbox('Purpose of the Loan',
				   ('','Debt Consolidation', 'Home Improvements', 'Business Loan', 'Buy a Car', 'Medical Bills', 'Buy House', 'Other'))

month_debt = st.number_input('How much debt monthly?')


cred_history = st.selectbox('Years of Credit History',
				           ('','0 a 10', '10.1 a 20', '20.1 a 30', '30+'))

if cred_history == '0 a 10':
	cred_history = 5
elif cred_history == '10.1 a 20':
	cred_history = 15
elif cred_history == '20.1 a 30':
	cred_history = 25
elif cred_history == '30+':
	cred_history = 35

delinquent = st.selectbox('Delinquent in the past 3 years?',
				   ('','Yes', 'No'))
if delinquent == 'Yes':
	delinquent = 21
elif delinquent == 'No':
	delinquent = 42

open_accounts = st.selectbox('Number of open accounts',
				   ('','Less than 10', '10 - 20', 'More than 20'))

if open_accounts == 'Less than 10':
	open_accounts = 8
elif open_accounts == '10 - 20':
	open_accounts = 15
elif open_accounts == 'More than 20':
    open_accounts = 25

credit_problems = st.selectbox('Any Credit Problems?',
				   ('','No', 'Yes, 1', 'Yes, more than 1'))

if credit_problems == 'No':
	credit_problems = 0
elif credit_problems == 'Yes, 1':
	credit_problems = 1
elif credit_problems == 'Yes, more than 1':
	credit_problems = 2

credit_balance = st.number_input('Current Credit Balance') #Mantenho

open_credit = st.number_input('Maximum Open Credit') 

bankruptcy = st.selectbox('Have you ever declared bankruptcy?',
				   ('','No', 'Yes'))

if bankruptcy == 'No':
	bankruptcy = 0
elif bankruptcy == 'Yes':
	bankruptcy = 1

tax_liens = st.selectbox('Have you ever been imposed a Tax Lien?',
				   ('','No', 'Yes'))

if tax_liens == 'No':
	tax_liens = 0
elif tax_liens == 'Yes':
	tax_liens = 1

##############################################################

input_teste = pd.DataFrame({'Loan.ID': [0],
						 'Current.Loan.Amount': [17879],
						 'Term': ['Short Term'],
						 'Credit.Score': [739.0],
						 'Years.in.current.job': ['6 years'],
						 'Home.Ownership': ['Home Mortgage'],
						 'Annual.Income': [95357.0],
						 'Purpose': ['Debt Consolidation'],
						 'Monthly.Debt': [1509.82],
						 'Years.of.Credit.History': [34.4],
						 'Months.since.last.delinquent': [5.0],
						 'Number.of.Open.Accounts': [26],
						 'Number.of.Credit.Problems': [0],
						 'Current.Credit.Balance': [23986],
						 'Maximum.Open.Credit': [40313],
						 'Bankruptcies': [0.0],
						 'Tax.Liens': [0.0]})


# st.write(input_teste)


input_customer = pd.DataFrame({'Loan.ID': [0],
						 'Current.Loan.Amount': [loan_amount],
						 'Term': [term],
						 'Credit.Score': [credit_profile],
						 'Years.in.current.job': [years_job],
						 'Home.Ownership': [home_ownership],
						 'Annual.Income': [annual_income],
						 'Purpose': [purpose],
						 'Monthly.Debt': [month_debt],
						 'Years.of.Credit.History': [cred_history],
						 'Months.since.last.delinquent': [delinquent],
						 'Number.of.Open.Accounts': [open_accounts],
						 'Number.of.Credit.Problems': [credit_problems],
						 'Current.Credit.Balance': [credit_balance],
						 'Maximum.Open.Credit': [open_credit],
						 'Bankruptcies': [bankruptcy],
						 'Tax.Liens': [tax_liens]})

'### Double check your personal information'
st.write(input_customer)
if st.button('Copy to a CSV file'):
	input_customer.to_csv('socorro.csv', index=False)



# passar o input para o joblib para fazer a previsão
st.text("\n")
st.text("")
st.text("")
st.text(" ")
st.text(" ")
st.write(" ")

# precisamos carregar o joblib
pipeline = joblib.load('model.joblib')


if st.button('Am I eligible?'):
    result = pipeline.predict_proba(input_customer)[0]
    st.write(result)
    if result[1] <= 0.2713:
    	result = 'No :(. Maybe in the next Life!'
    	st.error('Let\'s keep positive, maybe in the next Life!')
    	st.baloons()

    elif result[1] >= 0.6277:
    	result = 'Yes :). Congratulations!'
    	st.success('OMGGGG!')
    else:
    	result = 'Thanks for your time. We will contact you soon 🥲!'
    	st.warning("You haven't been approved yet! You are neither awesome nor terrible ")
    st.write(f'### {result}')


# mostrar os resultados na tela