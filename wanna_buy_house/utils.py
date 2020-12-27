
import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

#####################################################
#  Initial encodings and feature engineering  #
#####################################################

def encode_open_accounts(data): #1
    #TODO --> Create a docstring
    #Number of open accounts: buckets (0-10, 11-20, mais de 20)
    data['Number.of.Open.Accounts.Buckets'] = data['Number.of.Open.Accounts']

    def encoding_noab(x):
        if x < 10:
            return '0-10'
        elif x > 20:
            return 'mais de 20'
        else:
            return '11-20'

    data['Number.of.Open.Accounts.Buckets'] = data['Number.of.Open.Accounts.Buckets'].apply(encoding_noab)
    data = data.drop(columns='Number.of.Open.Accounts')
    return data

def credit_minus_loan(data): #2
    #TODO --> Create a docstring
    #Calculating current credit balance minus current loan amount
    #if 'Credit.Minus.Loan'> 0, then, 1 (esta pedindo emprestimo menor do que tem de saldo disponivel, entao, ok); otherwise, 0

    data['Credit.Minus.Loan'] = data['Current.Credit.Balance'] - data['Current.Loan.Amount']

    def encoding_cml(x):
        if x > 0:
            return 1
        else:
            return 0

    data['Credit.Minus.Loan'] = data['Credit.Minus.Loan'].apply(encoding_cml).copy()
    return data

def encode_credit_problems(data): #3
    #TODO --> Create a docstring
    #Number of Number.of.Credit.Problems: 0, 1 ou mais de 1
    data['Number.of.Credit.Problems.Buckets'] = data['Number.of.Credit.Problems'].copy()

    def encoding_nocp(x):
        if x ==0:
            return 0
        elif x ==1:
            return 1
        else:
            return 'mais de 1'

    data['Number.of.Credit.Problems.Buckets'] = data['Number.of.Credit.Problems.Buckets'].apply(encoding_nocp)
    data = data.drop(columns='Number.of.Credit.Problems').copy()
    return data

def encode_years_job(data): #4
    #TODO --> Create a docstring
    data['Years.current_job_enc'] = data['Years.in.current.job'].map({'less than 1 year': 0,
                                                                             '1 year': 1,
                                                                             '2 years': 2,
                                                                             '3 years': 3,
                                                                             '4 years': 4,
                                                                             '5 years': 5,
                                                                             '6 years': 6,
                                                                             '7 years': 7,
                                                                             '8 years': 8,
                                                                             '9 years': 9,
                                                                             '10+ years': 10})
    data = data.drop(columns='Years.in.current.job').copy()
    return data

def encode_purpose(data): #5
    #TODO --> Create Purpose

    categories = ['other', 'moving', 'small_business', 'Take a Trip', 'major_purchase',
                   'wedding', 'Educational Expenses', 'vacation', 'renewable_energy']

    data['Purpose_enc'] = data['Purpose']

    for cat in categories:
        data.loc[data['Purpose_enc'] == cat, 'Purpose_enc'] = 'Other'

    data = data.drop(columns='Purpose').copy()
    return data

def encode_tax_liens(data): #6
    
    def encode_tl(x):
        if x == 0:
            x_enc = 0
        else:
            x_enc = 1
        return x_enc

    data['Tax.Liens.Enc'] = data['Tax.Liens'].map(encode_tl)
    data = data.drop(columns='Tax.Liens').copy()
    return data

def encode_leverage(data): #7
    
    data['Leverage'] = (data['Monthly.Debt'] * 12) / data['Annual.Income']
    data = data.drop(columns='Monthly.Debt').copy()
    return data

def encode_credit_score(data): #8
    
    def encode_cs(x):
        if x >= 730:
            x_enc = 'Good'
        elif x >= 703:
            x_enc = 'Average'
        else:
            x_enc = 'Below Average'
        return x_enc

    data['Credit.Score.Enc'] = data['Credit.Score'].map(encode_cs)
    data = data.drop(columns='Credit.Score').copy()
    return data

def encode_years_credit_history(data): #9
    
    data['Year.Credit.History.Enc'] = pd.cut(x=data['Years.of.Credit.History'],
                                  bins=[-1, 10, 20, 30, 100],
                                  labels=['0 a 10', '10.1 a 20', '20.1 a 30', '30+'])

    data = data.drop(columns='Years.of.Credit.History').copy()
    return data

def encode_barkruptcies(data): #10

    def encode_ban(x):
        if x == 0:
            x_enc = 0
        else:
            x_enc = 1
        return x_enc

    data['Bankruptcies.enc'] = data['Bankruptcies'].map(encode_ban)
    data = data.drop(columns='Bankruptcies').copy()
    return data

def encode_years_last_delinquent(data): #11
    
    data['Years.since.last.delinquent'] = data['Months.since.last.delinquent'] / 12
    
    data.loc[data['Years.since.last.delinquent'].isnull(), 'Years.since.last.delinquent'] = 100
    data['Years.since.last.delinquent'].isnull().sum()

    def encode_msld(x):
        if x <= 3:
            return 1
        else:
            return 0

    data['Years.since.last.delinquent'] = data['Years.since.last.delinquent'].map(encode_msld)
    data = data.drop(columns='Months.since.last.delinquent').copy()
    return data

def encode_loan_vs_income(data): #12
    # (12) Current loan amount / Annual income
    data['Loan.vs.Income'] = data['Current.Loan.Amount'] / data['Annual.Income']
    
    return data

def encode_total(data):

    data = encode_open_accounts(data)
    data = credit_minus_loan(data)
    data = encode_credit_problems(data)
    data = encode_years_job(data)
    data = encode_purpose(data)
    data = encode_tax_liens(data)
    data = encode_leverage(data)
    data = encode_credit_score(data)
    data = encode_years_credit_history(data)
    data = encode_barkruptcies(data)
    data = encode_years_last_delinquent(data)
    data = encode_loan_vs_income(data)
    # 2 - Dropping Annual Income e Current Loan Amount
    data = data.drop(columns=['Loan.ID', 'Maximum.Open.Credit', 'Current.Credit.Balance', 'Current.Loan.Amount', 'Annual.Income']).copy()
    
    return data

#####################################################
#  Transforming Data to Sklearn easier readability  #
#####################################################

# 1 - Deleting scaling_robust since we are dropping Annual Income Scaled e Current Loan Amount Scaled
# def scaling_robust(data):
#     # Instanciate Scaler
#     scaler = RobustScaler()
#     # Transform features
#     data['Current.Loan.Amount.Scaled'] = scaler.fit_transform(data[['Current.Loan.Amount']])
#     data['Annual.Income.Scaled'] = scaler.fit_transform(data[['Annual.Income']])
#     data = data.drop(columns=['Current.Loan.Amount', 'Annual.Income']).copy()
#     return data

def ordinal_encoder(data): #14
    oe_ter = OrdinalEncoder(categories=[['Short Term', 'Long Term']])
    data['Term.Encoded'] = oe_ter.fit_transform(data[['Term']])
    data = data.drop(columns='Term').copy()
    return data

def label_encoder(data):
    #1
    data['Number.of.Open.Accounts.Labeled'] = data['Number.of.Open.Accounts.Buckets'].map({'0-10': 0,
                                                                     '11-20': 1,
                                                                     'mais de 20': 2})
    #3
    data['Number.of.Credit.Problems.Labeled'] = data['Number.of.Credit.Problems.Buckets'].map({0: 0,
                                                                                           1: 1,
                                                                                           'mais de 1': 2})
    #8
    data['Credit.Score.Labeled'] = data['Credit.Score.Enc'].map({'Below Average': 0,
                                                'Average': 1,
                                                'Good': 2})
    #9
    data['Year.Credit.History.Labeled'] = data['Year.Credit.History.Enc'].map({'0 a 10': 0,
                                                        '10.1 a 20': 1,
                                                        '20.1 a 30': 2,
                                                        '30+': 3})

    data = data.drop(columns=['Number.of.Open.Accounts.Buckets', 'Number.of.Credit.Problems.Buckets', \
                             'Credit.Score.Enc', 'Year.Credit.History.Enc']).copy()

    return data


def one_hot_encoder(data):

    ohe_ho = OneHotEncoder(sparse = False) #13
    ho_encoded = ohe_ho.fit_transform(data[['Home.Ownership']])
    data['H.O.' + ohe_ho.categories_[0]] = ho_encoded

    ohe_pur = OneHotEncoder(sparse = False) #5
    pur_encoded = ohe_pur.fit_transform(data[['Purpose_enc']])
    data['Purp.' + ohe_pur.categories_[0]] = pur_encoded

    data = data.drop(columns=['Home.Ownership', 'Purpose_enc']).copy()
    return data

def final_transformer(data):
    """ Scale and Encode the whole dataset using functions previously defined"""
    #data = scaling_robust(data) ---> Deleting scaling robust because we dropped the columns Annual Income and Current Loan Amount...
    data = ordinal_encoder(data)
    data = label_encoder(data)
    data = one_hot_encoder(data)

    return data

###############
##  Metrics  ##
###############

def custom_metric(y_true, y_pred):
# Customized metric - recall0= TP0/(TP0+FN0)

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for real, pred in zip(y_true, y_pred):
        if pred == 0:
            if real == 0: 
                tp += 1 #acertei TP
            else:
                fp += 1 #errei FP
        else:
            if real == 1: 
                tn += 1 #acertei TN
            else: 
                fn += 1 #errei FN
    
    if (tp + fn) == 0:
        return 0
    else:
        #print(tp / (tp + fn))
        return tp / (tp + fn)

################
#  DECORATORS  #
################

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed