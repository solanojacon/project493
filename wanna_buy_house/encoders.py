import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from wanna_buy_house.data import get_data, clean_data
from wanna_buy_house.utils import encode_total, final_transformer
from wanna_buy_house import utils

from imblearn.over_sampling import SMOTE


class InitialCleaning(BaseEstimator, TransformerMixin):
#Creation of a class to use in the Pipeline to clean original data

    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.X = X
        return clean_data(self.X)



class InitialEncoder(BaseEstimator, TransformerMixin):
#Creation of a framework class to use in the Pipeline to encode some features 
# and some feature engineering and return a confortable dataset to analyze

    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.X = X
        return encode_total(self.X)


class TransformEncoder(BaseEstimator, TransformerMixin):
#Creation of a framework class to use in the Pipeline to get dataset ready and
# comprehensible to machine learning algorithms

    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.X = X
        return final_transformer(self.X)


####################################
#####      True Encoders      ######
####################################


class OpenAccountEncoder(BaseEstimator, TransformerMixin):
#Creation of a class to use in the Pipeline to clean original data

    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        data = utils.encode_open_accounts(X)
        data['Number.of.Open.Accounts.Labeled'] = data['Number.of.Open.Accounts.Buckets'].map({'0-10': 0,
                                                                     '11-20': 1,
                                                                     'mais de 20': 2})
        return data[['Number.of.Open.Accounts.Labeled',]]


class CreditMinusEncoder(BaseEstimator, TransformerMixin):
#Creation of a class to use in the Pipeline to clean original data

    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        data = utils.credit_minus_loan(X)
        
        return data[['Credit.Minus.Loan',]]

class CreditProblemsEncoder(BaseEstimator, TransformerMixin):
#Creation of a class to use in the Pipeline to clean original data

    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        data = utils.encode_credit_problems(X)
        data['Number.of.Credit.Problems.Labeled'] = data['Number.of.Credit.Problems.Buckets'].map({0: 0,
                                                                                           1: 1,
                                                                                           'mais de 1': 2})
        return data[['Number.of.Credit.Problems.Labeled',]]

class YearsJobEncoder(BaseEstimator, TransformerMixin):
#Creation of a class to use in the Pipeline to clean original data

    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        data = utils.encode_years_job(X)
        
        return data[['Years.current_job_enc',]]


class PurposeEncoder(BaseEstimator, TransformerMixin):
#Creation of a class to use in the Pipeline to clean original data

    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        data = utils.encode_purpose(X)
        
        return data[['Purpose_enc',]]

class TaxLiensEncoder(BaseEstimator, TransformerMixin):
#Creation of a class to use in the Pipeline to clean original data

    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        data = utils.encode_tax_liens(X)
        
        return data[['Tax.Liens.Enc',]]

class LeverageEncoder(BaseEstimator, TransformerMixin):
#Creation of a class to use in the Pipeline to clean original data

    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        data = utils.encode_leverage(X)
        
        return data[['Leverage',]]

class CreditScoreEncoder(BaseEstimator, TransformerMixin):
#Creation of a class to use in the Pipeline to clean original data

    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        data = utils.encode_credit_score(X)
        data['Credit.Score.Labeled'] = data['Credit.Score.Enc'].map({'Below Average': 0,
                                                'Average': 1,
                                                'Good': 2})
        
        return data[['Credit.Score.Labeled',]]

class CreditHistoryEncoder(BaseEstimator, TransformerMixin):
#Creation of a class to use in the Pipeline to clean original data

    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        data = utils.encode_years_credit_history(X)
        data['Year.Credit.History.Labeled'] = data['Year.Credit.History.Enc'].map({'0 a 10': 0,
                                                        '10.1 a 20': 1,
                                                        '20.1 a 30': 2,
                                                        '30+': 3})
        
        return data[['Year.Credit.History.Labeled',]]

class BankruptciesEncoder(BaseEstimator, TransformerMixin):
#Creation of a class to use in the Pipeline to clean original data

    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        data = utils.encode_barkruptcies(X)
        
        return data[['Bankruptcies.enc',]]

class DelinquentEncoder(BaseEstimator, TransformerMixin):
#Creation of a class to use in the Pipeline to clean original data

    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        data = utils.encode_years_last_delinquent(X)
        
        return data[['Years.since.last.delinquent',]]

class LoanIncomeEncoder(BaseEstimator, TransformerMixin):
#Creation of a class to use in the Pipeline to clean original data

    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        data = utils.encode_loan_vs_income(X)
        
        return data[['Loan.vs.Income',]]

class SmotePiper(BaseEstimator, TransformerMixin):
#Creation of a class to use in the Pipeline to clean original data

    def __init__(self):
        self.X = None
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        sm = SMOTE()
        X, y = sm.fit_resample(X, y)

        
        #return data[['Loan.vs.Income',]]
#########################################
####### PIPELINE FUNCTION DRAFT #########
#########################################

def set_pipeline(**kwargs):

    # Define cleaning, initial encoding, feature engineering and transformation pipeline blocks here
    # Define the block sequences that will fill in the Pipeline
    feateng_steps = kwargs.get("feateng", ['open_accounts', "credit_minus", 'credit_problems', 'years_job',
                                           'purpose', 'tax_liens', 'leverage', 'credit_score', 'credit_history',
                                           'bankruptcies', 'delinquent', 'loan_vs_income', 'home_ownership', 'term'
                                           ])

    pipe_purpose = make_pipeline(PurposeEncoder(), OneHotEncoder())

    feateng_blocks = [('open_accounts', OpenAccountEncoder(), ['Number.of.Open.Accounts']),
                      ('credit_minus', CreditMinusEncoder(), ['Current.Credit.Balance', 'Current.Loan.Amount']),
                      ('credit_problems', CreditProblemsEncoder(), ['Number.of.Credit.Problems']),
                      ('years_job', YearsJobEncoder(), ['Years.in.current.job']),
                      ('purpose', pipe_purpose, ['Purpose']),
                      ('tax_liens', TaxLiensEncoder(), ['Tax.Liens']),
                      ('leverage', LeverageEncoder(), ['Monthly.Debt', 'Annual.Income']),
                      ('credit_score', CreditScoreEncoder(), ['Credit.Score']),
                      ('credit_history', CreditHistoryEncoder(), ['Years.of.Credit.History']),
                      ('bankruptcies', BankruptciesEncoder(), ['Bankruptcies']),
                      ('delinquent', DelinquentEncoder(), ['Months.since.last.delinquent']),
                      ('loan_vs_income', LoanIncomeEncoder(), ['Current.Loan.Amount', 'Annual.Income']),
                      ('home_ownership', OneHotEncoder(), ['Home.Ownership']),
                      ('term', OrdinalEncoder(categories=[['Short Term', 'Long Term']]), ['Term'])
                     ]

    new_blocks = []
                
    for bloc in feateng_blocks:
        if bloc[0] in feateng_steps:
            new_blocks.append(bloc)

    preprocessor = ColumnTransformer(new_blocks)
    
    blocks = [('feat_eng_bloc', preprocessor)]
    pipeline = Pipeline(blocks)
    
    return pipeline

if __name__ == "__main__":
    params = dict(nrows=None,              
                  local=True,
                  feateng=['credit_minus', 'credit_problems', 'years_job', 'leverage', 
                  'credit_score', 'credit_history', 'delinquent', 'loan_vs_income', 'term']
                  )

    data = get_data(**params)
    data = clean_data(data)
    pipe = set_pipeline(**params)
    data = pipe.fit_transform(data)
    

    