###Normal Imports###
import multiprocessing
import time
import warnings
from termcolor import colored
from tempfile import mkdtemp
import pandas as pd
import category_encoders as ce
from psutil import virtual_memory

###Own Package importing###
from wanna_buy_house.data import get_data, clean_data
from wanna_buy_house.utils import simple_time_tracker, encode_open_accounts, credit_minus_loan, encode_credit_problems, custom_metric
from wanna_buy_house.encoders import *

###Mlflow importing###
import joblib
# import mlflow
# from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

### Models Importing ###
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from imblearn import pipeline as pipe_imb

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate



# Mlflow wagon server
MLFLOW_URI = "https://mlflow.lewagon.co"

class Trainer(object):
    # Mlflow parameters identifying the experiment, you can add all the parameters you wish
    ESTIMATOR = "Logistic"
    EXPERIMENT_NAME = "WannaBuyHouse"

    def __init__(self, X, y, **kwargs):
        
        self.pipeline = None
        self.kwargs = kwargs
        self.grid = kwargs.get("gridsearch", False)  # apply gridsearch if True
        self.local = kwargs.get("local", True)  # if True training is done locally

        self.mlflow = kwargs.get("mlflow", False)  # if True log info to nlflow
        self.upload = kwargs.get("upload", False)  # if True log info to nlflow
        
        self.experiment_name = kwargs.get("experiment_name", self.EXPERIMENT_NAME)  # cf doc above
        self.model_params = None  
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get("split", True)  # cf doc above
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                                  test_size=0.3, random_state=42)
        self.nrows = self.X_train.shape[0]  # nb of rows to train on
        
        #Investigate
        self.log_kwargs_params()
        self.log_machine_specs()

    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)

        if estimator == "Logistic":
            model = LogisticRegression(class_weight = 'balanced',solver= 'lbfgs', penalty= 'none')
        elif estimator == "KNN":
            model = KNeighborsClassifier()
        elif estimator == 'Forest':
            model = RandomForestClassifier(n_jobs=-1,
                                           max_features='log2',
                                           n_estimators=55,
                                           max_depth=32,
                                           min_samples_split=3,
                                           min_samples_leaf=3,
                                           random_state=13)
        else:
            model = LogisticRegression()
        
        estimator_params = self.kwargs.get("estimator_params", {})
        self.mlflow_log_param("estimator", estimator)
        model.set_params(**estimator_params)
        print(colored(model.__class__.__name__, "red"))
        
        return model

    def set_pipeline(self):

        # Define cleaning, initial encoding, feature engineering and transformation pipeline blocks here
        # Define the block sequences that will fill in the Pipeline

        feateng_steps = self.kwargs.get("feateng", ['open_accounts', "credit_minus", 'credit_problems', 'years_job',
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
        
        #pipe_smote = make_pipeline(preprocessor, SmotePiper())
        
        blocks = [('feat_eng_bloc', preprocessor),
                  ('smote', SMOTE()),
                  ('model', self.get_estimator())
                  ]

        self.pipeline = pipe_imb.Pipeline(blocks)
        

    def add_grid_search(self, custom_scorer=False):
        """"
        Apply Gridsearch on self.params defined in get_estimator
        {'rgs__n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
          'rgs__max_features' : ['auto', 'sqrt'],
          'rgs__max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        """
        # Here to apply random search to pipeline, need to follow naming "rgs__paramname"
        c_scorer = self.kwargs.get("custom_scorer", False)
        
        if c_scorer:
            self.custom_scorer = make_scorer(custom_metric)
        params = {"rgs__" + k: v for k, v in self.model_params.items()}
        #params = {'n_neighbors': range(125, 135), 'p': [1, 2]}
        self.pipeline = RandomizedSearchCV(estimator=self.pipeline, param_distributions=params,
                                           n_iter=10,
                                           cv=2,
                                           verbose=1,
                                           random_state=42,
                                           n_jobs=None,
                                           class_weight = 'balanced',
                                           scoring = self.custom_scorer
                                           )
                                     

    @simple_time_tracker
    def train(self, gridsearch=False):
        tic = time.time()
        self.set_pipeline()
        if gridsearch:
            self.add_grid_search()
        self.pipeline.fit(self.X_train, self.y_train)
        # mlflow logs
        self.mlflow_log_metric("train_time", int(time.time() - tic))

    def evaluate(self):
        rmse_train = self.compute_metric(self.X_train, self.y_train)
        self.mlflow_log_metric("rmse_train", rmse_train)
        if self.split:
            
            self.pipeline.score(self.X_val, self.y_val)

            y_pred = self.pipeline.predict(self.X_val)

            # cv = cross_validate(self.pipeline,
            #                           self.X_train,
            #                           self.y_train,
            #                           scoring='f1_weighted',
            #                           cv=10)

            print('RANDOM FOREST W/ FEATURE SELECTION & GRID SEARCH - CLASSIFICATION REPORT')

            print(classification_report(self.y_val, y_pred))
            #rmse_val = self.compute_metric(self.X_val, self.y_val, show=True)
            #self.mlflow_log_metric("rmse_val", rmse_val)
            #print(colored("rmse train: {} || rmse val: {}".format(rmse_train, rmse_val), "blue"))
        else:
            print(colored("rmse train: {}".format(rmse_train), "blue"))

    def compute_metric(self, X_test, y_test, show=False):
        """
        Compute a custom metric and return its rounded value. 

        """
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")

        y_pred = self.pipeline.predict(X_test)

        if show: 
            res = pd.DataFrame(y_test)
            res["pred"] = y_pred
            print(colored(res.sample(5), "blue"))

        rmse = custom_metric(y_true=y_test, y_pred=y_pred)
        return round(rmse, 3)

    def save_model(self):
        """Save the model into a .joblib and upload it on Google Storage /models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

        if self.upload:
            storage_upload(model_version=MODEL_VERSION)
    

    ######################        
    ### MLFlow methods ###
    ######################

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def log_estimator_params(self):
        reg = self.get_estimator()
        self.mlflow_log_param('estimator_name', reg.__class__.__name__)
        params = reg.get_params()
        for k, v in params.items():
            self.mlflow_log_param(k, v)

    def log_kwargs_params(self):
        if self.mlflow:
            for k, v in self.kwargs.items():
                self.mlflow_log_param(k, v)

    def log_machine_specs(self):
        cpus = multiprocessing.cpu_count()
        mem = virtual_memory()
        ram = int(mem.total / 1000000000)
        self.mlflow_log_param("ram", ram)
        self.mlflow_log_param("cpus", cpus)

###################
##  Unit Testing ##
###################

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get and clean data
    experiment = "taxifare_test_jean"
    params = dict(nrows=None,
                  upload=False,
                  local=True,  # set to False to get data from GCP (Storage or BigQuery)
                  gridsearch=False,
                  estimator="Forest",
                  mlflow=False,  # set to True to log params to mlflow
                  experiment_name=None,
                  split = True,
                  custom_scorer=False,
                  feateng=['credit_minus', 'credit_problems', 'years_job', 'leverage', 
                           'credit_score', 'credit_history', 'delinquent', 'loan_vs_income', 'term']
                  )
    
    
    print("############   Loading Data   ############")
    data = get_data(**params)
    data = clean_data(data)
    X_train = data.drop(columns='Loan.Status')
    y_train = data['Loan.Status']
    del data
    print("shape: {}".format(X_train.shape))
    print("size: {} Mb".format(X_train.memory_usage().sum() / 1e6))
    # Train and save model, locally and
    t = Trainer(X=X_train, y=y_train, **params)
    del X_train, y_train
    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()
    print(colored("############   Saving model    ############", "green"))
    t.save_model()
    print(colored("############ Predicting Random Entry ############", "yellow"))
    data = get_data(**params)
    data = clean_data(data)
    loaded_model = joblib.load('model.joblib')
    pred = data.head(5)
    y_pred = loaded_model.predict(pred)
    approval = []
    for ind, decision in enumerate(y_pred):
        if decision == 0:
            approval.append(f'{ind + 1} - No :(')
        else:
            approval.append(f'{ind + 1} - Yes :)') 
    for ap in approval:
        print(colored(f'Results: {ap}', "magenta"))

    
