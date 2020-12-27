import pandas as pd
from google.cloud import storage
from wanna_buy_house.utils import simple_time_tracker, encode_total


@simple_time_tracker
def get_data(nrows=None, local=True, **kwargs):
    """method to get the training data (or a portion of it) locally or from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    if local:
        path = "raw_data/train.csv" 
    else:
        # TODO
        # Create a google cloud storage path
        #path = "gs://{}/{}".format(BUCKET_NAME, BUCKET_TRAIN_DATA_PATH)
        pass

    data_original = pd.read_csv(path, nrows=nrows)
    
    return data_original

def clean_data(data_original):
    #TO-DO Create a docstring
    # (1) Deleting rows with Current.Loan.Amount equal to 99999999
    NUMBER_ERR = 99_999_999
    data = data_original[data_original['Current.Loan.Amount'] != NUMBER_ERR].copy()
    # (2) Deleting rows where Credit.Score and Annual.Income are NA
    data = data[data['Credit.Score'].notna()]
    data = data[data['Annual.Income'].notna()]
    # (3) Na coluna Home.Ownership, HaveMortgage substituido por Home Mortgage
    data['Home.Ownership'] = data['Home.Ownership'].replace(['HaveMortgage'], 'Home Mortgage')
    # (4) If Years.in.current.job is null, fill out with "less than 1 year"
    data['Years.in.current.job'] = data['Years.in.current.job'].fillna('less than 1 year')
    data['Years.in.current.job'] = data['Years.in.current.job'].replace(['less than  1 year'], 'less than 1 year')
    # (5) If Bankruptcies are null, substitute by 0
    data['Bankruptcies'] = data['Bankruptcies'].fillna(0)
    # (6) If Tax.Liens are null, substitute by 0
    data['Tax.Liens'] = data['Tax.Liens'].fillna(0)
    # (7) If Credit.Score > 850 (maximum), divide by 10 (assuming there was a typing error)
    #(e.g., instead of typing 750, person typed 7500) # Revisar pr numeros proximos a 850
    def div(x):
        if x > 850:
            return x / 10
        else:
            return x
    data = data[data['Current.Credit.Balance'] < 5_000_000]
    data['Credit.Score'] = data['Credit.Score'].apply(div)
    data = data[data['Annual.Income'] < 4_000_000]
    data = data[data['Current.Loan.Amount'] / data['Annual.Income'] < 1]
    
    return data

def final_treatment():
    data = get_data()
    data = clean_data(data)
    data = encode_total(data)
    
    return data

#Testing
if __name__ == "__main__":
    
    params = dict(nrows=None,              
                  local=True)  # set to False to get data from GCP (Storage or BigQuery)
    data_original = get_data(**params)
    data = clean_data(data_original)
    #data_final = encode_total(data)
    print('Everything OK so far')
