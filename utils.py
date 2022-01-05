from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import pandas as pd

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def summarize_scores(scores):
    r2 = scores["test_r2"].mean()
    mae = np.abs(scores["test_neg_mean_absolute_error"]).mean()
    rmse = np.abs(scores["test_neg_root_mean_squared_error"]).mean()
    return r2, mae, rmse
    
def print_scores(scores):
    r2, mae, rmse = summarize_scores(scores)
    
    print('R2, MAE, RMSE')
    print(f'{r2:.2f}, {mae:.2f}, {rmse:.2f}')
    
def size_in_bytes(df):
    return df.values.nbytes + df.index.nbytes + df.columns.nbytes

def size_in_kilobytes(df):
    return size_in_bytes(df) / 1024

def size_in_megabytes(df):
    return size_in_bytes(df) / 1024 ** 2

def scores_from_model_collection_predictions(predictions):    
    models = predictions.columns.difference(['True Value'])
    model = []
    r2s = []
    maes = []
    rmses = []

    y_true = predictions['True Value']
    for model_name in models:
        y_pred = predictions[model_name]

        r2, rmse, mae= (r2_score(y_true, y_pred),
            mean_squared_error(y_true, y_pred, squared=False),
            mean_absolute_error(y_true, y_pred)
        )

        model.append(model_name)
        r2s.append(r2)
        maes.append(mae)
        rmses.append(rmse)

    scores = pd.DataFrame({'Model': models, 'R2': r2s, 'MAE': maes, 'RMSE': rmses}).set_index('Model')
    return scores

class GSheetConnector:
    def __init__(self, service_account_file, scopes = ['https://www.googleapis.com/auth/spreadsheets']):
        self.service_account_file = service_account_file
        self.scopes = scopes
        self.token = 'token.json'
        self.service = None
        self.creds = None
        
        self.__connect__()
    
    def get_sheets(self):
        return self.service.spreadsheets()
    
    def __connect__(self):
        # https://developers.google.com/identity/protocols/oauth2/service-account#authorizingrequests
        self.creds = service_account.Credentials.from_service_account_file(
            self.service_account_file, scopes=self.scopes)
        try:
            self.service = build('sheets', 'v4', credentials=self.creds)
        except HttpError as err:
            print(err)