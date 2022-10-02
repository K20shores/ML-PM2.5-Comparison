from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import pandas as pd

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
