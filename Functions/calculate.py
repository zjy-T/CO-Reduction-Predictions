import pandas as pd
import numpy as np
from tqdm.auto import  tqdm

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split

def ml_predict(dataframe, name, reg):
    dataframe['Gc-c_ML'] = np.nan
    for i, col in enumerate(dataframe['Gc-c_ML']):
        if pd.isnull(col):
            dataframe.iloc[i, -1] = reg.predict(pd.DataFrame(dataframe.iloc[i, 0:-2]).T)

    dataframe['Gc-c_Final'] = dataframe[name]
    for i, col in enumerate(dataframe['Gc-c_Final']):
        if pd.isnull(col):
            dataframe.iloc[i, -1] = reg.predict(pd.DataFrame(dataframe.iloc[i, 0:-3]).T)

def val_score(df, n, regressors, results_df):
    # Define the features and targets
    features = df.iloc[:, 0:-1]
    target = df.iloc[:, -1]

    # Obtain and save the cross_val_score for each regressor
    results = []
    fold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    for i in tqdm(regressors):
        temp = []

        mae = cross_val_score(i, features, target, cv=fold, scoring='neg_mean_absolute_error')
        temp.append(np.abs(mae).mean().round(4))

        rmse = cross_val_score(i, features, target, cv=fold, scoring='neg_root_mean_squared_error')
        temp.append(np.abs(rmse).mean().round(4))

        results.append(temp)

    results_df.loc[n] = results
