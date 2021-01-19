import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split

def ml_predict(dataframe, reg):
    dataframe['Gco_ML'] = np.nan
    for i, col in enumerate(dataframe['Gco_ML']):
        if pd.isnull(col):
            dataframe.iloc[i, -1] = reg.predict(pd.DataFrame(dataframe.iloc[i, 0:-2]).T)

    dataframe['Gco_Final'] = dataframe['Gco_DFT']
    for i, col in enumerate(dataframe['Gco_Final']):
        if pd.isnull(col):
            dataframe.iloc[i, -1] = reg.predict(pd.DataFrame(dataframe.iloc[i, 0:-3]).T)

def val_score(df, n, regressors, results_df):
    # Define the features and targets
    features = df.iloc[:, 0:-1]
    target = df.iloc[:, -1]

    # Obtain and save the cross_val_score for each regressor
    results = []
    fold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)

    for i in regressors:
        score = cross_val_score(i, features, target, cv=fold, scoring='neg_root_mean_squared_error')
        results.append(np.abs(score).mean())

    results_df.loc[n] = results
