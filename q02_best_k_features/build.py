# %load q02_best_k_features/build.py
# Default imports

import pandas as pd
import numpy as np
data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_regression, mutual_info_regression
#k=20
# Write your solution here:
def percentile_k_features(df, k=20):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    select_percentile = SelectPercentile(f_regression, k)
    select_percentile.fit(X,y)
    dataframe = pd.DataFrame()
    dataframe['cols'] = X.columns
    dataframe['selected'] = select_percentile.get_support()
    dataframe['score'] = select_percentile.scores_
    dataframe.sort_values('score', inplace=True, ascending=False)
    return list(dataframe[dataframe['selected'] == True]['cols'])


percentile_k_features(data)




