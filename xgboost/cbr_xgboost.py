# %%
import pandas as pd
import numpy as np
import xlwings as xl
import xgboost as xgb
xgb_regressor = xgb.XGBRegressor()

# %%

# read data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data_file = './xgboost/input-assets0.csv'
macro_file = './xgboost/input-macro.csv'
data_assets = pd.read_csv(data_file, header=0, index_col=0,parse_dates=True, dayfirst=True).dropna()
data_macro = pd.read_csv(macro_file, header=0, index_col=0,parse_dates=True, dayfirst=True).dropna()

r_assets, c_assets = data_assets.shape
r_macro, c_macro = data_macro.shape



# %%
returns = data_assets.pct_change().dropna()
X = returns[:-1]
y = returns[1:]
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
bst.fit(X_train, y_train)
preds = bst.predict(X_test)