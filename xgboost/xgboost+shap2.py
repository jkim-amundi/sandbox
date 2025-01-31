# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import xlwings as xl
import scipy.stats as stats
import os
import plotly.graph_objects as go
import datetime
from dateutil.relativedelta import relativedelta
from xgboost import XGBClassifier, XGBRegressor
from xgboost import plot_importance, DMatrix
from sklearn.metrics import mean_squared_error

# import xgboost 
from matplotlib import pyplot
import shap
import xlwings as xl
# from __future__ import division
os.chdir(r'P:/Amundi_Milan/Investment/SPECIALISTS/QUANT_RESEARCH/ASSET_ALLOCATION/crossasset/Jung/cbr/')
# os.chdir(r'/Users/jhkmm/Documents/python_mbpro/TAA-Report.git/TAA-Report')

# %%
# macro = pd.read_csv('input-macro_govy1.csv',header=0, index_col=0,parse_dates=True, dayfirst=True).dropna()
macro = pd.read_csv('input-macro_lvl0.csv',header=0, index_col=0,parse_dates=True, dayfirst=True).dropna()
# macro = pd.read_csv('input-macro_all.csv',header=0, index_col=0,parse_dates=True, dayfirst=True).dropna()
# macro = pd.read_csv('input-macro_eq.csv',header=0, index_col=0,parse_dates=True, dayfirst=True).dropna()
# macro = pd.read_excel('input-xgboost.xlsx',index_col=0, sheet_name='macro')
assets = pd.read_excel('input-xgboost.xlsx', index_col=0, sheet_name='assets')
# data_m = pd.read_csv('input-macro_hy.csv', header=0, index_col=0,parse_dates=True, dayfirst=True).dropna()

# %%
# X_last = data.iloc[-1,:-1]
# y_last = data.iloc[-1,-1]
# data = data.iloc[:-1]
macro_yoy = macro.pct_change(periods=12).dropna()
macro_qoq = macro.pct_change(periods=3).dropna()
macro_mom = macro.iloc[:,:-2].pct_change(periods=1).dropna()
# 0: equity, 1: gov, 2: ig, 3: hy, 4: commo, 5: cash
assets_mom = assets.iloc[:,0].pct_change().dropna()
x_data = macro_yoy

# asset shift -1 to align returns and macro data
assets_mom_orig = assets_mom.copy()
assets_mom = assets_mom.shift(-1).dropna()
# y = pd.DataFrame(data.iloc[:,-1])

# %%
def calc_corr(macro_data: pd.DataFrame, asset_data: pd.DataFrame, asset_idx: int):  
    r1,c1 = macro_data.shape
    r2, = asset_data.shape
    macro_data=macro_data[-min(r1,r2):]
    asset_data=asset_data[-min(r1,r2):]
    corr_pearson = np.zeros((min(r1,r2)-11,c1))
    macro_val = macro_data.values        
    asset_val = asset_data.values
    for h in range(min(r1,r2)-11):
        for i in range(c1):
            corr_pearson[h,i] = stats.pearsonr(
                macro_val[:h+12,i],
                asset_val[:h+12]).correlation
    return pd.DataFrame(data=corr_pearson, index=asset_data.index[-(min(r1,r2)-11):], columns=macro_data.columns)
corrs = calc_corr(macro_data=x_data,asset_data=assets_mom_orig, asset_idx=0)
# we only need the LAST row (for now)
last_corr = corrs.iloc[-1,:]
# %%
# assign y to (single) asset

r1,c1 = x_data.shape
r2, = assets_mom.shape
x_data = x_data.iloc[-min(r1,r2):,:]
assets_mom = assets_mom.iloc[-min(r1,r2):]
y = assets_mom
# %%
# calculate percentile ranks of each feature
X = pd.DataFrame(index=x_data.index,columns=x_data.columns)
for j in range(c1):
    for i in range(r1):
        X.iloc[i,j] = stats.percentileofscore(x_data.iloc[:i+1,j],x_data.iloc[i,j])/100
X = X.apply(pd.to_numeric)
X.columns = X.columns.astype(str)
# %%
# multiple percentile ranks 
X = X.mul(last_corr)
X_test = X.iloc[-1:,:]
X = X.iloc[:-1,:]
y = y[1:]
X_diff = pd.DataFrame(data=np.abs(X.values - X_test.values), index=X.index)
# %%
# y = y.iloc[-r:]
model = XGBRegressor()
model.fit(X, y)

model_diff = XGBRegressor()
model_diff.fit(X_diff, y)
# plot feature importance
plot_importance(model,title=assets_mom.name+": XGBoost Feature Importance")
pyplot.show()

pred = model.predict(X)
rmse = mean_squared_error(y, pred)
print(f"RMSE: {rmse:.5f}")

plt.plot(y.index, y)
plt.plot(y.index, pred)
plt.title(f'{assets_mom.name}: RMSE {rmse:.5%}')
plt.legend(["Realized", "XGBoost"], loc="lower right")
next = model.predict(X_test)
print(f"Predicted for {assets_mom.name} {macro.index[-1]}: {next[-1]:.3%}")
f_importance = model.get_booster().get_score(importance_type='gain')
importance_df = pd.DataFrame.from_dict(data=f_importance,  orient='index')
importance_df.plot.bar()
# %%
# Xd = DMatrix(X, label=y)
# # pred = model.predict(Xd, output_margin=True)
# explainer = shap.TreeExplainer(model)
# explanation = explainer(Xd)

# shap_values = explanation.values
# shap.plots.beeswarm(explanation)
# make sure the SHAP values add up to marginal predictions
# np.abs(shap_values.sum(axis=1) + explanation.base_values - pred).max()
# %%
# X_test = pd.DataFrame(data.iloc[-1,:-2])
# y_test = pd.DataFrame(data.iloc[-1,:-1])
                   
explainer = shap.Explainer(model.predict, X)
# Calculates the SHAP values - It takes some time
shap_values = explainer(X)
# %%
shap.plots.bar(shap_values)
shap.plots.bar(shap_values[-2])

# %%
shap.plots.waterfall(shap_values[-1])
# shap.plots.waterfall(shap_values[pd.DataFrame(data.iloc[-1,:])])
# %%
shap.summary_plot(shap_values, X)
# %%
shap.plots.bar(shap_values[-1])
# %%

explainer = shap.Explainer(model_diff.predict, X_diff)
# Calculates the SHAP values - It takes some time
shap_values_diff = explainer(X_diff)
# %%
shap.plots.bar(shap_values_diff)
shap.plots.bar(shap_values_diff[-1])

# %%
shap.plots.waterfall(shap_values[-1])
# shap.plots.waterfall(shap_values[pd.DataFrame(data.iloc[-1,:])])
# %%
shap.summary_plot(shap_values, X)
# %%
shap.plots.bar(shap_values[-1])