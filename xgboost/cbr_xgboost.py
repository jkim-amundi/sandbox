# %%
import pandas as pd
import numpy as np
import xlwings as xl
# from xgboost import XGBClassifier
# read data
# from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import xgboost as xgb
import sklearn.metrics as sk_metrics
import matplotlib.pyplot as plt
import shap

data_file = 'input-assets0.csv'
macro_file = 'input-macro.csv'
data_assets = pd.read_csv(data_file, header=0, index_col=0,parse_dates=True, dayfirst=True).dropna()
data_macro = pd.read_csv(macro_file, header=0, index_col=0,parse_dates=True, dayfirst=True).dropna()

r_assets, c_assets = data_assets.shape
r_macro, c_macro = data_macro.shape

r_diff = r_assets - r_macro
# in cases where we have a longer assets series than the macro
data_assets = data_assets.iloc[r_diff-1:]
returns = data_assets.pct_change().dropna()
# %%

X = data_macro[:-1]
y = returns.iloc[1:,0]
# %%
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=10, seed=123)

# %%
xg_reg.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_reg.predict(X_test)

# Compute the rmse: rmse
rmse = np.sqrt(sk_metrics.mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# %%
xgb.plot_tree(xg_reg, num_trees=0)


plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
xgb.plot_tree(xg_reg, fontsize=10)

plt.show()

# %%
import graphviz
import os
tree_dot = xgb.to_graphviz(xg_reg, num_trees=2)
dot_file_path = "xgboost_tree.dot"
tree_dot.save(dot_file_path)
# Convert dot file to png and display
with open(dot_file_path) as f:
    dot_graph = f.read()
# Use graphviz to display the tree
graph = graphviz.Source(dot_graph)
graph.render("xgboost_tree")
# Optionally, visualize the graph directly
graph
# %%
from dtreeviz import model
viz_model = model(
    xg_reg.get_booster(),
    X_train=X_train,
    y_train=y_train,
    feature_names=X.columns.to_list(),
    target_name='EQ',
    tree_index=2  # Visualizes the second tree (change this index for other trees)
)
viz_model.save()
# %%
from supertree import SuperTree
st = SuperTree(
    xg_reg, 
    X_train, 
    y_train, 
    X.columns.to_list(), 
    'EQ'
)
# Visualize the tree
st.show_tree(which_tree=2)
#save to html so you could simply open it in browser
#that way you will be able to expand tree, seedetail etc.
st.save_html() 
# %%
xgb.plot_importance(xg_reg, max_num_features=10) # top 10 most important features
plt.show()
# %%
shap_values = shap.TreeExplainer(xg_reg).shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
# %%
shap.summary_plot(shap_values, X_test)
# %%
shap.initjs()
explainer = shap.TreeExplainer(xg_reg)
shap_values = explainer.shap_values(X_test)
shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :])
# %%
top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))

# make SHAP plots of the three most important features
for i in range(1):
    shap.dependence_plot(top_inds[i], shap_values, X_train)
