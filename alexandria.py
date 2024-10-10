# %%
import pandas as pd
import xlwings as xl

raw_sentiment = pd.read_csv('mini.csv', header='infer',sep='\t')
raw_sentiment['duplicated'] = raw_sentiment["StoryID"].duplicated()
# %%
raw_sentiment["datetime"]=pd.to_datetime(raw_sentiment["Timestamp"])